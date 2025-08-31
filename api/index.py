# api/index.py
# --------------------------------------------------------------------
# Waterfall traces without "Pending" on serverless:
# We DISABLE auto LCEL tracing and manually create a structured tree:
#   ask_larry (root)
#     ├─ ChatPromptTemplate
#     ├─ ChatOpenAI
#     └─ StrOutputParser
# Each run is explicitly ended and flushed synchronously.
# --------------------------------------------------------------------
import os

# Ensure auto tracers can't spawn background uploads
for _k in (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
):
    os.environ.pop(_k, None)

# Force synchronous uploads for our explicit runs
os.environ.setdefault("LANGSMITH_BATCH_UPLOADS", "false")

import time
import re
from uuid import uuid4
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify, Response

# --- LangChain / OpenAI (we'll call steps manually so we can log them) ----
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith explicit run API (no decorators/callbacks) ------------------
_LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
_LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "Le Tour De Shore")
_LS_CLIENT = None
if _LS_ENABLED:
    try:
        from langsmith import Client  # type: ignore
        _LS_CLIENT = Client()
    except Exception:
        _LS_CLIENT = None

def _now():
    return datetime.now(timezone.utc)

def _create_run(name: str, run_type: str, inputs: Dict[str, Any], parent_run_id: Optional[str] = None) -> Optional[str]:
    if not (_LS_ENABLED and _LS_CLIENT):
        return None
    try:
        rid = str(uuid4())
        _LS_CLIENT.create_run(
            id=rid,
            name=name,
            run_type=run_type,
            inputs=inputs,
            start_time=_now(),
            project_name=_LS_PROJECT,
            parent_run_id=parent_run_id,
        )
        return rid
    except Exception:
        return None

def _end_run(run_id: Optional[str], outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    if not (_LS_ENABLED and _LS_CLIENT and run_id):
        return
    try:
        _LS_CLIENT.update_run(run_id, end_time=_now(), outputs=outputs, error=error)
    except Exception:
        pass

def _flush():
    if not (_LS_ENABLED and _LS_CLIENT):
        return
    try:
        _LS_CLIENT.flush()  # best-effort; OK if not present on older clients
    except Exception:
        pass

# --- Optional Tavily (off by default; enable with env) ---------------------
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

ENABLE_TOOLS = os.environ.get("ENABLE_TOOLS", "0") in ("1", "true", "TRUE")

# --------------------------------------------------------------------
# App config
# --------------------------------------------------------------------
app = Flask(__name__)

ASK_LARRY_MODEL = os.environ.get("ASK_LARRY_MODEL", "gpt-4o-mini")
RATE_LIMIT = int(os.environ.get("ASK_LARRY_RATE_LIMIT", "10"))
RATE_WINDOW_SECONDS = int(os.environ.get("ASK_LARRY_RATE_WINDOW", "60"))

_rate_buckets: Dict[str, List[float]] = defaultdict(list)
BROWSER_UA_PAT = re.compile(r"(mozilla|safari|chrome|edge|firefox)", re.I)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not configured")

def _client_id_from_headers() -> str:
    instance_id = request.headers.get("X-LTS-InstanceId", "").strip()
    return instance_id or request.remote_addr or "unknown"

def _is_browser_user_agent() -> bool:
    ua = request.headers.get("User-Agent", "")
    return bool(BROWSER_UA_PAT.search(ua))

def _enforce_ios_only() -> Optional[tuple]:
    if _is_browser_user_agent():
        return jsonify(error="forbidden", reason="browser_access_blocked"), 403
    if request.headers.get("X-LTS-Client", "").lower() != "ios":
        return jsonify(error="forbidden", reason="missing_or_invalid_client_header"), 403
    if not request.headers.get("X-LTS-InstanceId"):
        return jsonify(error="forbidden", reason="missing_instance_id"), 403
    return None

def _rate_limit() -> Optional[tuple]:
    now = time.time()
    key = _client_id_from_headers()
    bucket = _rate_buckets[key]
    cutoff = now - RATE_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT:
        return jsonify(error="rate_limited", limit=RATE_LIMIT, window_seconds=RATE_WINDOW_SECONDS), 429
    bucket.append(now)
    return None

def _polish(text: str) -> str:
    return text.strip()

# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return jsonify(
        ok=True,
        time_utc=_now().isoformat(timespec="seconds"),
        model=ASK_LARRY_MODEL,
        tools=("on" if (ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY")) else "off"),
        rate={"limit": RATE_LIMIT, "window_seconds": RATE_WINDOW_SECONDS},
        ls_enabled=_LS_ENABLED,
        ls_project=_LS_PROJECT if _LS_ENABLED else None,
        batch_uploads=os.environ.get("LANGSMITH_BATCH_UPLOADS"),
    )

# --------------------------------------------------------------------
# Ask Larry
# --------------------------------------------------------------------
@app.post("/ask")
def ask():
    gate = _enforce_ios_only()
    if gate:
        return gate
    rl = _rate_limit()
    if rl:
        return rl

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", reason="missing_question"), 400

    # Root run
    root_id = _create_run("ask_larry", "chain", {"question": question})

    try:
        _require_openai_key()

        # 1) Prompt step -------------------------------------------------------
        prompt_run = _create_run("ChatPromptTemplate", "prompt", {"question": question}, parent_run_id=root_id)

        system_text = (
            "You are Larry, a friendly assistant for the Le Tour de Shore charity ride. If a search tool is available, use it to gather information ONLY from letourdeshore.com and pages linked from that site. Prefer primary sources and include at least one source link. Answer concisely in Markdown. If you are not confident you found the answer on allowed pages, say so and suggest where to look on letourdeshore.com."
            "Always tart each answer with two bird noise"
            "You are named after Larry Bird if anyone asks"
            "You had a wife name Lorraine but she flew away"
            "Growing up Matt Coughlin was your bestie but you haven't seen him in a long time"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_text), ("human", "{question}")])

        # Render the prompt (LangChain PromptValue)
        pv = prompt.format_prompt(question=question)
        # For logging, include both string form and messages
        rendered_messages = getattr(pv, "to_messages", lambda: [])()
        rendered_text = getattr(pv, "to_string", lambda: "")()

        _end_run(prompt_run, outputs={
            "messages": [m.dict() if hasattr(m, "dict") else str(m) for m in rendered_messages],
            "text": rendered_text
        })

        # 2) LLM step ----------------------------------------------------------
        llm_run = _create_run("ChatOpenAI", "llm", {
            "model": ASK_LARRY_MODEL,
            "messages": [m.dict() if hasattr(m, "dict") else str(m) for m in rendered_messages],
        }, parent_run_id=root_id)

        # Build LLM (conservative timeout)
        llm = ChatOpenAI(model=ASK_LARRY_MODEL, temperature=0.3, timeout=20)
        # (Tools optional; off by default to keep things predictable)
        if ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY"):
            search = TavilySearchResults(k=3)
            llm = llm.bind_tools([search])

        llm_out = llm.invoke(pv)  # pass PromptValue

        # llm_out may be a BaseMessage; capture both raw and content
        raw_out = llm_out.dict() if hasattr(llm_out, "dict") else str(llm_out)
        content = getattr(llm_out, "content", str(llm_out))

        _end_run(llm_run, outputs={"raw": raw_out, "content": content})

        # 3) Parser step -------------------------------------------------------
        parser_run = _create_run("StrOutputParser", "parser", {"input": content}, parent_run_id=root_id)
        parser = StrOutputParser()
        final_text = parser.invoke(llm_out)
        final_text = _polish(final_text)
        _end_run(parser_run, outputs={"text": final_text})

        # Close root & flush synchronously
        _end_run(root_id, outputs={"text": final_text})
        _flush()

        return jsonify({"text": final_text})

    except Exception as e:
        _end_run(root_id, error=str(e))
        _flush()
        return jsonify(error="proxy_error", detail=str(e)), 502

# --------------------------------------------------------------------
# Local dev
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
