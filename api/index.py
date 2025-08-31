# api/index.py
# -----------------------------------------------------------------------------
# Serverless-safe tracing with a manual waterfall:
#   ask_larry (root)
#     ├─ ChatPromptTemplate
#     ├─ ChatOpenAI
#     └─ StrOutputParser
# We disable auto/LCEL background tracing and create/finish runs explicitly.
# -----------------------------------------------------------------------------

import os

# Disable auto LCEL/background tracing that can hang on serverless.
for _k in (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
):
    os.environ.pop(_k, None)

# Force synchronous uploads for explicit runs.
os.environ.setdefault("LANGSMITH_BATCH_UPLOADS", "false")

import time
import re
from uuid import uuid4
from urllib.parse import quote
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify, Response

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangSmith client (explicit run API)
_LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
_LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "Le Tour De Shore")
_LS_CLIENT = None
if _LS_ENABLED:
    try:
        from langsmith import Client  # type: ignore
        _LS_CLIENT = Client()  # API key from env
    except Exception:
        _LS_CLIENT = None

# Optional Tavily tool (kept off by default)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ASK_LARRY_MODEL = os.environ.get("ASK_LARRY_MODEL", "gpt-4o-mini")
RATE_LIMIT = int(os.environ.get("ASK_LARRY_RATE_LIMIT", "10"))
RATE_WINDOW_SECONDS = int(os.environ.get("ASK_LARRY_RATE_WINDOW", "60"))

MAX_QUESTION_CHARS = 800
LLM_TIMEOUT_SECS = 15
LLM_MAX_RETRIES = 2
ENABLE_TOOLS = os.environ.get("ENABLE_TOOLS", "0").lower() in ("1", "true")
APP_VERSION = os.environ.get("LTS_PROXY_VERSION", "1.0.0")

# -----------------------------------------------------------------------------
# App + State
# -----------------------------------------------------------------------------
app = Flask(__name__)

# simple in-memory limiter (fine for serverless cold starts)
_rate_buckets: Dict[str, List[float]] = defaultdict(list)
BROWSER_UA_PAT = re.compile(r"(mozilla|safari|chrome|edge|firefox)", re.I)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc)

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
    # Block obvious browsers & require our app headers
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
        return jsonify(
            error="rate_limited",
            limit=RATE_LIMIT,
            window_seconds=RATE_WINDOW_SECONDS,
        ), 429
    bucket.append(now)
    return None

def _polish(text: str) -> str:
    return text.strip()

# -----------------------------------------------------------------------------
# LangSmith explicit run helpers
# -----------------------------------------------------------------------------
def _create_run(
    name: str,
    run_type: str,
    inputs: Dict[str, Any],
    parent_run_id: Optional[str] = None,
) -> Optional[str]:
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
        _LS_CLIENT.flush()  # Ensure uploads complete before serverless freeze
    except Exception:
        pass

def _trace_url(run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    try:
        proj = quote(_LS_PROJECT)
        return f"https://smith.langchain.com/o/r?r={run_id}&p={proj}"
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Health / Version
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return jsonify(
        ok=True,
        time_utc=_now().isoformat(timespec="seconds"),
        model=ASK_LARRY_MODEL,
        tools=("on" if (ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY")) else "off"),
        rate={"limit": RATE_LIMIT, "window_seconds": RATE_WINDOW_SECONDS},
        tracing={"enabled": _LS_ENABLED, "project": _LS_PROJECT if _LS_ENABLED else None,
                 "batch_uploads": os.environ.get("LANGSMITH_BATCH_UPLOADS")},
        version=APP_VERSION,
    )

@app.get("/version")
def version():
    return jsonify(
        name="lts-proxy",
        version=APP_VERSION,
        model=ASK_LARRY_MODEL,
        tracing=bool(os.environ.get("LANGSMITH_API_KEY")),
    )

# -----------------------------------------------------------------------------
# Ask Larry (main endpoint)
# -----------------------------------------------------------------------------
@app.post("/ask")
def ask():
    # Gate / rate limit
    gate = _enforce_ios_only()
    if gate:
        return gate
    rl = _rate_limit()
    if rl:
        return rl

    # Payload
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", reason="missing_question"), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify(error="bad_request", reason=f"question_too_long (>{MAX_QUESTION_CHARS})"), 400

    # Root run
    root_id = _create_run("ask_larry", "chain", {"question": question})

    try:
        _require_openai_key()

        # ---------------- 1) Prompt step ----------------
        prompt_run = _create_run("ChatPromptTemplate", "prompt", {"question": question}, parent_run_id=root_id)
        system_text = ( "You are Larry, a friendly assistant for the Le Tour de Shore charity ride. If a search tool is available, use it to gather information ONLY from letourdeshore.com and pages linked from that site. Prefer primary sources and include at least one source link. Answer concisely in Markdown. If you are not confident you found the answer on allowed pages, say so and suggest where to look on letourdeshore.com." "Always tart each answer with two bird noise" "You are named after Larry Bird if anyone asks" "You had a wife name Lorraine but she flew away" "Growing up Matt Coughlin was your bestie but you haven't seen him in a long time" )
        prompt = ChatPromptTemplate.from_messages([("system", system_text), ("human", "{question}")])

        # Render PromptValue (messages + string)
        pv = prompt.format_prompt(question=question)
        rendered_messages = getattr(pv, "to_messages", lambda: [])()
        rendered_text = getattr(pv, "to_string", lambda: "")()

        _end_run(prompt_run, outputs={
            "messages": [m.dict() if hasattr(m, "dict") else str(m) for m in rendered_messages],
            "text": rendered_text
        })

        # ---------------- 2) LLM step -------------------
        llm_inputs = {
            "model": ASK_LARRY_MODEL,
            "messages": [m.dict() if hasattr(m, "dict") else str(m) for m in rendered_messages],
        }
        llm_run = _create_run("ChatOpenAI", "llm", llm_inputs, parent_run_id=root_id)

        llm = ChatOpenAI(
            model=ASK_LARRY_MODEL,
            temperature=0.3,
            timeout=LLM_TIMEOUT_SECS,
            max_retries=LLM_MAX_RETRIES,
        )

        if ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY"):
            search = TavilySearchResults(k=3)
            llm = llm.bind_tools([search])

        llm_out = llm.invoke(pv)  # pass PromptValue
        raw_out = llm_out.dict() if hasattr(llm_out, "dict") else str(llm_out)
        content = getattr(llm_out, "content", str(llm_out))

        _end_run(llm_run, outputs={"raw": raw_out, "content": content})

        # ---------------- 3) Parser step ----------------
        parser_run = _create_run("StrOutputParser", "parser", {"input": content}, parent_run_id=root_id)
        parser = StrOutputParser()
        final_text = _polish(parser.invoke(llm_out))
        _end_run(parser_run, outputs={"text": final_text})

        # Close root + flush synchronously
        _end_run(root_id, outputs={"text": final_text})
        _flush()

        return jsonify({
            "text": final_text,
            "request_id": root_id or str(uuid4()),
            "trace_url": _trace_url(root_id),
            "model": ASK_LARRY_MODEL,
        })

    except Exception as e:
        _end_run(root_id, error=str(e))
        _flush()
        return jsonify(error="proxy_error", detail=str(e)), 502

# -----------------------------------------------------------------------------
# Local dev entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
