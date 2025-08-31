# api/index.py
# --------------------------------------------------------------------
# Tracing: we WANT traces, but we do NOT want auto-LCEL background
# uploads that get stuck on serverless. So we disable auto tracing and
# create/finish a run explicitly, then flush synchronously.
# --------------------------------------------------------------------
import os

# Kill auto LCEL tracing so it can't spawn background runs that hang.
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

# ---- LangChain / OpenAI (the actual LLM call) ----------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---- LangSmith (explicit runs; no decorators/callbacks) ------------
_LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
_LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "Le Tour De Shore")

_LS_CLIENT = None
if _LS_ENABLED:
    try:
        from langsmith import Client  # type: ignore
        _LS_CLIENT = Client()
    except Exception:
        _LS_CLIENT = None


def _start_run(name: str, inputs: Dict[str, Any]) -> Optional[str]:
    """Create a LangSmith run via low-level API; return run_id or None."""
    if not (_LS_ENABLED and _LS_CLIENT):
        return None
    try:
        run_id = str(uuid4())
        _LS_CLIENT.create_run(
            id=run_id,
            name=name,
            inputs=inputs,
            run_type="chain",
            start_time=datetime.now(timezone.utc),
            project_name=_LS_PROJECT,
        )
        return run_id
    except Exception:
        return None


def _finish_run(run_id: Optional[str], outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """Finish the run and flush synchronously so serverless doesn't freeze uploads."""
    if not (_LS_ENABLED and _LS_CLIENT and run_id):
        return
    try:
        _LS_CLIENT.update_run(
            run_id,
            end_time=datetime.now(timezone.utc),
            outputs=outputs,
            error=error,
        )
        try:
            _LS_CLIENT.flush()  # may be a no-op on older clients; safe to try
        except Exception:
            pass
    except Exception:
        pass


# ---- Optional Tavily tools (off by default; enable with env) -------
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

# simple in-memory limiter (fine for serverless cold starts)
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
        return jsonify(error="rate_limited", limit=RATE_LIMIT, window_seconds=RATE_WINDOW_SECONDS), 429
    bucket.append(now)
    return None

def _build_chain():
    """LCEL chain. Tools off by default; enable with ENABLE_TOOLS=1 + TAVILY_API_KEY."""
    system = (
        "You are Larry, the friendly ride assistant for the Le Tour De Shore iOS app. "
        "Answer clearly in concise Markdown. If you're unsure, say so briefly."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

    # Add a conservative timeout so runs don't hang in serverless
    llm = ChatOpenAI(model=ASK_LARRY_MODEL, temperature=0.3, timeout=20)

    if ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY"):
        search = TavilySearchResults(k=3)
        llm = llm.bind_tools([search])

    return prompt | llm | StrOutputParser()

def _polish(text: str) -> str:
    return text.strip()


# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    flags = {
        "LANGSMITH_BATCH_UPLOADS": os.environ.get("LANGSMITH_BATCH_UPLOADS"),
        # these should be None/absent because we disabled auto tracing:
        "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_TRACING": os.environ.get("LANGCHAIN_TRACING"),
    }
    return jsonify(
        ok=True,
        time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=ASK_LARRY_MODEL,
        tools=("on" if (ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY")) else "off"),
        rate={"limit": RATE_LIMIT, "window_seconds": RATE_WINDOW_SECONDS},
        ls_enabled=_LS_ENABLED,
        ls_project=_LS_PROJECT if _LS_ENABLED else None,
        tracing_env=flags,
    )


# --------------------------------------------------------------------
# Ask Larry (main endpoint)
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

    # Start an explicit LangSmith run
    run_id = _start_run("ask_larry", {"question": question})

    try:
        _require_openai_key()

        chain = _build_chain()
        text = _polish(chain.invoke({"question": question}))

        # Finish + flush BEFORE returning (synchronous)
        _finish_run(run_id, outputs={"text": text})

        return jsonify({"text": text})

    except Exception as e:
        _finish_run(run_id, error=str(e))
        return jsonify(error="proxy_error", detail=str(e)), 502


# --------------------------------------------------------------------
# Local dev
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
