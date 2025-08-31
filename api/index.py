# api/index.py
import os
import time
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify, Response

# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith: version-agnostic tracer + explicit flush ---------------------
_LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
_LS_CLIENT = None  # lazily created only if enabled


def _make_tracer():
    """Create a LangSmith tracer across client versions, or return None."""
    if not _LS_ENABLED:
        return None
    global _LS_CLIENT
    # Make a client once (also gives us .flush()).
    if _LS_CLIENT is None:
        try:
            from langsmith import Client  # type: ignore
            _LS_CLIENT = Client()
        except Exception:
            _LS_CLIENT = None

    # Try modern helper first
    try:
        from langsmith.run_helpers import get_langchain_tracer  # type: ignore
        return get_langchain_tracer()
    except Exception:
        pass

    # Fallback to older LangChain tracer
    try:
        from langchain.callbacks.tracers import LangChainTracer  # type: ignore
        tracer = LangChainTracer()
        try:
            tracer.load_default_session()  # older API; ignore if absent
        except Exception:
            pass
        return tracer
    except Exception:
        return None


def _flush_traces_safely():
    """Ensure traces are uploaded before the process exits."""
    if _LS_CLIENT is None:
        return
    try:
        # Newer clients expose flush(); older may no-op or raise—ignore errors.
        _LS_CLIENT.flush()  # type: ignore[attr-defined]
    except Exception:
        pass


# --- Optional Tavily search (bind tools only if enabled) ---------------------
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

ENABLE_TOOLS = os.environ.get("ENABLE_TOOLS", "0") in ("1", "true", "TRUE")

# --------------------------------------------------------------------------------------
# Basic config
# --------------------------------------------------------------------------------------
app = Flask(__name__)

ASK_LARRY_MODEL = os.environ.get("ASK_LARRY_MODEL", "gpt-4o-mini")
RATE_LIMIT = int(os.environ.get("ASK_LARRY_RATE_LIMIT", "10"))
RATE_WINDOW_SECONDS = int(os.environ.get("ASK_LARRY_RATE_WINDOW", "60"))

# Simple in-memory rate limiter (reset on cold start, fine for serverless)
_rate_buckets: Dict[str, List[float]] = defaultdict(list)

BROWSER_UA_PAT = re.compile(r"(mozilla|safari|chrome|edge|firefox)", re.I)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
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
    # drop old events
    cutoff = now - RATE_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT:
        return (
            jsonify(
                error="rate_limited",
                limit=RATE_LIMIT,
                window_seconds=RATE_WINDOW_SECONDS,
            ),
            429,
        )
    bucket.append(now)
    return None

def _build_chain():
    """
    Build the LCEL chain. Tools are off by default; enable with ENABLE_TOOLS=1.
    """
    system = (
        "You are Larry, the friendly ride assistant for the Le Tour De Shore iOS app. "
        "Answer clearly in concise Markdown. If you're unsure, say so briefly."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Add a conservative timeout to avoid hanging runs in serverless.
    llm = ChatOpenAI(model=ASK_LARRY_MODEL, temperature=0.3, timeout=20)

    if ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY"):
        search = TavilySearchResults(k=3)
        llm = llm.bind_tools([search])

    chain = prompt | llm | StrOutputParser()
    return chain

def _polish(text: str) -> str:
    return text.strip()

# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return jsonify(
        ok=True,
        time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=ASK_LARRY_MODEL,
        tools=("on" if (ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY")) else "off"),
        rate={"limit": RATE_LIMIT, "window_seconds": RATE_WINDOW_SECONDS},
        tracing=("on" if _LS_ENABLED else "off"),
    )

# --------------------------------------------------------------------------------------
# Ask Larry
# --------------------------------------------------------------------------------------
@app.post("/ask")
def ask():
    # Gate & rate-limit first
    gate = _enforce_ios_only()
    if gate:
        return gate
    rl = _rate_limit()
    if rl:
        return rl

    # Validate payload
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", reason="missing_question"), 400

    try:
        _require_openai_key()

        chain = _build_chain()

        tracer = _make_tracer()
        config = {"callbacks": [tracer]} if tracer else None

        text = chain.invoke({"question": question}, config=config)
        resp: Response = jsonify({"text": _polish(text)})

        # IMPORTANT: flush traces before returning in serverless
        _flush_traces_safely()
        return resp

    except Exception as e:
        app.logger.exception("ask failed")
        resp_err: Response = jsonify(error="proxy_error", detail=str(e)), 502
        _flush_traces_safely()
        return resp_err

# --------------------------------------------------------------------------------------
# Flask entry (local dev)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
