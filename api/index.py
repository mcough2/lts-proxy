# api/index.py
import os
import time
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify

# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith (tracing) ---
from langsmith import traceable
from langsmith.run_helpers import get_langchain_tracer  # <-- correct import

# --- Optional Tavily search (used only if key is present) ---
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Basic config
# --------------------------------------------------------------------------------------
app = Flask(__name__)

ASK_LARRY_MODEL = os.environ.get("ASK_LARRY_MODEL", "gpt-4o-mini")
RATE_LIMIT = int(os.environ.get("ASK_LARRY_RATE_LIMIT", "10"))
RATE_WINDOW_SECONDS = int(os.environ.get("ASK_LARRY_RATE_WINDOW", "60"))

USE_LANGSMITH = bool(os.environ.get("LANGSMITH_API_KEY"))
USE_TAVILY = bool(os.environ.get("TAVILY_API_KEY")) and TAVILY_AVAILABLE

# Simple in-memory rate limiter (reset on cold start, fine for serverless)
_rate_buckets: Dict[str, list[float]] = defaultdict(list)

BROWSER_UA_PAT = re.compile(r"(mozilla|safari|chrome|edge|firefox)", re.I)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not configured")

def _client_id_from_headers() -> str:
    # Required headers from the iOS app
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
    Build the LCEL chain. If Tavily is configured, give the model a quick search tool.
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

    llm = ChatOpenAI(model=ASK_LARRY_MODEL, temperature=0.3)

    if USE_TAVILY:
        # Lightweight, top-3 web search when the model asks for it
        search = TavilySearchResults(k=3)
        # Bind tool for function-calling models; model will decide when to call
        llm = llm.bind_tools([search])

    chain = prompt | llm | StrOutputParser()
    return chain

def _polish(text: str) -> str:
    # Tiny post-processor to keep things tidy for the chat bubble
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
        tools=("tavily" if USE_TAVILY else "none"),
        rate={"limit": RATE_LIMIT, "window_seconds": RATE_WINDOW_SECONDS},
        tracing=("on" if USE_LANGSMITH else "off"),
    )

# --------------------------------------------------------------------------------------
# Ask Larry
# --------------------------------------------------------------------------------------
@traceable(name="ask_larry_handler", run_type="chain")
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

        # Build chain once per invocation (fast enough for serverless)
        chain = _build_chain()

        # LangSmith tracer is a function, not a Client method
        tracer = get_langchain_tracer() if USE_LANGSMITH else None
        config = {"callbacks": [tracer]} if tracer else None

        text = chain.invoke({"question": question}, config=config)
        return jsonify({"text": _polish(text)})

    except Exception as e:
        app.logger.exception("ask failed")
        return jsonify(error="proxy_error", detail=str(e)), 502

# --------------------------------------------------------------------------------------
# Flask entry (Vercel discovers app automatically)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Local dev only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
