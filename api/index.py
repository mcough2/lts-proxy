# api/index.py
# --- Tracing mode: explicit only (no auto LCEL tracing) ---
import os

# Kill auto LCEL tracing env if present so it can't create background runs
for _k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"):
    if _k in os.environ:
        os.environ.pop(_k, None)

# Force synchronous uploads for our explicit RunTree
os.environ.setdefault("LANGSMITH_BATCH_UPLOADS", "false")

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

# --- LangSmith: explicit RunTree lifecycle (no callbacks/decorators) ----------
_LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
_LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "Le Tour De Shore")
_LS_CLIENT = None
if _LS_ENABLED:
    try:
        from langsmith import Client  # type: ignore
        from langsmith.run_trees import RunTree  # type: ignore
        _LS_CLIENT = Client()
    except Exception:
        _LS_CLIENT = None
        RunTree = None  # type: ignore


def _start_run_tree(name: str, inputs: Dict[str, Any]):
    """Create a root RunTree if LangSmith is configured; otherwise None."""
    if not (_LS_ENABLED and _LS_CLIENT and 'RunTree' in globals()):
        return None
    try:
        return RunTree(name=name, run_type="chain", inputs=inputs, project_name=_LS_PROJECT)  # type: ignore
    except Exception:
        return None


def _upload_run_tree(run):
    """Upload the finished RunTree synchronously."""
    if not (_LS_ENABLED and _LS_CLIENT and run is not None):
        return
    try:
        _LS_CLIENT.create_run_tree(run)  # type: ignore
        # Ensure it leaves the process
        _LS_CLIENT.flush()  # type: ignore[attr-defined]
    except Exception:
        pass


# --- Optional Tavily (tooling disabled by default; enable via env) ----------
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
    """Build LCEL chain. Tools off by default; enable with ENABLE_TOOLS=1."""
    system = (
        "You are Larry, the friendly ride assistant for the Le Tour De Shore iOS app. "
        "Answer clearly in concise Markdown. If you're unsure, say so briefly."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]
    )

    # Conservative timeout to avoid hangs in serverless
    llm = ChatOpenAI(model=ASK_LARRY_MODEL, temperature=0.3, timeout=20)

    if ENABLE_TOOLS and TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY"):
        search = TavilySearchResults(k=3)
        llm = llm.bind_tools([search])

    return prompt | llm | StrOutputParser()

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
        batch_uploads=os.environ.get("LANGSMITH_BATCH_UPLOADS"),
        project=_LS_PROJECT if _LS_ENABLED else None,
    )

# --------------------------------------------------------------------------------------
# Ask Larry
# --------------------------------------------------------------------------------------
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

    run = _start_run_tree("ask_larry", {"question": question})

    try:
        _require_openai_key()
        chain = _build_chain()

        # Invoke model
        text = chain.invoke({"question": question})
        text = _polish(text)

        # Record outputs on the run (if enabled)
        if run is not None:
            run.add_outputs({"text": text})
            run.end()

        resp: Response = jsonify({"text": text})
        return resp

    except Exception as e:
        # Close the run with error (if enabled)
        if run is not None:
            try:
                run.end(error=str(e))
            except Exception:
                pass
        app.logger.exception("ask failed")
        return jsonify(error="proxy_error", detail=str(e)), 502

    finally:
        # Synchronous upload so the run never stays 'Pending'
        _upload_run_tree(run)

# --------------------------------------------------------------------------------------
# Local dev entry
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
