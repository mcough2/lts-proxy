import os, json, re, time
from flask import Flask, request, jsonify
import requests
from collections import defaultdict, deque

# ---- LangSmith (optional) ----
LS_ENABLED = bool(
    os.environ.get("LANGSMITH_API_KEY")
    or os.environ.get("LANGCHAIN_API_KEY")  # back-compat
)
LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "LeTourDeShore")

if LS_ENABLED:
    try:
        # RunTree gives us explicit control (start/end, metadata, tags)
        from langsmith.run_trees import RunTree
    except Exception:  # fail open if package missing
        LS_ENABLED = False

app = Flask(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.
Use web search to gather information ONLY from {DOMAIN} and pages linked from those sites.
Prefer primary sources, and include at least one source link in every answer.
Answer in concise Markdown."""

# ---- Simple in-memory rate limiting (resets per cold start) ----
RATE_LIMIT = 10         # requests
RATE_WINDOW = 60        # seconds
requests_log = defaultdict(lambda: deque())

def is_rate_limited(key: str) -> bool:
    now = time.time()
    q = requests_log[key]
    while q and q[0] <= now - RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False

# ---- Extract text from OpenAI response ----
def parse_responses_text(payload: dict) -> str:
    out_parts = []
    for item in payload.get("output", []) or []:
        msg = item.get("message", {})
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            text = "\n".join(
                [c.get("text", "") for c in msg["content"] if isinstance(c, dict) and c.get("text")]
            ).strip()
            if text:
                out_parts.append(text)
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("text"):
                out_parts.append(c["text"].strip())
    if out_parts:
        return "\n\n".join([p for p in out_parts if p])
    return payload.get("output_text") or payload.get("text", {}).get("value") or ""

def start_trace(question: str, meta: dict):
    """Start a LangSmith run (returns RunTree or None)."""
    if not LS_ENABLED:
        return None
    try:
        run = RunTree(
            name="ask_larry",
            project_name=LS_PROJECT,
            inputs={"question": question},
            tags=["ask-larry", "proxy"],
            metadata=meta,
        )
        run.post()  # create the run in LangSmith
        return run
    except Exception:
        return None

def end_trace(run, *, outputs=None, error=None, extra_meta=None):
    if run is None:
        return
    try:
        if extra_meta:
            run.update(metadata={**(run.metadata or {}), **extra_meta})
        if error:
            run.end(error=error)
        else:
            run.end(outputs=outputs or {})
    except Exception:
        pass

# ---- Main ask endpoint ----
@app.post("/ask")
def ask():
    t_req_start = time.time()

    if not OPENAI_KEY:
        return jsonify(error="server_not_configured", detail="Missing OPENAI_API_KEY"), 500

    # --- Basic client gating ---
    client = request.headers.get("X-LTS-Client", "").lower()
    instance_id = request.headers.get("X-LTS-InstanceId", "").strip()
    ua = (request.headers.get("User-Agent") or "").lower()
    ip = (request.headers.get("x-forwarded-for") or request.remote_addr or "").split(",")[0].strip()

    if client != "ios" or not instance_id:
        return jsonify(error="forbidden", detail="Missing or invalid client headers"), 403
    if "mozilla" in ua or "safari" in ua or "chrome" in ua:
        return jsonify(error="forbidden", detail="Browser access blocked"), 403

    # --- Rate limiting ---
    if is_rate_limited(instance_id):
        return jsonify(error="rate_limited",
                       detail=f"Exceeded {RATE_LIMIT} requests per {RATE_WINDOW}s"), 429

    # --- Body validation ---
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", detail="Provide JSON {\"question\":\"...\"}"), 400

    # ---- LangSmith start ----
    ls_meta = {
        "client": client,
        "instance_id": instance_id,
        "user_agent": ua[:200],
        "ip": ip,
        "rate_limit": {"limit": RATE_LIMIT, "window_s": RATE_WINDOW},
    }
    run = start_trace(question, ls_meta)

    # --- Build upstream request ---
    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}
        ],
        "tools": [{"type": "web_search"}]
    }

    try:
        t_up = time.time()
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )
        upstream_latency = time.time() - t_up
    except requests.RequestException as e:
        end_trace(run, error=f"upstream_unreachable: {str(e)}",
                  extra_meta={"upstream_latency_s": None})
        return jsonify(error="upstream_unreachable", detail=str(e)), 502

    if r.status_code // 100 != 2:
        end_trace(
            run,
            error=f"openai_error {r.status_code}",
            extra_meta={"upstream_latency_s": upstream_latency, "openai_body": safe_trunc(r.text)}
        )
        return jsonify(error="openai_error", status=r.status_code, detail=r.text), 502

    payload = r.json()
    text = parse_responses_text(payload)
    # Optional polish: remove dangling "(letourdeshore.com)" if present at the very end
    text = re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", text).strip()

    total_latency = time.time() - t_req_start

    # ---- LangSmith end ----
    end_trace(
        run,
        outputs={"text": text},
        extra_meta={
            "upstream_latency_s": upstream_latency,
            "total_latency_s": total_latency,
            "openai_model": payload.get("model") or MODEL
        }
    )

    return jsonify({"text": text})

def safe_trunc(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "…"
