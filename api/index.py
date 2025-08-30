import os, json, re, time
from flask import Flask, request, jsonify
import requests
from collections import defaultdict, deque

# -------- LangSmith (RunTree) --------
LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "LeTourDeShore")
RunTree = None
if LS_ENABLED:
    try:
        from langsmith.run_trees import RunTree
    except Exception:
        LS_ENABLED = False

app = Flask(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.
Use web search to gather information ONLY from {DOMAIN} and pages linked from those sites.
Prefer primary sources, and include at least one source link in every answer.
Answer in concise Markdown."""

# -------- simple in-memory rate limit (resets on cold start) --------
RATE_LIMIT = 10          # requests
RATE_WINDOW = 60         # seconds
_requests_log = defaultdict(lambda: deque())

def is_rate_limited(key: str) -> bool:
    now = time.time()
    q = _requests_log[key]
    while q and q[0] <= now - RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False

# -------- OpenAI response extraction --------
def parse_responses_text(payload: dict) -> str:
    parts = []
    for item in payload.get("output", []) or []:
        msg = item.get("message", {})
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            txt = "\n".join([c.get("text","") for c in msg["content"] if isinstance(c, dict) and c.get("text")]).strip()
            if txt: parts.append(txt)
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("text"):
                parts.append(c["text"].strip())
    if parts:
        return "\n\n".join([p for p in parts if p])
    return payload.get("output_text") or payload.get("text", {}).get("value") or ""

def safe_trunc(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "…"

# -------- main endpoint --------
@app.post("/ask")
def ask():
    if not OPENAI_KEY:
        return jsonify(error="server_not_configured", detail="Missing OPENAI_API_KEY"), 500

    # ---- iOS-only gate
    client = request.headers.get("X-LTS-Client", "").lower()
    instance_id = (request.headers.get("X-LTS-InstanceId") or "").strip()
    ua = (request.headers.get("User-Agent") or "").lower()
    ip = (request.headers.get("x-forwarded-for") or request.remote_addr or "").split(",")[0].strip()

    if client != "ios" or not instance_id:
        return jsonify(error="forbidden", detail="Missing or invalid client headers"), 403
    if "mozilla" in ua or "safari" in ua or "chrome" in ua:
        return jsonify(error="forbidden", detail="Browser access blocked"), 403

    if is_rate_limited(instance_id):
        return jsonify(error="rate_limited",
                       detail=f"Exceeded {RATE_LIMIT} requests per {RATE_WINDOW}s"), 429

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", detail="Provide JSON {\"question\":\"...\"}"), 400

    # ---- LangSmith root run
    run = None
    child = None
    try:
        if LS_ENABLED and RunTree:
            run = RunTree(
                name="ask_larry",
                run_type="chain",
                project_name=LS_PROJECT,
                inputs={"question": question},
                tags=["ask-larry", "proxy"],
                metadata={
                    "client": client,
                    "instance_id": instance_id,
                    "user_agent": ua[:200],
                    "ip": ip,
                    "rate_limit": {"limit": RATE_LIMIT, "window_s": RATE_WINDOW},
                },
            )
            run.post()  # <-- create run in LangSmith

        # ---- OpenAI call (child run)
        body = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}
            ],
            "tools": [{"type": "web_search"}]
        }

        t0 = time.time()
        if LS_ENABLED and RunTree and run is not None:
            child = RunTree(
                name="openai_responses",
                run_type="llm",
                project_name=LS_PROJECT,
                inputs={"endpoint": "/v1/responses", "model": MODEL},
                parent_run_id=run.id,
                tags=["openai"],
            )
            child.post()

        try:
            r = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                data=json.dumps(body),
                timeout=30,
            )
        except requests.RequestException as e:
            if child: 
                child.end(error=f"upstream_unreachable: {str(e)}")
            if run:
                run.end(error=f"upstream_unreachable: {str(e)}")
            return jsonify(error="upstream_unreachable", detail=str(e)), 502

        upstream_latency = time.time() - t0

        if r.status_code // 100 != 2:
            if child:
                child.end(error=f"openai_error {r.status_code}",
                          outputs={"status": r.status_code, "body": safe_trunc(r.text)})
            if run:
                run.update(metadata={**(run.metadata or {}), "upstream_latency_s": upstream_latency})
                run.end(error=f"openai_error {r.status_code}")
            return jsonify(error="openai_error", status=r.status_code, detail=r.text), 502

        payload = r.json()
        text = parse_responses_text(payload)
        text = re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", text).strip()

        if child:
            child.end(outputs={"latency_s": upstream_latency,
                               "model": payload.get("model") or MODEL})

        if run:
            run.update(metadata={**(run.metadata or {}), "upstream_latency_s": upstream_latency})
            run.end(outputs={"text": text})

        return jsonify({"text": text})

    except Exception as e:
        # Always close runs on error paths
        try:
            if child and not child.ended: child.end(error=str(e))
        except Exception:
            pass
        try:
            if run and not run.ended: run.end(error=str(e))
        except Exception:
            pass
        return jsonify(error="proxy_error", detail=str(e)), 502
