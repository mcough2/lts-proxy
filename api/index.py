import os, json, re, time, datetime
from flask import Flask, request, jsonify
import requests
from collections import defaultdict, deque

# ---- LangSmith (Client API) ----
LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))
LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "LeTourDeShore")
if LS_ENABLED:
    try:
        from langsmith import Client
        ls_client = Client()
    except Exception:
        LS_ENABLED = False
        ls_client = None

app = Flask(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.
Use web search to gather information ONLY from {DOMAIN} and pages linked from those sites.
Prefer primary sources, and include at least one source link in every answer.
Answer in concise Markdown."""

# ---- Simple in-memory rate limiting (resets per cold start) ----
RATE_LIMIT = 10
RATE_WINDOW = 60
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

def parse_responses_text(payload: dict) -> str:
    out_parts = []
    for item in payload.get("output", []) or []:
        msg = item.get("message", {})
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            text = "\n".join([c.get("text","") for c in msg["content"] if isinstance(c, dict) and c.get("text")]).strip()
            if text: out_parts.append(text)
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("text"):
                out_parts.append(c["text"].strip())
    if out_parts:
        return "\n\n".join([p for p in out_parts if p])
    return payload.get("output_text") or payload.get("text", {}).get("value") or ""

def safe_trunc(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "…"

# ---- Main ask endpoint ----
@app.post("/ask")
def ask():
    if not OPENAI_KEY:
        return jsonify(error="server_not_configured", detail="Missing OPENAI_API_KEY"), 500

    # Client gating
    client = request.headers.get("X-LTS-Client", "").lower()
    instance_id = request.headers.get("X-LTS-InstanceId", "").strip()
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
    run_id = None
    child_id = None
    try:
        if LS_ENABLED:
            run_id = ls_client.create_run(
                name="ask_larry",
                run_type="chain",                              # <<< REQUIRED
                inputs={"question": question},
                project_name=LS_PROJECT,
                tags=["ask-larry", "proxy"],
                metadata={
                    "client": client, "instance_id": instance_id,
                    "user_agent": ua[:200], "ip": ip,
                    "rate_limit": {"limit": RATE_LIMIT, "window_s": RATE_WINDOW},
                },
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )

        # --- OpenAI call (child run)
        body = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}
            ],
            "tools": [{"type": "web_search"}]
        }

        t_up = time.time()
        if LS_ENABLED:
            child_id = ls_client.create_run(
                name="openai_responses",
                run_type="llm",                               # <<< REQUIRED
                parent_run_id=run_id,
                project_name=LS_PROJECT,
                inputs={"endpoint": "/v1/responses", "model": MODEL},
                tags=["openai"],
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )

        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )
        upstream_latency = time.time() - t_up

        if r.status_code // 100 != 2:
            if LS_ENABLED and child_id:
                ls_client.update_run(
                    child_id,
                    error=f"openai_error {r.status_code}",
                    outputs={"status": r.status_code, "body": safe_trunc(r.text)},
                    end_time=datetime.datetime.now(datetime.timezone.utc),
                )
            raise RuntimeError(f"openai_error {r.status_code}: {r.text}")

        payload = r.json()
        text = parse_responses_text(payload)
        text = re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", text).strip()

        if LS_ENABLED and child_id:
            ls_client.update_run(
                child_id,
                outputs={"latency_s": upstream_latency, "model": payload.get("model") or MODEL},
                end_time=datetime.datetime.now(datetime.timezone.utc),
            )

        if LS_ENABLED and run_id:
            ls_client.update_run(
                run_id,
                outputs={"text": text},
                metadata={"upstream_latency_s": upstream_latency},
                end_time=datetime.datetime.now(datetime.timezone.utc),
            )

        return jsonify({"text": text})

    except Exception as e:
        if LS_ENABLED and child_id:
            try:
                ls_client.update_run(
                    child_id,
                    error=str(e),
                    end_time=datetime.datetime.now(datetime.timezone.utc),
                )
            except Exception:
                pass
        if LS_ENABLED and run_id:
            try:
                ls_client.update_run(
                    run_id,
                    error=str(e),
                    end_time=datetime.datetime.now(datetime.timezone.utc),
                )
            except Exception:
                pass
        msg = str(e)
        if "rate_limited" in msg:
            return jsonify(error="rate_limited", detail=msg), 429
        if "openai_error" in msg:
            return jsonify(error="openai_error", detail=msg), 502
        return jsonify(error="proxy_error", detail=msg), 502
