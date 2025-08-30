import os, json, re, time
from flask import Flask, request, jsonify
import requests
from collections import defaultdict, deque

app = Flask(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.
Use web search to gather information ONLY from {DOMAIN} and pages linked from those sites.
Prefer primary sources, and include at least one source link in every answer. Start each answer with a couple bird noises.
Answer in concise Markdown."""

# ---- Simple in-memory rate limiting (resets per serverless cold start) ----
RATE_LIMIT = 10         # requests
RATE_WINDOW = 60        # seconds
requests_log = defaultdict(lambda: deque())

def is_rate_limited(key: str) -> bool:
    now = time.time()
    q = requests_log[key]
    # drop old timestamps
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
            text = "\n".join([c.get("text","") for c in msg["content"] if isinstance(c, dict) and c.get("text")]).strip()
            if text: out_parts.append(text)
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("text"):
                out_parts.append(c["text"].strip())
    if out_parts:
        return "\n\n".join([p for p in out_parts if p])
    return payload.get("output_text") or payload.get("text", {}).get("value") or ""

# ---- Main ask endpoint ----
@app.post("/ask")
def ask():
    if not OPENAI_KEY:
        return jsonify(error="server_not_configured", detail="Missing OPENAI_API_KEY"), 500

    # --- Basic client gating ---
    client = request.headers.get("X-LTS-Client", "").lower()
    instance_id = request.headers.get("X-LTS-InstanceId", "").strip()
    ua = (request.headers.get("User-Agent") or "").lower()

    if client != "ios" or not instance_id:
        return jsonify(error="forbidden", detail="Missing or invalid client headers"), 403

    if "mozilla" in ua or "safari" in ua or "chrome" in ua:
        return jsonify(error="forbidden", detail="Browser access blocked"), 403

    # --- Rate limiting ---
    if is_rate_limited(instance_id):
        return jsonify(error="rate_limited",
                       detail=f"Exceeded {RATE_LIMIT} requests per {RATE_WINDOW}s"), 429

    # --- Validate request body ---
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", detail="Provide JSON {\"question\":\"...\"}"), 400

    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}
        ],
        "tools": [{"type": "web_search"}]
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )
    except requests.RequestException as e:
        return jsonify(error="upstream_unreachable", detail=str(e)), 502

    if r.status_code // 100 != 2:
        return jsonify(error="openai_error", status=r.status_code, detail=r.text), 502

    text = parse_responses_text(r.json())
    text = re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", text).strip()

    return jsonify({"text": text})
