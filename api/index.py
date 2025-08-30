import os, json, re, time, datetime, sys
from flask import Flask, request, jsonify
import requests
from collections import defaultdict, deque

app = Flask(__name__)

# ------------------ OpenAI / App config ------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.
Use web search to gather information ONLY from {DOMAIN} and pages linked from those sites.
Prefer primary sources, and include at least one source link in every answer.
Answer in concise Markdown."""

# ------------------ LangSmith wiring ------------------
LS_PROJECT = os.environ.get("LANGSMITH_PROJECT", "LeTourDeShore")
LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY"))  # if using Client
LS_BACKEND = None  # "client" | "runtree" | None

ls_client = None
RunTree = None

if LS_ENABLED:
    try:
        from langsmith import Client
        ls_client = Client()
        LS_BACKEND = "client"
    except Exception as e:
        print(f"[ls] Client import failed: {e}", file=sys.stderr)
        try:
            from langsmith.run_trees import RunTree  # fallback
            LS_BACKEND = "runtree"
        except Exception as e2:
            print(f"[ls] RunTree import failed: {e2}", file=sys.stderr)
            LS_ENABLED = False
            LS_BACKEND = None
else:
    # env var not set; try RunTree anyway in case user uses LS_ENDPOINT+API_KEY via other means
    try:
        from langsmith.run_trees import RunTree
        LS_ENABLED = True
        LS_BACKEND = "runtree"
    except Exception:
        LS_ENABLED = False
        LS_BACKEND = None

print(f"[startup] LS_ENABLED={LS_ENABLED} backend={LS_BACKEND} project={LS_PROJECT}", flush=True)

# ------------------ Rate limit (in-memory) ------------------
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

# ------------------ Helpers ------------------
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

# ------------------ Debug endpoints ------------------
@app.get("/ls-health")
def ls_health():
    return jsonify({
        "ls_enabled": LS_ENABLED,
        "backend": LS_BACKEND,
        "project": LS_PROJECT,
        "has_client": bool(ls_client),
        "has_runtree": bool(RunTree),
    })

@app.post("/ls-smoketest")
def ls_smoketest():
    """Create a tiny test run so you can confirm traces show up."""
    if not LS_ENABLED:
        return jsonify(ok=False, reason="tracing_disabled"), 200
    try:
        if LS_BACKEND == "client":
            rid = ls_client.create_run(
                name="ls_smoketest",
                run_type="chain",
                project_name=LS_PROJECT,
                inputs={"ping": "pong"},
                tags=["smoketest"],
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )
            ls_client.update_run(
                rid,
                outputs={"pong": True},
                end_time=datetime.datetime.now(datetime.timezone.utc),
            )
        else:
            run = RunTree(
                name="ls_smoketest",
                run_type="chain",
                project_name=LS_PROJECT,
                inputs={"ping": "pong"},
                tags=["smoketest"],
            )
            run.post(); run.end(outputs={"pong": True})
        return jsonify(ok=True), 200
    except Exception as e:
        print(f"[ls] smoketest error: {e}", file=sys.stderr)
        return jsonify(ok=False, error=str(e)), 500

# ------------------ Main endpoint ------------------
@app.post("/ask")
def ask():
    print("[ask] received", flush=True)
    if not OPENAI_KEY:
        return jsonify(error="server_not_configured", detail="Missing OPENAI_API_KEY"), 500

    # iOS gate
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

    # ---- start trace
    root = None
    child = None
    try:
        if LS_ENABLED and LS_BACKEND == "client":
            root = ls_client.create_run(
                name="ask_larry",
                run_type="chain",
                inputs={"question": question},
                project_name=LS_PROJECT,
                tags=["ask-larry","proxy"],
                metadata={"client":client,"instance_id":instance_id,"ip":ip,"ua":ua[:160]},
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )
        elif LS_ENABLED and LS_BACKEND == "runtree":
            root = RunTree(
                name="ask_larry",
                run_type="chain",
                project_name=LS_PROJECT,
                inputs={"question": question},
                tags=["ask-larry","proxy"],
                metadata={"client":client,"instance_id":instance_id,"ip":ip,"ua":ua[:160]},
            )
            root.post()

        body = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}
            ],
            "tools": [{"type": "web_search"}]
        }

        t0 = time.time()
        if LS_ENABLED and LS_BACKEND == "client":
            child = ls_client.create_run(
                name="openai_responses", run_type="llm",
                project_name=LS_PROJECT, parent_run_id=root,
                tags=["openai"], inputs={"endpoint":"/v1/responses","model":MODEL},
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )
        elif LS_ENABLED and LS_BACKEND == "runtree":
            child = RunTree(
                name="openai_responses", run_type="llm",
                project_name=LS_PROJECT, parent_run_id=root.id if root else None,
                tags=["openai"], inputs={"endpoint":"/v1/responses","model":MODEL},
            ); child.post()

        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )
        up_lat = time.time() - t0

        if r.status_code // 100 != 2:
            if LS_ENABLED and child:
                if LS_BACKEND == "client":
                    ls_client.update_run(child, error=f"openai_error {r.status_code}",
                                         outputs={"status": r.status_code, "body": safe_trunc(r.text)},
                                         end_time=datetime.datetime.now(datetime.timezone.utc))
                else:
                    child.end(error=f"openai_error {r.status_code}",
                              outputs={"status": r.status_code, "body": safe_trunc(r.text)})
            if LS_ENABLED and root:
                if LS_BACKEND == "client":
                    ls_client.update_run(root, error=f"openai_error {r.status_code}",
                                         metadata={"upstream_latency_s": up_lat},
                                         end_time=datetime.datetime.now(datetime.timezone.utc))
                else:
                    root.end(error=f"openai_error {r.status_code}",
                             metadata={"upstream_latency_s": up_lat})
            return jsonify(error="openai_error", status=r.status_code, detail=r.text), 502

        payload = r.json()
        text = parse_responses_text(payload)
        text = re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", text).strip()

        if LS_ENABLED and child:
            if LS_BACKEND == "client":
                ls_client.update_run(child,
                    outputs={"latency_s": up_lat, "model": payload.get("model") or MODEL},
                    end_time=datetime.datetime.now(datetime.timezone.utc))
            else:
                child.end(outputs={"latency_s": up_lat, "model": payload.get("model") or MODEL})

        if LS_ENABLED and root:
            if LS_BACKEND == "client":
                ls_client.update_run(root,
                    outputs={"text": text},
                    metadata={"upstream_latency_s": up_lat},
                    end_time=datetime.datetime.now(datetime.timezone.utc))
            else:
                root.end(outputs={"text": text},
                          metadata={"upstream_latency_s": up_lat})

        return jsonify({"text": text})

    except Exception as e:
        print(f"[ask] error: {e}", file=sys.stderr)
        # close runs on error
        try:
            if LS_ENABLED and child:
                if LS_BACKEND == "client":
                    ls_client.update_run(child, error=str(e),
                                         end_time=datetime.datetime.now(datetime.timezone.utc))
                else:
                    child.end(error=str(e))
        except Exception as ee:
            print(f"[ls] child end error: {ee}", file=sys.stderr)
        try:
            if LS_ENABLED and root:
                if LS_BACKEND == "client":
                    ls_client.update_run(root, error=str(e),
                                         end_time=datetime.datetime.now(datetime.timezone.utc))
                else:
                    root.end(error=str(e))
        except Exception as ee:
            print(f"[ls] root end error: {ee}", file=sys.stderr)
        return jsonify(error="proxy_error", detail=str(e)), 502
