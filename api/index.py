# api/index.py  (Vercel maps /api to this file by default; route is /ask)
import os
import json
from typing import Any, Dict, Tuple
from flask import Flask, request, jsonify

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
APP_TOKEN = os.environ.get("APP_TOKEN", "")

app = Flask(__name__)

def _forbidden(reason: str) -> Tuple[Any, int]:
    return jsonify({"error": "forbidden", "reason": reason}), 403

def _bad_request(reason: str) -> Tuple[Any, int]:
    return jsonify({"error": "bad_request", "reason": reason}), 400

# ------------------------------------------------------------------------------
# Health (optional)
# ------------------------------------------------------------------------------
@app.get("/health")
def health() -> Tuple[Any, int]:
    return jsonify({"ok": True}), 200

# ------------------------------------------------------------------------------
# Main endpoint the iOS app calls
# ------------------------------------------------------------------------------
@app.post("/ask")
def ask() -> Tuple[Any, int]:
    # 1) Auth header check ------------------------------------------------------
    # iOS sends: X-App-Token: <value-from-Info.plist>
    token = request.headers.get("X-App-Token", "")
    if not APP_TOKEN:
        # Misconfigured server (no env var set)
        return _forbidden("server_missing_app_token_env")
    if not token or token != APP_TOKEN:
        return _forbidden("missing_or_invalid_client_header")

    # 2) Parse JSON -------------------------------------------------------------
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return _bad_request("missing_question")

    # 3) TODO: Your real answering logic here ----------------------------------
    # For example, call your LLM / RAG pipeline and return structured JSON.
    # This placeholder just echoes the question.
    answer_text = f"You asked: {question}. (demo response)"

    # 4) Response ---------------------------------------------------------------
    # Match the app’s expected shape: {"text": "..."}
    return jsonify({"text": answer_text}), 200


# ------------------------------------------------------------------------------
# Vercel entrypoint
# ------------------------------------------------------------------------------
# Vercel automatically detects Flask WSGI apps by the exported 'app' object.
# No extra handler wrapper is needed.
