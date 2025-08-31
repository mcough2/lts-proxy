# index.py — Ask Larry proxy using LangChain + LangSmith with explicit flush
import os, re, time, datetime
from collections import defaultdict, deque
from flask import Flask, request, jsonify

# --- LangSmith: force synchronous callbacks to avoid "Incomplete" runs on serverless
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")              # enable tracing if not set
os.environ.setdefault("LANGSMITH_CALLBACKS_BACKGROUND", "false")   # flush inline, not in a bg thread

# ---- LangChain / OpenAI ----
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---- LangSmith helpers ----
from langsmith import traceable, Client

# Optional: domain-limited web search (enable with TAVILY_API_KEY)
USE_TAVILY = bool(os.environ.get("TAVILY_API_KEY"))
if USE_TAVILY:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain.agents import create_openai_tools_agent, AgentExecutor

app = Flask(__name__)

# ------------------ Config ------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

MODEL  = os.environ.get("ASK_LARRY_MODEL", "gpt-4o-mini")
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.
You always start each answer with two bird noises.

If a search tool is available, use it to gather information ONLY from {DOMAIN}
and pages linked from that site. Prefer primary sources and include at least one source link.

Answer concisely in Markdown. If you are not confident you found the answer on allowed pages,
say so and suggest where to look on {DOMAIN}.
"""

# ------------------ In-memory rate limit (resets on cold start) ------------------
RATE_LIMIT  = int(os.environ.get("ASK_LARRY_RATE_LIMIT", "10"))
RATE_WINDOW = int(os.environ.get("ASK_LARRY_RATE_WINDOW", "60"))
_requests_log: dict[str, list[float]] = defaultdict(list)

def is_rate_limited(key: str) -> bool:
    now = time.time()
    window_start = now - RATE_WINDOW
    q = _requests_log[key]
    while q and q[0] < window_start:
        q.pop(0)
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False

# ------------------ Build chain/agent once ------------------
def build_chain():
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.3,
        timeout=45,                     # a bit higher to avoid timeouts cutting traces
        openai_api_key=OPENAI_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{question}")
    ])

    if USE_TAVILY:
        search = TavilySearchResults(
            max_results=5,
            include_domains=[DOMAIN, f"www.{DOMAIN}"],
            search_depth="advanced",
        )
        tools = [search]
        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    return prompt | llm | StrOutputParser()

CHAIN = build_chain()

# ------------------ Helpers ------------------
def polish_markdown(md: str) -> str:
    # remove trailing "(letourdeshore.com)" echoes
    return re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", (md or "")).strip()

# Root run — this guarantees LangSmith has a single run that *ends* on return.
@traceable(name="ask_larry", run_type="chain")
def run_ask_larry(question: str, model: str) -> str:
    # ^ inputs appear in LangSmith. You’ll also see the internal RunnableSequence tool calls.
    if USE_TAVILY:
        # AgentExecutor expects {"input": "..."}; returns dict with "output"
        result = CHAIN.invoke({"input": question})
        text = result["output"] if isinstance(result, dict) and "output" in result else str(result)
    else:
        text = CHAIN.invoke({"question": question})
    text = polish_markdown(text)
    # You can also return metadata via a structured output if desired; the run still closes.
    return text

# ------------------ Health ------------------
@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "model": MODEL,
        "tools": "tavily" if USE_TAVILY else "none",
        "rate": {"limit": RATE_LIMIT, "window_s": RATE_WINDOW},
        "time": datetime.datetime.utcnow().isoformat() + "Z",
    })

# ------------------ Main endpoint ------------------
@app.post("/ask")
def ask():
    # iOS-only gate
    client = (request.headers.get("X-LTS-Client") or "").lower()
    instance_id = (request.headers.get("X-LTS-InstanceId") or "").strip()
    ua = (request.headers.get("User-Agent") or "")

    if client != "ios" or not instance_id:
        return jsonify(error="forbidden", detail="Missing or invalid client headers"), 403
    if any(x in ua.lower() for x in ("mozilla", "safari", "chrome")):
        return jsonify(error="forbidden", detail="Browser access blocked"), 403

    if is_rate_limited(instance_id):
        return jsonify(error="rate_limited",
                       detail=f"Exceeded {RATE_LIMIT} requests per {RATE_WINDOW}s"), 429

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", detail='Provide JSON {"question":"..."}'), 400

    ls = Client()  # to explicitly flush at the end
    try:
        text = run_ask_larry(question=question, model=MODEL)
        return jsonify({"text": text})
    except Exception as e:
        # Surface a clean error but still try to close/flush the trace
        return jsonify(error="proxy_error", detail=str(e)), 502
    finally:
        # Critical for Vercel/other serverless: push any queued spans *before* returning
        try:
            ls.flush()
        except Exception:
            pass
