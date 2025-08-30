import os, json, re, time, datetime
from flask import Flask, request, jsonify
from collections import defaultdict, deque

# ---- LangChain / OpenAI ----
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional: web search tool (domain-limited via Tavily)
USE_TAVILY = bool(os.environ.get("TAVILY_API_KEY"))
if USE_TAVILY:
    # langchain-community provides the Tavily tool
    from langchain_community.tools.tavily_search import TavilySearchResults

app = Flask(__name__)

# ------------------ App / Model config ------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
assert OPENAI_KEY, "Missing OPENAI_API_KEY"

MODEL = os.environ.get("ASK_LARRY_MODEL", "gpt-4o-mini")
DOMAIN = "letourdeshore.com"

SYSTEM = f"""You are Larry, a friendly assistant for the Le Tour de Shore charity ride.

If a search tool is available, use it to gather information ONLY from {DOMAIN} and pages
linked from that site. Prefer primary sources and include at least one source link.

Answer concisely in Markdown. If you are not confident you found the answer on allowed pages,
say so and suggest where to look on {DOMAIN}.
"""

# ------------------ Rate limit (in-memory; resets on cold start) ------------------
RATE_LIMIT = 10          # requests
RATE_WINDOW = 60         # seconds
_requests_log = defaultdict(list)

def is_rate_limited(key: str) -> bool:
    now = time.time()
    window_start = now - RATE_WINDOW
    q = _requests_log[key]
    # drop old
    while q and q[0] < window_start:
        q.pop(0)
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False

# ------------------ Build the chain once (threads are short-lived in Vercel) ------------------
def build_chain():
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.3,
        timeout=30,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{question}")
    ])

    if USE_TAVILY:
        # Domain-limited search results
        search = TavilySearchResults(
            max_results=5,
            include_domains=[DOMAIN, f"www.{DOMAIN}"],
            search_depth="advanced",
        )
        # Use LangChain's "tools" mechanism; ChatOpenAI will tool-call as needed
        from langchain.agents import create_openai_tools_agent, AgentExecutor
        tools = [search]
        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    # No tools available → simple chain
    return prompt | llm | StrOutputParser()

CHAIN = build_chain()

# ------------------ Helpers ------------------
def polish_markdown(md: str) -> str:
    # remove trailing "(letourdeshore.com)" if model echoes the domain alone
    md = re.sub(r"\s*\((?:https?://)?(?:www\.)?letourdeshore\.com\)\s*$", "", md).strip()
    return md

# ------------------ Health ------------------
@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "model": MODEL,
        "tools": "tavily" if USE_TAVILY else "none",
    })

# ------------------ Main endpoint ------------------
@app.post("/ask")
def ask():
    # iOS-only gate
    client = request.headers.get("X-LTS-Client", "").lower()
    instance_id = (request.headers.get("X-LTS-InstanceId") or "").strip()
    ua = (request.headers.get("User-Agent") or "")

    if client != "ios" or not instance_id:
        return jsonify(error="forbidden", detail="Missing or invalid client headers"), 403
    if any(x in ua.lower() for x in ["mozilla", "safari", "chrome"]):
        return jsonify(error="forbidden", detail="Browser access blocked"), 403

    if is_rate_limited(instance_id):
        return jsonify(error="rate_limited",
                       detail=f"Exceeded {RATE_LIMIT} requests per {RATE_WINDOW}s"), 429

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="bad_request", detail="Provide JSON {\"question\":\"...\"}"), 400

    try:
        if USE_TAVILY:
            # AgentExecutor expects a dict with "input"
            result = CHAIN.invoke({"input": question})
            text = result["output"] if isinstance(result, dict) else str(result)
        else:
            text = CHAIN.invoke({"question": question})

        text = polish_markdown(text)
        return jsonify({"text": text})

    except Exception as e:
        # Let the app surface a clean message
        return jsonify(error="proxy_error", detail=str(e)), 502
