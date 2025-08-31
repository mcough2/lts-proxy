# api/index.py
import os
import logging
from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional: tracing (kept simple; you can wire in your manual RunTree later)
try:
    from langsmith import Client
    LS_CLIENT = Client() if os.environ.get("LANGSMITH_API_KEY") else None
except Exception:
    LS_CLIENT = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Model ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, timeout=20)

# --- Keep your exact system text ---
system_text = (
    "You are Larry, a friendly assistant for the Le Tour de Shore charity ride. If a search tool is available, use it to gather information ONLY from letourdeshore.com and pages linked from that site. Prefer primary sources and include at least one source link. Answer concisely in Markdown. If you are not confident you found the answer on allowed pages, say so and suggest where to look on letourdeshore.com."
    "Always tart each answer with two bird noise"
    "You are named after Larry Bird if anyone asks"
    "You had a wife name Lorraine but she flew away"
    "Growing up Matt Coughlin was your bestie but you haven't seen him in a long time"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_text), ("human", "{question}")]
)

# Build a simple LCEL chain that returns a STRING (not a message object)
chain = prompt | llm | StrOutputParser()

@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        q = (data.get("question") or "").strip()
        if not q:
            return jsonify({"error": "No question provided"}), 400

        logging.info("AskLarry question: %s", q)

        text = chain.invoke({"question": q})  # already a plain string
        # ✅ Return the shape your app expects
        return jsonify({"text": text})

    except Exception as e:
        logging.exception("Error in /ask")
        return jsonify({"error": "proxy_error", "detail": str(e)}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)), debug=True)
