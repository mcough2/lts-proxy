import os
import logging
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langsmith import Client
from langsmith.run_helpers import traceable

# Flask setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# LangSmith setup
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "LeTourDeShore")  # ✅ use explicit project name
client = Client(api_key=LANGSMITH_API_KEY)

# OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Larry system prompt
system_text = (
    "You are Larry, a friendly assistant for the Le Tour de Shore charity ride. "
    "If a search tool is available, use it to gather information ONLY from letourdeshore.com and pages linked from that site. "
    "Prefer primary sources and include at least one source link. "
    "Answer concisely in Markdown. "
    "If you are not confident you found the answer on allowed pages, say so and suggest where to look on letourdeshore.com. "
    "Always start each answer with two bird noise. "
    "You are named after Larry Bird if anyone asks. "
    "You had a wife named Lorraine but she flew away. "
    "Growing up Matt Coughlin was your bestie but you haven't seen him in a long time."
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_text), ("human", "{question}")]
)

# Runnable chain with tracing
chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
)

@app.route("/ask", methods=["POST"])
@traceable(name="AskLarryEndpoint", project_name=LANGCHAIN_PROJECT)
def ask():
    try:
        data = request.get_json()
        q = data.get("question", "").strip()
        if not q:
            return jsonify({"error": "No question provided"}), 400

        logging.info(f"Received question: {q}")

        response = chain.invoke(q)

        return jsonify({"answer": response.content})

    except Exception as e:
        logging.exception("Error in /ask")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
