import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory

import config
from pipeline import run_agent
from intents.classify_llm import get_client

app = Flask(__name__, static_folder="static", static_url_path="")

# One shared Groq client, reused across requests (matches pipeline.py's own pattern)
_client = None


def get_shared_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    history = data.get("history") or []  # list of prior customer message strings, oldest first

    if not message:
        return jsonify({"error": "empty message"}), 400

    try:
        result = run_agent(message, context=history, client=get_shared_client())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "reply": result["draft_reply"],
        "status": result["draft_reply_status"],   # AUTO_SEND or SUGGESTION_FOR_HUMAN_REVIEW
        "intent": result["intent"],
        "escalate": result["escalate"],
        "reasons": result["escalation_reasons"],
        "grounded_on": result.get("grounded_on", []),
    })


if __name__ == "__main__":
    print(f"Brand: {config.BRAND_HANDLE}  |  Model: {config.GROQ_MODEL}")
    print("Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)