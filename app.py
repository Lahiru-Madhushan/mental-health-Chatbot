import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")


# ----------------------------
# Build EXACT notebook pipeline
# ----------------------------

def build_notebook_agent():
    # --- OpenAI key (notebook uses API_KEY) ---
    openai_api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing API key. Set API_KEY in .env (or OPENAI_API_KEY).")

    # --- LLM (same as notebook) ---
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, openai_api_key=openai_api_key)

    # --- Vectorstore: load persisted Chroma (no PDF processing) ---
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # notebook uses persist_directory="./chroma_db"
    persist_directory = str((Path(__file__).resolve().parent / "chroma_db"))
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # --- Retriever (same as notebook) ---
    retriever = vectorstore.as_retriever()

    # --- PromptTemplate (EXACT text from bot.ipynb) ---
    from langchain_core.prompts import PromptTemplate

    template = """
You are a compassionate mental health assistant grounded in Cognitive Behavioral Therapy (CBT).

Your purpose is to help users understand and manage their thoughts, emotions, and behaviors in a supportive, empowering way.

------------------------
CORE PRINCIPLES
------------------------
- Use ONLY the provided context when forming factual content.
- Never diagnose, label, or assume mental health conditions.
- Be empathetic, validating, and non-judgmental.
- Focus on CBT relationships: thoughts → feelings → behaviors.
- Help users gently identify thinking patterns.
- Offer short, practical CBT-based coping steps.
- Encourage reflection and autonomy — not dependency.
- Avoid overwhelming the user.
- If context is insufficient, respond supportively without inventing facts.

------------------------
RESPONSE STYLE
------------------------
- Warm, human, conversational tone.
- Clear and easy to follow.
- Prefer structure when helpful:
  Reflection → Insight → Small Action

------------------------
SAFETY & CRISIS HANDLING
------------------------
If the user expresses suicidal thoughts, intent to self-harm, or severe distress:

1. Respond with strong empathy and emotional validation.
2. Encourage reaching out to trusted people or professionals.
3. Provide Sri Lanka crisis support calmly:

   - National Mental Health Helpline: 1926 (24/7)
   - Sumithrayo emotional support: +94 707 308 308

4. Do NOT panic, moralize, or overwhelm.
5. Focus on safety, connection, and hope.

------------------------
Context:
{context}

User Question:
{question}
"""

    cbt_counseling_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    # --- Chain (same as notebook) ---
    from langchain_core.runnables import RunnablePassthrough

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | cbt_counseling_prompt
        | llm
    )

    # --- Tool (same as notebook) ---
    from langchain.tools import tool

    @tool
    def cbt_tool(user_input: str) -> str:
        """CBT counseling tool using retrieval + LLM"""
        return chain.invoke(user_input)

    # --- Agent (same as notebook snippet) ---
    from langchain.agents import create_agent, AgentState
    from langgraph.checkpoint.memory import InMemorySaver

    class CustomAgentState(AgentState):
        user_id: str
        preferences: dict

    agent = create_agent(
        llm,
        tools=[cbt_tool],
        state_schema=CustomAgentState,
        checkpointer=InMemorySaver(),
    )

    return agent


AGENT = build_notebook_agent()


def transcribe_audio(audio_path: str) -> str:
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return (result.get("text") or "").strip()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message required"}), 400

    # Match notebook’s invoke shape
    thread_id = str(data.get("thread_id") or data.get("session_id") or "1")
    user_id = data.get("user_id") or "user_123"
    preferences = data.get("preferences") or {"theme": "dark"}

    result = AGENT.invoke(
        {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "preferences": preferences,
        },
        {"configurable": {"thread_id": thread_id}},
    )

    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return jsonify({"response": str(result), "thread_id": thread_id})

    last = messages[-1]
    if isinstance(last, dict):
        response_text = (last.get("content") or "").strip()
    else:
        response_text = (getattr(last, "content", None) or str(last)).strip()

    return jsonify({"response": response_text, "thread_id": thread_id, "session_id": thread_id})


@app.route("/api/chat/voice", methods=["POST"])
def api_chat_voice():
    if "audio" not in request.files:
        return jsonify({"error": "audio required"}), 400

    file = request.files["audio"]
    thread_id = str(request.form.get("thread_id") or request.form.get("session_id") or "1")
    user_id = request.form.get("user_id") or "user_123"

    preferences = {"theme": "dark"}
    pref_raw = request.form.get("preferences")
    if pref_raw:
        try:
            import json

            preferences = json.loads(pref_raw)
        except Exception:
            preferences = {"theme": "dark"}

    suffix = Path(file.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    try:
        text = transcribe_audio(audio_path)
        result = AGENT.invoke(
            {
                "messages": [{"role": "user", "content": text}],
                "user_id": user_id,
                "preferences": preferences,
            },
            {"configurable": {"thread_id": thread_id}},
        )

        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            return jsonify(
                {"transcribed": text, "response": str(result), "thread_id": thread_id, "session_id": thread_id}
            )

        last = messages[-1]
        if isinstance(last, dict):
            response_text = (last.get("content") or "").strip()
        else:
            response_text = (getattr(last, "content", None) or str(last)).strip()

        return jsonify(
            {
                "transcribed": text,
                "response": response_text,
                "thread_id": thread_id,
                "session_id": thread_id,
            }
        )
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

