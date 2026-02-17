## Mental Health CBT Chatbot (Notebook-Exact Backend)

This project exposes your Jupyter notebook chatbot (`notebook/bot.ipynb`) as a **Flask web app** with:

- **Same logic & prompt** as the notebook (no extra behavior)
- **ChromaDB** vector store loaded from disk (no PDF re-processing)
- **Text chat** endpoint
- **Voice message** endpoint (browser → audio blob → Whisper → agent)

All RAG + agent logic is built exactly like in the notebook:

- `llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, openai_api_key=API_KEY)`
- `vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))`
- `retriever = vectorstore.as_retriever()`
- `cbt_counseling_prompt = PromptTemplate(template=<same template text>, input_variables=["context","question"])`
- `chain = ({"context": retriever, "question": RunnablePassthrough()} | cbt_counseling_prompt | llm)`
- `@tool cbt_tool(user_input: str) -> str: return chain.invoke(user_input)`
- `agent = create_agent(llm, tools=[cbt_tool], state_schema=CustomAgentState, checkpointer=InMemorySaver())`
- Requests call `agent.invoke({...}, {"configurable": {"thread_id": ...}})` exactly like the notebook example.

No extra prompts, safety layers, or transformations are added around the agent, so responses should match what the notebook would produce for the same inputs (subject to OpenAI model randomness).

---

### Project structure

- `app.py` – Flask app + notebook-identical RAG pipeline + agent
- `notebook/bot.ipynb` – Original notebook (source of the prompt and chain)
- `chroma_db/` – Persisted ChromaDB created by the notebook (already embedded chunks)
- `static/`
  - `index.html` – Minimal chat UI (text + voice)
  - `styles.css` – Basic dark styling
  - `app.js` – Frontend logic (text send, voice recording, thread_id handling)
- `requirements.txt` – Python dependencies

---

### Prerequisites

- Python 3.10+ recommended
- Existing **ChromaDB** directory at `./chroma_db` (created previously by `bot.ipynb`)
- OpenAI API key with access to `gpt-3.5-turbo`

---

### Setup

From the project root (`d:\my projjects\python\Mental Health`):

1. **Create and activate a virtualenv** (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables** in `.env` (project root):

   ```env
   API_KEY=sk-...your-openai-key...
   ```

   `app.py` looks for `API_KEY` first, then `OPENAI_API_KEY`.

4. **Ensure ChromaDB exists** at:

   ```text
   d:\my projjects\python\Mental Health\chroma_db
   ```

   This must be the same DB the notebook used; the app **does not re-process PDFs**.

---

### Running the app

From the project root:

```bash
python app.py
```

Then open in your browser:

```text
http://localhost:5000
```

You’ll see:

- A text area for messages
- **Send** button for text chat
- **Voice** button to record audio (using `MediaRecorder` in the browser)

---

### API endpoints

- `GET /`
  - Serves `static/index.html` (frontend UI).

- `POST /api/chat`
  - JSON body:

    ```json
    {
      "message": "Hello",
      "thread_id": "1",
      "user_id": "user_123",
      "preferences": { "theme": "dark" }
    }
    ```

  - Returns:

    ```json
    {
      "response": "<assistant text>",
      "thread_id": "1",
      "session_id": "1"
    }
    ```

  - This calls:
    - `agent.invoke({ "messages": [{ "role": "user", "content": message }], "user_id": ..., "preferences": ... }, { "configurable": { "thread_id": thread_id }})`

- `POST /api/chat/voice`
  - Multipart form:
    - `audio`: recorded audio blob (`audio/webm` or similar)
    - `thread_id` (optional)
    - `user_id` (optional)
    - `preferences` (optional JSON string)
  - Returns:

    ```json
    {
      "transcribed": "<text from Whisper>",
      "response": "<assistant text>",
      "thread_id": "1",
      "session_id": "1"
    }
    ```

---

### Frontend behavior (static/app.js)

- Maintains a persisted `thread_id` in `localStorage` so the agent’s `InMemorySaver` sees a consistent `thread_id` across messages.
- Sends text messages to `/api/chat`.
- Records audio with `MediaRecorder`, sends it as a blob to `/api/chat/voice`, shows the transcription, and then the assistant’s reply.

---

### Notes about “same output as bot.ipynb”

- The core components (**prompt, chain, tool, agent, and `agent.invoke` call**) are **line-by-line equivalent** to the notebook cells:
  - Same prompt text
  - Same model + temperature
  - Same retrieval call
  - Same tool implementation
  - Same agent construction and invocation
- No extra safety prompts, no additional pre/post-processing of the user message or the model output.
- Because OpenAI models are non-deterministic, there can still be slight variation between runs unless you also control the seed and exact SDK version, but structurally the backend matches `bot.ipynb`.

