(function () {
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const stopBtn = document.getElementById("stopBtn");
  const recordingEl = document.getElementById("recording");

  // Persist thread_id so agent memory matches notebook thread_id usage
  let threadId = localStorage.getItem("thread_id") || "1";

  function addMessage(role, text) {
    const row = document.createElement("div");
    row.className = "msg " + role;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendText() {
    const message = (inputEl.value || "").trim();
    if (!message) return;
    inputEl.value = "";
    addMessage("user", message);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        thread_id: threadId,
        user_id: "user_123",
        preferences: { theme: "dark" },
      }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      addMessage("bot", data.error || "Error");
      return;
    }
    if (data.thread_id) {
      threadId = String(data.thread_id);
      localStorage.setItem("thread_id", threadId);
    }
    addMessage("bot", data.response || "");
  }

  sendBtn.addEventListener("click", sendText);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  });

  // Voice recording
  let recorder = null;
  let chunks = [];

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = () => stream.getTracks().forEach((t) => t.stop());
    recorder.start();
    recordingEl.classList.remove("hidden");
  }

  async function stopAndSend() {
    if (!recorder) return;
    const r = recorder;
    recorder = null;

    await new Promise((resolve) => {
      r.onstop = resolve;
      r.stop();
    });

    recordingEl.classList.add("hidden");

    const blob = new Blob(chunks, { type: "audio/webm" });
    chunks = [];

    addMessage("user", "[voice]");

    const form = new FormData();
    form.append("audio", blob, "voice.webm");
    form.append("thread_id", threadId);
    form.append("user_id", "user_123");
    form.append("preferences", JSON.stringify({ theme: "dark" }));

    const res = await fetch("/api/chat/voice", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      addMessage("bot", data.error || "Error");
      return;
    }
    if (data.thread_id) {
      threadId = String(data.thread_id);
      localStorage.setItem("thread_id", threadId);
    }
    // show transcription then response
    if (data.transcribed) addMessage("user", "🎤 " + data.transcribed);
    addMessage("bot", data.response || "");
  }

  voiceBtn.addEventListener("click", async () => {
    try {
      if (!recorder) await startRecording();
      else await stopAndSend();
    } catch (e) {
      addMessage("bot", "Microphone error: " + (e && e.message ? e.message : String(e)));
    }
  });

  stopBtn.addEventListener("click", stopAndSend);
})();

