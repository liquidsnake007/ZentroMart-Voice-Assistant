const state = { history: [], order_verified: false, verification_attempts: 0, recording: false };
const micButton = document.getElementById("micButton");
const fileInput = document.getElementById("fileInput");
const clearButton = document.getElementById("clearButton");
const statusText = document.getElementById("statusText");
const visualizerLabel = document.getElementById("visualizerLabel");
const orb = document.querySelector(".orb");
const conversation = document.getElementById("conversation");
const emptyState = document.getElementById("emptyState");
const bars = document.getElementById("bars");
let mediaRecorder;
let chunks = [];
let audioContext;
let analyser;
let animationFrame;
let stream;

for (let i = 0; i < 46; i++) {
  const bar = document.createElement("span");
  bar.className = "bar";
  bar.style.height = `${8 + Math.random() * 12}px`;
  bars.appendChild(bar);
}

function setStatus(text, active = false) {
  statusText.textContent = text;
  document.querySelector(".status-dot").style.background = active ? "#ff8a9d" : "var(--mint)";
}

function drawIntensity() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  const intensity = data.reduce((sum, value) => sum + value, 0) / data.length / 255;
  orb.style.transform = `scale(${1 + intensity * .35})`;
  [...bars.children].forEach((bar, index) => {
    const sample = data[(index * 3) % data.length] / 255;
    bar.style.height = `${8 + sample * 70}px`;
  });
  animationFrame = requestAnimationFrame(drawIntensity);
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 128;
  audioContext.createMediaStreamSource(stream).connect(analyser);
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];
  mediaRecorder.ondataavailable = event => chunks.push(event.data);
  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
    try {
      const buffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
      handleAudio(audioBufferToWav(buffer), "recording.wav");
    } catch (error) {
      addMessage("agent", "System", `Could not prepare the recording: ${error.message}`);
      setStatus("Recording failed");
    }
    stream.getTracks().forEach(track => track.stop());
    audioContext.close();
    cancelAnimationFrame(animationFrame);
    analyser = null;
  };
  mediaRecorder.start();
  state.recording = true;
  micButton.classList.add("recording");
  visualizerLabel.textContent = "Listening… tap to finish";
  setStatus("Listening", true);
  drawIntensity();
}

function stopRecording() {
  if (mediaRecorder && state.recording) {
    mediaRecorder.stop();
    state.recording = false;
    micButton.classList.remove("recording");
    visualizerLabel.textContent = "Processing your voice…";
    setStatus("Processing");
  }
}

function audioBufferToWav(buffer) {
  const channels = buffer.numberOfChannels;
  const length = buffer.length * channels * 2 + 44;
  const output = new ArrayBuffer(length);
  const view = new DataView(output);
  let offset = 0;
  const write = value => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset++, value.charCodeAt(i));
  };
  const uint16 = value => { view.setUint16(offset, value, true); offset += 2; };
  const uint32 = value => { view.setUint32(offset, value, true); offset += 4; };
  write("RIFF"); uint32(length - 8); write("WAVE"); write("fmt ");
  uint32(16); uint16(1); uint16(channels); uint32(buffer.sampleRate);
  uint32(buffer.sampleRate * channels * 2); uint16(channels * 2); uint16(16);
  write("data"); uint32(length - offset - 4);
  const channelData = Array.from({ length: channels }, (_, index) => buffer.getChannelData(index));
  for (let i = 0; i < buffer.length; i++) {
    for (let channel = 0; channel < channels; channel++) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([view], { type: "audio/wav" });
}

function addMessage(kind, label, text, sources = [], audio = null) {
  emptyState.style.display = "none";
  const item = document.createElement("article");
  item.className = `message ${kind}`;
  item.innerHTML = `<div class="message-label">${label}</div><p></p>`;
  item.querySelector("p").textContent = text;
  if (sources.length) {
    const sourceRow = document.createElement("div");
    sourceRow.className = "source-row";
    sourceRow.innerHTML = sources.map(source => `<span class="source">${source}</span>`).join("");
    item.appendChild(sourceRow);
  }
  if (audio) {
    const player = document.createElement("audio");
    player.className = "play";
    player.controls = true;
    player.src = `data:audio/wav;base64,${audio}`;
    item.appendChild(player);
  }
  conversation.appendChild(item);
  item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function handleAudio(blob, filename) {
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const encoded = reader.result.split(",")[1];
      const response = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio: encoded,
          filename,
          history: state.history,
          order_verified: state.order_verified,
          verification_attempts: state.verification_attempts,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Unable to process audio.");
      addMessage("user", "You said", result.transcript || "No speech detected.");
      addMessage("agent", "VoiceOps", result.response, result.sources, result.audio);
      state.history = result.history || state.history;
      state.order_verified = result.order_verified;
      state.verification_attempts = result.verification_attempts;
      visualizerLabel.textContent = "Tap the mic to begin";
      setStatus("Ready to listen");
    } catch (error) {
      addMessage("agent", "System", error.message);
      visualizerLabel.textContent = "Tap the mic to try again";
      setStatus("Something went wrong");
    }
  };
  reader.readAsDataURL(blob);
}

micButton.addEventListener("click", () => state.recording ? stopRecording() : startRecording().catch(error => setStatus(error.message)));
fileInput.addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) {
    addMessage("user", "Uploaded audio", file.name);
    visualizerLabel.textContent = "Processing your upload…";
    setStatus("Processing");
    handleAudio(file, file.name);
  }
});
clearButton.addEventListener("click", () => {
  state.history = [];
  state.order_verified = false;
  state.verification_attempts = 0;
  conversation.innerHTML = "";
  conversation.appendChild(emptyState);
  emptyState.style.display = "block";
  setStatus("Ready to listen");
  visualizerLabel.textContent = "Tap the mic to begin";
});
