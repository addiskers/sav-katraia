// --- Main Application Logic ---

const statusDiv = document.getElementById("status");
const statusText = statusDiv.querySelector(".status-text");
const authSection = document.getElementById("auth-section");
const appSection = document.getElementById("app-section");
const sessionEndSection = document.getElementById("session-end-section");
const restartBtn = document.getElementById("restartBtn");
const micBtn = document.getElementById("micBtn");
const cameraBtn = document.getElementById("cameraBtn");
const screenBtn = document.getElementById("screenBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const videoPreview = document.getElementById("video-preview");
const videoPlaceholder = document.getElementById("video-placeholder");
const connectBtn = document.getElementById("connectBtn");
const chatLog = document.getElementById("chat-log");
const callTimerEl = document.getElementById("call-timer");
const languageIndicator = document.getElementById("language-indicator");

let currentAgentMessageDiv = null;
let currentUserMessageDiv = null;

// New state
let callStartTime = null;
let callTimerInterval = null;
let currentLanguage = "HI";
let vehicleData = null;
let bookingData = null;
let serviceCostData = null;
let audioVisualizerAnimationId = null;

// Call transcript & outcome tracking
let callTranscript = []; // {role: "user"|"agent", text: "", time: ""}
let toolCallsLog = [];   // {name, args, result}

const mediaHandler = new MediaHandler();
const sarvamClient = new SarvamClient({
  onOpen: () => {
    setStatus("connected", "Connected");
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");

    startCallTimer();

    // Send initial trigger to force agent to start talking
    sarvamClient.sendText(
      `Hi, I have picked up the phone. Please start the call.`
    );

    // Delay mic until greeting text arrives so ambient noise / early speech
    // cannot barge-in-cancel the opening line before TTS starts.
    // Fallback: open mic after 2.5s even if greeting is slow.
    let micStarted = false;
    const startMicOnce = () => {
      if (micStarted) return;
      micStarted = true;
      startMic();
    };
    window.__startMicAfterGreeting = startMicOnce;
    setTimeout(startMicOnce, 2500);
  },
  onMessage: (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "agent" && window.__startMicAfterGreeting) {
          window.__startMicAfterGreeting();
          window.__startMicAfterGreeting = null;
        }
        handleJsonMessage(msg);
      } catch (e) {
        console.error("Parse error:", e);
      }
    } else {
      mediaHandler.playAudio(event.data);
    }
  },
  onClose: (e) => {
    console.log("WS Closed:", e);
    setStatus("disconnected", "Disconnected");
    // Delay to allow any in-flight tool_call messages to be processed before rendering summary
    setTimeout(() => showSessionEnd(), 300);
  },
  onError: (e) => {
    console.error("WS Error:", e);
    setStatus("error", "Connection Error");
  },
});

// --- Status Helper ---
function setStatus(className, text) {
  statusDiv.className = "status-badge " + className;
  statusText.textContent = text;
}

// --- Call Timer ---
function startCallTimer() {
  callStartTime = Date.now();
  callTimerEl.textContent = "00:00";
  callTimerInterval = setInterval(() => {
    const elapsed = Date.now() - callStartTime;
    const mins = Math.floor(elapsed / 60000).toString().padStart(2, "0");
    const secs = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, "0");
    callTimerEl.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopCallTimer() {
  if (callTimerInterval) {
    clearInterval(callTimerInterval);
    callTimerInterval = null;
  }
}

function getCallDuration() {
  if (!callStartTime) return "00:00";
  const elapsed = Date.now() - callStartTime;
  const mins = Math.floor(elapsed / 60000).toString().padStart(2, "0");
  const secs = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

// --- Language Detection ---
function detectLanguage(text) {
  if (!text) return;
  let gujarati = 0, devanagari = 0, latin = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (code >= 0x0a80 && code <= 0x0aff) gujarati++;
    else if (code >= 0x0900 && code <= 0x097f) devanagari++;
    else if (code >= 0x0041 && code <= 0x007a) latin++;
  }
  const total = gujarati + devanagari + latin;
  if (total === 0) return;

  let lang = currentLanguage;
  if (gujarati > devanagari && gujarati > latin) lang = "GU";
  else if (devanagari > gujarati && devanagari > latin) lang = "HI";
  else if (latin > 5) lang = "EN";

  if (lang !== currentLanguage) {
    currentLanguage = lang;
    languageIndicator.textContent = lang;
  }
}

// --- Audio Visualizer ---
function initAudioVisualizer() {
  const canvas = document.getElementById("audio-visualizer");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  function draw() {
    audioVisualizerAnimationId = requestAnimationFrame(draw);
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    ctx.clearRect(0, 0, width, height);

    // Subtle grid lines
    ctx.strokeStyle = "rgba(0, 212, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let y = 0; y < height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Output (agent speaking) — frequency bars
    const outputAnalyser = mediaHandler.getOutputAnalyser();
    if (outputAnalyser) {
      const bufferLength = outputAnalyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      outputAnalyser.getByteFrequencyData(dataArray);

      const barCount = Math.min(bufferLength, 64);
      const barWidth = width / barCount;
      for (let i = 0; i < barCount; i++) {
        const barHeight = (dataArray[i] / 255) * height * 0.85;
        if (barHeight < 1) continue;

        const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
        gradient.addColorStop(0, "rgba(0, 212, 255, 0.7)");
        gradient.addColorStop(0.5, "rgba(124, 58, 237, 0.5)");
        gradient.addColorStop(1, "rgba(124, 58, 237, 0.1)");
        ctx.fillStyle = gradient;
        ctx.fillRect(i * barWidth + 1, height - barHeight, barWidth - 2, barHeight);
      }
    }

    // Input (user speaking) — waveform line
    const inputAnalyser = mediaHandler.getInputAnalyser();
    if (inputAnalyser) {
      const bufferLength = inputAnalyser.fftSize;
      const dataArray = new Float32Array(bufferLength);
      inputAnalyser.getFloatTimeDomainData(dataArray);

      ctx.strokeStyle = "rgba(16, 185, 129, 0.5)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      for (let i = 0; i < bufferLength; i++) {
        const y = (dataArray[i] * 0.5 + 0.5) * height;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * sliceWidth, y);
      }
      ctx.stroke();
    }
  }

  draw();
}

function stopAudioVisualizer() {
  if (audioVisualizerAnimationId) {
    cancelAnimationFrame(audioVisualizerAnimationId);
    audioVisualizerAnimationId = null;
  }
}

// --- Tool Call Renderers ---
function renderVehicleInfoPanel(data) {
  const el = document.getElementById("vehicle-info-content");
  if (!el || !data) return;

  let html = `<div class="info-grid">
    <span class="info-label">Owner</span>
    <span class="info-value">${data.owner_name || "—"}</span>
    <span class="info-label">Vehicle</span>
    <span class="info-value">${data.model || "—"} (${data.year || ""})</span>
    <span class="info-label">Reg. No.</span>
    <span class="info-value highlight">${data.vehicle_number || "—"}</span>
    <span class="info-label">KM Reading</span>
    <span class="info-value">${data.current_km_system ? data.current_km_system.toLocaleString() + " km" : "—"}</span>
    <span class="info-label">Warranty</span>
    <span class="info-value ${data.warranty_active ? "success" : "warning"}">${data.warranty_active ? "Active" : "Expired"}${data.warranty_expiry ? " (till " + data.warranty_expiry + ")" : ""}</span>
  </div>`;

  // Service history timeline
  if (data.service_history && data.service_history.length > 0) {
    html += `<div class="service-timeline">
      <div class="service-timeline-title">Service History</div>`;
    for (const svc of data.service_history) {
      html += `<div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-content">
          <div class="tl-type">${svc.type}</div>
          <div class="tl-detail">${svc.date} &middot; ${svc.km ? svc.km.toLocaleString() + " km" : ""} &middot; ${svc.workshop || ""}</div>
        </div>
      </div>`;
    }
    html += `</div>`;
  }

  // Next service
  if (data.next_service) {
    const ns = data.next_service;
    html += `<div class="next-service-badge">
      <div class="ns-title">Next: ${ns.type}</div>
      <div class="ns-detail">Due at ${ns.due_km ? ns.due_km.toLocaleString() : "—"} km &middot; Est. &#8377;${ns.estimated_cost_min || "?"}–&#8377;${ns.estimated_cost_max || "?"}</div>
    </div>`;
  }

  el.innerHTML = html;
}

function renderServiceCostPanel(data) {
  const el = document.getElementById("service-info-content");
  if (!el || !data) return;

  el.innerHTML = `<div class="info-grid">
    <span class="info-label">Cost Range</span>
    <span class="info-value highlight">&#8377;${data.min || "?"} – &#8377;${data.max || "?"}</span>
    <span class="info-label">Includes</span>
    <span class="info-value">${data.includes || "—"}</span>
  </div>`;
}

function renderBookingPanel(data) {
  const panel = document.getElementById("booking-panel");
  const el = document.getElementById("booking-content");
  if (!panel || !el || !data) return;

  panel.classList.remove("hidden");

  el.innerHTML = `<div class="info-grid">
    <span class="info-label">Booking ID</span>
    <span class="info-value highlight">${data.booking_id || "—"}</span>
    <span class="info-label">Pickup</span>
    <span class="info-value">${data.pickup_date || "��"} at ${data.pickup_time || "—"}</span>
    <span class="info-label">Driver</span>
    <span class="info-value">${data.driver_name || "—"} (${data.driver_phone || ""})</span>
    <span class="info-label">Workshop</span>
    <span class="info-value">${data.workshop || "—"}</span>
    ${data.special_instructions ? `<span class="info-label">Notes</span><span class="info-value">${data.special_instructions}</span>` : ""}
  </div>`;
}

// --- Message Handling ---
function handleJsonMessage(msg) {
  if (msg.type === "status") {
    const statusMsgDiv = document.createElement("div");
    statusMsgDiv.className = "message system";
    statusMsgDiv.innerHTML = `<span class="msg-text" style="color:#f59e0b;font-style:italic;">${escapeHtml(msg.text)}</span>`;
    chatLog.appendChild(statusMsgDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
    return;
  } else if (msg.type === "interrupted") {
    mediaHandler.stopAudioPlayback();
    currentAgentMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "turn_complete") {
    currentAgentMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "user") {
    // Stop leftover agent audio as soon as a user turn is recognized
    // (covers lag if "interrupted" arrived late or was missed).
    mediaHandler.stopAudioPlayback();
    if (currentUserMessageDiv) {
      const textEl = currentUserMessageDiv.querySelector(".msg-text");
      if (textEl) textEl.textContent += msg.text;
      // Append to last transcript entry
      if (callTranscript.length && callTranscript[callTranscript.length - 1].role === "user") {
        callTranscript[callTranscript.length - 1].text += msg.text;
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage("user", msg.text);
      callTranscript.push({ role: "user", text: msg.text, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    }
  } else if (msg.type === "agent") {
    if (currentAgentMessageDiv) {
      const textEl = currentAgentMessageDiv.querySelector(".msg-text");
      if (textEl) textEl.textContent += msg.text;
      if (callTranscript.length && callTranscript[callTranscript.length - 1].role === "agent") {
        callTranscript[callTranscript.length - 1].text += msg.text;
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentAgentMessageDiv = appendMessage("agent", msg.text);
      callTranscript.push({ role: "agent", text: msg.text, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    }
    detectLanguage(msg.text);
  } else if (msg.type === "tool_call") {
    toolCallsLog.push({ name: msg.name, args: msg.args, result: msg.result });
    if (msg.name === "get_vehicle_info") {
      vehicleData = msg.result;
      renderVehicleInfoPanel(msg.result);
    } else if (msg.name === "schedule_pickup") {
      bookingData = msg.result;
      renderBookingPanel(msg.result);
    } else if (msg.name === "get_service_cost_estimate") {
      serviceCostData = msg.result;
      renderServiceCostPanel(msg.result);
    }
  }
}

function appendMessage(type, text) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.innerHTML = `<span class="msg-text">${escapeHtml(text)}</span><span class="msg-time">${time}</span>`;
  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msgDiv;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- Call Me (Twilio outbound) ---
const phoneInput = document.getElementById("phoneInput");
const callMeBtn = document.getElementById("callMeBtn");
const callMeStatus = document.getElementById("callMeStatus");

if (callMeBtn) {
  callMeBtn.onclick = async () => {
    let phone = phoneInput.value.trim();
    if (!phone) {
      callMeStatus.textContent = "Enter your phone number";
      callMeStatus.className = "call-me-status error";
      return;
    }
    // Auto-add +91 if user just typed digits
    phone = phone.replace(/[\s\-()]/g, "");
    if (!phone.startsWith("+")) {
      phone = "+91" + phone;
    }

    callMeBtn.disabled = true;
    callMeStatus.textContent = "Calling...";
    callMeStatus.className = "call-me-status loading";

    try {
      const res = await fetch("/call-me", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone }),
      });
      const data = await res.json();
      if (data.success) {
        callMeStatus.textContent = "Calling " + phone + " — pick up your phone! Opening live transcript...";
        callMeStatus.className = "call-me-status success";
        // Open live transcript dashboard in new tab
        setTimeout(() => window.open("/live", "_blank"), 1000);
      } else {
        callMeStatus.textContent = data.error || "Failed to call";
        callMeStatus.className = "call-me-status error";
      }
    } catch (e) {
      callMeStatus.textContent = "Network error: " + e.message;
      callMeStatus.className = "call-me-status error";
    }
    callMeBtn.disabled = false;
  };

  // Enter key triggers call
  if (phoneInput) {
    phoneInput.onkeypress = (e) => {
      if (e.key === "Enter") callMeBtn.click();
    };
  }
}

// --- Connect ---
connectBtn.onclick = async () => {
  setStatus("disconnected", "Connecting...");
  connectBtn.disabled = true;

  try {
    await mediaHandler.initializeAudio();
    sarvamClient.connect();
  } catch (error) {
    console.error("Connection error:", error);
    setStatus("error", "Failed: " + error.message);
    connectBtn.disabled = false;
  }
};

// --- Mic ---
async function startMic() {
  try {
    await mediaHandler.startAudio((data) => {
      if (sarvamClient.isConnected()) {
        sarvamClient.send(data);
      }
    });
    micBtn.classList.add("active");
    micBtn.dataset.active = "true";

    // Start visualizer after mic is active (analyser nodes ready)
    initAudioVisualizer();
  } catch (e) {
    console.error("Could not start audio capture", e);
  }
}

micBtn.onclick = async () => {
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.classList.remove("active");
    micBtn.dataset.active = "false";
  } else {
    await startMic();
  }
};

// --- Camera ---
cameraBtn.onclick = async () => {
  if (cameraBtn.dataset.active === "true") {
    mediaHandler.stopVideo(videoPreview);
    cameraBtn.classList.remove("active");
    cameraBtn.dataset.active = "false";
    screenBtn.dataset.active = "false";
    screenBtn.classList.remove("active");
    videoPlaceholder.classList.remove("hidden");
  } else {
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      screenBtn.classList.remove("active");
      screenBtn.dataset.active = "false";
    }
    try {
      await mediaHandler.startVideo(videoPreview, (base64Data) => {
        if (sarvamClient.isConnected()) {
          sarvamClient.sendImage(base64Data);
        }
      });
      cameraBtn.classList.add("active");
      cameraBtn.dataset.active = "true";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      console.error("Could not access camera", e);
    }
  }
};

// --- Screen Share ---
screenBtn.onclick = async () => {
  if (screenBtn.dataset.active === "true") {
    mediaHandler.stopVideo(videoPreview);
    screenBtn.classList.remove("active");
    screenBtn.dataset.active = "false";
    cameraBtn.dataset.active = "false";
    cameraBtn.classList.remove("active");
    videoPlaceholder.classList.remove("hidden");
  } else {
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      cameraBtn.classList.remove("active");
      cameraBtn.dataset.active = "false";
    }
    try {
      await mediaHandler.startScreen(
        videoPreview,
        (base64Data) => {
          if (sarvamClient.isConnected()) {
            sarvamClient.sendImage(base64Data);
          }
        },
        () => {
          screenBtn.classList.remove("active");
          screenBtn.dataset.active = "false";
          videoPlaceholder.classList.remove("hidden");
        }
      );
      screenBtn.classList.add("active");
      screenBtn.dataset.active = "true";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      console.error("Could not share screen", e);
    }
  }
};

// --- Text Input ---
sendBtn.onclick = sendText;
textInput.onkeypress = (e) => {
  if (e.key === "Enter") sendText();
};

function sendText() {
  const text = textInput.value;
  if (text && sarvamClient.isConnected()) {
    sarvamClient.sendText(text);
    appendMessage("user", text);
    textInput.value = "";
  }
}

// --- Disconnect ---
disconnectBtn.onclick = () => {
  sarvamClient.disconnect();
};

// --- Session End ---
function showSessionEnd() {
  appSection.classList.add("hidden");
  sessionEndSection.classList.remove("hidden");
  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  stopCallTimer();
  stopAudioVisualizer();

  const summaryEl = document.getElementById("call-summary");
  if (!summaryEl) return;

  // Determine call outcome — check both bookingData and toolCallsLog as fallback
  const hasBooking = bookingData || toolCallsLog.some(t => t.name === "schedule_pickup" && t.result && t.result.success);
  if (!bookingData && hasBooking) {
    bookingData = toolCallsLog.find(t => t.name === "schedule_pickup").result;
  }
  const outcome = hasBooking ? "Booking Confirmed" : "No Booking Made";
  const outcomeClass = hasBooking ? "outcome-success" : "outcome-neutral";

  let html = "";

  // Outcome badge
  html += `<div class="outcome-badge ${outcomeClass}">${outcome}</div>`;

  // Call overview
  html += `<div class="summary-section">
    <div class="summary-section-title">Call Overview</div>`;
  html += summaryItem("Duration", getCallDuration());
  html += summaryItem("Language", langLabel(currentLanguage));
  html += summaryItem("Messages", `${callTranscript.length} exchanges`);
  html += summaryItem("Tool Calls", `${toolCallsLog.length}`);
  html += `</div>`;

  // Vehicle info
  if (vehicleData) {
    html += `<div class="summary-section">
      <div class="summary-section-title">Vehicle</div>`;
    html += summaryItem("Owner", vehicleData.owner_name);
    html += summaryItem("Vehicle", `${vehicleData.model} (${vehicleData.year})`);
    html += summaryItem("Reg. No.", vehicleData.vehicle_number);
    html += summaryItem("Odometer", `${vehicleData.current_km_system?.toLocaleString()} km`);
    html += summaryItem("Warranty", vehicleData.warranty_active ? `Active till ${vehicleData.warranty_expiry}` : "Expired");
    html += `</div>`;
  }

  // Booking details
  if (bookingData) {
    html += `<div class="summary-section">
      <div class="summary-section-title">Booking Details</div>`;
    html += summaryItem("Booking ID", bookingData.booking_id);
    html += summaryItem("Pickup Date", bookingData.pickup_date);
    html += summaryItem("Pickup Time", bookingData.pickup_time);
    html += summaryItem("Driver", `${bookingData.driver_name} (${bookingData.driver_phone})`);
    html += summaryItem("Workshop", bookingData.workshop);
    if (bookingData.special_instructions) {
      html += summaryItem("Instructions", bookingData.special_instructions);
    }
    html += `</div>`;
  }

  // Service cost
  if (serviceCostData) {
    html += `<div class="summary-section">
      <div class="summary-section-title">Service Estimate</div>`;
    html += summaryItem("Cost Range", `\u20B9${serviceCostData.min} – \u20B9${serviceCostData.max}`);
    html += summaryItem("Includes", serviceCostData.includes);
    html += `</div>`;
  }

  // Full transcript (collapsible)
  if (callTranscript.length > 0) {
    html += `<div class="summary-section">
      <button class="transcript-toggle" onclick="this.parentElement.classList.toggle('expanded')">
        <span class="toggle-icon">&#9654;</span> View Full Transcript (${callTranscript.length} messages)
      </button>
      <div class="transcript-full">`;
    for (const entry of callTranscript) {
      const roleLabel = entry.role === "user" ? "Customer" : "Advisor";
      const roleClass = entry.role === "user" ? "tr-user" : "tr-agent";
      html += `<div class="tr-line ${roleClass}">
        <span class="tr-time">${entry.time}</span>
        <span class="tr-role">${roleLabel}:</span>
        <span class="tr-text">${escapeHtml(entry.text)}</span>
      </div>`;
    }
    html += `</div></div>`;
  }

  summaryEl.innerHTML = html;
}

function summaryItem(label, value) {
  return `<div class="summary-item"><span class="summary-label">${label}</span><span class="summary-value">${value || "—"}</span></div>`;
}

function langLabel(code) {
  const map = { HI: "Hindi", EN: "English", GU: "Gujarati", MR: "Marathi" };
  return map[code] || code;
}

// --- Reset ---
function resetUI() {
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");

  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  videoPlaceholder.classList.remove("hidden");
  stopCallTimer();
  stopAudioVisualizer();

  micBtn.classList.remove("active");
  micBtn.dataset.active = "false";
  cameraBtn.classList.remove("active");
  cameraBtn.dataset.active = "false";
  screenBtn.classList.remove("active");
  screenBtn.dataset.active = "false";

  chatLog.innerHTML = "";
  connectBtn.disabled = false;
  callTimerEl.textContent = "00:00";
  languageIndicator.textContent = "HI";
  currentLanguage = "HI";
  vehicleData = null;
  bookingData = null;
  serviceCostData = null;
  currentAgentMessageDiv = null;
  currentUserMessageDiv = null;
  callTranscript = [];
  toolCallsLog = [];

  // Reset info panels
  const vic = document.getElementById("vehicle-info-content");
  if (vic) vic.innerHTML = '<div class="info-placeholder">Waiting for data...</div>';
  const sic = document.getElementById("service-info-content");
  if (sic) sic.innerHTML = '<div class="info-placeholder">No service info yet</div>';
  const bp = document.getElementById("booking-panel");
  if (bp) bp.classList.add("hidden");

  // Reset mobile state
  const colRight = document.querySelector(".col-right");
  if (colRight) colRight.classList.remove("mobile-expanded");
  const videoPanel = document.querySelector(".video-panel");
  if (videoPanel) videoPanel.classList.remove("mobile-video-active");
  document.querySelectorAll(".mobile-tab").forEach((t, i) => {
    t.classList.toggle("active", i === 0);
  });
  // Show all panels again (desktop mode)
  Object.values({ v: "vehicle-info-panel", s: "service-info-panel", b: "booking-panel" }).forEach((id) => {
    const p = document.getElementById(id);
    if (p) p.style.display = "";
  });
}

restartBtn.onclick = () => {
  resetUI();
};

// --- Mobile: Tab switching for info panels ---
(function initMobileTabs() {
  const tabs = document.querySelectorAll(".mobile-tab");
  const colRight = document.querySelector(".col-right");
  const panelMap = {
    vehicle: "vehicle-info-panel",
    service: "service-info-panel",
    booking: "booking-panel",
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      // Toggle expand/collapse
      if (tab.classList.contains("active") && colRight.classList.contains("mobile-expanded")) {
        colRight.classList.remove("mobile-expanded");
        return;
      }

      // Switch active tab
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      colRight.classList.add("mobile-expanded");

      // Show only selected panel
      const selectedId = panelMap[tab.dataset.tab];
      Object.values(panelMap).forEach((id) => {
        const panel = document.getElementById(id);
        if (!panel) return;
        if (id === selectedId) {
          panel.style.display = "";
        } else {
          panel.style.display = "none";
        }
      });
    });
  });
})();

// --- Mobile: Video PiP (picture-in-picture style) ---
function updateMobileVideoState(isActive) {
  const videoPanel = document.querySelector(".video-panel");
  if (!videoPanel) return;
  if (window.innerWidth <= 768) {
    if (isActive) {
      videoPanel.classList.add("mobile-video-active");
    } else {
      videoPanel.classList.remove("mobile-video-active");
    }
  }
}

// Patch camera/screen handlers to update mobile video state
const origCameraClick = cameraBtn.onclick;
cameraBtn.onclick = async () => {
  await origCameraClick();
  updateMobileVideoState(cameraBtn.dataset.active === "true");
};

const origScreenClick = screenBtn.onclick;
screenBtn.onclick = async () => {
  await origScreenClick();
  updateMobileVideoState(screenBtn.dataset.active === "true");
};
