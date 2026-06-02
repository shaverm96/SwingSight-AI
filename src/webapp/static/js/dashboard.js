const state = {
  videoUploadId: null,
  analysisId: null,
  mediaStream: null,
  recording: false,
};

const uploadInput = document.getElementById("uploadInput");
const uploadTrigger = document.getElementById("uploadTrigger");
const recordTrigger = document.getElementById("recordTrigger");
const startGuideButton = document.getElementById("startGuideButton");
const cancelRecordButton = document.getElementById("cancelRecordButton");
const recordPanel = document.getElementById("recordPanel");
const resultsPanel = document.getElementById("resultsPanel");
const livePreview = document.getElementById("livePreview");
const analysisPreview = document.getElementById("analysisPreview");
const recordStep = document.getElementById("recordStep");
const statusText = document.getElementById("statusText");
const analysisIdValue = document.getElementById("analysisIdValue");
const detectedClubValue = document.getElementById("detectedClubValue");
const detectedClubDetail = document.getElementById("detectedClubDetail");
const swingScoreValue = document.getElementById("swingScoreValue");
const swingScoreDetail = document.getElementById("swingScoreDetail");
const swingGradeValue = document.getElementById("swingGradeValue");
const takeawayList = document.getElementById("takeawayList");
const focusText = document.getElementById("focusText");
const overlayLink = document.getElementById("overlayLink");
const downloadPdfButton = document.getElementById("downloadPdfButton");
const downloadDocxButton = document.getElementById("downloadDocxButton");
const advancedMetrics = document.getElementById("advancedMetrics");
const advancedTracking = document.getElementById("advancedTracking");
const advancedModels = document.getElementById("advancedModels");

uploadTrigger.addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) {
    return;
  }
  analysisPreview.src = URL.createObjectURL(file);
  await handleUploadFlow(file);
});

recordTrigger.addEventListener("click", async () => {
  recordPanel.classList.remove("hidden");
  await initCamera();
});

startGuideButton.addEventListener("click", async () => {
  if (state.recording) {
    return;
  }
  try {
    state.recording = true;
    await runGuidedCapture();
  } finally {
    state.recording = false;
  }
});

cancelRecordButton.addEventListener("click", () => {
  stopCamera();
  recordPanel.classList.add("hidden");
  updateStatus("Recording canceled. Choose another option.");
  updateStep("Ready");
});

downloadPdfButton.addEventListener("click", () => requestReport("pdf"));
downloadDocxButton.addEventListener("click", () => requestReport("docx"));

async function handleUploadFlow(file) {
  try {
    updateStatus("Uploading your swing...");
    state.videoUploadId = await uploadFile("/api/upload-video", "video", file);
    await runAnalysis("/api/analyze", { video_upload_id: state.videoUploadId });
  } catch (error) {
    console.error(error);
    updateStatus("Upload failed. Please try again.");
  }
}

async function initCamera() {
  try {
    if (state.mediaStream) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    state.mediaStream = stream;
    livePreview.srcObject = stream;
    await livePreview.play();
    updateStatus("Camera ready. Tap Start to begin.");
  } catch (err) {
    console.error("Camera error", err);
    updateStatus("Unable to access camera. You can still upload a video.");
  }
}

function stopCamera() {
  if (!state.mediaStream) {
    return;
  }
  state.mediaStream.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
  livePreview.srcObject = null;
}

async function runGuidedCapture() {
  if (!state.mediaStream) {
    await initCamera();
    if (!state.mediaStream) {
      return;
    }
  }

  updateStep("Show club end");
  updateStatus("Show the end of your club.");
  const clubResult = await attemptClubRecognition();
  if (!clubResult) {
    updateStatus("Could not confirm club. Try again or upload a video.");
    updateStep("Ready");
    return;
  }

  const clubLabel = clubResult.predicted_club || clubResult.category || "Unknown";
  updateStatus(`Club confirmed: ${clubLabel}.`);

  updateStep("Step back");
  updateStatus("Step back so your full body is visible.");
  const bodyVisible = await attemptBodyCheck();
  if (!bodyVisible) {
    updateStatus("Step back so your full body is visible.");
    updateStep("Ready");
    return;
  }

  updateStep("Ready");
  updateStatus("Ready. Swing away.");
  await wait(1000);
  const videoBlob = await recordSwing(5500);

  updateStep("Uploading");
  updateStatus("Uploading your swing...");
  const uploadPayload = await uploadRecordedSwing(videoBlob);
  state.videoUploadId = uploadPayload.upload_id;
  analysisPreview.srcObject = null;
  analysisPreview.src = uploadPayload.preview_url || `/uploads/${uploadPayload.file_name}`;

  updateStep("Analyzing");
  await runAnalysis("/api/analyze-swing", { video_upload_id: state.videoUploadId });
  updateStep("Done");
}

async function attemptClubRecognition() {
  for (let i = 0; i < 6; i++) {
    const resp = await postFrame("/api/club-detect");
    const result = resp.result || resp;
    const status = result?.status || (result?.confidence >= 0.6 ? "confirmed" : "uncertain");
    if (status === "confirmed") {
      return result;
    }
    await wait(600);
  }
  return null;
}

async function attemptBodyCheck() {
  for (let i = 0; i < 8; i++) {
    const resp = await postFrame("/api/body-check");
    const check = resp.check || resp;
    if (check?.visible) {
      return true;
    }
    await wait(500);
  }
  return false;
}

async function postFrame(endpoint) {
  const blob = await captureFrameBlob();
  const fd = new FormData();
  fd.append("frame", blob, "frame.png");
  const resp = await fetch(endpoint, { method: "POST", body: fd });
  return resp.json();
}

async function captureFrameBlob() {
  const width = livePreview.videoWidth || 640;
  const height = livePreview.videoHeight || 480;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(livePreview, 0, 0, width, height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

async function recordSwing(durationMs) {
  if (!state.mediaStream) {
    throw new Error("No media stream available");
  }
  const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";
  const recorder = new MediaRecorder(state.mediaStream, { mimeType });
  const chunks = [];
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.start();
  await wait(durationMs);
  recorder.stop();
  await new Promise((resolve) => (recorder.onstop = resolve));
  return new Blob(chunks, { type: mimeType });
}

async function uploadRecordedSwing(blob) {
  const formData = new FormData();
  formData.append("video", blob, "swing.webm");
  const response = await fetch("/api/record-swing", { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error("Recording upload failed");
  }
  return response.json();
}

async function runAnalysis(endpoint, payload) {
  updateStatus("Analyzing your swing...");
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error("Analysis request failed");
  }
  const result = await response.json();
  state.analysisId = result.analysis_id;
  renderResults(result);
  updateStatus("Analysis complete.");
  resultsPanel.classList.remove("hidden");
  downloadPdfButton.disabled = false;
  downloadDocxButton.disabled = false;
}

async function uploadFile(endpoint, fieldName, file) {
  const formData = new FormData();
  formData.append(fieldName, file);
  const response = await fetch(endpoint, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }
  const payload = await response.json();
  return payload.upload_id;
}

async function requestReport(format) {
  if (!state.analysisId) {
    updateStatus("Run an analysis before downloading a report.");
    return;
  }
  try {
    updateStatus(`Generating ${format.toUpperCase()} report...`);
    const response = await fetch(`/api/reports/${state.analysisId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format }),
    });
    if (!response.ok) {
      throw new Error(`Report generation failed: ${response.status}`);
    }
    const payload = await response.json();
    window.location.href = payload.download_url;
    updateStatus(`${format.toUpperCase()} report generated and download started.`);
  } catch (error) {
    console.error(error);
    updateStatus(`Unable to generate ${format.toUpperCase()} report.`);
  }
}

function renderResults(result) {
  const detectedClub = result?.club || result?.detected_club || "Unknown";
  const score = Number(result?.swing_score);
  const overlayFiles = Array.isArray(result?.overlay_files) ? result.overlay_files : [];

  analysisIdValue.textContent = result.analysis_id || "--";
  detectedClubValue.textContent = detectedClub;
  detectedClubDetail.textContent = detectedClub;
  swingScoreValue.textContent = Number.isFinite(score) ? `${Math.round(score)}` : "--";
  swingScoreDetail.textContent = Number.isFinite(score) ? `${Math.round(score)}` : "--";
  swingGradeValue.textContent = scoreToGrade(score);

  if (overlayFiles.length) {
    const firstOverlay = overlayFiles[0];
    overlayLink.href = `/${firstOverlay}`;
    overlayLink.classList.remove("hidden");
  } else {
    overlayLink.href = "#";
    overlayLink.classList.add("hidden");
  }

  renderFeedbackSection(takeawayList, result?.strengths || []);
  focusText.textContent = result?.next_focus || "Keep your tempo smooth and finish balanced.";

  advancedMetrics.textContent = JSON.stringify(result?.advanced_metrics || {}, null, 2);
  advancedTracking.textContent = JSON.stringify(result?.advanced_metrics?.pose || {}, null, 2);
  advancedModels.textContent = JSON.stringify(
    {
      club: result?.club,
      swing_score: result?.swing_score,
      strengths: result?.strengths,
      improvements: result?.improvements,
      overlay_files: result?.overlay_files,
    },
    null,
    2,
  );
}

function renderFeedbackSection(listEl, items) {
  listEl.innerHTML = "";
  const safeItems = Array.isArray(items) ? items : [];
  if (!safeItems.length) {
    const li = document.createElement("li");
    li.textContent = "No feedback available yet.";
    listEl.appendChild(li);
    return;
  }
  for (const item of safeItems) {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  }
}

function updateStatus(message) {
  statusText.textContent = message;
}

function updateStep(step) {
  recordStep.textContent = `Step: ${step}`;
}

function scoreToGrade(score) {
  if (!Number.isFinite(score)) {
    return "Awaiting analysis";
  }
  if (score >= 90) {
    return "Excellent";
  }
  if (score >= 80) {
    return "Strong";
  }
  if (score >= 70) {
    return "Improving";
  }
  return "Needs work";
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}