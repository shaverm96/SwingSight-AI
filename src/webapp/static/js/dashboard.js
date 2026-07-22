const state = {
  mediaStream: null,
  selectedCameraId: null,
  recording: false,
  uploadClub: null,
  recordedClub: null,
  analysisProgressStartedAt: null,
  analysisProgressInterval: null,
  analysisProgressAdvanceTimer: null,
};

const uploadInput = document.getElementById("uploadInput");
const uploadTrigger = document.getElementById("uploadTrigger");
const uploadClubSelect = document.getElementById("uploadClubSelect");
const recordTrigger = document.getElementById("recordTrigger");
const startGuideButton = document.getElementById("startGuideButton");
const cancelRecordButton = document.getElementById("cancelRecordButton");
const recordPanel = document.getElementById("recordPanel");
const livePreview = document.getElementById("livePreview");
const recordStep = document.getElementById("recordStep");
const recordClubStatus = document.getElementById("recordClubStatus");
const statusText = document.getElementById("statusText");
const analysisProgress = document.getElementById("analysisProgress");
const analysisProgressTitle = document.getElementById("analysisProgressTitle");
const analysisProgressCopy = document.getElementById("analysisProgressCopy");
const analysisProgressBar = document.getElementById("analysisProgressBar");
const analysisProgressElapsed = document.getElementById("analysisProgressElapsed");

uploadClubSelect.addEventListener("change", () => {
  state.uploadClub = uploadClubSelect.value || null;
  uploadTrigger.disabled = !state.uploadClub;
});

uploadTrigger.addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files?.[0];
  if (!file || !state.uploadClub) {
    return;
  }
  try {
    showAnalysisProgress("upload");
    updateStatus(`Uploading your ${state.uploadClub} swing…`);
    const videoUploadId = await uploadFile("/api/upload-video", "video", file);
    setAnalysisProgressStage("analysis");
    await runAnalysis("/api/analyze", {
      video_upload_id: videoUploadId,
      club_category: state.uploadClub,
    });
  } catch (error) {
    console.error(error);
    hideAnalysisProgress();
    updateStatus(error.message || "We could not upload that video. Please try again.");
  } finally {
    uploadInput.value = "";
  }
});

recordTrigger.addEventListener("click", openRecorder);

startGuideButton.addEventListener("click", async () => {
  if (state.recording) {
    return;
  }
  try {
    state.recording = true;
    await runGuidedCapture();
  } catch (error) {
    console.error(error);
    hideAnalysisProgress();
    updateStatus(error.message || "We could not finish that swing review. Please try again.");
  } finally {
    state.recording = false;
  }
});

cancelRecordButton.addEventListener("click", () => {
  stopCamera();
  recordPanel.classList.add("hidden");
  updateStatus("Ready when you are");
  updateStep("Ready");
});

async function openRecorder() {
  state.recordedClub = null;
  updateStep("Scan club");
  recordClubStatus.textContent = "Show the club head to the camera, then tap Scan club & start.";
  startGuideButton.textContent = "Scan club & start";
  recordPanel.classList.remove("hidden");
  recordPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  await initCamera();
}

async function initCamera() {
  try {
    if (state.mediaStream) {
      return;
    }
    const stream = await requestCameraStream(state.selectedCameraId);
    state.selectedCameraId = stream.getVideoTracks()[0]?.getSettings?.().deviceId || state.selectedCameraId;
    state.mediaStream = stream;
    livePreview.srcObject = stream;
    await livePreview.play();
    updateStatus("Camera ready — show the club head, then start.");
  } catch (error) {
    console.error("Camera error", error);
    updateStatus("We could not open the camera. You can still upload a video.");
  }
}

function stopCamera() {
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
  livePreview.srcObject = null;
}

async function requestCameraStream(deviceId) {
  const primary = deviceId ? { video: { deviceId: { exact: deviceId } }, audio: false } : { video: true, audio: false };
  try {
    return await navigator.mediaDevices.getUserMedia(primary);
  } catch (error) {
    if (deviceId) {
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    throw error;
  }
}

async function runGuidedCapture() {
  if (!state.mediaStream) {
    await initCamera();
    if (!state.mediaStream) {
      return;
    }
  }
  updateStep("Scan club");
  updateStatus("Checking your club with the five-way club model…");
  const clubResult = await detectClubFromCamera();
  if (!clubResult) {
    return;
  }

  state.recordedClub = clubResult.club;
  const confidence = Number.isFinite(clubResult.confidence) ? ` (${Math.round(clubResult.confidence * 100)}% confidence)` : "";
  recordClubStatus.textContent = `${clubResult.club} detected${confidence}. Step back so your full body is visible.`;
  updateStep("Check stance");
  const bodyVisible = await attemptBodyCheck();
  if (!bodyVisible) {
    recordClubStatus.textContent = "Step back so your full body is visible, then try again.";
    updateStatus("We need a clearer full-body view.");
    updateStep("Ready");
    return;
  }

  updateStep("Recording");
  updateStatus("Ready. Make your swing.");
  await wait(900);
  const videoBlob = await recordSwing(5500);

  updateStep("Uploading");
  showAnalysisProgress("upload");
  updateStatus("Uploading your swing…");
  const uploadPayload = await uploadRecordedSwing(videoBlob);
  setAnalysisProgressStage("analysis");
  await runAnalysis("/api/analyze-swing", {
    video_upload_id: uploadPayload.upload_id,
    club_category: state.recordedClub,
  });
}

async function attemptBodyCheck() {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const response = await postFrame("/api/body-check");
    const check = response.check || response;
    if (check?.visible) {
      return true;
    }
    await wait(500);
  }
  return false;
}

async function detectClubFromCamera() {
  try {
    const response = await postFrame("/api/club-detect");
    const result = response?.result || {};
    const club = result.club || result.detected_club;
    if (result.status !== "confirmed" || !club || club === "Not detected") {
      const reason = result.reasoning || "The club was not clear enough to confirm.";
      recordClubStatus.textContent = `Could not confirm the club. ${reason} Try again with the club head centered in the frame.`;
      updateStatus("Club scan needs a clearer view.");
      return null;
    }
    return {
      club,
      confidence: Number(result.confidence),
    };
  } catch (error) {
    console.error(error);
    recordClubStatus.textContent = "Club scan is unavailable. Check that the club model is installed, then try again.";
    updateStatus("Club scan is unavailable.");
    return null;
  }
}

async function postFrame(endpoint) {
  const blob = await captureFrameBlob();
  const formData = new FormData();
  formData.append("frame", blob, "frame.png");
  const response = await fetch(endpoint, { method: "POST", body: formData });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "Camera check failed.");
  }
  return payload;
}

async function captureFrameBlob() {
  const width = livePreview.videoWidth || 640;
  const height = livePreview.videoHeight || 480;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(livePreview, 0, 0, width, height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

async function recordSwing(durationMs) {
  if (!state.mediaStream) {
    throw new Error("No camera stream available.");
  }
  const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  const recorder = new MediaRecorder(state.mediaStream, { mimeType });
  const chunks = [];
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.start();
  await wait(durationMs);
  recorder.stop();
  await new Promise((resolve) => { recorder.onstop = resolve; });
  return new Blob(chunks, { type: mimeType });
}

async function uploadRecordedSwing(blob) {
  const formData = new FormData();
  formData.append("video", blob, "swing.webm");
  const response = await fetch("/api/record-swing", { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error("Recording upload failed.");
  }
  return response.json();
}

async function uploadFile(endpoint, fieldName, file) {
  const formData = new FormData();
  formData.append(fieldName, file);
  const response = await fetch(endpoint, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }
  return response.json().then((payload) => payload.upload_id);
}

async function runAnalysis(endpoint, payload) {
  showAnalysisProgress("analysis");
  updateStatus("Analyzing your swing…");
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result.message || result.error || "Analysis request failed.");
    }
    setAnalysisProgressStage("complete");
    updateStatus("Your review is ready.");
    window.setTimeout(() => {
      window.location.assign(`/analysis/${encodeURIComponent(result.analysis_id)}`);
    }, 420);
  } catch (error) {
    hideAnalysisProgress();
    throw error;
  }
}

const ANALYSIS_PROGRESS_STAGES = {
  upload: { title: "Getting your swing ready", copy: "Securely preparing your video for review.", progress: 26 },
  analysis: { title: "Reading your swing", copy: "Finding the movements that matter most.", progress: 62 },
  overlay: { title: "Building your review", copy: "Adding motion tracking and coach-style feedback.", progress: 86 },
  complete: { title: "Your swing review is ready", copy: "Opening your personalized feedback now.", progress: 100 },
};

function showAnalysisProgress(stage = "upload") {
  if (!state.analysisProgressStartedAt) {
    state.analysisProgressStartedAt = Date.now();
  }
  analysisProgress.classList.remove("hidden");
  analysisProgress.setAttribute("aria-hidden", "false");
  document.body.setAttribute("aria-busy", "true");
  setAnalysisProgressStage(stage);
  if (!state.analysisProgressInterval) {
    state.analysisProgressInterval = window.setInterval(updateAnalysisProgressElapsed, 1000);
  }
}

function setAnalysisProgressStage(stage) {
  const details = ANALYSIS_PROGRESS_STAGES[stage] || ANALYSIS_PROGRESS_STAGES.analysis;
  const stageOrder = { upload: 1, analysis: 2, overlay: 3, complete: 4 };
  const currentOrder = stageOrder[stage] || 2;
  analysisProgressTitle.textContent = details.title;
  analysisProgressCopy.textContent = details.copy;
  analysisProgressBar.style.width = `${details.progress}%`;
  analysisProgress.querySelectorAll("[data-progress-step]").forEach((item) => {
    const itemOrder = stageOrder[item.dataset.progressStep] || 0;
    item.classList.toggle("is-complete", currentOrder > itemOrder);
    item.classList.toggle("is-active", currentOrder === itemOrder || (stage === "complete" && itemOrder === 3));
  });
  window.clearTimeout(state.analysisProgressAdvanceTimer);
  if (stage === "analysis") {
    state.analysisProgressAdvanceTimer = window.setTimeout(() => setAnalysisProgressStage("overlay"), 4200);
  }
}

function updateAnalysisProgressElapsed() {
  const seconds = Math.max(1, Math.floor((Date.now() - state.analysisProgressStartedAt) / 1000));
  analysisProgressElapsed.textContent = seconds < 10 ? "This usually takes less than a minute" : `Still working — ${seconds}s elapsed`;
}

function hideAnalysisProgress() {
  window.clearTimeout(state.analysisProgressAdvanceTimer);
  window.clearInterval(state.analysisProgressInterval);
  state.analysisProgressAdvanceTimer = null;
  state.analysisProgressInterval = null;
  state.analysisProgressStartedAt = null;
  analysisProgress.classList.add("hidden");
  analysisProgress.setAttribute("aria-hidden", "true");
  document.body.removeAttribute("aria-busy");
}

function updateStatus(message) {
  statusText.textContent = message;
}

function updateStep(step) {
  recordStep.textContent = `Step: ${step}`;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

if (new URLSearchParams(window.location.search).get("record") === "1") {
  openRecorder();
}