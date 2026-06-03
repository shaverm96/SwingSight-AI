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
const originalViewButton = document.getElementById("originalViewButton");
const overlayViewButton = document.getElementById("overlayViewButton");
const downloadOverlayButton = document.getElementById("downloadOverlayButton");
const visualizationStatus = document.getElementById("visualizationStatus");
const downloadPdfButton = document.getElementById("downloadPdfButton");
const downloadDocxButton = document.getElementById("downloadDocxButton");
const advancedMetrics = document.getElementById("advancedMetrics");
const advancedTracking = document.getElementById("advancedTracking");
const advancedModels = document.getElementById("advancedModels");
const advancedOverlay = document.getElementById("advancedOverlay");
const advancedDebug = document.getElementById("advancedDebug");

state.visualizationMode = "original";
state.originalVideoUrl = null;
state.overlayVideoUrl = null;
state.overlayValidation = null;

wireVideoDebug(analysisPreview, "analysisPreview");

uploadTrigger.addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) {
    return;
  }
  state.originalVideoUrl = URL.createObjectURL(file);
  state.overlayVideoUrl = null;
  setVisualizationMode("original");
  resultsPanel.classList.remove("hidden");
  downloadPdfButton.disabled = true;
  downloadDocxButton.disabled = true;
  analysisIdValue.textContent = "Uploading...";
  detectedClubValue.textContent = "--";
  detectedClubDetail.textContent = "--";
  swingScoreValue.textContent = "--";
  swingScoreDetail.textContent = "--";
  swingGradeValue.textContent = "Waiting for analysis";
  focusText.textContent = "--";
  renderFeedbackSection(takeawayList, ["Uploading the video preview so you can confirm the file is visible."]);
  visualizationStatus.textContent = "Original video preview";
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  await handleUploadFlow(file);
});

originalViewButton.addEventListener("click", () => setVisualizationMode("original"));
overlayViewButton.addEventListener("click", () => {
  if (!state.overlayVideoUrl) {
    return;
  }
  setVisualizationMode("overlay");
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
    updateStatus(error.message || "Upload failed. Please try again.");
    resultsPanel.classList.remove("hidden");
    setVisualizationMode("original");
  }
}

function setVisualizationMode(mode) {
  state.visualizationMode = mode;
  const showOverlay = mode === "overlay" && Boolean(state.overlayVideoUrl);
  const nextSource = showOverlay ? state.overlayVideoUrl : state.originalVideoUrl;

  if (nextSource) {
    console.debug("[overlay-debug] video src assigned", { mode, nextSource });
    analysisPreview.srcObject = null;
    analysisPreview.src = nextSource;
  }

  originalViewButton.classList.toggle("is-active", !showOverlay);
  overlayViewButton.classList.toggle("is-active", showOverlay);
  overlayViewButton.disabled = !state.overlayVideoUrl;
  downloadOverlayButton.classList.toggle("hidden", !state.overlayVideoUrl);
  visualizationStatus.textContent = showOverlay
    ? "Pose overlay: body tracking and motion trails"
    : "Original video preview";
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
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.message || errorPayload.error || "Analysis request failed");
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
  const detectedClub = result?.club || result?.detected_club || "Not detected";
  const scoreValue = result?.swing_score;
  const score = scoreValue === null || scoreValue === undefined ? Number.NaN : Number(scoreValue);
  const overlayFiles = Array.isArray(result?.overlay_files) ? result.overlay_files : [];
  const scoreLabel = result?.score_label || scoreToGrade(score, result);
  const visualization = result?.visualization || {};
  const overlayVideoUrl = visualization?.overlay_video_url || overlayFiles.find((item) => /\.(mp4|mov|webm)$/i.test(item)) || null;
  const originalVideoUrl = visualization?.original_video_url || state.originalVideoUrl;
  const overlayValidation = result?.overlay_validation || result?.debug?.fallback_status?.overlay_validation || null;
  state.overlayValidation = overlayValidation;

  analysisIdValue.textContent = result.analysis_id || "--";
  detectedClubValue.textContent = detectedClub;
  detectedClubDetail.textContent = detectedClub;
  swingScoreValue.textContent = Number.isFinite(score) ? `${Math.round(score)}` : "--";
  swingScoreDetail.textContent = Number.isFinite(score) ? `${Math.round(score)}` : "--";
  swingGradeValue.textContent = scoreLabel;

  renderFeedbackSection(takeawayList, result?.strengths || []);
  focusText.textContent = result?.next_focus || "Record from a clear side angle with your full body in frame.";

  if (originalVideoUrl) {
    state.originalVideoUrl = originalVideoUrl;
  }
  if (overlayVideoUrl) {
    state.overlayVideoUrl = overlayVideoUrl.startsWith("/") ? overlayVideoUrl : `/${overlayVideoUrl}`;
    downloadOverlayButton.href = state.overlayVideoUrl;
    downloadOverlayButton.download = state.overlayVideoUrl.split("/").pop() || "overlay.mp4";
  } else {
    state.overlayVideoUrl = null;
    downloadOverlayButton.href = "#";
  }

  setVisualizationMode("original");
  visualizationStatus.textContent = state.overlayVideoUrl
    ? `Pose overlay ready. Detection rate: ${formatDetectionRate(result?.tracking?.detection_rate)}. Click Pose Overlay to view it.`
    : "Pose tracking could not be generated from this video.";

  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });

  advancedMetrics.textContent = JSON.stringify(
    {
      video_metadata: result?.video_metadata || {},
      advanced_metrics: result?.advanced_metrics || {},
      warnings: result?.warnings || [],
    },
    null,
    2,
  );
  advancedTracking.textContent = JSON.stringify(result?.tracking || {}, null, 2);
  advancedModels.textContent = JSON.stringify(
    {
      status: result?.status,
      club: result?.club,
      swing_score: result?.swing_score,
      score_label: result?.score_label,
      strengths: result?.strengths,
      improvements: result?.improvements,
      overlay_files: result?.overlay_files,
      model_outputs: result?.model_outputs || {},
    },
    null,
    2,
  );
  advancedOverlay.textContent = JSON.stringify(
    {
      overlay_video_url: state.overlayVideoUrl,
      overlay_validation: overlayValidation || {},
      overlay_files,
    },
    null,
    2,
  );
  advancedDebug.textContent = JSON.stringify(result?.debug || {}, null, 2);
}

function wireVideoDebug(videoEl, label) {
  if (!videoEl || videoEl.dataset.debugWired === "true") {
    return;
  }
  videoEl.dataset.debugWired = "true";

  const logEvent = (eventName) => {
    const error = videoEl.error;
    console.debug(`[overlay-debug] ${label}:${eventName}`, {
      src: videoEl.currentSrc || videoEl.src,
      readyState: videoEl.readyState,
      networkState: videoEl.networkState,
      currentTime: videoEl.currentTime,
      duration: videoEl.duration,
      paused: videoEl.paused,
      errorCode: error ? error.code : null,
      errorMessage: error ? error.message || null : null,
    });
    if (eventName === "error") {
      const message = describeVideoError(error);
      visualizationStatus.textContent = `Overlay video failed to load. ${message}`;
      console.error("[overlay-debug] video error", {
        label,
        src: videoEl.currentSrc || videoEl.src,
        errorCode: error ? error.code : null,
        errorMessage: error ? error.message || null : null,
        overlayValidation: state.overlayValidation,
      });
    }
  };

  ["loadstart", "loadedmetadata", "loadeddata", "canplay", "playing", "waiting", "stalled", "emptied", "error", "ended"].forEach((eventName) => {
    videoEl.addEventListener(eventName, () => logEvent(eventName));
  });
}

function describeVideoError(error) {
  if (!error) {
    return "No MediaError details were reported.";
  }
  const descriptions = {
    1: "The playback was aborted.",
    2: "The overlay file could not be loaded.",
    3: "The overlay video decode failed.",
    4: "The browser does not support this overlay format.",
  };
  return descriptions[error.code] || "The browser reported a video loading error.";
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

function scoreToGrade(score, result) {
  if (!Number.isFinite(score)) {
    if (result?.status !== "success") {
      return "Analysis incomplete";
    }
    return "Needs clearer video";
  }
  if (score >= 85) {
    return "Excellent";
  }
  if (score >= 80) {
    return "Strong";
  }
  if (score >= 70) {
    return "Improving";
  }
  return "Needs clearer video";
}

function formatDetectionRate(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(1)}%`;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}