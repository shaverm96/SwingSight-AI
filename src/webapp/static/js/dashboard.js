const state = {
  videoUploadId: null,
  analysisId: null,
  mediaStream: null,
  selectedCameraId: null,
  availableCameras: [],
  recording: false,
  uploadClub: null,
  recordedClub: null,
};

const uploadInput = document.getElementById("uploadInput");
const uploadTrigger = document.getElementById("uploadTrigger");
const uploadClubSelect = document.getElementById("uploadClubSelect");
const recordTrigger = document.getElementById("recordTrigger");
const startGuideButton = document.getElementById("startGuideButton");
const cancelRecordButton = document.getElementById("cancelRecordButton");
const recordPanel = document.getElementById("recordPanel");
const resultsPanel = document.getElementById("resultsPanel");
const livePreview = document.getElementById("livePreview");
const analysisPreview = document.getElementById("analysisPreview");
const recordStep = document.getElementById("recordStep");
const recordClubStatus = document.getElementById("recordClubStatus");
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
const expandOverlayButton = document.getElementById("expandOverlayButton");
const downloadOverlayButton = document.getElementById("downloadOverlayButton");
const visualizationStatus = document.getElementById("visualizationStatus");
const downloadPdfButton = document.getElementById("downloadPdfButton");
const downloadDocxButton = document.getElementById("downloadDocxButton");
const advancedMetrics = document.getElementById("advancedMetrics");
const advancedTracking = document.getElementById("advancedTracking");
const advancedHands = document.getElementById("advancedHands");
const advancedModels = document.getElementById("advancedModels");
const advancedOverlay = document.getElementById("advancedOverlay");
const advancedDebug = document.getElementById("advancedDebug");
const singleVideoView = document.getElementById("singleVideoView");
const overlayModal = document.getElementById("overlayModal");
const overlayModalPreview = document.getElementById("overlayModalPreview");
const modalOriginalButton = document.getElementById("modalOriginalButton");
const modalOverlayButton = document.getElementById("modalOverlayButton");
const closeOverlayModalButton = document.getElementById("closeOverlayModalButton");

state.visualizationMode = "original";
state.originalVideoUrl = null;
state.overlayVideoUrl = null;
state.overlayValidation = null;
state.overlayVariants = {};
state.overlayStyle = "smoothed";
state.isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
state.modalViewMode = "original";
state.modalOpen = false;
state.originalVideoInfo = null;
state.overlayVideoInfo = null;

wireVideoDebug(analysisPreview, "analysisPreview");
wireVideoDebug(overlayModalPreview, "overlayModalPreview");

uploadClubSelect.addEventListener("change", () => {
  state.uploadClub = uploadClubSelect.value || null;
  uploadTrigger.disabled = !state.uploadClub;
});

uploadTrigger.addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  const club = state.uploadClub;
  if (!file || !club) {
    return;
  }
  state.overlayStyle = "smoothed";
  state.originalVideoUrl = URL.createObjectURL(file);
  state.overlayVideoUrl = null;
  state.overlayVariants = {};
  state.overlayValidation = null;
  setVisualizationMode("original");
  resultsPanel.classList.remove("hidden");
  downloadPdfButton.disabled = true;
  downloadDocxButton.disabled = true;
  analysisIdValue.textContent = "Uploading...";
  detectedClubValue.textContent = club;
  detectedClubDetail.textContent = club;
  swingScoreValue.textContent = "--";
  swingScoreDetail.textContent = "--";
  swingGradeValue.textContent = "Waiting for analysis";
  focusText.textContent = "--";
  renderFeedbackSection(takeawayList, ["Uploading the video preview so you can confirm the file is visible."]);
  visualizationStatus.textContent = "Original video preview";
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  await handleUploadFlow(file, club);
});

originalViewButton.addEventListener("click", () => setVisualizationMode("original"));
overlayViewButton.addEventListener("click", () => {
  if (!state.overlayVideoUrl) {
    return;
  }
  setVisualizationMode("overlay");
});
expandOverlayButton.addEventListener("click", () => openOverlayModal());
closeOverlayModalButton.addEventListener("click", () => closeOverlayModal());
overlayModal.addEventListener("click", (event) => {
  if (event.target === overlayModal || event.target.classList.contains("overlay-modal__backdrop")) {
    closeOverlayModal();
  }
});
modalOriginalButton.addEventListener("click", () => setModalVideoMode("original"));
modalOverlayButton.addEventListener("click", () => setModalVideoMode("overlay"));

recordTrigger.addEventListener("click", async () => {
  state.recordedClub = null;
  recordStep.textContent = "Step: Scan club";
  recordClubStatus.textContent = "Show the club head to the camera, then select Scan Club & Start.";
  startGuideButton.textContent = "Scan Club & Start";
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

async function handleUploadFlow(file, club) {
  try {
    updateStatus(`Uploading your ${club} swing...`);
    state.overlayStyle = "smoothed";
    state.overlayVariants = {};
    state.overlayValidation = null;
    state.videoUploadId = await uploadFile("/api/upload-video", "video", file);
    await runAnalysis("/api/analyze", {
      video_upload_id: state.videoUploadId,
      club_category: club,
    });
  } catch (error) {
    console.error(error);
    updateStatus(error.message || "Upload failed. Please try again.");
    resultsPanel.classList.remove("hidden");
    setVisualizationMode("original");
  }
}

function setVisualizationMode(mode) {
  state.visualizationMode = mode;
  const nextSource = mode === "overlay" ? state.overlayVideoUrl : state.originalVideoUrl;

  loadVideoSource(analysisPreview, nextSource, {
    mode,
    label: "main",
    fallbackInfo: mode === "overlay" ? state.overlayVideoInfo : state.originalVideoInfo,
  });

  originalViewButton.classList.toggle("is-active", mode === "original");
  overlayViewButton.classList.toggle("is-active", mode === "overlay");
  overlayViewButton.disabled = !state.overlayVideoUrl;
  downloadOverlayButton.classList.toggle("hidden", !state.overlayVideoUrl);
  visualizationStatus.textContent = mode === "overlay"
    ? "Processed overlay video"
    : "Original uploaded video";

  if (state.modalOpen) {
    setModalVideoMode(state.modalViewMode);
  }
}

function openOverlayModal() {
  if (!overlayModal) {
    return;
  }
  state.modalOpen = true;
  overlayModal.classList.remove("hidden");
  overlayModal.setAttribute("aria-hidden", "false");
  setModalVideoMode(state.visualizationMode === "overlay" ? "overlay" : "original");
}

function closeOverlayModal() {
  if (!overlayModal) {
    return;
  }
  state.modalOpen = false;
  overlayModal.classList.add("hidden");
  overlayModal.setAttribute("aria-hidden", "true");
  overlayModalPreview.pause();
  overlayModalPreview.removeAttribute("src");
  overlayModalPreview.srcObject = null;
}

function setModalVideoMode(mode) {
  if (!overlayModalPreview) {
    return;
  }
  state.modalViewMode = mode;
  const nextSource = mode === "overlay" ? state.overlayVideoUrl : state.originalVideoUrl;
  loadVideoSource(overlayModalPreview, nextSource, {
    mode,
    label: "modal",
    fallbackInfo: mode === "overlay" ? state.overlayVideoInfo : state.originalVideoInfo,
  });
  modalOriginalButton.classList.toggle("is-active", mode === "original");
  modalOverlayButton.classList.toggle("is-active", mode === "overlay");
  modalOverlayButton.disabled = !state.overlayVideoUrl;
}

function loadVideoSource(videoEl, sourceUrl, options) {
  if (!videoEl) {
    return false;
  }

  const mode = options?.mode || "unknown";
  const label = options?.label || "video";
  const fallbackInfo = options?.fallbackInfo || {};
  const normalizedSource = sourceUrl || "";

  console.info("[overlay-video] current video mode", { label, mode, currentVideoMode: state.visualizationMode });
  console.info("[overlay-video] current video url", { label, mode, currentVideoUrl: normalizedSource });

  if (!normalizedSource) {
    const message = label === "main"
      ? "Original video failed to load."
      : "Overlay video failed to load.";
    updateVideoFailure(message, fallbackInfo, label);
    console.error("[overlay-video] load failed (missing source)", { label, mode, fallbackInfo });
    return false;
  }

  videoEl.preload = "auto";
  videoEl.pause();
  videoEl.srcObject = null;
  videoEl.removeAttribute("src");

  return new Promise((resolve) => {
    let settled = false;
    const cleanup = () => {
      videoEl.removeEventListener("loadedmetadata", onLoadedMetadata);
      videoEl.removeEventListener("loadeddata", onLoadedData);
      videoEl.removeEventListener("canplay", onCanPlay);
      videoEl.removeEventListener("error", onError);
    };

    const finish = (ok) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve(ok);
    };

    const onLoadedMetadata = () => {
      console.info("[overlay-video] video metadata loaded", {
        label,
        mode,
        currentVideoUrl: videoEl.currentSrc || videoEl.src,
        videoWidth: videoEl.videoWidth,
        videoHeight: videoEl.videoHeight,
        duration: videoEl.duration,
      });
      console.info("[overlay-video] video metadata loaded success", { label, mode });
    };

    const onLoadedData = () => {
      console.info("[overlay-video] video load success", {
        label,
        mode,
        currentVideoUrl: videoEl.currentSrc || videoEl.src,
      });
      videoEl.play().catch((error) => {
        console.info("[overlay-video] autoplay blocked", { label, mode, error: error?.message || String(error) });
      });
      finish(true);
    };

    const onCanPlay = () => {
      console.debug("[overlay-video] canplay", { label, mode, currentVideoUrl: videoEl.currentSrc || videoEl.src });
    };

    const onError = () => {
      const error = videoEl.error;
      const message = label === "main"
        ? "Original video failed to load."
        : "Overlay video failed to load.";
      console.error("[overlay-video] video load failure", {
        label,
        mode,
        currentVideoUrl: videoEl.currentSrc || videoEl.src,
        errorCode: error ? error.code : null,
        errorMessage: error ? error.message || null : null,
        fallbackInfo,
      });
      updateVideoFailure(message, fallbackInfo, label);
      finish(false);
    };

    videoEl.addEventListener("loadedmetadata", onLoadedMetadata, { once: true });
    videoEl.addEventListener("loadeddata", onLoadedData, { once: true });
    videoEl.addEventListener("canplay", onCanPlay, { once: true });
    videoEl.addEventListener("error", onError, { once: true });

    videoEl.src = normalizedSource;
    videoEl.load();
  });
}

function updateVideoFailure(message, fallbackInfo, label) {
  const info = fallbackInfo || {};
  const parts = [message];
  if (info.video_path) {
    parts.push(`File path: ${info.video_path}`);
  }
  if (info.size_mb !== null && info.size_mb !== undefined) {
    parts.push(`File size: ${info.size_mb} MB`);
  }
  if (info.error) {
    parts.push(`Backend response: ${info.error}`);
  }
  if (label === "main") {
    visualizationStatus.textContent = parts.join(" ");
  } else if (label === "modal") {
    console.warn("[overlay-video] modal playback failure", parts.join(" "));
  }
}

async function initCamera() {
  try {
    if (state.mediaStream) {
      return;
    }
    state.availableCameras = await discoverCameras();
    const preferredCamera = state.selectedCameraId || state.availableCameras[0]?.deviceId || null;
    const stream = await requestCameraStream(preferredCamera);
    const trackSettings = stream.getVideoTracks()[0]?.getSettings?.() || {};
    state.selectedCameraId = trackSettings.deviceId || preferredCamera;
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

async function discoverCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === "videoinput");
  } catch (err) {
    console.warn("Unable to enumerate cameras", err);
    return [];
  }
}

async function requestCameraStream(deviceId) {
  const primaryConstraints = deviceId
    ? { video: { deviceId: { exact: deviceId } }, audio: false }
    : { video: true, audio: false };

  try {
    return await navigator.mediaDevices.getUserMedia(primaryConstraints);
  } catch (err) {
    if (deviceId) {
      console.warn("Preferred camera unavailable, falling back to default camera", err);
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    throw err;
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
  updateStatus("Show the club head to the camera. Scanning now...");
  const club = await detectClubFromCamera();
  if (!club) {
    updateStep("Scan club");
    return;
  }

  state.recordedClub = club;
  detectedClubValue.textContent = club;
  detectedClubDetail.textContent = club;
  recordClubStatus.textContent = `Club confirmed: ${club}.`;
  startGuideButton.textContent = "Club Confirmed";
  updateStep("Step back");
  updateStatus("Club confirmed. Step back so your full body is visible.");
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
  state.overlayStyle = "smoothed";
  state.overlayVariants = {};
  state.overlayValidation = null;
  state.videoUploadId = uploadPayload.upload_id;
  analysisPreview.srcObject = null;
  analysisPreview.src = uploadPayload.preview_url || `/uploads/${uploadPayload.file_name}`;

  updateStep("Analyzing");
  await runAnalysis("/api/analyze-swing", {
    video_upload_id: state.videoUploadId,
    club_category: state.recordedClub,
  });
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

async function detectClubFromCamera() {
  try {
    const response = await postFrame("/api/club-detect");
    const result = response?.result || {};
    const club = result.club || result.detected_club;
    if (result.status !== "confirmed" || !club || club === "Not detected") {
      const reason = result.reasoning || "The club was not clear enough to confirm.";
      recordClubStatus.textContent = `Could not confirm the club. ${reason} Try again with the club head centered in the frame.`;
      updateStatus("Club scan needs a clearer view. Try again.");
      return null;
    }
    return club;
  } catch (error) {
    console.error(error);
    recordClubStatus.textContent = "Club scan is unavailable. Check that the five-way model is installed, then try again.";
    updateStatus("Club scan is unavailable. Try again after the model is installed.");
    return null;
  }
}

async function postFrame(endpoint) {
  const blob = await captureFrameBlob();
  const fd = new FormData();
  fd.append("frame", blob, "frame.png");
  const resp = await fetch(endpoint, { method: "POST", body: fd });
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
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
  state.overlayVariants = visualization?.overlay_variants || result?.overlay_variants || {};
  const overlayVideoUrl = visualization?.smoothed_overlay_video_url || visualization?.overlay_video_url || overlayFiles.find((item) => /\.(mp4|mov|webm)$/i.test(item)) || null;
  const originalVideoUrl = visualization?.original_video_url || state.originalVideoUrl;
  const overlayValidation = result?.overlay_validation || result?.debug?.fallback_status?.overlay_validation || null;
  state.overlayValidation = overlayValidation;
  const trackingStats = result?.tracking || {};
  const qualityMetrics = trackingStats?.quality_metrics || {};
  const trackingDebugUrl = trackingStats?.tracking_debug_video_url || state.overlayVariants?.[state.overlayStyle]?.tracking_debug_video_url || null;
  const rawOverlayUrl = visualization?.raw_overlay_video_url || state.overlayVariants?.raw?.overlay_video_url || null;
  const smoothedOverlayUrl = visualization?.smoothed_overlay_video_url || state.overlayVariants?.smoothed?.overlay_video_url || state.overlayVariants?.simple?.overlay_video_url || null;
  state.originalVideoInfo = {
    video_path: result?.debug?.file_paths?.video_path || result?.video_metadata?.video_path || null,
    size_mb: result?.video_metadata?.size_mb ?? null,
    error: result?.video_metadata?.error || null,
    response: result?.video_metadata || {},
  };
  state.overlayVideoInfo = {
    video_path: overlayValidation?.overlay_path || null,
    size_mb: overlayValidation?.size_mb ?? null,
    error: overlayValidation?.valid === false ? "Overlay validation failed." : null,
    response: overlayValidation || {},
  };

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

  state.overlayStyle = "smoothed";
  setVisualizationMode("original");
  visualizationStatus.textContent = state.overlayVideoUrl
    ? "Overlay ready. Click Overlay to view the processed swing."
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
  advancedHands.textContent = JSON.stringify(
    {
      hand_tracking_rate: qualityMetrics?.hand_tracking_rate ?? qualityMetrics?.hands_tracked_rate ?? null,
      left_hand_tracking_rate: qualityMetrics?.left_hand_tracking_rate ?? null,
      right_hand_tracking_rate: qualityMetrics?.right_hand_tracking_rate ?? null,
      left_hand_confidence: qualityMetrics?.left_hand_confidence ?? null,
      right_hand_confidence: qualityMetrics?.right_hand_confidence ?? null,
      hand_swap_corrections: qualityMetrics?.hand_swap_corrections ?? 0,
      hand_interpolations: qualityMetrics?.hand_interpolations ?? 0,
      hand_rejections: qualityMetrics?.hand_rejections ?? 0,
      hand_recoveries: qualityMetrics?.hand_recoveries ?? 0,
      wrist_recovery_count: qualityMetrics?.wrist_recovery_count ?? 0,
      stale_hand_corrections: qualityMetrics?.stale_hand_corrections ?? 0,
      frames_with_missing_wrist_detection: qualityMetrics?.frames_with_missing_wrist_detection ?? 0,
      hand_tracking_debug_video_url: trackingStats?.hand_tracking_debug_video_url || null,
      wrist_tracking_debug_video_url: trackingStats?.wrist_tracking_debug_video_url || null,
      hand_background_debug_video_url: trackingStats?.hand_background_debug_video_url || null,
      hand_tracking_debug_csv: trackingStats?.hand_tracking_debug_csv || null,
      wrist_tracking_debug_csv: trackingStats?.wrist_tracking_debug_csv || null,
      hand_background_debug_csv: trackingStats?.hand_background_debug_csv || null,
    },
    null,
    2,
  );
  advancedModels.textContent = JSON.stringify(
    {
      status: result?.status,
      club: result?.club,
      swing_score: result?.swing_score,
      score_label: result?.score_label,
      strengths: result?.strengths,
      improvements: result?.improvements,
      overlay_files: result?.overlay_files,
      overlay_variants: state.overlayVariants || {},
      model_outputs: result?.model_outputs || {},
      wrist_tracking_debug_video_url: trackingStats?.wrist_tracking_debug_video_url || null,
      hand_background_debug_video_url: trackingStats?.hand_background_debug_video_url || null,
    },
    null,
    2,
  );
  advancedOverlay.textContent = JSON.stringify(
    {
      overlay_style: "smoothed",
      overlay_video_url: state.overlayVideoUrl,
      overlay_validation: overlayValidation || {},
      raw_overlay_video_url: rawOverlayUrl,
      smoothed_overlay_video_url: smoothedOverlayUrl,
      tracking_debug_video_url: trackingDebugUrl,
      wrist_tracking_debug_video_url: trackingStats?.wrist_tracking_debug_video_url || null,
      hand_background_debug_video_url: trackingStats?.hand_background_debug_video_url || null,
      overlay_quality_metrics: qualityMetrics,
      overlay_files: overlayFiles,
      overlay_variants: state.overlayVariants || {},
    },
    null,
    2,
  );
  advancedTracking.textContent = JSON.stringify(
    {
      total_frames_processed: trackingStats?.total_frames_processed ?? qualityMetrics?.total_frames_processed ?? result?.video_metadata?.frame_count ?? null,
      landmarks_tracked: trackingStats?.landmarks_tracked ?? qualityMetrics?.landmarks_tracked ?? null,
      body_part_tracking_rates: qualityMetrics?.body_part_tracking_rates ?? {},
      body_landmark_coordinates_csv_url: trackingStats?.body_landmark_coordinates_csv_url || null,
      body_landmark_coordinates_wide_csv_url: trackingStats?.body_landmark_coordinates_wide_csv_url || null,
      tracking: result?.tracking || {},
    },
    null,
    2,
  );
  advancedDebug.textContent = JSON.stringify(
    {
      debug: result?.debug || {},
      tracking: result?.tracking || {},
    },
    null,
    2,
  );
}

function wireVideoDebug(videoEl, label) {
  if (!videoEl || videoEl.dataset.debugWired === "true") {
    return;
  }
  videoEl.dataset.debugWired = "true";

  const logEvent = (eventName) => {
    const error = videoEl.error;
    console.debug(`[overlay-debug] ${label}:${eventName}`, {
      currentVideoMode: state.visualizationMode,
      currentVideoUrl: videoEl.currentSrc || videoEl.src,
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
      if (state.originalVideoUrl && videoEl === analysisPreview) {
        analysisPreview.srcObject = null;
        analysisPreview.src = state.originalVideoUrl;
        if (state.isSafari) {
          analysisPreview.load();
        }
      }
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

function formatPercentage(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(1)}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(2);
}

function formatCount(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Math.round(Number(value))}`;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}