const state = {
  videoUploadId: null,
  clubImageUploadId: null,
  analysisId: null,
  workflowState: "idle",
  mediaStream: null,
  recorder: null,
};

const videoInput = document.getElementById("videoInput");
const clubImageInput = document.getElementById("clubImageInput");
const clubCategorySelect = document.getElementById("clubCategory");

const videoPreview = document.getElementById("videoPreview");
const clubPreview = document.getElementById("clubPreview");
const statusText = document.getElementById("statusText");
const summaryOutput = document.getElementById("summaryOutput");
const metricsOutput = document.getElementById("metricsOutput");
const feedbackList = document.getElementById("feedbackList");
const metricList = document.getElementById("metricList");
const overallScoreValue = document.getElementById("overallScoreValue");
const overallGrade = document.getElementById("overallGrade");
const finalClubValue = document.getElementById("finalClubValue");
const frameCountValue = document.getElementById("frameCountValue");
const analysisIdValue = document.getElementById("analysisIdValue");

const uploadButton = document.getElementById("uploadButton");
const analyzeButton = document.getElementById("analyzeButton");
const downloadPdfButton = document.getElementById("downloadPdfButton");
const downloadDocxButton = document.getElementById("downloadDocxButton");

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  if (!file) {
    return;
  }
  videoPreview.src = URL.createObjectURL(file);
  updateStatus("Video selected. Click upload to continue.");
});

clubImageInput.addEventListener("change", () => {
  const file = clubImageInput.files[0];
  if (!file) {
    return;
  }
  clubPreview.src = URL.createObjectURL(file);
  updateStatus("Club image selected. Click upload to continue.");
});

uploadButton.addEventListener("click", async () => {
  const videoFile = videoInput.files[0];
  const imageFile = clubImageInput.files[0];

  if (!videoFile) {
    updateStatus("Please select a swing video before uploading.");
    return;
  }

  try {
    updateStatus("Uploading files...");

    state.videoUploadId = await uploadFile("/api/upload-video", "video", videoFile);

    if (imageFile) {
      state.clubImageUploadId = await uploadFile("/api/upload-club-image", "image", imageFile);
    } else {
      state.clubImageUploadId = null;
    }

    analyzeButton.disabled = false;
    updateStatus("Upload complete. You can now run analysis.");
  } catch (error) {
    console.error(error);
    updateStatus("Upload failed. Check the selected files and try again.");
  }
});

// Guided camera workflow for club recognition, body check, recording, and analysis
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    state.mediaStream = stream;
    videoPreview.srcObject = stream;
    videoPreview.play();
    state.workflowState = "idle";
    updateStatus("Camera ready. Click Record Swing to begin.");
  } catch (err) {
    console.error("Camera error", err);
    updateStatus("Unable to access camera. You can still upload a video file.");
  }
}

function captureFrameBlob() {
  const w = videoPreview.videoWidth || 640;
  const h = videoPreview.videoHeight || 480;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoPreview, 0, 0, w, h);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

async function postFrameForClubDetection() {
  const blob = await captureFrameBlob();
  const fd = new FormData();
  fd.append("frame", blob, "frame.png");
  const resp = await fetch("/api/club-detect", { method: "POST", body: fd });
  return resp.json();
}

async function postFrameForBodyCheck() {
  const blob = await captureFrameBlob();
  const fd = new FormData();
  fd.append("frame", blob, "frame.png");
  const resp = await fetch("/api/body-check", { method: "POST", body: fd });
  return resp.json();
}

async function recordSwingAutomated(durationMs = 5000) {
  if (!state.mediaStream) throw new Error("No media stream available");
  const options = { mimeType: "video/webm;codecs=vp9" };
  const recorder = new MediaRecorder(state.mediaStream, options);
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();
  updateStatus("Recording... Swing now.");
  await new Promise((res) => setTimeout(res, durationMs));
  recorder.stop();
  await new Promise((res) => (recorder.onstop = res));
  const blob = new Blob(chunks, { type: "video/webm" });
  return blob;
}

analyzeButton.addEventListener("click", async () => {
  // Begin guided workflow
  if (!state.mediaStream) {
    updateStatus("Camera not initialized. Clicking will try to access the camera.");
    await initCamera();
    return;
  }

  try {
    state.workflowState = "club_recognition";
    updateStatus("Show the butt/end of your club to the camera.");

    // Try multiple frames until confident or timeout
    let clubResult = null;
    const maxAttempts = 6;
    for (let i = 0; i < maxAttempts; i++) {
      const resp = await postFrameForClubDetection();
      clubResult = resp.result || resp;
      if (clubResult && clubResult.confidence >= 0.6) break;
      await new Promise((r) => setTimeout(r, 700));
    }

    if (!clubResult || clubResult.confidence < 0.5) {
      updateStatus("Could not confirm club. Please hold the butt/end of the club closer to the camera.");
      state.workflowState = "idle";
      return;
    }

    updateStatus(`Club confirmed as: ${clubResult.category} (conf: ${Number(clubResult.confidence).toFixed(2)})`);
    state.workflowState = "club_confirmed";

    // Body detection
    state.workflowState = "body_detection";
    updateStatus("Checking body visibility. Step back until your full body is visible.");
    let visible = false;
    for (let i = 0; i < 8; i++) {
      const bodyResp = await postFrameForBodyCheck();
      const check = bodyResp.check || bodyResp;
      if (check && check.visible) {
        visible = true;
        break;
      }
      await new Promise((r) => setTimeout(r, 500));
    }

    if (!visible) {
      updateStatus("Body not fully visible. Step back until your full body is visible.");
      state.workflowState = "idle";
      return;
    }

    // Start recording automatically for a short capture window
    state.workflowState = "recording";
    updateStatus("Recording swing in 1s. Prepare to swing.");
    await new Promise((r) => setTimeout(r, 1000));
    const videoBlob = await recordSwingAutomated(5000);

    // Upload recorded swing
    updateStatus("Uploading recorded swing for analysis...");
    const fd = new FormData();
    fd.append("video", videoBlob, "swing.webm");
    const uploadResp = await fetch("/api/record-swing", { method: "POST", body: fd });
    const uploadPayload = await uploadResp.json();
    state.videoUploadId = uploadPayload.upload_id;

    // Trigger analysis
    state.workflowState = "analyzing";
    updateStatus("Analyzing swing. This may take a moment...");
    const analyzeResp = await fetch("/api/analyze-swing", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_upload_id: state.videoUploadId, club_category: clubCategorySelect.value || null }),
    });
    const analysisResult = await analyzeResp.json();
    state.analysisId = analysisResult.analysis_id;
    renderSummary(analysisResult);
    renderMetrics(analysisResult.metrics || {});
    renderMetricList(analysisResult.metrics || {});
    renderFeedback(analysisResult.feedback || []);
    updateStatus("Analysis complete.");
    state.workflowState = "results";
    downloadPdfButton.disabled = false;
    downloadDocxButton.disabled = false;
  } catch (err) {
    console.error(err);
    updateStatus("Guided capture failed. You can still upload a video for analysis.");
    state.workflowState = "idle";
  }
});

downloadPdfButton.addEventListener("click", () => {
  requestReport("pdf");
});

downloadDocxButton.addEventListener("click", () => {
  requestReport("docx");
});

async function uploadFile(endpoint, fieldName, file) {
  const formData = new FormData();
  formData.append(fieldName, file);

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

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
      headers: {
        "Content-Type": "application/json",
      },
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

function renderSummary(result) {
  const summary = {
    analysis_id: result.analysis_id,
    final_club_category: result.final_club_category,
    pose_frame_count: result.pose_frame_count,
    score: result.score,
    body_tracking: result.body_tracking,
  };
  summaryOutput.textContent = JSON.stringify(summary, null, 2);

  if (analysisIdValue) {
    analysisIdValue.textContent = result.analysis_id || "-";
  }
  if (finalClubValue) {
    finalClubValue.textContent = result.final_club_category || "-";
  }
  if (frameCountValue) {
    frameCountValue.textContent = `${result.pose_frame_count || 0}`;
  }

  const score = result?.score?.overall_score;
  if (overallScoreValue) {
    overallScoreValue.textContent = Number.isFinite(score) ? `${Math.round(score)}` : "--";
  }
  if (overallGrade) {
    overallGrade.textContent = scoreToGrade(score);
  }
}

function renderMetrics(metrics) {
  metricsOutput.textContent = JSON.stringify(metrics, null, 2);
}

function renderMetricList(metrics) {
  if (!metricList) {
    return;
  }

  metricList.innerHTML = "";
  const entries = Object.entries(metrics);
  if (!entries.length) {
    const li = document.createElement("li");
    li.textContent = "No metrics available yet.";
    metricList.appendChild(li);
    return;
  }

  for (const [name, value] of entries) {
    const li = document.createElement("li");
    const normalized = Number(value);
    const displayValue = Number.isFinite(normalized) ? normalized.toFixed(2) : value;
    li.textContent = `${toTitleCase(name)}: ${displayValue}`;
    metricList.appendChild(li);
  }
}

function renderFeedback(items) {
  feedbackList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "No feedback items generated yet.";
    feedbackList.appendChild(li);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    feedbackList.appendChild(li);
  }
}

function updateStatus(message) {
  statusText.textContent = message;
}

function scoreToGrade(score) {
  if (!Number.isFinite(score)) {
    return "Awaiting Analysis";
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
  return "Needs Work";
}

function toTitleCase(text) {
  return text
    .replaceAll("_", " ")
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
