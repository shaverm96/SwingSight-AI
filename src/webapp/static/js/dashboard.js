const state = {
  videoUploadId: null,
  clubImageUploadId: null,
  analysisId: null,
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

analyzeButton.addEventListener("click", async () => {
  if (!state.videoUploadId) {
    updateStatus("Upload a video first.");
    return;
  }

  try {
    updateStatus("Running analysis locally. This may take a minute...");

    const payload = {
      video_upload_id: state.videoUploadId,
      club_image_upload_id: state.clubImageUploadId,
      club_category: clubCategorySelect.value || null,
    };

    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Analysis failed with status ${response.status}`);
    }

    const result = await response.json();
    state.analysisId = result.analysis_id;

    renderSummary(result);
    renderMetrics(result.metrics || {});
    renderMetricList(result.metrics || {});
    renderFeedback(result.feedback || []);

    downloadPdfButton.disabled = false;
    downloadDocxButton.disabled = false;
    updateStatus("Analysis complete.");
  } catch (error) {
    console.error(error);
    updateStatus("Analysis failed. Check backend logs for details.");
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
