const page = document.querySelector("[data-analysis-id]");
const analysisId = page?.dataset.analysisId;
const state = {
  result: null,
  mode: "original",
  originalVideoUrl: null,
  overlayVideoUrl: null,
  modalOpen: false,
  kpis: {},
};

const reviewStatus = document.getElementById("reviewStatus");
const reviewClub = document.getElementById("reviewClub");
const reviewScore = document.getElementById("reviewScore");
const reviewGrade = document.getElementById("reviewGrade");
const scoreContext = document.getElementById("scoreContext");
const scoreSource = document.getElementById("scoreSource");
const coachSummary = document.getElementById("coachSummary");
const focusText = document.getElementById("focusText");
const heroKpiList = document.getElementById("heroKpiList");
const clubNote = document.getElementById("clubNote");
const mediaFeedbackList = document.getElementById("mediaFeedbackList");
const strengthList = document.getElementById("strengthList");
const improvementList = document.getElementById("improvementList");
const tipList = document.getElementById("tipList");
const drillList = document.getElementById("drillList");
const analysisPreview = document.getElementById("analysisPreview");
const originalViewButton = document.getElementById("originalViewButton");
const overlayViewButton = document.getElementById("overlayViewButton");
const visualizationStatus = document.getElementById("visualizationStatus");
const downloadOverlayButton = document.getElementById("downloadOverlayButton");
const downloadPdfButton = document.getElementById("downloadPdfButton");
const downloadDocxButton = document.getElementById("downloadDocxButton");
const advancedMetrics = document.getElementById("advancedMetrics");
const advancedTracking = document.getElementById("advancedTracking");
const advancedModels = document.getElementById("advancedModels");
const overlayModal = document.getElementById("overlayModal");
const overlayModalPreview = document.getElementById("overlayModalPreview");
const expandOverlayButton = document.getElementById("expandOverlayButton");
const closeOverlayModalButton = document.getElementById("closeOverlayModalButton");
const modalOriginalButton = document.getElementById("modalOriginalButton");
const modalOverlayButton = document.getElementById("modalOverlayButton");
const kpiInfoModal = document.getElementById("kpiInfoModal");
const kpiInfoTitle = document.getElementById("kpiInfoTitle");
const kpiInfoDescription = document.getElementById("kpiInfoDescription");
const closeKpiInfoButton = document.getElementById("closeKpiInfoButton");

originalViewButton.addEventListener("click", () => setVideoMode("original"));
overlayViewButton.addEventListener("click", () => setVideoMode("overlay"));
expandOverlayButton.addEventListener("click", openOverlayModal);
closeOverlayModalButton.addEventListener("click", closeOverlayModal);
modalOriginalButton.addEventListener("click", () => setModalMode("original"));
modalOverlayButton.addEventListener("click", () => setModalMode("overlay"));
closeKpiInfoButton?.addEventListener("click", closeKpiInfo);
heroKpiList?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-kpi-key]");
  if (button) openKpiInfo(button.dataset.kpiKey);
});
kpiInfoModal?.addEventListener("click", (event) => {
  if (event.target === kpiInfoModal || event.target.classList.contains("kpi-info-modal__backdrop")) closeKpiInfo();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !kpiInfoModal?.classList.contains("hidden")) closeKpiInfo();
});
overlayModal.addEventListener("click", (event) => {
  if (event.target === overlayModal || event.target.classList.contains("overlay-modal__backdrop")) {
    closeOverlayModal();
  }
});
downloadPdfButton.addEventListener("click", () => requestReport("pdf"));
downloadDocxButton.addEventListener("click", () => requestReport("docx"));

loadReview();

async function loadReview() {
  if (!analysisId) {
    showReviewError("We couldn’t find that swing review.");
    return;
  }
  try {
    const response = await fetch(`/api/results/${encodeURIComponent(analysisId)}`);
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result.error || "This swing review is no longer available.");
    }
    state.result = result;
    renderReview(result);
  } catch (error) {
    console.error(error);
    showReviewError(error.message || "We couldn’t load this swing review.");
  }
}

function renderReview(result) {
  const detailed = result.gemini_analysis || result.gemini || result.detailed_analysis || result.coaching_details || {};
  const score = Number(result.swing_score);
  const strengths = firstList(detailed.strengths, detailed.what_worked, result.strengths);
  const improvements = firstList(detailed.improvements, detailed.key_adjustments, result.improvements);
  const tips = firstList(detailed.tips, detailed.practice_cues, detailed.coaching_tips, result.tips, improvements);
  const drills = firstList(detailed.drills, detailed.practice_plan, result.drills, result.practice_plan);
  const summary = firstText(detailed.summary, detailed.overview, detailed.coach_summary, result.gemini_summary, result.next_focus);
  const original = result.visualization?.original_video_url || result.original_video_url || result.video_url;
  const overlay = result.visualization?.smoothed_overlay_video_url || result.visualization?.overlay_video_url || result.overlay_variants?.smoothed?.overlay_video_url || result.overlay_variants?.simple?.overlay_video_url || firstVideo(result.overlay_files);

  const geminiStatus = result.gemini?.status;
  const geminiIssue = geminiStatus === "invalid_response" || geminiStatus === "unavailable";
  reviewStatus.textContent = geminiStatus === "success"
    ? "Gemini coaching included"
    : geminiIssue
      ? "Gemini coaching needs a retry"
      : result.status === "success" ? "Vision review ready" : "Review needs attention";
  reviewClub.textContent = result.club || result.detected_club || "Not detected";
  reviewScore.textContent = Number.isFinite(score) ? Math.round(score) : "--";
  reviewGrade.textContent = result.score_label || "Review ready";
  const geminiScore = detailed?.overall_score;
  scoreSource.textContent = result.score_source === "gemini"
    ? "Gemini score · measured pose evidence"
    : geminiStatus === "not_configured"
      ? "Local CV score · Gemini key is not configured"
      : geminiIssue
        ? "Local CV score · Gemini response needs retry"
        : "Local CV score";
  scoreContext.textContent = result.score_source === "gemini"
    ? detailed.score_rationale || result.score_rationale || "Gemini scored the observed body-movement evidence."
    : geminiIssue
      ? "Your swing was tracked, but Gemini’s response could not be read. Select Record new swing to retry."
      : result.video_processed
        ? "Motion tracking and coaching are ready to review."
        : "Use the notes below to capture a clearer next swing.";
  if (geminiScore === null || geminiScore === undefined) {
    scoreSource.textContent = geminiStatus === "success"
      ? "Gemini did not score this swing · not enough pose evidence"
      : scoreSource.textContent;
  }
  coachSummary.textContent = summary || "Your personalized swing feedback is ready.";
  focusText.textContent = result.next_focus || firstText(detailed.next_focus, detailed.priority) || "Use the feedback below to guide your next practice swing.";
  renderHeroKpis(result);
  if (clubNote) {
    clubNote.textContent = result.club_note || firstText(detailed.coach_note, detailed.context) || "";
  }
  const nearbyFeedback = [...strengths.slice(0, 1), ...improvements.slice(0, 2)];
  renderList(mediaFeedbackList, nearbyFeedback, "Your coach’s notes will appear beside the video when the review is ready.");
  renderList(strengthList, strengths, "Your review will highlight the best parts of this swing.");
  renderList(improvementList, improvements, "Your review will identify the highest-value change to make next.");
  renderList(tipList, tips, "Use this space for Gemini practice cues and simple on-course reminders.");
  renderDrills(drills, improvements);
  state.originalVideoUrl = normalizeUrl(original);
  state.overlayVideoUrl = normalizeUrl(overlay);
  setVideoMode(state.overlayVideoUrl ? "overlay" : "original");
  overlayViewButton.disabled = !state.overlayVideoUrl;
  downloadOverlayButton.classList.toggle("hidden", !state.overlayVideoUrl);
  if (state.overlayVideoUrl) {
    downloadOverlayButton.href = state.overlayVideoUrl;
    downloadOverlayButton.download = state.overlayVideoUrl.split("/").pop() || "swing-overlay.mp4";
  }
  advancedMetrics.textContent = JSON.stringify({ advanced_metrics: result.advanced_metrics || {}, warnings: result.warnings || [] }, null, 2);
  advancedTracking.textContent = JSON.stringify(result.tracking || {}, null, 2);
  advancedModels.textContent = JSON.stringify({ model_outputs: result.model_outputs || {}, detailed_coaching: detailed }, null, 2);
}

function numberMetric(metrics, ...names) {
  for (const name of names) {
    const value = Number(metrics?.[name]);
    if (Number.isFinite(value)) return Math.max(0, Math.min(100, value));
  }
  return null;
}

function kpiStatus(score) {
  if (!Number.isFinite(score)) return "Needs more data";
  if (score >= 85) return "Strong";
  if (score >= 70) return "On track";
  if (score >= 50) return "Developing";
  return "Needs work";
}

function renderHeroKpis(result) {
  if (!heroKpiList) return;

  const metrics = result.advanced_metrics || result.advanced || {};
  const overall = Number(result.swing_score);
  const spineScore = numberMetric(metrics, "spine_maintenance_score")
    ?? (() => {
      const angle = Number(metrics.spine_angle_variation_deg);
      return Number.isFinite(angle) ? Math.max(0, Math.min(100, 100 - angle * 8)) : null;
    })();

  const kpis = [
    {
      key: "sequence",
      name: "Kinematic Sequence",
      score: numberMetric(metrics, "kinematic_sequence_score", "tempo_estimate"),
      detail: "Measures the order and efficiency with which your body segments accelerate and decelerate—pelvis, then torso, then arms. SwingSight estimates the sequence from motion peaks in the side-view pose data.",
    },
    {
      key: "xfactor",
      name: "X-Factor Separation",
      score: numberMetric(metrics, "x_factor_score"),
      detail: "The rotational difference between your pelvis and upper torso near the top of the backswing—the coil that helps create speed. This is a 2D video proxy, so a square side view gives the cleanest reading.",
    },
    {
      key: "spine",
      name: "Spine Angle Maintenance",
      score: spineScore,
      detail: "Tracks how consistently you maintain spinal tilt and posture from address toward impact, helping you keep a repeatable strike zone and swing plane.",
    },
    {
      key: "weightShift",
      name: "Lateral Weight Shift",
      score: numberMetric(metrics, "lateral_weight_shift_score", "balance_score", "weight_shift"),
      detail: "Monitors your lateral movement and pressure-transfer proxy toward the lead side, looking for a centered move without too much sway or thrust.",
    },
    {
      key: "overall",
      name: "Overall Swing Score",
      score: Number.isFinite(overall) ? Math.max(0, Math.min(100, overall)) : null,
      detail: "A 0–100 all-in SwingSight rating that combines measured sequencing, posture, and arm mechanics against the app’s movement-quality model. It is a video-based estimate, not a launch-monitor reading.",
    },
  ];

  state.kpis = Object.fromEntries(kpis.map((kpi) => [kpi.key, kpi]));
  heroKpiList.innerHTML = "";

  kpis.forEach((kpi) => {
    const score = Number.isFinite(kpi.score) ? Math.round(kpi.score) : null;
    const item = document.createElement("li");
    item.className = "hero-kpi-item";
    item.style.setProperty("--kpi-progress", String(score ?? 0));

    const heading = document.createElement("div");
    heading.className = "hero-kpi-heading";

    const title = document.createElement("strong");
    title.textContent = kpi.name;

    const info = document.createElement("button");
    info.className = "kpi-info-button";
    info.type = "button";
    info.dataset.kpiKey = kpi.key;
    info.setAttribute("aria-label", "About " + kpi.name);
    info.setAttribute("aria-haspopup", "dialog");
    info.textContent = "i";

    heading.append(title, info);

    const ring = document.createElement("div");
    ring.className = "hero-kpi-ring";
    ring.setAttribute("aria-label", score === null ? kpi.name + ": more data needed" : kpi.name + ": " + score + " out of 100");
    ring.innerHTML = "<strong>" + (score === null ? "—" : score) + "</strong>";

    const status = document.createElement("p");
    const statusLabel = score === null ? "More data" : kpiStatus(score);
    status.className = "hero-kpi-status status-" + statusLabel.toLowerCase().replace(/\s+/g, "-");
    status.textContent = statusLabel;

    item.append(heading, ring, status);
    heroKpiList.appendChild(item);
  });
}

function openKpiInfo(key) {
  const kpi = state.kpis[key];
  if (!kpi || !kpiInfoModal) return;
  kpiInfoTitle.textContent = kpi.name;
  kpiInfoDescription.textContent = kpi.detail;
  kpiInfoModal.classList.remove("hidden");
  kpiInfoModal.setAttribute("aria-hidden", "false");
  closeKpiInfoButton?.focus();
}

function closeKpiInfo() {
  if (!kpiInfoModal) return;
  kpiInfoModal.classList.add("hidden");
  kpiInfoModal.setAttribute("aria-hidden", "true");
}

function firstText(...values) {
  return values.find((value) => typeof value === "string" && value.trim()) || "";
}

function firstList(...values) {
  for (const value of values) {
    if (Array.isArray(value) && value.length) return value;
    if (typeof value === "string" && value.trim()) return [value];
  }
  return [];
}

function formatItem(item) {
  if (typeof item === "string") return item;
  if (item && typeof item === "object") return item.title || item.name || item.tip || item.description || item.text || JSON.stringify(item);
  return String(item || "");
}

function renderList(element, items, fallback) {
  element.innerHTML = "";
  const values = items.length ? items : [fallback];
  values.slice(0, 5).forEach((item) => {
    const entry = document.createElement("li");
    entry.textContent = formatItem(item);
    element.appendChild(entry);
  });
}

function renderDrills(drills, improvements) {
  drillList.innerHTML = "";
  const values = drills.length ? drills : improvements.slice(0, 3).map((item, index) => ({
    title: ["Slow rehearsal", "Feel the position", "Bring it to the ball"][index] || "Practice the move",
    description: formatItem(item),
  }));
  const fallback = values.length ? values : [{ title: "Your practice plan is coming", description: "Gemini drill recommendations will appear here with your detailed analysis." }];
  fallback.slice(0, 4).forEach((drill, index) => {
    const item = typeof drill === "string" ? { title: `Practice cue ${index + 1}`, description: drill } : drill;
    const card = document.createElement("article");
    card.className = "drill-card";
    card.innerHTML = `<span class="drill-number">0${index + 1}</span><h3></h3><p></p>`;
    card.querySelector("h3").textContent = item.title || item.name || `Practice cue ${index + 1}`;
    card.querySelector("p").textContent = item.description || item.instructions || item.tip || formatItem(item);
    drillList.appendChild(card);
  });
}

function normalizeUrl(value) {
  if (!value || typeof value !== "string") return null;
  return value.startsWith("/") || value.startsWith("http") || value.startsWith("blob:") ? value : `/${value}`;
}

function firstVideo(files) {
  return Array.isArray(files) ? files.find((item) => typeof item === "string" && /\.(mp4|mov|webm)$/i.test(item)) : null;
}

function setVideoMode(mode) {
  state.mode = mode;
  const source = mode === "overlay" ? state.overlayVideoUrl : state.originalVideoUrl;
  const label = mode === "overlay" ? "Processed overlay video" : "Original swing video";
  if (!source) {
    visualizationStatus.textContent = "Video is not available for this review.";
    return;
  }
  analysisPreview.pause();
  analysisPreview.src = source;
  analysisPreview.load();
  visualizationStatus.textContent = label;
  originalViewButton.classList.toggle("is-active", mode === "original");
  overlayViewButton.classList.toggle("is-active", mode === "overlay");
  if (state.modalOpen) setModalMode(mode);
}

function openOverlayModal() {
  state.modalOpen = true;
  overlayModal.classList.remove("hidden");
  overlayModal.setAttribute("aria-hidden", "false");
  setModalMode(state.mode);
}

function closeOverlayModal() {
  state.modalOpen = false;
  overlayModal.classList.add("hidden");
  overlayModal.setAttribute("aria-hidden", "true");
  overlayModalPreview.pause();
  overlayModalPreview.removeAttribute("src");
}

function setModalMode(mode) {
  const source = mode === "overlay" ? state.overlayVideoUrl : state.originalVideoUrl;
  if (source) {
    overlayModalPreview.src = source;
    overlayModalPreview.load();
  }
  modalOriginalButton.classList.toggle("is-active", mode === "original");
  modalOverlayButton.classList.toggle("is-active", mode === "overlay");
  modalOverlayButton.disabled = !state.overlayVideoUrl;
}

async function requestReport(format) {
  try {
    const response = await fetch(`/api/reports/${encodeURIComponent(analysisId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Report generation failed.");
    window.location.assign(payload.download_url);
  } catch (error) {
    console.error(error);
    reviewStatus.textContent = error.message || "Unable to generate report";
  }
}

function showReviewError(message) {
  reviewStatus.textContent = "Review unavailable";
  coachSummary.textContent = message;
  focusText.textContent = "Record a new swing to start a fresh review.";
}