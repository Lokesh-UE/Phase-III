const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("fileInput");
const previewImage = $("previewImage");
const predictBtn = $("predictBtn");
const downloadPdfBtn = $("downloadPdfBtn");
const faceDetection = $("faceDetection");
const accuracyMode = $("accuracyMode");
const errorMsg = $("errorMsg");
const timingText = $("timingText");
const loadingOverlay = $("loadingOverlay");

const emptyState = $("emptyState");
const resultsContent = $("resultsContent");
const analysisSection = $("analysisSection");

const predictionEmoji = $("predictionEmoji");
const predictionText = $("predictionText");
const confidenceText = $("confidenceText");
const certaintyBadge = $("certaintyBadge");
const faceNotice = $("faceNotice");
const reliabilityBanner = $("reliabilityBanner");
const interpretationText = $("interpretationText");
const metricsGrid = $("metricsGrid");
const scoreBars = $("scoreBars");
const top3Cards = $("top3Cards");

const annotatedImage = $("annotatedImage");
const preprocessedImage = $("preprocessedImage");
const heatmapImage = $("heatmapImage");
const gradcamImage = $("gradcamImage");

const gradcamPanel = $("gradcamPanel");
const probabilityChart = $("probabilityChart");
const radarChart = $("radarChart");
const donutChart = $("donutChart");
const confidenceGauge = $("confidenceGauge");

const API_BASE = (window.FER_API_URL || "").replace(/\/$/, "");

let selectedFile = null;
let lastResult = null;

function show(el) {
  if (el) el.classList.remove("hidden");
}

function hide(el) {
  if (el) el.classList.add("hidden");
}

function showError(message) {
  if (!errorMsg) return;
  errorMsg.textContent = message;
  show(errorMsg);
}

function clearError() {
  if (!errorMsg) return;
  errorMsg.textContent = "";
  hide(errorMsg);
}

function setLoading(isLoading) {
  if (loadingOverlay) loadingOverlay.classList.toggle("hidden", !isLoading);
  if (predictBtn) {
    predictBtn.disabled = isLoading || predictBtn.hasAttribute("data-model-missing");
  }
}

function setImageSrc(element, base64) {
  if (!element) return;
  const figure = element.closest("figure");
  if (base64) {
    element.src = `data:image/png;base64,${base64}`;
    element.classList.remove("hidden");
    if (figure) figure.classList.remove("hidden");
  } else {
    element.removeAttribute("src");
    if (figure) figure.classList.add("hidden");
  }
}

function defaultAnalysis(data) {
  const scores = data.scores || [];
  const top3 = scores.slice(0, 3).map((item) => ({
    emotion: item.emotion,
    percent: item.percent,
  }));
  return {
    certainty: "Medium",
    certainty_note: "Detailed analysis loaded from fallback values.",
    interpretation: "Prediction completed.",
    second_emotion: scores[1]?.emotion || "—",
    second_confidence_percent: scores[1]?.percent || 0,
    margin_percent:
      scores.length >= 2
        ? Math.round((scores[0].percent - scores[1].percent) * 10) / 10
        : 0,
    entropy: 0,
    normalized_entropy: 0,
    top3,
  };
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please upload a valid image file.");
    return;
  }

  selectedFile = file;
  clearError();

  const reader = new FileReader();
  reader.onload = (event) => {
    if (previewImage) {
      previewImage.src = event.target.result;
      show(previewImage);
    }
    const dropzoneContent = dropzone?.querySelector(".dropzone-content");
    if (dropzoneContent) hide(dropzoneContent);
  };
  reader.readAsDataURL(file);
}

if (dropzone && fileInput) {
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    handleFile(event.dataTransfer.files[0]);
  });
}

if (fileInput) {
  fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
}

function renderScores(scores, topEmotion) {
  if (!scoreBars) return;
  scoreBars.innerHTML = "";
  scores.forEach((item) => {
    const row = document.createElement("div");
    row.className = "score-row";

    const label = document.createElement("div");
    label.className = "score-label";
    label.textContent = `${item.emoji} ${item.emotion}`;

    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    if (item.emotion === topEmotion) fill.classList.add("top");
    fill.style.width = `${item.percent}%`;
    if (item.color) fill.style.background = item.color;
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "score-value";
    value.textContent = `${item.percent}%`;

    row.append(label, track, value);
    scoreBars.appendChild(row);
  });
}

function renderMetrics(analysis) {
  if (!metricsGrid) return;
  const items = [
    { label: "Runner-up", value: `${analysis.second_emotion} (${analysis.second_confidence_percent}%)` },
    { label: "Margin", value: `${analysis.margin_percent}%` },
    { label: "Entropy", value: analysis.entropy },
    { label: "Uncertainty", value: `${Math.round(analysis.normalized_entropy * 100)}%` },
  ];

  metricsGrid.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<span>${item.label}</span><strong>${item.value}</strong>`;
    metricsGrid.appendChild(card);
  });
}

function renderTop3(top3) {
  if (!top3Cards) return;
  top3Cards.innerHTML = "";
  if (!top3 || top3.length === 0) {
    top3Cards.innerHTML = '<p class="muted">No ranking data available.</p>';
    return;
  }
  top3.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "top3-card";
    card.innerHTML = `
      <span class="rank">#${index + 1}</span>
      <strong>${item.emotion}</strong>
      <span class="top3-percent">${item.percent}%</span>
    `;
    top3Cards.appendChild(card);
  });
}

function setCertaintyBadge(level, note) {
  if (!certaintyBadge) return;
  const levelMap = { High: "high", Medium: "medium", Low: "low" };
  const cssLevel = levelMap[level] || "medium";
  certaintyBadge.textContent = level || "—";
  certaintyBadge.className = `certainty-badge ${cssLevel}`;
  certaintyBadge.title = note || "";
}

function renderResult(data) {
  lastResult = data;
  hide(emptyState);
  show(resultsContent);
  show(analysisSection);
  if (downloadPdfBtn) show(downloadPdfBtn);

  if (predictionEmoji) predictionEmoji.textContent = data.emoji || "😐";
  if (predictionText) predictionText.textContent = data.emotion || "Unknown";
  if (confidenceText) {
    const raw = data.raw_emotion && data.raw_emotion !== data.emotion
      ? ` (best match: ${data.raw_emotion})`
      : "";
    confidenceText.textContent = `${data.confidence_percent ?? 0}% confidence${raw}`;
  }

  if (timingText) {
    timingText.textContent = data.processing_ms
      ? `Analysis completed in ${data.processing_ms} ms`
      : "";
    show(timingText);
  }

  const analysis = data.analysis || defaultAnalysis(data);
  setCertaintyBadge(analysis.certainty || "Medium", analysis.certainty_note || "");
  if (interpretationText) interpretationText.textContent = data.reliability_note || analysis.interpretation || "";

  if (reliabilityBanner) {
    reliabilityBanner.textContent = data.is_reliable
      ? "Verified: high-confidence identification passed quality checks."
      : (data.reliability_note || "Low confidence — upload a clearer face photo.");
    reliabilityBanner.className = `reliability-banner ${data.is_reliable ? "reliable" : "unreliable"}`;
    show(reliabilityBanner);
  }

  if (faceNotice) {
    faceNotice.textContent = data.face_detected
      ? "Face detected and cropped automatically."
      : "No face detected — enable face detection and upload a clear frontal face.";
    show(faceNotice);
  }

  renderMetrics(analysis);
  renderScores(data.scores || [], data.raw_emotion || data.emotion);
  renderTop3(analysis.top3 || []);

  setImageSrc(annotatedImage, data.annotated_image);
  setImageSrc(preprocessedImage, data.preprocessed_image);
  setImageSrc(heatmapImage, data.heatmap_image);
  setImageSrc(gradcamImage, data.gradcam_image);

  const charts = data.charts || {};
  setImageSrc(gradcamPanel, charts.gradcam_panel);
  setImageSrc(probabilityChart, charts.probability_chart);
  setImageSrc(radarChart, charts.radar_chart);
  setImageSrc(donutChart, charts.donut_chart);
  setImageSrc(confidenceGauge, charts.confidence_gauge);

  if (data.chart_error) showError(`Charts warning: ${data.chart_error}`);

  if (analysisSection && (Object.keys(charts).length > 0 || data.gradcam_image)) {
    show(analysisSection);
    analysisSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function downloadPdfReport() {
  if (!lastResult) {
    showError("Run analysis first before downloading the PDF.");
    return;
  }
  try {
    const response = await fetch(`${API_BASE}/download-pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastResult),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "PDF download failed.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "facial_expression_report.pdf";
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    showError(error.message);
  }
}

if (downloadPdfBtn) {
  downloadPdfBtn.addEventListener("click", downloadPdfReport);
}

if (predictBtn) {
  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Upload an image first.");
      return;
    }

    clearError();
    setLoading(true);

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("face_detection", faceDetection?.checked ? "true" : "false");
    formData.append("accuracy_mode", accuracyMode?.checked ? "true" : "false");

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed.");
      renderResult(data);
    } catch (error) {
      showError(error.message || "Something went wrong. Is the server running?");
    } finally {
      setLoading(false);
    }
  });
}

if (predictBtn?.disabled) {
  predictBtn.setAttribute("data-model-missing", "true");
}

async function checkApiHealth() {
  const statusPill = $("statusPill");
  const apiAlert = $("apiAlert");
  if (!API_BASE) {
    if (statusPill) {
      statusPill.textContent = "API URL missing";
      statusPill.className = "status-pill missing";
    }
    show(apiAlert);
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/health`, { method: "GET" });
    const data = await response.json();
    if (!response.ok || data.status !== "ok") throw new Error("unhealthy");

    if (statusPill) {
      statusPill.textContent = data.model_ready ? "API ready" : "API online (no model)";
      statusPill.className = `status-pill ${data.model_ready ? "ready" : "missing"}`;
    }
    hide(apiAlert);

    if (predictBtn && data.model_ready) {
      predictBtn.disabled = false;
      predictBtn.removeAttribute("data-model-missing");
    }
  } catch {
    if (statusPill) {
      statusPill.textContent = "API offline";
      statusPill.className = "status-pill missing";
    }
    show(apiAlert);
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.setAttribute("data-model-missing", "true");
    }
  }
}

checkApiHealth();
