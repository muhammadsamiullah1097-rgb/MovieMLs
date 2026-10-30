const API = "";
const state = { models: null, task: "classification" };
const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(API + path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Something went wrong.");
  return data;
}

function setStatus(id, message, isError = false) {
  const el = $(id);
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function showView(view) {
  document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
  document.querySelectorAll(".nav").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $(view + "View").classList.add("active");
  const names = { studio: "Model Studio", preprocessing: "Data Preparation", features: "Feature Analysis", recommend: "Movie Recommender", datasets: "Dataset Files" };
  $("pageTitle").textContent = names[view];
  if (view === "preprocessing") loadPreprocessing();
  if (view === "datasets") loadDatasetInfo();
}

document.querySelectorAll(".nav").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));

function updateModels() {
  const task = $("taskSelect").value;
  state.task = task;
  const models = task === "classification" ? state.models.classification : task === "regression" ? state.models.regression : state.models.text;
  $("modelSelect").innerHTML = models.map((item) => `<option value="${item.id}">${item.name}</option>`).join("");
  if (task === "classification") {
    $("targetSelect").innerHTML = '<option value="rating_tier">Rating tier (from vote_average)</option>';
    $("taskHint").textContent = "Predict whether a movie is Low, Medium or High rated.";
    $("scalerField").classList.remove("hidden");
    $("smoteRow").classList.remove("hidden");
  } else if (task === "regression") {
    $("targetSelect").innerHTML = '<option value="vote_average">Movie vote_average</option>';
    $("taskHint").textContent = "Predict a movie's numeric average rating.";
    $("scalerField").classList.remove("hidden");
    $("smoteRow").classList.add("hidden");
  } else {
    $("targetSelect").innerHTML = '<option value="primary_genre">Primary genre from overview</option>';
    $("taskHint").textContent = "Predict the main movie genre from its overview text.";
    $("scalerField").classList.add("hidden");
    $("smoteRow").classList.add("hidden");
  }
}

function displayMetrics(result) {
  const m = result.metrics;
  const entries = result.task === "regression"
    ? [["R-squared", m.r2], ["MAE", m.mae], ["RMSE", m.rmse], ["MSE", m.mse], ["Median AE", m.median_absolute_error]]
    : [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["Weighted F1", m.f1], ["Balanced Acc.", m.balanced_accuracy], ["MCC", m.mcc]];
  $("metricGrid").innerHTML = entries.map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value === null || value === undefined ? "—" : Number(value).toFixed(3)}</strong></div>`).join("");
  if (result.task !== "regression" && m.confusion_matrix) {
    const labels = result.labels || [];
    $("confusionWrap").classList.remove("hidden");
    $("confusionWrap").innerHTML = `<div class="result-subhead"><h4>Confusion matrix</h4><span>Actual rows · Predicted columns</span></div><div class="table-wrap"><table><thead><tr><th>Actual / Predicted</th>${labels.map((label) => `<th>${label}</th>`).join("")}</tr></thead><tbody>${m.confusion_matrix.map((row, index) => `<tr><th>${labels[index] || index}</th>${row.map((value) => `<td>${value}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  } else {
    $("confusionWrap").classList.add("hidden");
    $("confusionWrap").innerHTML = "";
  }
  $("resultMeta").textContent = `${result.train_rows.toLocaleString()} TRAIN ROWS  ·  ${result.test_rows.toLocaleString()} TEST ROWS  ·  ${result.seconds}s  ·  ${result.features.length} INPUT FEATURES${result.smote_applied ? "  ·  SMOTE ON" : ""}`;
  const rows = result.preview || [];
  const headers = rows.length ? Object.keys(rows[0]) : [];
  $("previewHead").innerHTML = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  $("previewBody").innerHTML = rows.map((row) => `<tr>${headers.map((h) => `<td>${row[h] ?? "—"}</td>`).join("")}</tr>`).join("");
  $("resultTitle").textContent = result.model_name;
  $("resultNote").textContent = `${result.note} Saved model: ${result.saved_model || "comparison run"}.`;
  $("resultState").textContent = "COMPLETED";
  $("emptyState").classList.add("hidden");
  $("compareContent").classList.add("hidden");
  $("resultContent").classList.remove("hidden");
}

async function runSelected() {
  const button = $("runBtn");
  button.disabled = true;
  button.textContent = "Training model...";
  setStatus("runStatus", "Training with a fixed train/test split. The test rows stay separate.");
  $("resultState").textContent = "RUNNING";
  try {
    const result = await api("/api/run", { method: "POST", body: JSON.stringify({
      task: state.task,
      target: $("targetSelect").value,
      model: $("modelSelect").value,
      test_size: Number($("testSize").value) / 100,
      scaler: $("scalerSelect").value,
      use_smote: $("smoteToggle").checked,
    }) });
    displayMetrics(result);
    setStatus("runStatus", `${result.model_name} completed successfully.`);
  } catch (error) {
    $("resultState").textContent = "ERROR";
    setStatus("runStatus", error.message, true);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Run selected model <span>→</span>';
  }
}

async function compareAll() {
  const button = $("compareBtn");
  button.disabled = true;
  button.textContent = "Comparing models...";
  setStatus("runStatus", "Running each algorithm with the same split settings. This can take longer than one model.");
  $("resultState").textContent = "RUNNING";
  try {
    const result = await api("/api/compare", { method: "POST", body: JSON.stringify({
      task: state.task,
      target: $("targetSelect").value,
      test_size: Number($("testSize").value) / 100,
      scaler: $("scalerSelect").value,
      use_smote: $("smoteToggle").checked,
    }) });
    $("resultContent").classList.add("hidden");
    $("emptyState").classList.add("hidden");
    $("compareContent").classList.remove("hidden");
    $("resultTitle").textContent = "Algorithm comparison";
    $("resultState").textContent = "COMPLETED";
    $("compareMetric").textContent = `SORTED BY ${result.metric.toUpperCase()}`;
    $("compareBody").innerHTML = result.results.map((item, index) => `<tr><td>${index + 1}</td><td>${item.model_name}</td><td><strong>${Number(item.score).toFixed(4)}</strong></td><td>${item.seconds}s</td></tr>`).join("");
    $("bestModel").textContent = `Top test result: ${result.best_model.model_name} · ${result.metric}: ${Number(result.best_model.score).toFixed(4)}`;
    $("compareErrors").textContent = result.errors.length ? `Models not completed: ${result.errors.map((item) => item.model).join(", ")}` : result.note;
    setStatus("runStatus", `Compared ${result.results.length} models successfully.`);
  } catch (error) {
    $("resultState").textContent = "ERROR";
    setStatus("runStatus", error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Compare all";
  }
}

async function loadPreprocessing() {
  try {
    const data = await api("/api/preprocessing");
    const cards = [["MOVIE RECORDS", data.rows], ["MODEL-READY ROWS", data.model_rows], ["DUPLICATE ROWS", data.duplicate_rows]];
    $("prepCards").innerHTML = cards.map(([label, value]) => `<div class="stat-card"><span>${label}</span><strong>${Number(value).toLocaleString()}</strong></div>`).join("");
    $("missingBody").innerHTML = data.missing.map((item) => `<tr><td>${item.column}</td><td>${Number(item.count).toLocaleString()}</td></tr>`).join("");
    const headers = data.preview_columns;
    $("prepHead").innerHTML = `<tr>${headers.map((item) => `<th>${item}</th>`).join("")}</tr>`;
    $("prepBody").innerHTML = data.preview.map((row) => `<tr>${headers.map((item) => `<td>${row[item] ?? "—"}</td>`).join("")}</tr>`).join("");
  } catch (error) {
    $("prepCards").innerHTML = `<div class="result-note">${error.message}</div>`;
  }
}

async function loadFeatures() {
  const button = $("runFeatures");
  button.disabled = true;
  button.textContent = "Analysing...";
  try {
    const data = await api("/api/features");
    const pca = data.pca;
    $("featureOutput").innerHTML = `
      <div class="panel"><span class="eyebrow">RFE · FEATURE SELECTION</span><h3>Selected input features</h3><div class="feature-list">${data.selected_features.map((name, i) => `<div class="feature-row"><span>${i + 1}. ${name}</span><span>Selected</span></div>`).join("")}</div><p class="result-note">Recursive Feature Elimination ranks transformed features using Logistic Regression.</p></div>
      <div class="panel"><span class="eyebrow">PCA · DIMENSIONALITY REDUCTION</span><h3>Two principal components</h3><div class="metric-grid"><div class="metric-card"><span>PC1 variance</span><strong>${pca.explained_variance[0]}</strong></div><div class="metric-card"><span>PC2 variance</span><strong>${pca.explained_variance[1]}</strong></div><div class="metric-card"><span>Total variance</span><strong>${pca.total_explained_variance}</strong></div></div><div class="result-subhead"><h4>First 10 projections</h4></div><div class="pca-points">${pca.sample_projection.map((point, i) => `${i + 1}. PC1 ${point[0]} · PC2 ${point[1]}`).join("<br>")}</div><p class="result-note">PCA is applied to the numeric movie features after median imputation and standard scaling.</p></div>`;
  } catch (error) {
    $("featureOutput").innerHTML = `<div class="panel result-note">${error.message}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Run feature analysis →";
  }
}

async function loadDatasetInfo() {
  try {
    const data = await api("/api/summary");
    $("datasetMovieCount").textContent = Number(data.movie_rows).toLocaleString();
    $("movieCount").textContent = Number(data.movie_rows).toLocaleString();
    $("ratingCount").textContent = Number(data.rating_rows).toLocaleString();
    $("datasetFiles").innerHTML = data.files.map((file) => `<div class="file-card"><div class="file-icon">CSV</div><div><h4>${file.name}</h4><p>${file.purpose}</p><small>${file.size_mb} MB</small></div></div>`).join("");
  } catch (error) {
    $("datasetFiles").textContent = error.message;
  }
}

async function recommendContent() {
  const button = $("contentBtn");
  button.disabled = true;
  button.textContent = "Finding similar movies...";
  setStatus("recommendStatus", "Preparing the content similarity index on first use...");
  try {
    const result = await api("/api/recommend/content", { method: "POST", body: JSON.stringify({ title: $("movieTitle").value, limit: 10 }) });
    renderRecommendations(result);
    setStatus("recommendStatus", result.note);
  } catch (error) {
    setStatus("recommendStatus", error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Find similar movies →";
  }
}

async function recommendCollaborative() {
  const button = $("collabBtn");
  button.disabled = true;
  button.textContent = "Finding user patterns...";
  setStatus("recommendStatus", "Building recommendations from MovieLens ratings...");
  try {
    const result = await api("/api/recommend/collaborative", { method: "POST", body: JSON.stringify({ user_id: Number($("userId").value), limit: 10 }) });
    renderRecommendations(result);
    setStatus("recommendStatus", result.note);
  } catch (error) {
    setStatus("recommendStatus", error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Recommend for this user →";
  }
}

function renderRecommendations(result) {
  const movies = result.recommendations || [];
  const selected = result.selected_movie ? `<h3>Because you selected: ${result.selected_movie}</h3>` : `<h3>Recommendations for user ${result.user_id}</h3><p class="muted">Based on ${result.based_on_ratings} ratings</p>`;
  $("recommendOutput").innerHTML = `${selected}<div class="movie-grid">${movies.map((movie) => `<article class="movie-card"><h4>${movie.title}</h4><div class="movie-meta">${movie.year || "Year unknown"} · ${movie.genre} · ${movie.similarity !== undefined ? `SIM ${movie.similarity}` : `SCORE ${movie.score}`}</div><p>${movie.overview || "No overview available."}</p></article>`).join("")}</div>`;
}

let searchDelay;
$("movieTitle").addEventListener("input", () => {
  clearTimeout(searchDelay);
  const value = $("movieTitle").value.trim();
  if (value.length < 2) { $("movieSuggestions").innerHTML = ""; return; }
  searchDelay = setTimeout(async () => {
    try {
      const items = await api(`/api/movies/search?q=${encodeURIComponent(value)}&limit=7`);
      $("movieSuggestions").innerHTML = items.map((item) => `<button class="suggestion" type="button">${item.title}${item.year ? ` (${item.year})` : ""}</button>`).join("");
      document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => { $("movieTitle").value = button.textContent.replace(/\s\(\d{4}\)$/, ""); $("movieSuggestions").innerHTML = ""; }));
    } catch (_) { $("movieSuggestions").innerHTML = ""; }
  }, 220);
});

$("taskSelect").addEventListener("change", updateModels);
$("testSize").addEventListener("input", () => { $("testSizeLabel").textContent = `${$("testSize").value}%`; });
$("runBtn").addEventListener("click", runSelected);
$("compareBtn").addEventListener("click", compareAll);
$("refreshPrep").addEventListener("click", loadPreprocessing);
$("runFeatures").addEventListener("click", loadFeatures);
$("contentBtn").addEventListener("click", recommendContent);
$("collabBtn").addEventListener("click", recommendCollaborative);

async function init() {
  try {
    const [models, data] = await Promise.all([api("/api/models"), api("/api/summary")]);
    state.models = models;
    updateModels();
    $("apiStatus").textContent = "API connected";
    $("movieCount").textContent = Number(data.movie_rows).toLocaleString();
    $("ratingCount").textContent = Number(data.rating_rows).toLocaleString();
    $("datasetMovieCount").textContent = Number(data.movie_rows).toLocaleString();
    $("datasetFiles").innerHTML = data.files.map((file) => `<div class="file-card"><div class="file-icon">CSV</div><div><h4>${file.name}</h4><p>${file.purpose}</p><small>${file.size_mb} MB</small></div></div>`).join("");
  } catch (error) {
    $("apiStatus").textContent = "API disconnected";
    setStatus("runStatus", error.message, true);
  }
}

init();
