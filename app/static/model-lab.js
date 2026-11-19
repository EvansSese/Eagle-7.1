const modelLeague = document.querySelector("#model-league");
const modelLeagueTree = document.querySelector("#model-league-tree");
const modelLeagueSearch = document.querySelector("#model-league-search");
const modelLeagueCount = document.querySelector("#model-league-count");
const modelError = document.querySelector("#model-lab-error");
const modelResults = document.querySelector("#model-results");
const modelScores = document.querySelector("#model-scores");
const modelScoresContext = document.querySelector("#model-scores-context");
const modelScoresCount = document.querySelector("#model-scores-count");
const modelJobMessage = document.querySelector("#model-job-message");
const modelJobState = document.querySelector("#model-job-state");
const modelJobProgress = document.querySelector("#model-job-progress");
const modelActionButtons = [...document.querySelectorAll(".model-action")];
const modelWorkflowSelections = [...document.querySelectorAll(".model-select")];
const modelRunSelected = document.querySelector("#model-run-selected");
const modelRunAll = document.querySelector("#model-run-all");
const modelJobCurrent = document.querySelector("#model-job-current");
const modelProgressLabel = document.querySelector("#model-job-progress-label");
const modelActionLabels = Object.fromEntries(modelActionButtons.map((button) => [
  button.dataset.action, button.closest(".model-card").querySelector("h2").textContent,
]));
const modelButtonLabels = Object.fromEntries(modelActionButtons.map((button) => [
  button.dataset.action, button.textContent,
]));
let modelJobId = null;
let modelRequestId = 0;
let modelSeasonRequestId = 0;
let modelScoresRequestId = 0;
let modelWorkflowRequestId = 0;
let modelWorkflowStates = {};
let modelWorkflowLoading = false;
let modelExpandedCountry = "";
function showModelError(error) {
  modelError.textContent = error.message || String(error);
  modelError.hidden = false;
}

function clearModelError() {
  modelError.hidden = true;
}

function updateModelActionButtons() {
  const hasLeague = Boolean(modelLeague.value);
  const busy = modelJobId !== null;
  const enabled = hasLeague && !busy && !modelWorkflowLoading;
  for (const button of modelActionButtons) {
    const state = modelWorkflowStates[button.dataset.action];
    button.disabled = !enabled;
    let label = modelButtonLabels[button.dataset.action];
    button.textContent = state === "completed" ? "Re-run" : label;
    const card = button.closest(".model-card");
    const status = card.querySelector(".model-card-status");
    card.classList.toggle("completed", state === "completed");
    card.classList.toggle("running", state === "running");
    status.textContent = state === "completed" ? "✓ Completed" :
      (state === "running" ? "Running" : (state === "failed" ? "Failed" : ""));
  }
  for (const checkbox of modelWorkflowSelections) checkbox.disabled = !enabled;
  modelRunSelected.disabled = !enabled || !modelWorkflowSelections.some((input) => input.checked);
  modelRunAll.disabled = !enabled;
}

async function loadModelWorkflowStates() {
  const requestId = ++modelWorkflowRequestId;
  modelWorkflowStates = {};
  modelWorkflowLoading = Boolean(modelLeague.value);
  updateModelActionButtons();
  if (!modelLeague.value) return;
  const params = new URLSearchParams({ league_id: modelLeague.value });
  try {
    const data = await requestJson(`/api/model-lab-workflows?${params}`);
    if (requestId !== modelWorkflowRequestId) return;
    modelWorkflowStates = data.states;
    if (data.active_job_id && modelJobId === null) {
      modelJobId = data.active_job_id;
      void pollModelJob(modelJobId);
    }
    updateModelActionButtons();
  } catch (error) {
    if (requestId === modelWorkflowRequestId) showModelError(error);
  } finally {
    if (requestId === modelWorkflowRequestId) {
      modelWorkflowLoading = false;
      updateModelActionButtons();
    }
  }
}

function renderModelReadiness(status) {
  const rows = [
    ["Feature tables (all seasons)", status.feature_tables.total > 0 &&
      status.feature_tables.available === status.feature_tables.total,
    `${status.feature_tables.available} of ${status.feature_tables.total} available`],
    ["Training dataset", status.datasets.training, status.datasets.training ? "Available" : "Not prepared"],
    ["Prediction dataset", status.datasets.prediction, status.datasets.prediction ? "Available" : "Not prepared"],
    ["Classifier", status.trained.classifier, status.trained.classifier ? "Trained" : "Not trained"],
    ["Regressors", status.trained.regression, status.trained.regression ? "At least one trained" : "Not trained"],
    ["Neural networks", status.trained.neural, status.trained.neural ? "Trained" : "Not all trained"],
  ];
  const table = createElement("table", "model-readiness-table");
  const caption = createElement("caption", "", "Data and model readiness");
  const body = document.createElement("tbody");
  for (const [label, ready, detail] of rows) {
    const row = document.createElement("tr");
    const heading = createElement("th", "", label);
    heading.scope = "row";
    const value = document.createElement("td");
    const icon = createElement("span", `model-readiness-icon ${ready ? "ready" : "warning"}`, ready ? "✓" : "⚠");
    icon.setAttribute("aria-hidden", "true");
    value.append(icon, document.createTextNode(detail));
    row.append(heading, value);
    body.append(row);
  }
  table.append(caption, body);
  const note = createElement("p", "model-readiness-note", "Readiness covers all available seasons for this league.");
  document.querySelector("#model-selection-status").replaceChildren(table, note);
}

async function loadModelStatus() {
  const requestId = ++modelRequestId;
  if (!modelLeague.value) {
    document.querySelector("#model-selection-status").textContent =
      "Select a league to see dataset readiness.";
    return;
  }
  const leagueId = Number(modelLeague.value);
  document.querySelector("#model-selection-status").textContent = "Checking local data…";
  try {
    const params = new URLSearchParams({ league_id: leagueId });
    const status = await requestJson(`/api/model-lab-status?${params}`);
    if (requestId !== modelRequestId) return;
    renderModelReadiness(status);
  } catch (error) {
    if (requestId === modelRequestId) {
      document.querySelector("#model-selection-status").textContent = "Data readiness unavailable.";
      showModelError(error);
    }
  }
}

function formatModelScore(value, format) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(3);
}

function renderModelScores(groups) {
  modelScores.replaceChildren();
  const availableGroups = groups.filter((group) => group.rows.length);
  const modelCount = availableGroups.reduce((total, group) => total + group.rows.length, 0);
  modelScoresCount.textContent = String(modelCount);
  if (!modelCount) {
    modelScores.append(
      createElement(
        "p",
        "model-score-empty",
        "No saved model evaluations are available for this league yet.",
      ),
    );
    return;
  }

  for (const group of availableGroups) {
    const isNeural = group.name === "Neural networks";
    const isRegression = group.name === "Goal regressors";
    const columns = [
      ["model", "Model", "text"],
      ["feature_bundle", "Features", "text"],
    ];
    if (isNeural) {
      columns.push(
        ["holdout_accuracy", "Test / holdout", "percent"],
        ["holdout_log_loss", "Log loss", "decimal"],
        ["double_chance_accuracy", "Double chance", "percent"],
        ["exact_score_accuracy", "Exact score", "percent"],
      );
    } else {
      columns.push(
        ["validation_accuracy", "Validation", "percent"],
        ["holdout_accuracy", "Test / holdout", "percent"],
      );
      if (isRegression) {
        columns.push(
          ["holdout_goal_mae", "Goal MAE", "decimal"],
          ["exact_score_accuracy", "Exact score", "percent"],
        );
      } else {
        columns.push(["holdout_log_loss", "Log loss", "decimal"]);
      }
    }

    const section = createElement("section", "model-score-group");
    section.append(createElement("h3", "", group.name));
    const table = createElement("table", "model-score-table");
    const head = document.createElement("thead");
    const header = document.createElement("tr");
    for (const [, label] of columns) header.append(createElement("th", "", label));
    head.append(header);
    table.append(head);
    const body = document.createElement("tbody");
    for (const row of group.rows) {
      const tableRow = document.createElement("tr");
      for (const [key, , format] of columns) {
        const value = row[key];
        const display = format === "text"
          ? (value || "--")
          : formatModelScore(value, format);
        const cell = createElement("td", key === "model" ? "model-score-model" : "", display);
        if (key === "model") cell.title = row.feature_bundle
          ? `${row.model} · ${row.feature_bundle}`
          : row.model;
        tableRow.append(cell);
      }
      body.append(tableRow);
    }
    table.append(body);
    const tableWrap = createElement("div", "model-score-table-wrap");
    tableWrap.append(table);
    section.append(tableWrap);
    modelScores.append(section);
  }
}

async function loadModelScores(leagueId = modelLeague.value) {
  const requestId = ++modelScoresRequestId;
  modelScores.replaceChildren();
  if (!leagueId) {
    modelScoresContext.textContent =
      "Select a league. Scores distinguish validation from chronological test; -- means unavailable.";
    modelScoresCount.textContent = "--";
    modelScores.append(createElement("p", "model-score-empty", "Choose a league to load its model scores."));
    return;
  }

  const league = leagues.find((item) => item.league_id === Number(leagueId));
  modelScoresContext.textContent = league
    ? `${league.country_name} · ${league.league_name} · validation / chronological test`
    : "Loading saved model scores...";
  modelScoresCount.textContent = "...";
  modelScores.append(createElement("p", "model-score-empty", "Loading model scores..."));
  try {
    const params = new URLSearchParams({ league_id: leagueId });
    const data = await requestJson(`/api/model-lab-metrics?${params}`);
    if (requestId !== modelScoresRequestId || modelLeague.value !== String(leagueId)) return;
    renderModelScores(data.groups);
  } catch (error) {
    if (requestId !== modelScoresRequestId) return;
    modelScoresCount.textContent = "!";
    modelScores.replaceChildren(
      createElement("p", "model-score-empty", `Could not load model scores: ${error.message}`),
    );
  }
}

function renderModelMetrics(metrics) {
  modelResults.replaceChildren();
  if (!Array.isArray(metrics) || !metrics.length) {
    modelResults.hidden = true;
    return;
  }
  const isRegression = modelResults.dataset.modelType === "regression";
  const isNeural = modelResults.dataset.modelType === "neural";
  const columns = isNeural
    ? [
      ["model", "Model"],
      ["test_fixture_count", "Test fixtures"],
      ["outcome_accuracy", "3-way accuracy"],
      ["double_chance_accuracy", "Double-chance accuracy"],
      ["correct_score_accuracy", "Exact-score accuracy"],
      ["home_goal_mae", "Home goal MAE"],
      ["away_goal_mae", "Away goal MAE"],
      ["outcome_log_loss", "Log loss"],
    ]
    : isRegression
    ? [
      ["model", "Model"],
      ["feature_bundle", "Feature bundle"],
      ["outcome_accuracy", "Outcome accuracy"],
      ["mae_overall", "Goal MAE"],
      ["rmse_overall", "Goal RMSE"],
      ["scoreline_exact_accuracy", "Exact scoreline"],
    ]
    : [
      ["model_name", "Model"],
      ["feature_bundle", "Feature bundle"],
      ["accuracy", "Accuracy"],
      ["balanced_accuracy", "Balanced accuracy"],
      ["macro_f1", "Macro F1"],
      ["log_loss", "Log loss"],
      ["brier_score", "Brier score"],
    ];
  const table = createElement("table", "model-metrics-table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const [, label] of columns) headerRow.append(createElement("th", "", label));
  head.append(headerRow);
  table.append(head);
  const body = document.createElement("tbody");
  for (const row of metrics) {
    const tableRow = document.createElement("tr");
    for (const [key] of columns) {
      const value = row[key];
      const display = typeof value === "number" ? value.toFixed(3) : (value ?? "—");
      tableRow.append(createElement("td", "", display));
    }
    body.append(tableRow);
  }
  table.append(body);
  const heading = isNeural
    ? "Neural-network chronological holdout metrics"
    : isRegression
    ? "Regression chronological holdout metrics"
    : "Classification chronological holdout metrics";
  modelResults.append(createElement("h3", "", heading), table);
  modelResults.hidden = false;
}

function renderAllModelPredictions(result) {
  modelResults.replaceChildren();
  modelResults.append(
    createElement(
      "p",
      "model-result-summary",
      `${result.message} Evaluated model impact on ${Number(result.holdout_fixture_count).toLocaleString()} shared chronological holdout fixtures.`,
    ),
  );

  const appendTable = (heading, columns, rows) => {
    modelResults.append(createElement("h3", "", heading));
    if (!Array.isArray(rows) || !rows.length) {
      modelResults.append(createElement("p", "model-result-summary", "No results available."));
      return;
    }
    const table = createElement("table", "model-metrics-table");
    const head = document.createElement("thead");
    const header = document.createElement("tr");
    for (const [, label] of columns) header.append(createElement("th", "", label));
    head.append(header);
    table.append(head);
    const body = document.createElement("tbody");
    for (const row of rows) {
      const tableRow = document.createElement("tr");
      for (const [key, , format] of columns) {
        const value = row[key];
        let display = value;
        if (value === null || value === undefined) {
          display = "--";
        } else if (format === "percent") {
          display = `${(Number(value) * 100).toFixed(1)}%`;
        } else if (format === "impact") {
          const points = Number(value) * 100;
          display = `${points > 0 ? "+" : ""}${points.toFixed(2)} pp`;
        } else if (format === "decimal") {
          display = Number(value).toFixed(3);
        } else if (format === "integer") {
          display = Number(value).toLocaleString();
        }
        tableRow.append(createElement("td", "", display));
      }
      body.append(tableRow);
    }
    table.append(body);
    modelResults.append(table);
  };

  appendTable(
    "Upcoming fixture predictions",
    [
      ["date", "Date"],
      ["home_team", "Home"],
      ["away_team", "Away"],
      ["predicted_outcome", "3-way"],
      ["outcome_confidence", "Confidence", "percent"],
      ["predicted_double_chance", "Double chance"],
      ["double_chance_confidence", "DC confidence", "percent"],
      ["predicted_correct_score", "Correct score"],
      ["family_agreement_count", "Families backing pick", "integer"],
    ],
    result.predictions,
  );
  appendTable(
    "Model impact on chronological holdout",
    [
      ["model", "Model"],
      ["holdout_fixtures", "Fixtures", "integer"],
      ["outcome_accuracy", "Accuracy", "percent"],
      ["outcome_log_loss", "Log loss", "decimal"],
      ["draw_precision", "Draw precision", "percent"],
      ["draw_recall", "Draw recall", "percent"],
      ["draw_f1", "Draw F1", "decimal"],
      ["draw_predictions", "Draw picks", "integer"],
      ["accuracy_impact_if_removed", "Accuracy impact if removed", "impact"],
      ["log_loss_impact_if_removed", "Log-loss impact if removed", "impact"],
      ["draw_precision_impact_if_removed", "Draw precision impact", "impact"],
      ["draw_recall_impact_if_removed", "Draw recall impact", "impact"],
    ],
    result.importance,
  );
  modelResults.append(
    createElement(
      "p",
      "model-result-summary",
      "Positive impact means the model improves the ensemble metric; for log loss, positive means a lower loss. Draw precision and recall are measured on the chronological holdout.",
    ),
  );
  modelResults.hidden = false;
}

function showModelJobResult(job) {
  modelResults.replaceChildren();
  modelResults.hidden = true;
  if (job.result?.model_type === "all-model-predictions") {
    renderAllModelPredictions(job.result);
  } else if (job.result?.metrics) {
    modelResults.dataset.modelType = job.result.model_type || "classifier";
    renderModelMetrics(job.result.metrics);
  } else if (job.result) {
    const result = job.result;
    const summary = result.training_rows !== undefined
      ? `Prepared ${Number(result.training_rows).toLocaleString()} fixture rows across ` +
        `seasons ${(result.training_seasons || []).join(", ")}. ` +
        "Training uses completed fixtures with valid scores."
      : (result.message || "Operation completed.");
    modelResults.append(createElement("p", "model-result-summary", summary));
    modelResults.hidden = false;
  }
}

async function pollModelJob(jobId) {
  try {
    const job = await requestJson(`/api/model-lab-job-status?job_id=${encodeURIComponent(jobId)}`);
    if (modelJobId !== jobId) return;
    const activeAction = job.action === "batch" ? job.active_action : job.action;
    const actionName = modelActionLabels[activeAction] || "Workflow";
    if (job.action === "batch") {
      for (const [action, state] of Object.entries(job.workflow_states)) {
        if (state !== "queued") modelWorkflowStates[action] = state;
      }
    } else {
      modelWorkflowStates[activeAction] = job.status;
    }
    modelJobCurrent.textContent = job.action === "batch"
      ? `Step ${job.current_step} of ${job.actions.length}: ${actionName}`
      : actionName;
    modelJobMessage.textContent = job.message;
    modelJobState.textContent = job.status.toUpperCase();
    modelJobProgress.hidden = job.status !== "running";
    modelProgressLabel.hidden = job.status !== "running";
    const total = Number(job.total);
    const progress = Math.min(100, Math.max(0, activeAction === "fetch" && total
      ? Math.round(Number(job.completed) * 100 / total) : Number(job.progress) || 0));
    modelJobProgress.classList.toggle("indeterminate", !total);
    document.querySelector("#model-job-progress-bar").style.width = total ? `${progress}%` : "";
    if (total) {
      modelJobProgress.setAttribute("aria-valuenow", String(progress));
      modelProgressLabel.textContent = `${job.completed} of ${total} completed · ${progress}%`;
    } else {
      modelJobProgress.removeAttribute("aria-valuenow");
      modelProgressLabel.textContent = "Progress is not available for this step.";
    }
    updateModelActionButtons();
    if (job.status === "running") {
      window.setTimeout(() => pollModelJob(jobId), 1000);
      return;
    }
    modelJobId = null;
    updateModelActionButtons();
    if (modelLeague.value === String(job.league_id)) {
      showModelJobResult(job);
      if (job.action === "batch" && job.result) {
        const completedActions = Object.keys(job.workflow_results);
        const lastAction = completedActions[completedActions.length - 1];
        if (lastAction) modelResults.prepend(createElement(
          "p", "model-result-summary", `Latest result: ${modelActionLabels[lastAction]}.`,
        ));
      }
    } else {
      modelResults.replaceChildren();
      modelResults.hidden = true;
    }
    void loadModelWorkflowStates();
    if (job.status === "failed") {
      const remaining = job.action === "batch" ? job.actions.length - job.current_step : 0;
      showModelError(new Error(`${actionName} failed: ${job.message}${remaining ? ` ${remaining} remaining workflow(s) were not run.` : ""}`));
      void loadModelStatus();
      return;
    }
    clearModelError();
    await loadModelStatus();
    await loadModelScores();
  } catch (error) {
    if (modelJobId !== jobId) return;
    showModelError(error);
    if (error.status === 404) {
      modelJobId = null;
      modelJobState.textContent = "UNAVAILABLE";
      modelJobMessage.textContent = "The job is no longer available; the server may have restarted.";
      modelJobProgress.hidden = true;
      modelProgressLabel.hidden = true;
      void loadModelWorkflowStates();
      return;
    }
    window.setTimeout(() => pollModelJob(jobId), 2000);
  }
}

async function startModelJob(action, actions = []) {
  if (!modelLeague.value || modelJobId !== null) return;
  if (action === "batch" && !actions.length) return;
  clearModelError();
  modelResults.replaceChildren();
  modelResults.hidden = true;
  modelJobId = "starting";
  modelJobCurrent.textContent = "Starting workflows";
  modelJobMessage.textContent = "Starting operation…";
  modelJobState.textContent = "RUNNING";
  modelJobProgress.hidden = false;
  modelJobProgress.classList.add("indeterminate");
  modelJobProgress.removeAttribute("aria-valuenow");
  modelProgressLabel.hidden = false;
  modelProgressLabel.textContent = "Preparing workflow.";
  updateModelActionButtons();
  try {
    const response = await requestJson("/api/model-lab-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        action,
        actions,
        league_id: Number(modelLeague.value),
      }),
    });
    modelJobId = response.job_id;
    updateModelActionButtons();
    await pollModelJob(modelJobId);
  } catch (error) {
    modelJobId = null;
    modelJobMessage.textContent = "Operation could not be started.";
    modelJobState.textContent = "FAILED";
    modelJobProgress.hidden = true;
    modelProgressLabel.hidden = true;
    updateModelActionButtons();
    showModelError(error);
  }
}

async function loadModelLeagues() {
  try {
    const data = await requestJson("/api/leagues");
    modelLeague.replaceChildren(makeOption("Select a league"));
    leagues = data.leagues;
    for (const league of leagues) {
      const option = createElement("option", "", `${league.league_name} · #${league.league_id}`);
      option.value = String(league.league_id);
      modelLeague.append(option);
    }
    renderModelLeagueTree();
    try {
      const saved = JSON.parse(sessionStorage.getItem("model-lab-selection") || "null");
      if (saved && leagues.some((league) => String(league.league_id) === saved.league)) {
        modelLeague.value = saved.league;
        const league = leagues.find((item) => String(item.league_id) === saved.league);
        modelExpandedCountry = league.country_name;
        renderModelLeagueTree();
        await loadModelSeasons(saved.league);
      }
    } catch (error) {
      console.warn("Could not restore Model Lab selection:", error);
    }
  } catch (error) {
    modelLeague.replaceChildren(makeOption("Could not load leagues"));
    modelLeagueTree.replaceChildren(
      createElement("div", "pane-placeholder", `Could not load leagues. ${error.message}`),
    );
    modelLeagueCount.textContent = "Unavailable";
    showModelError(error);
  }
}

function renderModelLeagueTree() {
  const query = modelLeagueSearch.value.trim().toLocaleLowerCase();
  const filtered = leagues.filter((league) =>
    `${league.country_name} ${league.league_name}`.toLocaleLowerCase().includes(query),
  );
  filtered.sort((a, b) =>
    a.country_name.localeCompare(b.country_name, undefined, { sensitivity: "base" }) ||
    a.league_id - b.league_id,
  );
  const groups = groupByCountry(filtered);
  modelLeagueTree.replaceChildren();
  if (!groups.size) {
    modelLeagueTree.append(createElement("div", "pane-placeholder", "No matching countries or leagues."));
    modelLeagueCount.textContent = "0 countries";
    return;
  }
  modelLeagueCount.textContent =
    `${groups.size.toLocaleString()} countries · ${filtered.length.toLocaleString()} leagues`;

  for (const [countryName, countryLeagues] of groups) {
    const country = createElement("section", "tree-country");
    const isExpanded = modelExpandedCountry === countryName;
    const countryButton = makeTreeButton(
      "",
      "tree-row country-row",
      "model-country",
      countryName,
    );
    const flag = createElement("span", "tree-flag");
    addImage(flag, countryLeagues[0].country_flag, "tree-flag-image", `${countryName} flag`);
    countryButton.append(
      createElement("span", "tree-chevron", isExpanded ? "⌄" : "›"),
      flag,
      createElement("span", "tree-label", countryName),
      createElement("span", "tree-count", String(countryLeagues.length)),
    );
    country.append(countryButton);

    if (isExpanded) {
      const leagueList = createElement("div", "tree-children league-children");
      for (const league of countryLeagues) {
        const selected = modelLeague.value === String(league.league_id);
        const leagueButton = makeTreeButton(
          "",
          "tree-row league-row",
          "model-league",
          String(league.league_id),
          selected,
        );
        const logo = createElement("span", "tree-league-logo");
        addImage(logo, league.league_logo, "tree-league-logo-image", `${league.league_name} logo`);
        leagueButton.append(
          createElement("span", "tree-chevron", selected ? "●" : "›"),
          logo,
          createElement("span", "tree-label", league.league_name),
          createElement("span", "tree-count", `#${league.league_id}`),
        );
        leagueList.append(leagueButton);
      }
      country.append(leagueList);
    }
    modelLeagueTree.append(country);
  }
}

function saveModelSelection() {
  try {
    sessionStorage.setItem("model-lab-selection", JSON.stringify({
      league: modelLeague.value,
    }));
  } catch (error) {
    console.warn("Could not save Model Lab selection:", error);
  }
}

async function loadModelSeasons(leagueId) {
  const requestId = ++modelSeasonRequestId;
  modelWorkflowRequestId += 1;
  modelWorkflowLoading = false;
  modelWorkflowStates = {};
  modelResults.replaceChildren();
  modelResults.hidden = true;
  modelJobCurrent.textContent = "Choose a workflow to begin.";
  modelJobMessage.textContent = "";
  modelJobState.textContent = "READY";
  modelJobProgress.hidden = true;
  modelProgressLabel.hidden = true;
  document.querySelector("#model-selection-status").textContent = "Loading league readiness…";
  void loadModelScores(leagueId);
  updateModelActionButtons();
  try {
    const data = await requestJson(`/api/seasons?league_id=${encodeURIComponent(leagueId)}`);
    if (requestId !== modelSeasonRequestId || modelLeague.value !== String(leagueId)) return;
    if (!data.seasons.length) throw new Error("No seasons are available for this league.");
    saveModelSelection();
    updateModelActionButtons();
    void loadModelWorkflowStates();
    await loadModelStatus();
  } catch (error) {
    if (requestId !== modelSeasonRequestId) return;
    document.querySelector("#model-selection-status").textContent = "Data readiness unavailable.";
    showModelError(error);
  }
}

modelLeagueTree.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  if (button.dataset.action === "model-country") {
    modelExpandedCountry =
      modelExpandedCountry === button.dataset.id ? "" : button.dataset.id;
    renderModelLeagueTree();
    return;
  }
  if (button.dataset.action !== "model-league") return;
  if (modelJobId !== null) return;
  const leagueId = button.dataset.id;
  const league = leagues.find((item) => item.league_id === Number(leagueId));
  if (!league) return;
  modelExpandedCountry = league.country_name;
  modelLeague.value = leagueId;
  renderModelLeagueTree();
  clearModelError();
  modelRequestId += 1;
  modelSeasonRequestId += 1;
  loadModelSeasons(leagueId);
});
modelLeagueSearch.addEventListener("input", renderModelLeagueTree);
modelLeague.addEventListener("change", () => {
  clearModelError();
  modelRequestId += 1;
  modelSeasonRequestId += 1;
  if (modelLeague.value) {
    const league = leagues.find((item) => item.league_id === Number(modelLeague.value));
    if (league) modelExpandedCountry = league.country_name;
    renderModelLeagueTree();
    loadModelSeasons(modelLeague.value);
  }
  else {
    updateModelActionButtons();
    loadModelStatus();
    void loadModelScores("");
  }
});
for (const button of modelActionButtons) {
  button.addEventListener("click", () => startModelJob(button.dataset.action));
}
for (const selection of modelWorkflowSelections) {
  selection.addEventListener("change", updateModelActionButtons);
}
modelRunSelected.addEventListener("click", () => {
  void startModelJob("batch", modelWorkflowSelections.filter((input) => input.checked).map((input) => input.dataset.action));
});
modelRunAll.addEventListener("click", () => {
  void startModelJob("batch", modelWorkflowSelections.map((input) => input.dataset.action));
});
