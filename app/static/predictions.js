const predictionLeagueTree = document.querySelector("#prediction-league-tree");
const predictionLeagueSearch = document.querySelector("#prediction-league-search");
const predictionLeagueCount = document.querySelector("#prediction-league-count");
const predictionFixtureList = document.querySelector("#prediction-fixture-list");
const predictionFixtureCount = document.querySelector("#prediction-fixture-count");
const predictionFixtureFilter = document.querySelector("#prediction-fixture-filter");
const predictionFixturesContext = document.querySelector("#prediction-fixtures-context");
const predictionDetails = document.querySelector("#prediction-details");
const predictionDetailsContext = document.querySelector("#prediction-details-context");
const predictionError = document.querySelector("#prediction-error");
let predictionExpandedCountry = "";
let predictionLeagueId = "";
let predictionSeason = "";
let predictionSeasons = [];
let predictionSeasonsLeagueId = "";
let predictionSelectedFixtureId = "";
let predictionDetailLeagueId = "";
let predictionDetailSeason = "";
let predictionDetailFixtureId = "";
let predictionRequestId = 0;
let predictionFixtureData = null;
let predictionDetailRequestId = 0;
function renderPredictionLeagueTree() {
  const query = predictionLeagueSearch.value.trim().toLocaleLowerCase();
  const filtered = leagues
    .filter((league) =>
      `${league.country_name} ${league.league_name}`.toLocaleLowerCase().includes(query),
    )
    .sort((a, b) =>
      a.country_name.localeCompare(b.country_name, undefined, { sensitivity: "base" }) ||
      a.league_id - b.league_id,
    );
  const groups = groupByCountry(filtered);
  predictionLeagueTree.replaceChildren();
  if (!groups.size) {
    predictionLeagueTree.append(createElement("div", "pane-placeholder", "No matching countries or leagues."));
    predictionLeagueCount.textContent = "0 leagues";
    return;
  }
  predictionLeagueCount.textContent = `${groups.size} countries · ${filtered.length} leagues`;
  for (const [countryName, countryLeagues] of groups) {
    const country = createElement("section", "tree-country");
    const expanded = predictionExpandedCountry === countryName;
    const countryButton = makeTreeButton(
      "",
      "tree-row country-row",
      "prediction-country",
      countryName,
      expanded,
    );
    const flag = createElement("span", "tree-flag");
    addImage(flag, countryLeagues[0].country_flag, "tree-flag-image", `${countryName} flag`);
    countryButton.append(
      createElement("span", "tree-chevron", expanded ? "⌄" : "›"),
      flag,
      createElement("span", "tree-label", countryName),
      createElement("span", "tree-count", String(countryLeagues.length)),
    );
    country.append(countryButton);
    if (expanded) {
      const leagueList = createElement("div", "tree-children league-children");
      for (const league of countryLeagues) {
        const selected = predictionLeagueId === String(league.league_id);
        const leagueNode = createElement("div", "tree-league-node");
        const leagueButton = makeTreeButton(
          "",
          "tree-row league-row",
          "prediction-league",
          String(league.league_id),
          selected,
        );
        const logo = createElement("span", "tree-league-logo");
        addImage(logo, league.league_logo, "tree-league-logo-image", `${league.league_name} logo`);
        leagueButton.append(
          createElement("span", "tree-chevron", selected ? "⌄" : "›"),
          logo,
          createElement("span", "tree-label", league.league_name),
          createElement("span", "tree-count", `#${league.league_id}`),
        );
        leagueNode.append(leagueButton);
        if (selected) {
          const seasonList = createElement("div", "tree-children season-children");
          if (predictionSeasonsLeagueId !== predictionLeagueId) {
            seasonList.append(createElement("span", "tree-empty", "Loading seasons…"));
          } else if (!predictionSeasons.length) {
            seasonList.append(createElement("span", "tree-empty", "No seasons available"));
          }
          for (const item of predictionSeasons) {
            const seasonButton = makeTreeButton(
              "",
              "tree-row season-row",
              "prediction-season",
              String(item.season),
              predictionSeason === String(item.season),
            );
            seasonButton.dataset.leagueId = String(league.league_id);
            seasonButton.append(
              createElement("span", "tree-chevron", predictionSeason === String(item.season) ? "●" : "○"),
              createElement("span", "tree-label", String(item.season)),
              createElement(
                "span",
                item.fetched ? "tree-season-state fetched" : "tree-season-state",
                item.fetched ? "Ready" : "No fixtures",
              ),
            );
            seasonList.append(seasonButton);
          }
          leagueNode.append(seasonList);
        }
        leagueList.append(leagueNode);
      }
      country.append(leagueList);
    }
    predictionLeagueTree.append(country);
  }
}

async function loadPredictionSeasons(leagueId) {
  const requestId = ++predictionRequestId;
  predictionSeasons = [];
  predictionSeasonsLeagueId = "";
  predictionSeason = "";
  predictionSelectedFixtureId = "";
  predictionDetailLeagueId = "";
  predictionDetailSeason = "";
  predictionDetailFixtureId = "";
  predictionDetailRequestId += 1;
  renderPredictionLeagueTree();
  predictionFixtureList.replaceChildren(createElement("div", "pane-placeholder", "Select a season to load predictions."));
  predictionFixtureCount.textContent = "—";
  predictionDetails.replaceChildren(createElement("div", "pane-placeholder", "Select a fixture or favorite."));
  predictionDetailsContext.textContent = "Select a fixture or favorite";
  try {
    const data = await requestJson(`/api/seasons?league_id=${encodeURIComponent(leagueId)}`);
    if (requestId !== predictionRequestId || predictionLeagueId !== String(leagueId)) return;
    predictionSeasons = data.seasons;
    predictionSeasonsLeagueId = String(leagueId);
    renderPredictionLeagueTree();
  } catch (error) {
    if (requestId !== predictionRequestId) return;
    showPredictionError(error);
  }
}

function showPredictionError(error) {
  predictionError.textContent = error.message || String(error);
  predictionError.hidden = false;
}

function clearPredictionError() {
  predictionError.hidden = true;
}

function formatConfidence(value) {
  if (value === null || value === undefined) return "—";
  const confidence = Number(value);
  return Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "—";
}

function evaluatePrediction(fixture, prediction) {
  const status = (fixture.status || "").toUpperCase();
  if (status === "NS") return { state: "not-played" };
  if (status !== "FT") return { state: "unavailable" };
  const home = fixture.home_goals;
  const away = fixture.away_goals;
  const outcome = prediction?.classification?.outcomes?.[0]?.code;
  const chance = prediction?.classification?.double_chance?.[0]?.code;
  if (!Number.isInteger(home) || !Number.isInteger(away) || home < 0 || away < 0) {
    return { state: "unavailable" };
  }
  const actual = home > away ? "H" : home < away ? "A" : "D";
  const covered = { "1X": ["H", "D"], "X2": ["D", "A"], "12": ["H", "A"] };
  return {
    state: "played",
    outcomeCorrect: ["H", "D", "A"].includes(outcome) ? outcome === actual : null,
    doubleChanceCorrect: covered[chance] ? covered[chance].includes(actual) : null,
  };
}

function testBadge(prediction) {
  if (prediction?.is_test === true) {
    const badge = createElement("span", "prediction-test-badge", "test");
    badge.title = "Fixture was in at least one model's recorded test set; other models may have trained on it.";
    return badge;
  }
  if (prediction?.is_test === null) {
    const badge = createElement("span", "prediction-test-unknown", "Test set unknown");
    badge.title = "Some model test records are missing; test membership cannot be determined.";
    return badge;
  }
  return null;
}

function renderPredictionMatchup(fixture) {
  const teams = createElement("span", "prediction-fixture-teams");
  for (const [side, teamName] of [["home", fixture.home_team], ["away", fixture.away_team]]) {
    const name = teamName || (side === "home" ? "Home team" : "Away team");
    const team = createElement("span", `prediction-fixture-team ${side}`);
    const logo = createElement("span", "prediction-team-logo-wrap");
    addImage(
      logo,
      fixture[`${side}_logo`],
      "prediction-team-logo",
      `${name} logo`,
      String(name).slice(0, 1).toUpperCase(),
    );
    team.append(createElement("span", "prediction-team-name", name), logo);
    teams.append(team);
    if (side === "home") {
      const center = createElement("span", "prediction-fixture-center");
      center.append(
        createElement(
          "span",
          "prediction-team-separator",
          hasFinalScore(fixture) && fixture.home_goals != null && fixture.away_goals != null
            ? `${fixture.home_goals} - ${fixture.away_goals}`
            : "vs",
        ),
        createElement("span", "prediction-fixture-date", formatFixtureDate(fixture.date, false)),
        createElement("span", "prediction-fixture-time", formatFixtureTime(fixture.date)),
      );
      teams.append(center);
    }
  }
  return teams;
}

function renderPredictionFixtures(data) {
  predictionFixtureList.replaceChildren();
  const allFixtures = data.fixtures || [];
  const fixtures = allFixtures.filter((fixture) => {
    if (predictionFixtureFilter.value === "played") return hasFinalScore(fixture);
    if (predictionFixtureFilter.value === "upcoming") return (fixture.status || "").toUpperCase() === "NS";
    return true;
  });
  predictionFixtureCount.textContent = fixtures.length.toLocaleString();
  if (!fixtures.length) {
    predictionFixtureList.append(
      createElement(
        "div",
        "pane-placeholder",
        "No predictions are available for this fixture filter.",
      ),
    );
    return;
  }
  const rounds = new Map();
  for (const fixture of fixtures) {
    const roundName = fixture.round || "Other";
    if (!rounds.has(roundName)) rounds.set(roundName, []);
    rounds.get(roundName).push(fixture);
  }
  for (const [roundIndex, [roundName, roundFixtures]] of [...rounds.entries()].entries()) {
    const expanded = roundIndex === 0;
    const roundGroup = createElement("section", "prediction-round-group");
    const roundHeader = createElement("button", "prediction-round-heading");
    roundHeader.type = "button";
    roundHeader.dataset.action = "toggle-prediction-round";
    roundHeader.setAttribute("aria-expanded", String(expanded));
    const evaluated = allFixtures
      .filter((fixture) => (fixture.round || "Other") === roundName)
      .map((fixture) => evaluatePrediction(fixture, fixture.prediction));
    const makeAccuracy = (label, description, field) => {
      const eligible = evaluated.filter((evaluation) => typeof evaluation[field] === "boolean");
      const correct = eligible.filter((evaluation) => evaluation[field]).length;
      const badge = createElement("span", "prediction-round-accuracy", `${label}-(${correct}/${eligible.length})`);
      badge.title = `Correct ${description} predictions / eligible full-time fixtures in this round`;
      badge.setAttribute("aria-label", `${correct} correct ${description} predictions out of ${eligible.length} eligible full-time fixtures`);
      return badge;
    };
    roundHeader.append(
      createElement("span", "prediction-round-chevron", expanded ? "⌄" : "›"),
      createElement("span", "prediction-round-name", roundName),
      makeAccuracy("3W", "3-way", "outcomeCorrect"),
      makeAccuracy("DC", "double chance", "doubleChanceCorrect"),
      createElement("span", "round-count", String(roundFixtures.length)),
    );
    const matches = createElement("div", "prediction-round-fixtures");
    matches.hidden = !expanded;
    for (const fixture of roundFixtures) {
      const prediction = fixture.prediction;
      const card = createElement(
        "article",
        `prediction-fixture-card${
          String(fixture.fixture_id) === predictionSelectedFixtureId &&
          String(predictionLeagueId) === predictionDetailLeagueId &&
          String(predictionSeason) === predictionDetailSeason ? " selected" : ""
        }`,
      );
      const selectButton = createElement("button", "prediction-fixture-select");
      selectButton.type = "button";
      selectButton.dataset.action = "prediction-fixture";
      selectButton.dataset.id = String(fixture.fixture_id);
      selectButton.dataset.leagueId = String(predictionLeagueId);
      selectButton.dataset.season = String(predictionSeason);
      selectButton.append(renderPredictionMatchup(fixture));
      card.append(selectButton);
      card.append(createElement("div", "prediction-fixture-venue", fixture.venue || "Venue not available"));
      card.append(createElement("div", "prediction-fixture-venue", `Status: ${fixture.status || "Unknown"}`));
      const badge = testBadge(prediction);
      const evaluation = evaluatePrediction(fixture, prediction);
      if (badge || evaluation.outcomeCorrect === true || evaluation.doubleChanceCorrect === true) {
        const markers = createElement("div", "prediction-fixture-markers");
        if (badge) markers.append(badge);
        const indicators = createElement("div", "prediction-correct-indicators");
        for (const [label, description, correct] of [
          ["3W", "3-way", evaluation.outcomeCorrect],
          ["DC", "double chance", evaluation.doubleChanceCorrect],
        ]) {
          if (correct !== true) continue;
          const indicator = createElement("span", "prediction-correct-indicator", label);
          indicator.title = `Correct ${description} prediction`;
          indicator.setAttribute("aria-label", indicator.title);
          indicators.append(indicator);
        }
        markers.append(indicators);
        card.append(markers);
      }
      card.append(createElement("p", "prediction-card-summary", prediction.summary));
      const picks = createElement("div", "prediction-fixture-picks");
      const topOutcome = prediction.classification.outcomes[0];
      const topDoubleChance = prediction.classification.double_chance[0];
      for (const [label, value, confidence] of [
        ["3-way", topOutcome.label, topOutcome.confidence],
        ["Double chance", topDoubleChance.code, topDoubleChance.confidence],
      ]) {
        const pick = createElement("div", "prediction-pick");
        pick.append(
          createElement("span", "", label),
          createElement("strong", "", value),
          createElement("span", "confidence-badge", formatConfidence(confidence)),
        );
        picks.append(pick);
      }
      card.append(picks);
      matches.append(card);
    }
    roundGroup.append(roundHeader, matches);
    predictionFixtureList.append(roundGroup);
  }
}

async function loadPredictionFixtures(leagueId, season) {
  const requestId = ++predictionRequestId;
  predictionFixtureData = null;
  predictionFixturesContext.textContent = "Loading model predictions…";
  predictionFixtureCount.textContent = "…";
  predictionFixtureList.replaceChildren(
    createElement("div", "pane-placeholder", "Running trained models on season fixtures…"),
  );
  clearPredictionError();
  try {
    const params = new URLSearchParams({ league_id: leagueId, season });
    const data = await requestJson(`/api/prediction-fixtures?${params}`);
    if (
      requestId !== predictionRequestId ||
      predictionLeagueId !== String(leagueId) ||
      predictionSeason !== String(season)
    ) return;
    predictionFixturesContext.textContent =
      `${data.country_name} · ${data.league_name} · ${season}`;
    predictionFixtureData = data;
    renderPredictionFixtures(data);
  } catch (error) {
    if (requestId !== predictionRequestId) return;
    predictionFixtureList.replaceChildren(createElement("div", "pane-placeholder", error.message));
    predictionFixtureCount.textContent = "!";
    predictionFixturesContext.textContent = "Predictions unavailable";
    showPredictionError(error);
  }
}

function renderPredictionRankList(title, predictions, labelFor) {
  const section = createElement("section", "prediction-analysis-card");
  section.append(createElement("h3", "", title));
  const list = createElement("ol", "prediction-rank-list");
  for (const item of predictions || []) {
    const row = document.createElement("li");
    row.append(
      createElement("span", "", labelFor(item)),
      createElement("strong", "", formatConfidence(item.confidence)),
    );
    list.append(row);
  }
  section.append(list);
  return section;
}

function renderPredictionDetails(data) {
  const fixture = data.fixture;
  const prediction = data.prediction;
  predictionDetails.replaceChildren();
  const header = createElement("section", "prediction-fixture-card prediction-detail-card");
  const heading = createElement("div", "prediction-detail-heading");
  heading.append(createElement("span", "prediction-detail-round", fixture.round || "Fixture"));
  const badge = testBadge(prediction);
  if (badge) heading.append(badge);
  const favoriteButton = createElement(
    "button",
    `favorite-toggle${data.is_favorite ? " active" : ""}`,
    data.is_favorite ? "♥" : "♡",
  );
  favoriteButton.type = "button";
  favoriteButton.dataset.action = "toggle-favorite";
  favoriteButton.setAttribute("aria-label", "Add fixture to active favorite lists");
  favoriteButton.setAttribute("aria-pressed", String(data.is_favorite));
  heading.append(favoriteButton);
  header.append(heading);
  header.append(renderPredictionMatchup(fixture));
  header.append(
    createElement("div", "prediction-fixture-venue", fixture.venue || "Venue unavailable"),
    createElement("div", "prediction-fixture-venue", `Status: ${fixture.status || "Unknown"}`),
  );
  predictionDetails.append(header);

  if (!prediction) {
    predictionDetails.append(
      createElement("div", "detail-empty", "No trained-model prediction is available for this fixture."),
    );
  } else {
    const summary = createElement("section", "prediction-analysis-card");
    summary.append(
      createElement("h3", "", "Prediction summary"),
      createElement("p", "", prediction.summary),
    );
    predictionDetails.append(summary);
    predictionDetails.append(
      renderPredictionRankList(
        "3-way outcome · all models",
        prediction.classification.outcomes,
        (item) => item.label,
      ),
      renderPredictionRankList(
        "Double chance · all models",
        prediction.classification.double_chance,
        (item) => `${item.code} · ${item.label}`,
      ),
    );
    const score = createElement("section", "prediction-analysis-card");
    score.append(
      createElement("h3", "", "Correct score · all models"),
      createElement("p", "", prediction.regression.score),
    );
    const regressionModels = createElement("div", "prediction-model-list");
    for (const model of prediction.regression.models) {
      const row = createElement("div", "prediction-model-row");
      row.append(
        createElement("span", "", `${model.feature_bundle} · ${model.model}`),
        createElement(
          "strong",
          "",
          `${model.score} · ${formatConfidence(model.validation_accuracy)} validation accuracy`,
        ),
      );
      regressionModels.append(row);
    }
    score.append(regressionModels);
    predictionDetails.append(score);
  }

  const evaluation = evaluatePrediction(fixture, prediction);
  const result = createElement("section", "prediction-analysis-card prediction-evaluation");
  result.append(createElement("h3", "", "Prediction result"));
  if (evaluation.state === "not-played") {
    result.append(createElement("p", "", "Not yet played"));
  } else if (evaluation.state === "unavailable") {
    result.append(createElement("p", "", "Result unavailable for evaluation"));
  } else {
    for (const [label, correct] of [
      ["3-way", evaluation.outcomeCorrect],
      ["Double chance", evaluation.doubleChanceCorrect],
    ]) {
      const state = correct === null ? "unavailable" : correct ? "correct" : "incorrect";
      const text = correct === null ? "Unavailable" : correct ? "Correct" : "Incorrect";
      result.append(createElement("p", `prediction-result-${state}`, `${label}: ${text}`));
    }
  }
  predictionDetails.append(result);

  const statsHeading = createElement("div", "stats-section-heading");
  statsHeading.append(
    createElement("h3", "", "Fixture feature stats"),
    createElement("span", "stats-group-count", `${data.stats.length} groups`),
  );
  predictionDetails.append(statsHeading);
  if (!data.stats.length) {
    predictionDetails.append(
      createElement("div", "detail-empty", "No processed stats are available for this fixture."),
    );
    return;
  }
  const stats = createElement("div", "detail-stats-list");
  for (const group of data.stats) {
    const card = createElement("details", "stats-card");
    card.open = group.name === "outcome-probability";
    const summary = createElement("summary", "stats-card-title");
    summary.append(
      createElement("span", "stats-disclosure", "›"),
      createElement("span", "", formatMetricName(group.name)),
      createElement("span", "metric-count", String(group.metrics.length)),
    );
    card.append(summary);
    const metrics = createElement("dl", "metrics-grid");
    for (const metric of group.metrics) {
      const item = createElement("div", "metric-item");
      item.append(
        createElement("dt", "", formatMetricName(metric.name)),
        createElement("dd", "", formatMetricValue(metric.value)),
      );
      metrics.append(item);
    }
    card.append(metrics);
    stats.append(card);
  }
  predictionDetails.append(stats);
}

async function selectPredictionFixture(leagueId, season, fixtureId) {
  predictionSelectedFixtureId = String(fixtureId);
  predictionDetailLeagueId = String(leagueId);
  predictionDetailSeason = String(season);
  predictionDetailFixtureId = String(fixtureId);
  renderFavoriteFixtures();
  predictionFixtureList.querySelectorAll(".prediction-fixture-card").forEach((card) => {
    card.classList.toggle(
      "selected",
      card.querySelector("[data-action='prediction-fixture']")?.dataset.id === predictionDetailFixtureId &&
        card.querySelector("[data-action='prediction-fixture']")?.dataset.leagueId === predictionDetailLeagueId &&
        card.querySelector("[data-action='prediction-fixture']")?.dataset.season === predictionDetailSeason,
    );
  });
  const requestId = ++predictionDetailRequestId;
  predictionDetailsContext.textContent = "Loading fixture prediction…";
  predictionDetails.replaceChildren(createElement("div", "pane-placeholder", "Loading fixture statistics and model predictions…"));
  clearPredictionError();
  try {
    const params = new URLSearchParams({
      league_id: leagueId,
      season,
      fixture_id: fixtureId,
    });
    const data = await requestJson(`/api/prediction-fixture?${params}`);
    if (requestId !== predictionDetailRequestId) return;
    renderPredictionDetails(data);
    predictionDetailsContext.textContent = `${data.fixture.home_team} vs ${data.fixture.away_team}`;
  } catch (error) {
    if (requestId !== predictionDetailRequestId) return;
    predictionDetails.replaceChildren(createElement("div", "pane-placeholder", error.message));
    predictionDetailsContext.textContent = "Fixture analysis unavailable";
    showPredictionError(error);
  }
}

predictionLeagueSearch.addEventListener("input", renderPredictionLeagueTree);
predictionFixtureFilter.addEventListener("change", () => {
  if (predictionFixtureData) renderPredictionFixtures(predictionFixtureData);
});
predictionLeagueTree.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, id } = button.dataset;
  if (action === "prediction-country") {
    predictionExpandedCountry =
      predictionExpandedCountry === id ? "" : id;
    renderPredictionLeagueTree();
    return;
  }
  if (action === "prediction-league") {
    const league = leagues.find((item) => item.league_id === Number(id));
    if (!league) return;
    predictionExpandedCountry = league.country_name;
    predictionLeagueId = id;
    predictionSeason = "";
    predictionRequestId += 1;
    predictionFixtureData = null;
    renderPredictionLeagueTree();
    loadPredictionSeasons(id);
    return;
  }
  if (action === "prediction-season") {
    predictionLeagueId = button.dataset.leagueId;
    predictionSeason = id;
    predictionSelectedFixtureId = "";
    predictionDetailLeagueId = "";
    predictionDetailSeason = "";
    predictionDetailFixtureId = "";
    predictionDetailRequestId += 1;
    predictionDetails.replaceChildren(createElement("div", "pane-placeholder", "Select a fixture or favorite."));
    predictionDetailsContext.textContent = "Select a fixture or favorite";
    renderPredictionLeagueTree();
    const league = leagues.find((item) => item.league_id === Number(predictionLeagueId));
    predictionFixturesContext.textContent = `${league?.league_name || "League"} · ${predictionSeason}`;
    loadPredictionFixtures(predictionLeagueId, predictionSeason);
  }
});
predictionFixtureList.addEventListener("click", (event) => {
  const roundButton = event.target.closest("button[data-action='toggle-prediction-round']");
  if (roundButton) {
    const expanded = roundButton.getAttribute("aria-expanded") !== "true";
    roundButton.setAttribute("aria-expanded", String(expanded));
    roundButton.nextElementSibling.hidden = !expanded;
    roundButton.querySelector(".prediction-round-chevron").textContent = expanded ? "⌄" : "›";
    return;
  }
  const button = event.target.closest("button[data-action='prediction-fixture']");
  if (!button) return;
  selectPredictionFixture(
    button.dataset.leagueId,
    button.dataset.season,
    button.dataset.id,
  );
});
predictionDetails.addEventListener("click", (event) => {
  if (event.target.closest("button[data-action='toggle-favorite']")) {
    toggleFavoriteFixture();
  }
});
