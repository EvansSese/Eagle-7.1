const countryTree = document.querySelector("#country-tree");
const searchInput = document.querySelector("#league-search");
const browserCount = document.querySelector("#browser-count");
const fixtureList = document.querySelector("#fixture-list");
const fixtureCount = document.querySelector("#fixture-count");
const fixturesContext = document.querySelector("#fixtures-context");
const fixtureDetails = document.querySelector("#fixture-details");
const detailsContext = document.querySelector("#details-context");
const explorerError = document.querySelector("#explorer-error");
const fetchPanel = document.querySelector("#fetch-panel");
const fetchSeasonList = document.querySelector("#fetch-seasons");
const fetchButton = document.querySelector("#fetch-selected");
const fetchProgress = document.querySelector("#fetch-progress");

let currentSeasons = [];
let seasonsLoadedLeagueId = "";
let selectedCountry = "";
let selectedLeagueId = "";
let selectedSeason = "";
let selectedFixtureId = "";
let expandedCountry = "";
let expandedLeagueId = "";
let fetchJobId = null;
let seasonRequestId = 0;
let fixtureRequestId = 0;
let detailRequestId = 0;

function showError(error) {
  explorerError.textContent = error.message || String(error);
  explorerError.hidden = false;
}

function clearError() {
  explorerError.hidden = true;
}

function renderCountryTree() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  const filtered = leagues.filter((league) =>
    `${league.country_name} ${league.league_name}`.toLocaleLowerCase().includes(query),
  );
  filtered.sort((a, b) => a.country_name.localeCompare(b.country_name, undefined, { sensitivity: "base" }) || a.league_id - b.league_id);
  const groups = groupByCountry(filtered);
  countryTree.replaceChildren();

  if (!groups.size) {
    countryTree.append(createElement("div", "pane-placeholder", "No matching countries or leagues."));
    browserCount.textContent = "0 countries";
    return;
  }

  browserCount.textContent = `${groups.size.toLocaleString()} countries · ${filtered.length.toLocaleString()} leagues`;
  for (const [countryName, countryLeagues] of groups) {
    const country = createElement("section", "tree-country");
    const isCountryExpanded = expandedCountry === countryName;
    const countryButton = makeTreeButton(
      "",
      "tree-row country-row",
      "country",
      countryName,
      selectedCountry === countryName,
    );
    const flag = createElement("span", "tree-flag");
    addImage(flag, countryLeagues[0].country_flag, "tree-flag-image", `${countryName} flag`);
    const arrow = createElement("span", "tree-chevron", isCountryExpanded ? "⌄" : "›");
    countryButton.append(arrow, flag, createElement("span", "tree-label", countryName));
    countryButton.append(createElement("span", "tree-count", String(countryLeagues.length)));
    country.append(countryButton);

    if (isCountryExpanded) {
      const leagueList = createElement("div", "tree-children league-children");
      for (const league of countryLeagues) {
        const leagueNode = createElement("div", "tree-league-node");
        const isLeagueExpanded = Number(expandedLeagueId) === league.league_id;
        const leagueButton = makeTreeButton(
          "",
          "tree-row league-row",
          "league",
          String(league.league_id),
          Number(selectedLeagueId) === league.league_id,
        );
        const logo = createElement("span", "tree-league-logo");
        addImage(logo, league.league_logo, "tree-league-logo-image", "");
        leagueButton.append(
          createElement("span", "tree-chevron", isLeagueExpanded ? "⌄" : "›"),
          logo,
          createElement("span", "tree-label", league.league_name),
        );
        if (league.seasons_available) {
          leagueButton.append(createElement("span", "tree-season-indicator", `${league.season_count}`));
        }
        leagueNode.append(leagueButton);

        if (isLeagueExpanded) {
          const seasonsLoaded = seasonsLoadedLeagueId === String(league.league_id);
          const seasons = seasonsLoaded ? currentSeasons : [];
          const seasonList = createElement("div", "tree-children season-children");
          if (!seasonsLoaded) {
            seasonList.append(createElement("span", "tree-empty", "Loading seasons…"));
          } else if (!seasons.length) {
            seasonList.append(createElement("span", "tree-empty", "No seasons available"));
          }
          for (const item of seasons) {
            const seasonButton = makeTreeButton(
              "",
              "tree-row season-row",
              "season",
              String(item.season),
              Number(selectedLeagueId) === league.league_id && Number(selectedSeason) === item.season,
            );
            seasonButton.append(
              createElement("span", "tree-chevron", Number(selectedSeason) === item.season && Number(selectedLeagueId) === league.league_id ? "●" : "○"),
              createElement("span", "tree-label", String(item.season)),
            );
            seasonButton.append(
              createElement(
                "span",
                item.fetched ? "tree-season-state fetched" : "tree-season-state",
                item.fetched ? "Fetched" : "No fixtures",
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
    countryTree.append(country);
  }
}

function populateFetchSeasons() {
  fetchSeasonList.replaceChildren();
  const missing = currentSeasons.filter((item) => !item.fetched);
  for (const item of currentSeasons) {
    const label = createElement("label", "fetch-season-option");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "fetch-season";
    checkbox.value = String(item.season);
    checkbox.checked = missing.length ? !item.fetched : false;
    const status = createElement(
      "span",
      item.fetched ? "fetch-season-status fetched" : "fetch-season-status missing",
      item.fetched ? "Fetched" : "Not fetched",
    );
    label.append(checkbox, createElement("span", "fetch-season-year", String(item.season)), status);
    fetchSeasonList.append(label);
  }
  document.querySelector("#select-all-seasons").disabled = fetchJobId !== null;
  updateFetchSelection();
}

function updateFetchSelection() {
  const selected = fetchSeasonList.querySelectorAll('input[name="fetch-season"]:checked').length;
  document.querySelector("#fetch-selection-count").textContent = selected
    ? `${selected} selected`
    : "Select seasons";
  fetchButton.disabled = selected === 0 || fetchJobId !== null;
  const checkboxes = [...fetchSeasonList.querySelectorAll('input[name="fetch-season"]')];
  document.querySelector("#select-all-seasons").textContent =
    checkboxes.length && checkboxes.every((checkbox) => checkbox.checked) ? "Clear" : "All";
}

function renderFixtureGroups(data) {
  fixtureList.replaceChildren();
  if (!data.available || !data.rounds.length) {
    fixtureCount.textContent = "0";
    fixtureList.append(
      createElement("div", "pane-placeholder no-fixtures-placeholder", "No fixtures fetched for this season."),
    );
    return;
  }

  const total = data.rounds.reduce((sum, round) => sum + round.fixtures.length, 0);
  fixtureCount.textContent = total.toLocaleString();
  for (const round of data.rounds) {
    const group = createElement("section", "round-group");
    const heading = createElement("h3", "round-heading");
    heading.append(
      createElement("span", "round-marker", ""),
      createElement("span", "round-name", round.name),
      createElement("span", "round-count", String(round.fixtures.length)),
    );
    group.append(heading);

    const matches = createElement("div", "round-fixtures");
    for (const fixture of round.fixtures) {
      const isPlayed = hasFinalScore(fixture);
      const selected = Number(selectedFixtureId) === fixture.fixture_id;
      const button = makeTreeButton(
        "",
        "match-row",
        "fixture",
        String(fixture.fixture_id),
        selected,
      );
      button.dataset.round = round.name;
      const summary = isPlayed
        ? `${fixture.home_team} ${fixture.home_goals ?? "—"}-${fixture.away_goals ?? "—"} ${fixture.away_team}`
        : `${fixture.home_team} - ${fixture.away_team}`;
      button.append(
        createElement("span", "match-summary", summary),
        createElement("span", isPlayed ? "match-result" : "match-kickoff", isPlayed ? fixture.status : formatFixtureDate(fixture.date)),
      );
      matches.append(button);
    }
    group.append(matches);
    fixtureList.append(group);
  }
}

function renderFixtureDetails(data) {
  const fixture = data.fixture;
  fixtureDetails.replaceChildren();
  const score = hasFinalScore(fixture);
  const matchHeader = createElement("section", "detail-match-card");
  const meta = createElement("div", "detail-match-meta");
  meta.append(
    createElement("span", "detail-round", fixture.round),
    createElement("span", "detail-status", fixture.status || "Scheduled"),
  );

  const teams = createElement("div", "detail-teams");
  for (const side of ["home", "away"]) {
    const teamBlock = createElement("div", `detail-team ${side}`);
    const logo = createElement("div", "detail-team-logo");
    const teamName = fixture[`${side}_team`] || (side === "home" ? "Home team" : "Away team");
    addImage(
      logo,
      fixture[`${side}_logo`],
      "detail-team-logo-image",
      `${teamName} logo`,
      teamName.slice(0, 1).toUpperCase(),
    );
    teamBlock.append(
      logo,
      createElement("span", "detail-team-name", teamName),
    );
    teams.append(teamBlock);
  }
  const scoreLine = createElement("strong", "detail-score", score
    ? `${fixture.home_goals ?? "—"}-${fixture.away_goals ?? "—"}`
    : "-");
  teams.insertBefore(scoreLine, teams.children[1]);
  matchHeader.append(meta, teams);

  const matchMeta = createElement("div", "detail-venue");
  matchMeta.append(
    createElement("span", "", formatFixtureDate(fixture.date)),
    createElement("span", "", fixture.venue || "Venue not available"),
    createElement("span", "", `Fixture ${fixture.fixture_id}`),
  );
  matchHeader.append(matchMeta);
  fixtureDetails.append(matchHeader);

  const statsHeading = createElement("div", "stats-section-heading");
  statsHeading.append(
    createElement("h3", "", "Match statistics"),
    createElement("span", "stats-group-count", `${data.stats.length} groups`),
  );
  fixtureDetails.append(statsHeading);

  if (!data.stats.length) {
    fixtureDetails.append(createElement("div", "detail-empty", "No processed stats are available for this fixture yet."));
    return;
  }

  const stats = createElement("div", "detail-stats-list");
  for (const group of data.stats) {
    const card = createElement("details", "stats-card");
    card.open = group.name === "fixtures" || group.name === "outcome-probability";
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
  fixtureDetails.append(stats);
}

async function loadSeasons(leagueId) {
  const requestId = ++seasonRequestId;
  fixtureRequestId += 1;
  detailRequestId += 1;
  currentSeasons = [];
  seasonsLoadedLeagueId = "";
  fetchPanel.hidden = true;
  try {
    const data = await requestJson(`/api/seasons?league_id=${encodeURIComponent(leagueId)}`);
    if (requestId !== seasonRequestId || Number(selectedLeagueId) !== Number(leagueId)) return;
    currentSeasons = data.seasons;
    seasonsLoadedLeagueId = String(leagueId);
    fetchPanel.hidden = currentSeasons.length === 0;
    populateFetchSeasons();
    renderCountryTree();
  } catch (error) {
    if (requestId === seasonRequestId) showError(error);
  }
}

async function loadSeasonFixtures(leagueId, season) {
  const requestId = ++fixtureRequestId;
  detailRequestId += 1;
  selectedFixtureId = "";
  fixtureDetails.replaceChildren(
    createElement("div", "pane-placeholder", "Select a fixture to see match data."),
  );
  detailsContext.textContent = "Select a fixture to see match data";
  fixtureList.replaceChildren(
    createElement("div", "pane-placeholder", "Loading fixtures…"),
  );
  fixtureCount.textContent = "…";
  const league = leagues.find((item) => item.league_id === Number(leagueId));
  fixturesContext.textContent = `${league?.league_name || "League"} · ${season}`;
  clearError();
  try {
    const params = new URLSearchParams({ league_id: leagueId, season });
    const data = await requestJson(`/api/season-fixtures?${params}`);
    if (requestId !== fixtureRequestId || selectedSeason !== String(season)) return;
    renderFixtureGroups(data);
  } catch (error) {
    if (requestId === fixtureRequestId) showError(error);
  }
}

async function loadFixtureDetails(leagueId, season, fixtureId) {
  const requestId = ++detailRequestId;
  fixtureDetails.replaceChildren(
    createElement("div", "pane-placeholder", "Loading match details…"),
  );
  detailsContext.textContent = "Loading match statistics…";
  clearError();
  try {
    const params = new URLSearchParams({
      league_id: leagueId,
      season,
      fixture_id: fixtureId,
    });
    const data = await requestJson(`/api/fixture?${params}`);
    if (requestId !== detailRequestId || selectedFixtureId !== String(fixtureId)) return;
    renderFixtureDetails(data);
    const { home_team: homeTeam, away_team: awayTeam } = data.fixture;
    detailsContext.textContent = `${homeTeam} vs ${awayTeam}`;
  } catch (error) {
    if (requestId === detailRequestId) showError(error);
  }
}

function updateFetchProgress(job) {
  const percentage = Math.max(0, Math.min(100, Number(job.progress) || 0));
  document.querySelector("#fetch-progress-message").textContent = job.message;
  document.querySelector("#fetch-progress-value").textContent = `${percentage}%`;
  document.querySelector("#fetch-progress-bar").style.width = `${percentage}%`;
  document.querySelector(".progress-track").setAttribute("aria-valuenow", String(percentage));
}

async function pollFetchJob(jobId, leagueId, season) {
  try {
    const job = await requestJson(`/api/fetch-status?job_id=${encodeURIComponent(jobId)}`);
    updateFetchProgress(job);
    if (job.status === "running") {
      window.setTimeout(() => pollFetchJob(jobId, leagueId, season), 900);
      return;
    }

    fetchJobId = null;
    fetchSeasonList.querySelectorAll('input[name="fetch-season"]').forEach((checkbox) => {
      checkbox.disabled = false;
    });
    document.querySelector("#select-all-seasons").disabled = false;
    updateFetchSelection();
    if (Number(selectedLeagueId) === Number(leagueId)) {
      await loadSeasons(leagueId);
      if (selectedSeason === String(season)) {
        await loadSeasonFixtures(leagueId, season);
      }
    }
    if (job.status === "failed") {
      showError(new Error(`Fixture fetch failed: ${job.message}`));
      return;
    }
  } catch (error) {
    showError(error);
    window.setTimeout(() => pollFetchJob(jobId, leagueId, season), 2000);
  }
}

async function startFixtureFetch() {
  const seasons = [...fetchSeasonList.querySelectorAll('input[name="fetch-season"]:checked')]
    .map((checkbox) => Number(checkbox.value));
  if (!seasons.length || !selectedLeagueId) return;

  fetchButton.disabled = true;
  clearError();
  fetchProgress.hidden = false;
  updateFetchProgress({ progress: 0, message: "Starting fixture fetch…" });
  try {
    const response = await requestJson("/api/fetch-fixtures", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ league_id: Number(selectedLeagueId), seasons }),
    });
    fetchJobId = response.job_id;
    fetchSeasonList.querySelectorAll('input[name="fetch-season"]').forEach((checkbox) => {
      checkbox.disabled = true;
    });
    document.querySelector("#select-all-seasons").disabled = true;
    await pollFetchJob(response.job_id, selectedLeagueId, selectedSeason);
  } catch (error) {
    fetchButton.disabled = false;
    showError(error);
  }
}

countryTree.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, id } = button.dataset;
  clearError();

  if (action === "country") {
    seasonRequestId += 1;
    fixtureRequestId += 1;
    detailRequestId += 1;
    const countryName = id;
    if (expandedCountry === countryName) {
      expandedCountry = "";
      selectedCountry = "";
      selectedLeagueId = "";
      expandedLeagueId = "";
      selectedSeason = "";
      selectedFixtureId = "";
      currentSeasons = [];
      seasonsLoadedLeagueId = "";
      fetchPanel.hidden = true;
      fixtureList.replaceChildren(createElement("div", "pane-placeholder", "Select a country, league, and season."));
      fixturesContext.textContent = "Select a season to browse matches";
      fixtureCount.textContent = "—";
      fixtureDetails.replaceChildren(createElement("div", "pane-placeholder", "Fixture details will appear here."));
      detailsContext.textContent = "Select a fixture to see match data";
    } else {
      expandedCountry = countryName;
      selectedCountry = countryName;
      expandedLeagueId = "";
      selectedLeagueId = "";
      selectedSeason = "";
      selectedFixtureId = "";
      currentSeasons = [];
      seasonsLoadedLeagueId = "";
      fetchPanel.hidden = true;
      fixtureList.replaceChildren(createElement("div", "pane-placeholder", "Select a league and season to browse matches."));
      fixturesContext.textContent = "Select a season to browse matches";
      fixtureCount.textContent = "—";
      fixtureDetails.replaceChildren(createElement("div", "pane-placeholder", "Fixture details will appear here."));
      detailsContext.textContent = "Select a fixture to see match data";
    }
  } else if (action === "league") {
    fixtureRequestId += 1;
    detailRequestId += 1;
    const league = leagues.find((item) => item.league_id === Number(id));
    if (!league) return;
    if (Number(expandedLeagueId) === league.league_id) {
      expandedLeagueId = "";
      selectedLeagueId = "";
      selectedSeason = "";
      selectedFixtureId = "";
      currentSeasons = [];
      seasonsLoadedLeagueId = "";
      fetchPanel.hidden = true;
      fixtureList.replaceChildren(createElement("div", "pane-placeholder", "Select a league and season to browse matches."));
      fixturesContext.textContent = "Select a season to browse matches";
      fixtureCount.textContent = "—";
      fixtureDetails.replaceChildren(createElement("div", "pane-placeholder", "Fixture details will appear here."));
      detailsContext.textContent = "Select a fixture to see match data";
    } else {
      expandedCountry = league.country_name;
      selectedCountry = league.country_name;
      expandedLeagueId = String(league.league_id);
      selectedLeagueId = String(league.league_id);
      selectedSeason = "";
      selectedFixtureId = "";
      currentSeasons = [];
      seasonsLoadedLeagueId = "";
      fetchPanel.hidden = true;
      fetchSeasonList.replaceChildren();
      fixtureList.replaceChildren(createElement("div", "pane-placeholder", "Select a season to browse matches."));
      fixturesContext.textContent = `${league.league_name} · select a season`;
      fixtureCount.textContent = "—";
      fixtureDetails.replaceChildren(createElement("div", "pane-placeholder", "Select a fixture to see match data."));
      detailsContext.textContent = "Select a fixture to see match data";
      loadSeasons(league.league_id);
    }
  } else if (action === "season") {
    const season = id;
    selectedLeagueId = String(expandedLeagueId);
    selectedSeason = season;
    selectedFixtureId = "";
    renderCountryTree();
    loadSeasonFixtures(selectedLeagueId, season);
  }

  renderCountryTree();
});

fixtureList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action='fixture']");
  if (!button) return;
  selectedFixtureId = button.dataset.id;
  fixtureList.querySelectorAll(".match-row").forEach((row) => {
    row.classList.toggle("selected", row === button);
  });
  loadFixtureDetails(selectedLeagueId, selectedSeason, selectedFixtureId);
});

searchInput.addEventListener("input", renderCountryTree);
fetchSeasonList.addEventListener("change", updateFetchSelection);
fetchButton.addEventListener("click", startFixtureFetch);
document.querySelector("#select-all-seasons").addEventListener("click", () => {
  const checkboxes = [...fetchSeasonList.querySelectorAll('input[name="fetch-season"]')];
  const selectAll = !checkboxes.every((checkbox) => checkbox.checked);
  checkboxes.forEach((checkbox) => { checkbox.checked = selectAll; });
  updateFetchSelection();
});
