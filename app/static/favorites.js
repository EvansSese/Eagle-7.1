const favoriteFixtureList = document.querySelector("#favorite-fixture-list");
const favoriteFixtureCount = document.querySelector("#favorite-fixture-count");
const favoriteListForm = document.querySelector("#favorite-list-form");
const favoriteListName = document.querySelector("#favorite-list-name");

let favorites = [];
const activeFavoriteListIds = new Set();

function renderFavoriteFixtures() {
  favoriteFixtureList.replaceChildren();
  favoriteFixtureCount.textContent = String(favorites.length);
  if (!favorites.length) {
    favoriteFixtureList.append(createElement("div", "pane-placeholder", "Create a list to save fixtures."));
    return;
  }
  const knownIds = new Set(favorites.map((list) => list.list_id));
  for (const listId of activeFavoriteListIds) {
    if (!knownIds.has(listId)) activeFavoriteListIds.delete(listId);
  }
  for (const list of favorites) {
    const section = createElement("section", "favorite-list-group");
    const heading = createElement("div", "favorite-list-heading");
    const label = createElement("label", "favorite-list-activate");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = activeFavoriteListIds.has(list.list_id);
    checkbox.dataset.action = "activate-favorite-list";
    checkbox.dataset.listId = list.list_id;
    label.append(checkbox, createElement("strong", "", list.name));
    const controls = createElement("div", "favorite-list-controls");
    const rename = createElement("button", "favorite-list-control", "Rename");
    rename.type = "button";
    rename.dataset.action = "rename-favorite-list";
    rename.dataset.listId = list.list_id;
    const remove = createElement("button", "favorite-list-control danger", "Delete");
    remove.type = "button";
    remove.dataset.action = "delete-favorite-list";
    remove.dataset.listId = list.list_id;
    controls.append(rename, remove);
    heading.append(label, controls);
    section.append(heading);
    if (!list.fixtures.length) {
      section.append(createElement("div", "favorite-list-empty", "No fixtures in this list."));
    }

    const countries = new Map();
    for (const fixture of list.fixtures) {
      if (!countries.has(fixture.country_name)) countries.set(fixture.country_name, new Map());
      const leaguesByName = countries.get(fixture.country_name);
      if (!leaguesByName.has(fixture.league_name)) leaguesByName.set(fixture.league_name, []);
      leaguesByName.get(fixture.league_name).push(fixture);
    }
    for (const [countryName, leaguesByName] of countries) {
      const country = createElement("section", "favorite-group");
      country.append(createElement("h3", "favorite-group-title", countryName));
      for (const [leagueName, fixtures] of leaguesByName) {
        const league = createElement("section", "favorite-group");
        league.append(createElement("h4", "favorite-group-title", leagueName));
        for (const fixture of fixtures) {
          const row = createElement("div", "favorite-fixture-row");
          const button = createElement("button", "favorite-fixture-button");
          button.type = "button";
          button.dataset.action = "favorite-fixture";
          button.dataset.leagueId = String(fixture.league_id);
          button.dataset.season = String(fixture.season);
          button.dataset.fixtureId = String(fixture.fixture_id);
          button.disabled = fixture.fixture_available === false;
          if (!fixture.fixture_available) {
            button.title = `Fixture ${fixture.fixture_id} is no longer available; remove it from this list.`;
          }
          button.classList.toggle(
            "selected",
            String(fixture.fixture_id) === predictionDetailFixtureId &&
              String(fixture.league_id) === predictionDetailLeagueId &&
              String(fixture.season) === predictionDetailSeason,
          );
          button.append(
            createElement(
              "span",
              "favorite-fixture-teams",
              fixture.fixture_available === false
                ? `Unavailable fixture · #${fixture.fixture_id}`
                : `${fixture.home_team} vs ${fixture.away_team}`,
            ),
            createElement(
              "span",
              "favorite-fixture-meta",
              `${fixture.season} · ${formatFixtureDate(fixture.date)}${fixture.round ? ` · ${fixture.round}` : ""}`,
            ),
          );
          const removeFixture = createElement("button", "favorite-fixture-remove", "×");
          removeFixture.type = "button";
          removeFixture.title = "Remove fixture from this list";
          removeFixture.setAttribute("aria-label", "Remove fixture from this list");
          removeFixture.dataset.action = "remove-favorite-fixture";
          removeFixture.dataset.listId = list.list_id;
          removeFixture.dataset.leagueId = String(fixture.league_id);
          removeFixture.dataset.season = String(fixture.season);
          removeFixture.dataset.fixtureId = String(fixture.fixture_id);
          row.append(button, removeFixture);
          league.append(row);
        }
        country.append(league);
      }
      section.append(country);
    }
    favoriteFixtureList.append(section);
  }
}

async function loadFavoriteFixtures() {
  try {
    const data = await requestJson("/api/favorites");
    favorites = data.lists;
    renderFavoriteFixtures();
  } catch (error) {
    favoriteFixtureList.replaceChildren(createElement("div", "pane-placeholder", error.message));
    showPredictionError(error);
  }
}

async function toggleFavoriteFixture() {
  if (!predictionDetailFixtureId) return;
  const listIds = [...activeFavoriteListIds];
  if (!listIds.length) {
    showPredictionError(new Error("Activate at least one favorite list before saving a fixture."));
    return;
  }
  const leagueId = predictionDetailLeagueId;
  const season = predictionDetailSeason;
  const fixtureId = predictionDetailFixtureId;
  try {
    await requestJson("/api/favorites", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        action: "add-fixture",
        list_ids: listIds,
        league_id: Number(leagueId),
        season: Number(season),
        fixture_id: Number(fixtureId),
      }),
    });
    await Promise.all([
      loadFavoriteFixtures(),
      selectPredictionFixture(leagueId, season, fixtureId),
    ]);
  } catch (error) {
    showPredictionError(error);
  }
}

async function updateFavoriteList(action, values) {
  await requestJson("/api/favorites", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ action, ...values }),
  });
  await loadFavoriteFixtures();
}

favoriteListForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = favoriteListName.value.trim();
  if (!name) return;
  try {
    await updateFavoriteList("create-list", { name });
    favoriteListName.value = "";
  } catch (error) {
    showPredictionError(error);
  }
});
favoriteFixtureList.addEventListener("change", (event) => {
  const checkbox = event.target.closest("input[data-action='activate-favorite-list']");
  if (!checkbox) return;
  if (checkbox.checked) activeFavoriteListIds.add(checkbox.dataset.listId);
  else activeFavoriteListIds.delete(checkbox.dataset.listId);
});
favoriteFixtureList.addEventListener("click", (event) => {
  const actionButton = event.target.closest("button[data-action]");
  if (actionButton?.dataset.action === "rename-favorite-list") {
    const list = favorites.find((item) => item.list_id === actionButton.dataset.listId);
    if (!list) return;
    const name = window.prompt("Rename favorite list", list.name);
    if (name === null || !name.trim()) return;
    updateFavoriteList("rename-list", { list_id: list.list_id, name: name.trim() })
      .catch(showPredictionError);
    return;
  }
  if (actionButton?.dataset.action === "delete-favorite-list") {
    const listId = actionButton.dataset.listId;
    const list = favorites.find((item) => item.list_id === listId);
    if (!list || !window.confirm(`Delete "${list.name}" and all its saved fixtures?`)) return;
    updateFavoriteList("delete-list", { list_id: listId })
      .then(() => activeFavoriteListIds.delete(listId))
      .catch(showPredictionError);
    return;
  }
  if (actionButton?.dataset.action === "remove-favorite-fixture") {
    const { listId, leagueId, season, fixtureId } = actionButton.dataset;
    updateFavoriteList("remove-fixture", {
      list_id: listId,
      league_id: Number(leagueId),
      season: Number(season),
      fixture_id: Number(fixtureId),
    }).then(() => {
      if (
        predictionDetailLeagueId === leagueId &&
        predictionDetailSeason === season &&
        predictionDetailFixtureId === fixtureId
      ) {
        return selectPredictionFixture(leagueId, season, fixtureId);
      }
    }).catch(showPredictionError);
    return;
  }
  const button = event.target.closest("button[data-action='favorite-fixture']");
  if (!button) return;
  selectPredictionFixture(
    button.dataset.leagueId,
    button.dataset.season,
    button.dataset.fixtureId,
  );
});
