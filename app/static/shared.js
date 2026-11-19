let leagues = [];

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = text;
  return element;
}

function makeOption(label, value = "") {
  const option = createElement("option", "", label);
  option.value = value;
  return option;
}

function addImage(parent, source, className, alt, fallbackText = "") {
  if (!source || !/^https?:\/\//i.test(source)) {
    if (fallbackText) parent.append(createElement("span", `${className}-fallback`, fallbackText));
    return;
  }
  const image = document.createElement("img");
  image.className = className;
  image.src = source;
  image.alt = alt;
  image.loading = "lazy";
  image.referrerPolicy = "no-referrer";
  image.addEventListener("error", () => {
    image.remove();
    if (fallbackText) parent.append(createElement("span", `${className}-fallback`, fallbackText));
  }, { once: true });
  parent.append(image);
}

async function requestJson(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setScreen() {
  const hash = window.location.hash;
  const isModelLab = hash === "#model-lab-screen";
  const isPredictions = hash === "#prediction-screen";
  document.querySelector("#dashboard-screen").hidden = isModelLab || isPredictions;
  document.querySelector("#prediction-screen").hidden = !isPredictions;
  document.querySelector("#model-lab-screen").hidden = !isModelLab;
  document.querySelector("#dashboard-nav").classList.toggle("active", !isModelLab && !isPredictions);
  document.querySelector("#prediction-nav").classList.toggle("active", isPredictions);
  document.querySelector("#model-lab-nav").classList.toggle("active", isModelLab);
  document.querySelector("#dashboard-nav").toggleAttribute("aria-current", !isModelLab && !isPredictions);
  document.querySelector("#prediction-nav").toggleAttribute("aria-current", isPredictions);
  document.querySelector("#model-lab-nav").toggleAttribute("aria-current", isModelLab);
  document.querySelector("#breadcrumb-title").textContent =
    isModelLab ? "Model lab" : (isPredictions ? "Predictions" : "Match explorer");
  if (isPredictions) {
    loadFavoriteFixtures();
    if (predictionLeagueId && predictionSeason) {
      loadPredictionFixtures(predictionLeagueId, predictionSeason);
    }
  }
}

function groupByCountry(items) {
  const groups = new Map();
  for (const league of items) {
    if (!groups.has(league.country_name)) groups.set(league.country_name, []);
    groups.get(league.country_name).push(league);
  }
  return groups;
}

function makeTreeButton(label, className, action, id, selected = false) {
  const button = createElement("button", `${className}${selected ? " selected" : ""}`, label);
  button.type = "button";
  button.dataset.action = action;
  button.dataset.id = id;
  return button;
}

function formatFixtureDate(value, withTime = true) {
  if (!value) return "Date and time TBC";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date and time TBC";
  return date.toLocaleString(undefined, withTime
    ? { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }
    : { year: "numeric", month: "short", day: "numeric" });
}

function formatFixtureTime(value) {
    if (!value) return "Time TBC";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "Time TBC";
    return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function hasFinalScore(fixture) {
  return ["FT", "AET", "PEN", "AWD", "WO"].includes((fixture.status || "").toUpperCase());
}

function formatMetricName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\bpm\b/gi, "per match")
    .replace(/\bx\b/g, "Home")
    .replace(/\by\b/g, "Away")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}

async function loadLeagues() {
  try {
    const data = await requestJson("/api/leagues", { headers: { Accept: "application/json" } });
    if (!Array.isArray(data.leagues)) throw new Error("The server returned an invalid league list.");
    leagues = data.leagues;
    renderCountryTree();
    renderPredictionLeagueTree();
  } catch (error) {
    countryTree.replaceChildren();
    countryTree.append(createElement("div", "pane-placeholder", `Could not load leagues. ${error.message}`));
    browserCount.textContent = "Unavailable";
    predictionLeagueTree.replaceChildren(createElement("div", "pane-placeholder", `Could not load leagues. ${error.message}`));
    predictionLeagueCount.textContent = "Unavailable";
  }
}

window.addEventListener("hashchange", setScreen);
