const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(tag = "div", text = "") {
    this.tag = tag;
    this.textContent = text;
    this.children = [];
    this.dataset = {};
    this.listeners = {};
    this.classList = { toggle() {} };
    this.hidden = false;
    this.value = "all";
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

const elements = new Map();
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  },
  createElement: (tag) => new Element(tag),
};
const fixture = { fixture_id: 1, date: "2026-01-01", home_team: "A", away_team: "B",
  status: "NS", round: "Round 1" };
let savedMarkets = [];
let timer;
const context = vm.createContext({
  document, URLSearchParams, console,
  window: { setInterval: (callback) => { timer = callback; } },
  createElement: (tag, css, text) => new Element(tag, text),
  addImage() {}, renderFavoriteFixtures() {},
  hasFinalScore: () => false, formatFixtureDate: () => "", formatFixtureTime: () => "",
  formatMetricName: (value) => value, formatMetricValue: (value) => value,
  requestJson: async (url) => url.startsWith("/api/market-predictions")
    ? { prediction_season: 2026, predictions: savedMarkets }
    : { fixture, prediction: null, market_predictions: savedMarkets, is_favorite: false, stats: [] },
});
vm.runInContext(fs.readFileSync("app\\static\\predictions.js", "utf8"), context);
assert.equal(typeof timer, "function");
assert.equal(document.querySelector("#prediction-refresh").listeners.click.length, 1);
vm.runInContext(`
  predictionLeagueId = "39";
  predictionSeason = "2026";
  predictionFixtureData = { fixtures: [{ ...${JSON.stringify(fixture)}, prediction: null, market_predictions: [] }] };
  renderPredictionFixtures(predictionFixtureData);
`, context);

(async () => {
  await vm.runInContext('selectPredictionFixture("39", "2026", "1")', context);
  await vm.runInContext('selectPredictionFixture("39", "2026", "1")', context);
  assert.equal(document.querySelector("#prediction-refresh").listeners.click.length, 1);
  savedMarkets = [{ fixture_id: 1, target: "result", prediction: "H",
    probabilities: { H: 0.6, D: 0.2, A: 0.2, "1X": 0.8 } }];
  timer();
  await new Promise(setImmediate);
  assert.equal(vm.runInContext("predictionFixtureData.fixtures[0].market_predictions[0].target", context), "result");
  assert.equal(vm.runInContext("predictionDetailData.market_predictions[0].prediction", context), "H");
  const details = JSON.stringify(document.querySelector("#prediction-details").children);
  assert.match(details, /Saved market models/);
  assert.match(details, /Home win/);
  assert.match(details, /1X: 80%/);
  assert.equal(vm.runInContext('marketValue("btts", "1")', context), "Yes");
  assert.equal(vm.runInContext('marketValue("over_2_5", "0")', context), "Under 2.5");
  assert.equal(vm.runInContext('marketLabel("over_2_5")', context), "Over / under 2.5");
  savedMarkets = [];
  await vm.runInContext("refreshSavedMarkets()", context);
  assert.equal(vm.runInContext("predictionDetailData.market_predictions.length", context), 0);
})().catch((error) => { console.error(error); process.exitCode = 1; });
