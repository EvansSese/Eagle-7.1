const assert = require("node:assert/strict");
const fs = require("node:fs");
const test = require("node:test");
const vm = require("node:vm");

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
}

const shared = fs.readFileSync("app/static/shared.js", "utf8");
const source = fs.readFileSync("app/static/predictions.js", "utf8");
const extract = (text, start, end) => text.slice(text.indexOf(start), text.indexOf(end));
const script = [
  extract(shared, "function createElement(", "function makeOption("),
  extract(shared, "function hasFinalScore(", "function formatMetricName("),
  extract(source, "function evaluatePrediction(", "async function loadPredictionFixtures("),
].join("\n");

function render(filter, fixtures) {
  const list = new Element("div");
  const count = new Element("span");
  const context = {
    document: { createElement: (tag) => new Element(tag) },
    predictionFixtureList: list,
    predictionFixtureCount: count,
    predictionFixtureFilter: { value: filter },
    predictionSelectedFixtureId: "",
    predictionLeagueId: "1",
    predictionSeason: "2024",
    predictionDetailLeagueId: "",
    predictionDetailSeason: "",
    addImage: () => {},
    formatFixtureDate: (date) => date,
    formatFixtureTime: (date) => date,
    formatConfidence: (value) => String(value),
  };
  vm.createContext(context);
  vm.runInContext(script, context);
  context.renderPredictionFixtures({ fixtures });
  return { list, count };
}

test("filters statuses and preserves chronological round and fixture order", () => {
  const prediction = {
    summary: "Prediction",
    classification: {
      outcomes: [{ label: "Home", confidence: 0.5 }],
      double_chance: [{ code: "1X", confidence: 0.8 }],
    },
  };
  const fixture = (id, status, round) => ({
    fixture_id: id, status, round, date: `2020-01-0${id}`,
    home_team: "Home", away_team: "Away", prediction,
  });

  test("counts all eligible FT fixtures regardless of filter and marks recorded tests", () => {
    const prediction = (outcome, chance, isTest) => ({
      summary: "Prediction", is_test: isTest,
      classification: {
        outcomes: [{ code: outcome, label: outcome, confidence: 0.5 }],
        double_chance: [{ code: chance, confidence: 0.8 }],
      },
    });
    const fixture = (id, status, home, away, pick) => ({
      fixture_id: id, status, home_goals: home, away_goals: away, round: "Round 1",
      home_team: "Home", away_team: "Away", prediction: pick,
    });
    const fixtures = [
      fixture(1, "FT", 2, 1, prediction("H", "1X", true)),
      fixture(2, "FT", 1, 1, prediction("A", "X2", false)),
      fixture(3, "FT", null, 0, prediction("A", "12", null)),
      fixture(4, "NS", null, null, prediction("H", "1X", false)),
      fixture(5, "AET", 0, 1, prediction("H", "1X", false)),
    ];
    const all = render("all", fixtures);
    assert.equal(all.list.children[0].children[0].children[2].textContent, "3W-(1/2)");
    assert.equal(all.list.children[0].children[0].children[3].textContent, "DC-(2/2)");
    const markers = all.list.children[0].children[1].children[0].children[3];
    assert.equal(markers.children[0].textContent, "test");
    assert.deepEqual(markers.children[1].children.map((circle) => circle.textContent), ["3W", "DC"]);
    assert.deepEqual(
      all.list.children[0].children[1].children[1].children[3].children[0].children.map((circle) => circle.textContent),
      ["DC"],
    );
    const upcoming = render("upcoming", fixtures);
    assert.equal(upcoming.list.children[0].children[0].children[2].textContent, "3W-(1/2)");
    assert.equal(upcoming.list.children[0].children[0].children[3].textContent, "DC-(2/2)");
  });

  test("evaluates only FT scores and double chance coverage", () => {
    const context = {};
    vm.createContext(context);
    vm.runInContext(extract(source, "function evaluatePrediction(", "function testBadge("), context);
    const prediction = { classification: { outcomes: [{ code: "H" }], double_chance: [{ code: "1X" }] } };
    const evaluate = (status, home, away) => context.evaluatePrediction(
      { status, home_goals: home, away_goals: away }, prediction,
    );
    assert.equal(evaluate("FT", 1, 1).outcomeCorrect, false);
    assert.equal(evaluate("FT", 1, 1).doubleChanceCorrect, true);
    assert.equal(evaluate("FT", 0, 2).doubleChanceCorrect, false);
    assert.equal(evaluate("NS", 0, 0).state, "not-played");
    assert.equal(evaluate("AET", 2, 1).state, "unavailable");
    assert.equal(evaluate("FT", null, 0).state, "unavailable");
  });
  const fixtures = [
    fixture(1, "NS", "Round 1"),
    fixture(2, "FT", "Round 2"),
    fixture(3, "NS", "Round 1"),
    fixture(4, "AET", "Round 2"),
    fixture(5, "TBD", "Round 3"),
  ];
  const ids = (list) => list.children.flatMap((group) =>
    group.children[1].children.map((card) => Number(card.children[0].dataset.id)));
  const all = render("all", fixtures);
  assert.equal(all.count.textContent, "5");
  assert.deepEqual(all.list.children.map((group) => group.children[0].children[1].textContent), [
    "Round 1", "Round 2", "Round 3",
  ]);
  assert.deepEqual(ids(all.list), [1, 3, 2, 4, 5]);
  assert.deepEqual(ids(render("upcoming", fixtures).list), [1, 3]);
  assert.deepEqual(ids(render("played", fixtures).list), [2, 4]);
  assert.equal(render("played", [fixtures[0]]).count.textContent, "0");
});
