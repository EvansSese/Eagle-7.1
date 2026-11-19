"""Verify that dashboard feature assets load and work together."""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import duckdb

from app import server


class AssetParser(HTMLParser):
    """Collect local stylesheets and scripts in the order the page loads them."""

    def __init__(self):
        super().__init__()
        self.styles = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "link" and attributes.get("href", "").startswith("/static/"):
            self.styles.append(attributes["href"])
        if tag == "script" and attributes.get("src", "").startswith("/static/"):
            self.scripts.append(attributes["src"])
            if "defer" not in attributes:
                raise AssertionError("Dashboard scripts must preserve deferred load order")


class DashboardAssetTests(unittest.TestCase):
    def test_feature_assets_are_served_with_correct_content_types(self):
        http_server = server.ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
        thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(http_server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(http_server.shutdown)
        root = f"http://127.0.0.1:{http_server.server_port}"
        with urlopen(root) as response:
            parser = AssetParser()
            parser.feed(response.read().decode("utf-8"))

        self.assertEqual(parser.scripts, [
            "/static/shared.js", "/static/model-lab.js", "/static/explorer.js",
            "/static/predictions.js", "/static/favorites.js", "/static/dashboard.js",
        ])
        self.assertEqual(parser.styles, [
            "/static/dashboard.css", "/static/explorer-layout.css",
            "/static/predictions.css", "/static/favorites.css",
            "/static/components.css", "/static/explorer.css", "/static/messages.css",
            "/static/model-lab.css", "/static/responsive.css",
            "/static/typography.css", "/static/theme.css",
        ])
        for path in parser.styles + parser.scripts:
            with self.subTest(path=path), urlopen(root + path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
                expected = "text/css" if path.endswith(".css") else "text/javascript"
                self.assertEqual(response.headers.get_content_type(), expected)

    @unittest.skipUnless(
        os.environ.get("EAGLE_BROWSER_TEST") == "1"
        and (
            shutil.which("chrome")
            or shutil.which("chromium")
            or Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe").is_file()
            or Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe").is_file()
        ),
        "Set EAGLE_BROWSER_TEST=1 and install a Chromium browser to run the interaction smoke test",
    )
    def test_dashboard_features_work_together_in_browser(self):
        browser = (
            shutil.which("chrome") or shutil.which("chromium") or
            str(next(path for path in (
                Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            ) if path.is_file()))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalogue = root / "leagues.ddb"
            country_dir = root / "countries"
            country_dir.mkdir()
            with duckdb.connect(str(catalogue)) as connection:
                connection.execute(
                    """CREATE TABLE leagues (
                        league_id INTEGER, league_name VARCHAR, country_name VARCHAR,
                        country_flag VARCHAR, league_logo VARCHAR, seasons INTEGER[]
                    )"""
                )
                connection.execute("INSERT INTO leagues VALUES (1, 'League', 'Test', NULL, NULL, [2024])")
            with duckdb.connect(str(country_dir / "Test.ddb")) as connection:
                connection.execute(
                    """CREATE TABLE "1-2024-fixtures" (
                        fixture_id INTEGER, date TIMESTAMP, venue VARCHAR, round VARCHAR,
                        home_team VARCHAR, away_team VARCHAR, home_goals INTEGER,
                        away_goals INTEGER, status VARCHAR
                    )"""
                )
                connection.execute(
                    """INSERT INTO "1-2024-fixtures" VALUES
                       (10, '2024-01-01', 'Ground', 'Round 1', 'Home', 'Away', 2, 1, 'FT')"""
                )

            instrumentation = """
<script>
window.dashboardErrors = [];
window.addEventListener("error", event => window.dashboardErrors.push(event.message));
window.addEventListener("unhandledrejection", event => window.dashboardErrors.push(String(event.reason)));
window.addEventListener("load", async () => {
  const waitFor = async (selector) => {
    for (let i = 0; i < 100; i++) {
      const element = document.querySelector(selector);
      if (element) return element;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    throw new Error("Missing " + selector);
  };
  try {
    (await waitFor('#country-tree [data-action="country"]')).click();
    (await waitFor('#country-tree [data-action="league"]')).click();
    (await waitFor('#country-tree [data-action="season"]')).click();
    (await waitFor('#fixture-list [data-action="fixture"]')).click();
    await waitFor('#fixture-details .detail-match-card');
    document.querySelector('#theme-toggle').click();
    if (document.body.dataset.theme !== 'light') throw new Error("Theme did not change");
    document.querySelector('#prediction-nav').click();
    (await waitFor('#prediction-league-tree [data-action="prediction-country"]')).click();
    (await waitFor('#prediction-league-tree [data-action="prediction-league"]')).click();
    await waitFor('#prediction-league-tree [data-action="prediction-season"]');
    const prediction = {
      summary: 'Test prediction', is_test: true,
      classification: {
        outcomes: [{ code: 'H', label: 'Home win', confidence: 0.6 }],
        double_chance: [{ code: '1X', label: 'Home or draw', confidence: 0.8 }],
      },
      regression: { score: '1-0', models: [] },
    };
    const fixture = {
      fixture_id: 10, round: 'Round 1', home_team: 'Home', away_team: 'Away',
      home_goals: 1, away_goals: 1, status: 'FT', prediction,
    };
    const unplayed = {
      ...fixture, fixture_id: 11, home_goals: null, away_goals: null, status: 'NS',
      prediction: { ...prediction, is_test: null },
    };
    predictionFixtureFilter.value = 'all';
    renderPredictionFixtures({ fixtures: [fixture, unplayed] });
    const accuracy = () => [...document.querySelectorAll('.prediction-round-accuracy')].map(badge => badge.textContent).join(', ');
    if (accuracy() !== '3W-(0/1), DC-(1/1)') throw new Error('Wrong round accuracy: ' + accuracy());
    if (document.querySelector('.prediction-test-badge').textContent !== 'test') throw new Error('Missing test badge');
    if (!document.querySelector('.prediction-test-unknown')) throw new Error('Missing unknown test state');
    let cards = document.querySelectorAll('.prediction-fixture-card');
    if ([...cards[0].querySelectorAll('.prediction-correct-indicator')].map(item => item.textContent).join(',') !== 'DC') throw new Error('Wrong correct indicators');
    if (cards[1].querySelector('.prediction-correct-indicator')) throw new Error('Unplayed fixture has correct indicators');
    const bothCorrect = { ...fixture, fixture_id: 13, home_goals: 2, away_goals: 1, prediction: { ...prediction, is_test: false } };
    const neitherCorrect = { ...fixture, fixture_id: 14, home_goals: 0, away_goals: 1 };
    const otherStatus = { ...bothCorrect, fixture_id: 15, status: 'AET' };
    renderPredictionFixtures({ fixtures: [bothCorrect, neitherCorrect, otherStatus] });
    cards = document.querySelectorAll('.prediction-fixture-card');
    if ([...cards[0].querySelectorAll('.prediction-correct-indicator')].map(item => item.textContent).join(',') !== '3W,DC') throw new Error('Missing correct indicators for non-test fixture');
    if (cards[0].querySelector('.prediction-test-badge')) throw new Error('Unexpected test badge');
    if (cards[1].querySelector('.prediction-correct-indicator') || cards[2].querySelector('.prediction-correct-indicator')) throw new Error('Incorrect or non-FT fixture has correct indicators');
    predictionFixtureFilter.value = 'upcoming';
    renderPredictionFixtures({ fixtures: [fixture, unplayed] });
    if (accuracy() !== '3W-(0/1), DC-(1/1)') throw new Error('Round accuracy changed with filter');
    const missingChance = { ...fixture, fixture_id: 12, prediction: {
      ...prediction, classification: { ...prediction.classification, double_chance: [{ code: 'invalid' }] },
    } };
    predictionFixtureFilter.value = 'all';
    renderPredictionFixtures({ fixtures: [fixture, missingChance] });
    if (accuracy() !== '3W-(0/2), DC-(1/1)') throw new Error('Wrong eligible prediction counts');
    renderPredictionFixtures({ fixtures: [unplayed] });
    if (accuracy() !== '3W-(0/0), DC-(0/0)') throw new Error('Wrong empty prediction counts');
    renderPredictionDetails({ fixture, prediction, stats: [], is_favorite: false });
    const detailCard = document.querySelector('.prediction-detail-card');
    if (detailCard.querySelector('.prediction-team-separator').textContent !== '1 - 1') throw new Error('Missing fixture score');
    if ([...detailCard.querySelectorAll('.prediction-team-name')].map(item => item.textContent).join(' vs ') !== 'Home vs Away') throw new Error('Missing fixture teams');
    if (detailCard.querySelectorAll('.prediction-team-logo-fallback').length !== 2) throw new Error('Missing team logo fallbacks');
    if (!detailCard.textContent.includes('Round 1') || !detailCard.textContent.includes('Status: FT')) throw new Error('Missing fixture metadata');
    if (detailCard.querySelector('.favorite-toggle').getAttribute('aria-pressed') !== 'false') throw new Error('Missing favorite control');
    if (!detailCard.querySelector('.prediction-test-badge')) throw new Error('Missing detail test badge');
    if (!document.querySelector('.prediction-evaluation').textContent.includes('3-way: Incorrect')) throw new Error('Wrong 3-way result');
    if (!document.querySelector('.prediction-evaluation').textContent.includes('Double chance: Correct')) throw new Error('Wrong double-chance result');
    renderPredictionDetails({ fixture: unplayed, prediction: unplayed.prediction, stats: [], is_favorite: false });
    if (document.querySelector('.prediction-detail-card .prediction-team-separator').textContent !== 'vs') throw new Error('Unexpected unplayed score');
    if (!document.querySelector('.prediction-detail-card').textContent.includes('Status: NS')) throw new Error('Missing unplayed status');
    if (!document.querySelector('.prediction-evaluation').textContent.includes('Not yet played')) throw new Error('Missing unplayed status');
    renderPredictionDetails({ fixture, prediction: null, stats: [], is_favorite: true });
    if (document.querySelector('.prediction-detail-card .favorite-toggle').getAttribute('aria-pressed') !== 'true') throw new Error('Favorite state lost without prediction');
    if (!document.querySelector('.prediction-details').textContent.includes('No trained-model prediction')) throw new Error('Missing unavailable prediction message');
    renderPredictionDetails({ fixture: { ...fixture, status: 'AET' }, prediction, stats: [], is_favorite: false });
    if (!document.querySelector('.prediction-evaluation').textContent.includes('unavailable')) throw new Error('Unexpected non-FT evaluation');
    document.querySelector('#favorite-list-name').value = 'Test list';
    document.querySelector('#favorite-list-form').requestSubmit();
    await waitFor('.favorite-list-group');
    document.querySelector('#model-lab-nav').click();
    (await waitFor('#model-league-tree [data-action="model-country"]')).click();
    (await waitFor('#model-league-tree [data-action="model-league"]')).click();
    if (document.querySelector('#model-league').value !== '1') throw new Error("Model Lab selection failed");
    if (window.dashboardErrors.length) throw new Error(window.dashboardErrors.join('; '));
    document.documentElement.dataset.dashboardTest = 'passed';
  } catch (error) {
    document.documentElement.dataset.dashboardTest = 'failed: ' + error.message;
  }
});
</script>
"""

            class InstrumentedHandler(server.DashboardHandler):
                def do_GET(self):
                    if self.path == "/":
                        template = (server.WEB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
                        template = template.replace("</head>", instrumentation + "</head>")
                        template = "\n".join(line for line in template.splitlines() if "fonts.googleapis.com" not in line and "fonts.gstatic.com" not in line)
                        content = template.encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                    else:
                        super().do_GET()

            http_server = server.ThreadingHTTPServer(("127.0.0.1", 0), InstrumentedHandler)
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            with patch.dict(os.environ, {
                "EAGLE_LEAGUES_DB": str(catalogue),
                "EAGLE_COUNTRY_DATABASES": str(country_dir),
                "EAGLE_DATASET_DATABASES": str(root / "datasets"),
            }):
                thread.start()
                try:
                    result = subprocess.run(
                        [browser, "--headless", "--disable-gpu", "--no-sandbox",
                         "--disable-background-networking", "--no-first-run",
                         f"--user-data-dir={root / 'browser'}",
                         "--virtual-time-budget=15000", "--dump-dom",
                         f"http://127.0.0.1:{http_server.server_port}/"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                    )
                finally:
                    http_server.shutdown()
                    thread.join()
                    http_server.server_close()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('data-dashboard-test="passed"', result.stdout, result.stdout[:2000])


if __name__ == "__main__":
    unittest.main()
