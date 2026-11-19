"""Tests for the league catalogue model and dashboard endpoint."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import duckdb

from app import server
from app.features.leagues import model


class LeagueCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "leagues.ddb"
        with duckdb.connect(str(self.database)) as connection:
            connection.execute(
                """CREATE TABLE leagues (
                    league_id INTEGER, league_name VARCHAR, country_name VARCHAR,
                    country_flag VARCHAR, league_logo VARCHAR, seasons INTEGER[]
                )"""
            )
            connection.execute(
                """INSERT INTO leagues VALUES
                    (7, 'Second', 'England', 'flag', 'logo', [2023, 2024]),
                    (3, 'First', 'England', NULL, NULL, []),
                    (1, NULL, NULL, NULL, NULL, NULL)"""
            )

    def test_catalogue_order_and_fallbacks(self):
        leagues = model.get_leagues(self.database)
        self.assertEqual([league["league_id"] for league in leagues], [1, 3, 7])
        self.assertEqual(leagues[0]["league_name"], "Unknown league")
        self.assertEqual(leagues[0]["country_name"], "Unknown country")
        self.assertFalse(leagues[0]["seasons_available"])
        self.assertEqual(leagues[-1]["season_count"], 2)
        self.assertEqual(server.get_leagues(self.database), leagues)

    def test_http_endpoint_returns_catalogue_and_database_errors(self):
        http_server = server.ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
        thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(http_server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(http_server.shutdown)
        url = f"http://127.0.0.1:{http_server.server_port}/api/leagues"

        with patch.object(server, "get_database_path", return_value=self.database):
            with urlopen(url) as response:
                body = json.load(response)
                self.assertEqual(response.status, 200)
                self.assertEqual(body["leagues"], model.get_leagues(self.database))
            with patch.object(server, "get_leagues", return_value=[]) as loader:
                with urlopen(url) as response:
                    self.assertEqual(json.load(response), {"leagues": []})
                loader.assert_called_once_with(self.database)

        empty_database = Path(self.directory.name) / "empty.ddb"
        with duckdb.connect(str(empty_database)):
            pass
        with patch.object(server, "get_database_path", return_value=empty_database):
            with self.assertRaises(HTTPError) as raised:
                urlopen(url)
            self.assertEqual(raised.exception.code, 500)
            self.assertEqual(
                json.load(raised.exception),
                {"error": "Unable to load leagues from the DuckDB database."},
            )
            raised.exception.close()


if __name__ == "__main__":
    unittest.main()
