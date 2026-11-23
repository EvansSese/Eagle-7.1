import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from core.fixture_dates import fixture_dates_utc, fixture_time_utc
from processing.h2h.combine_fixtures import load_all_fixtures
from processing.process_seasons_data import process_seasons_data
from training.deep_learning.train_sequence_models import _build_sequences, _read_fixture_tables
from training.prepare_feature_dataset import prepare_datasets


class FixtureDatesTests(unittest.TestCase):
    def test_normalize_mixed_offsets_and_missing_values(self):
        dates = pd.Series([
            pd.Timestamp("2024-01-01 10:00"),
            pd.Timestamp("2024-01-01 13:00", tz="Africa/Nairobi"),
            pd.Timestamp("2024-01-01 05:00", tz="America/New_York"),
            pd.NaT,
        ])
        result = fixture_dates_utc(dates)
        self.assertEqual(result.iloc[:3].tolist(), [pd.Timestamp("2024-01-01 10:00")] * 3)
        self.assertTrue(pd.isna(result.iloc[3]))
        self.assertIsNone(result.dt.tz)
        with self.assertRaises(ValueError):
            fixture_dates_utc(pd.Series(["invalid"]))
        self.assertEqual(fixture_time_utc(pd.Timestamp("2024-01-01 13:00", tz="Africa/Nairobi")),
                         pd.Timestamp("2024-01-01 10:00"))
        self.assertEqual(fixture_time_utc(pd.Timestamp("2024-01-01 10:00")),
                         pd.Timestamp("2024-01-01 10:00"))

    def test_cross_season_history_preserves_same_kickoff_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "country.ddb"
            with duckdb.connect(str(path)) as con:
                con.execute('CREATE TABLE "1-2024-fixtures" AS SELECT 1 AS fixture_id, '
                            "TIMESTAMP '2024-01-01 10:00:00' AS date, "
                            "'A' AS home_team, 'B' AS away_team, 1 AS home_goals, "
                            "0 AS away_goals, 'FT' AS status")
                con.execute('CREATE TABLE "1-2025-fixtures" AS SELECT 2 AS fixture_id, '
                            "TIMESTAMPTZ '2024-01-01 13:00:00+03' AS date, "
                            "'A' AS home_team, 'B' AS away_team, 0 AS home_goals, "
                            "1 AS away_goals, 'FT' AS status UNION ALL SELECT 3, "
                            "TIMESTAMPTZ '2024-01-02 13:00:00+03', 'A', 'B', 1, 1, 'FT'")
            combined = load_all_fixtures(path, league_id=1)
            self.assertEqual(combined["fixture_id"].tolist(), [1, 2, 3])
            self.assertEqual(combined.loc[0, "date"], combined.loc[1, "date"])
            self.assertIsNone(combined["date"].dt.tz)
            raw = _read_fixture_tables(path, 1)
            self.assertEqual(raw["date"].tolist(), combined["date"].tolist())
            sequences, _, _, _, _, _ = _build_sequences(raw, 10, set())
            self.assertEqual(sequences["home_history"][0, :, -1].sum(), 0)
            self.assertEqual(sequences["home_history"][1, :, -1].sum(), 0)
            self.assertEqual(sequences["home_history"][2, :, -1].sum(), 2)

    def test_processing_and_preparation_with_mixed_season_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, country, dataset = root / "leagues.ddb", root / "country.ddb", root / "datasets.ddb"
            with duckdb.connect(str(catalog)) as con:
                con.execute("CREATE TABLE leagues (league_id INTEGER, country_name VARCHAR, "
                            "league_name VARCHAR, seasons INTEGER[])")
                con.execute("INSERT INTO leagues VALUES (1, 'Test', 'League', [2024, 2025])")
            with duckdb.connect(str(country)) as con:
                for season in (2024, 2025):
                    timestamp = ("TIMESTAMP '2024-01-01'" if season == 2024
                                 else "TIMESTAMPTZ '2025-01-01 03:00:00+03'")
                    start = 1 if season == 2024 else 11
                    con.execute(f'''CREATE TABLE "1-{season}-fixtures" AS
                        SELECT 1 AS league_id, i AS fixture_id,
                        {timestamp} + (i - {start}) * INTERVAL '1 day' AS date,
                        'Venue' AS venue, 'Round ' || CAST(i - {start} AS VARCHAR) AS round,
                        'A' AS home_team, 'B' AS away_team, 1 AS home_goals,
                        0 AS away_goals, 'FT' AS status
                        FROM range({start}, {start + 10}) t(i)''')
            progress = []
            process_seasons_data(1, [2024, 2025], catalog, country,
                                 progress_callback=lambda done, total, message: progress.append((done, total)))
            self.assertEqual(progress[-1], (23, 23))
            result = prepare_datasets(1, country, dataset)
            self.assertEqual(result["prediction_rows"], 10)
            with duckdb.connect(str(dataset)) as con:
                date = con.execute('SELECT date FROM "1-prediction-dataset" WHERE fixture_id = 11').fetchone()[0]
            self.assertEqual(date, pd.Timestamp("2025-01-01"))


if __name__ == "__main__":
    unittest.main()
