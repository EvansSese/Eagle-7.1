import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from app import server
from training.prepare_feature_dataset import FEATURE_SUFFIXES, join_season, prepare_datasets


class PrepareFeatureDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "country.ddb"
        self.target = Path(self.temp.name) / "datasets.ddb"
        with duckdb.connect(str(self.source)) as con:
            for season, start, count in ((2023, 1, 10), (2024, 11, 10), (2025, 21, 3)):
                table = f"1-{season}-fixtures"
                con.execute(f'''CREATE TABLE "{table}" AS
                    SELECT 1 AS league_id, i AS fixture_id,
                    TIMESTAMP '2023-01-01' + i * INTERVAL '1 day' AS date,
                    'stadium' AS venue, 'round' AS round, 'A' AS home_team,
                    'B' AS away_team, 2 AS home_goals, 1 AS away_goals, 'FT' AS status
                    FROM range({start}, {start + count}) t(i)''')
                for suffix in FEATURE_SUFFIXES:
                    con.execute(f'''CREATE TABLE "1-{season}-{suffix}" AS
                        SELECT fixture_id, home_team, home_goals, fixture_id * 2 AS metric,
                        NULL::DOUBLE AS missing_value FROM "{table}"''')

    def test_all_features_joined_and_split_chronologically(self):
        with self.assertLogs("training.prepare_feature_dataset", level="WARNING"):
            result = prepare_datasets(1, self.source, self.target)
        self.assertEqual((result["training_rows"], result["validation_rows"], result["test_rows"]), (14, 3, 3))
        self.assertEqual(result["prediction_season"], 2025)
        with duckdb.connect(str(self.target)) as con:
            for name, first, last in (
                ("training", 1, 14), ("validation", 15, 17),
                ("test", 18, 20), ("prediction", 21, 23),
            ):
                table = f"1-{name}-dataset"
                data = con.execute(f'SELECT fixture_id, goals__metric, points__metric, h2h__metric FROM "{table}" ORDER BY date, fixture_id').fetchall()
                self.assertEqual((data[0][0], data[-1][0]), (first, last))
                self.assertEqual(data[0][1:], (first * 2,) * 3)
                columns = {row[0] for row in con.execute(f'DESCRIBE "{table}"').fetchall()}
                self.assertTrue(all(f"{suffix.replace('-', '_')}__metric" in columns for suffix in FEATURE_SUFFIXES))
                self.assertNotIn("goals__home_goals", columns)
                self.assertIn("home_goals", columns)
                self.assertEqual(con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE goals__missing_value IS NULL').fetchone()[0], len(data))
            self.assertTrue(server._has_chronological_dataset(con, 1))
            self.assertFalse(server._has_all_season_dataset(con, 1, [2023, 2024, 2025]))

    def test_excludes_incomplete_history_and_includes_full_latest_season(self):
        with duckdb.connect(str(self.source)) as con:
            con.execute('UPDATE "1-2023-fixtures" SET status = ? WHERE fixture_id = 1', ["NS"])
            con.execute('UPDATE "1-2023-fixtures" SET home_goals = NULL WHERE fixture_id = 2')
            con.execute('UPDATE "1-2025-fixtures" SET status = ? WHERE fixture_id = 23', ["NS"])
        with self.assertLogs("training.prepare_feature_dataset", level="WARNING"):
            result = prepare_datasets(1, self.source, self.target)
        self.assertEqual(result["excluded_historical_rows"], 2)
        self.assertEqual((result["training_rows"], result["validation_rows"], result["test_rows"]), (12, 3, 3))
        with duckdb.connect(str(self.target)) as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM "1-prediction-dataset"').fetchone()[0], 3)
            self.assertEqual(con.execute('SELECT MIN(fixture_id) FROM "1-training-dataset"').fetchone()[0], 3)

    def test_duplicate_or_missing_feature_fails_before_replacing_output(self):
        with duckdb.connect(str(self.target)) as con:
            con.execute('CREATE TABLE "1-training-dataset" AS SELECT 999 AS fixture_id')
        with duckdb.connect(str(self.source)) as con:
            con.execute('INSERT INTO "1-2023-points" SELECT * FROM "1-2023-points" LIMIT 1')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_datasets(1, self.source, self.target)
        with duckdb.connect(str(self.target)) as con:
            self.assertEqual(con.execute('SELECT fixture_id FROM "1-training-dataset"').fetchone()[0], 999)
        with duckdb.connect(str(self.source)) as con:
            con.execute('ALTER TABLE "1-2023-points" RENAME TO "1-2023-points-old"')
            with self.assertRaisesRegex(ValueError, "Missing feature table"):
                join_season(con, 1, 2023)

    def test_same_kickoff_split_is_by_fixture_id(self):
        with duckdb.connect(str(self.source)) as con:
            con.execute('UPDATE "1-2024-fixtures" SET date = TIMESTAMP \'2023-01-01\'')
        with self.assertLogs("training.prepare_feature_dataset", level="WARNING"):
            prepare_datasets(1, self.source, self.target)
        with duckdb.connect(str(self.target)) as con:
            training = {row[0] for row in con.execute('SELECT fixture_id FROM "1-training-dataset"').fetchall()}
            self.assertTrue(set(range(11, 21)).issubset(training))
            self.assertTrue(set(range(1, 5)).issubset(training))

    def test_preparation_normalizes_mixed_season_timezones(self):
        with duckdb.connect(str(self.source)) as con:
            for season in (2024, 2025):
                con.execute(f'ALTER TABLE "1-{season}-fixtures" ALTER COLUMN date '
                            "TYPE TIMESTAMPTZ USING timezone('UTC', date)")
        with self.assertLogs("training.prepare_feature_dataset", level="WARNING"):
            result = prepare_datasets(1, self.source, self.target)
        self.assertEqual((result["training_rows"], result["validation_rows"], result["test_rows"]), (14, 3, 3))
        with duckdb.connect(str(self.target)) as con:
            self.assertEqual(con.execute('SELECT date FROM "1-test-dataset" WHERE fixture_id = 20').fetchone()[0],
                             pd.Timestamp("2023-01-21"))
            self.assertEqual(con.execute('SELECT date FROM "1-prediction-dataset" WHERE fixture_id = 21').fetchone()[0],
                             pd.Timestamp("2023-01-22"))

    def test_left_join_retains_unmatched_fixture(self):
        with duckdb.connect(str(self.source)) as con:
            con.execute('UPDATE "1-2023-points" SET fixture_id = 999 WHERE fixture_id = 1')
            with self.assertLogs("training.prepare_feature_dataset", level="WARNING") as logs:
                joined = join_season(con, 1, 2023).set_index("fixture_id")
            self.assertEqual(len(joined), 10)
            self.assertTrue(pd.isna(joined.loc[1, "points__metric"]))
            self.assertTrue(any("missing 1 fixture rows" in line for line in logs.output))
            self.assertTrue(any("rows outside" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
