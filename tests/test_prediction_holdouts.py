"""Verify that prediction test badges use recorded participating-model holdouts."""

import unittest

import duckdb

from app.features.predictions.holdouts import read_test_fixture_ids


class PredictionHoldoutTests(unittest.TestCase):
    def test_union_unknown_membership_and_optional_models(self):
        with duckdb.connect(":memory:") as connection:
            for name, ids in (
                ("ensemble-classifier-test-predictions", [1, 2]),
                ("goal-regression-test", [2, 3]),
                ("deep-learning-test-predictions", [4]),
                ("sequence-model-test-predictions", [4, 5, 5]),
                ("outcome-network-test-predictions", [6]),
            ):
                connection.execute(f'CREATE TABLE "1-{name}" (fixture_id INTEGER)')
                connection.executemany(f'INSERT INTO "1-{name}" VALUES (?)', [(i,) for i in ids])

            ids, missing = read_test_fixture_ids(connection, 1, False)
            self.assertEqual(ids, {1, 2, 3, 4, 5})
            self.assertEqual(missing, [])
            ids, missing = read_test_fixture_ids(connection, 1, True)
            self.assertEqual(ids, {1, 2, 3, 4, 5, 6})
            self.assertEqual(missing, [])
            connection.execute('DROP TABLE "1-outcome-network-test-predictions"')
            ids, missing = read_test_fixture_ids(connection, 1, True)
            self.assertEqual(ids, {1, 2, 3, 4, 5})
            self.assertEqual(missing, ["1-outcome-network-test-predictions"])

    def test_malformed_test_table_is_not_silently_ignored(self):
        with duckdb.connect(":memory:") as connection:
            connection.execute('CREATE TABLE "1-ensemble-classifier-test-predictions" (other INTEGER)')
            with self.assertRaises(duckdb.Error):
                read_test_fixture_ids(connection, 1, False)
