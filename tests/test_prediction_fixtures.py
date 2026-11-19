import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from app import server
from training import prepare_dataset
from training.classification import train_ensemble
from training.classification.preprocess_dataset import engineer_features as classifier_features
from training.regression.train_goal_models import engineer_features as regression_features


class FixturePredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = pd.DataFrame(
            {
                "fixture_id": [1, 2, 3, 4],
                "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2099-01-01", None]),
                "status": ["FT", "NS", "NS", "TBD"],
                "result": ["H", None, None, None],
                "home_goals": [2, None, None, None],
                "away_goals": [1, None, None, None],
                "home_team": ["Home"] * 4,
                "away_team": ["Away"] * 4,
                "round": ["Round 2", "Round 1", "Round 2", "Round 3"],
                "venue": ["Venue"] * 4,
            }
        )

    def test_preparation_preserves_full_selected_season(self):
        database = self.root / "datasets.ddb"
        with patch.object(prepare_dataset, "get_league_details", return_value=([2023, 2024], "Test")), \
             patch.object(prepare_dataset, "join_tables", side_effect=[self.rows.iloc[:1], self.rows]):
            result = prepare_dataset.prepare_league_dataset(1, 2024, dataset_db_path=database)
        with duckdb.connect(str(database), read_only=True) as connection:
            ids = connection.execute('SELECT fixture_id FROM "1-prediction-dataset" ORDER BY fixture_id').fetchall()
            metadata = connection.execute('SELECT prediction_season, full_season FROM "1-dataset-metadata"').fetchone()
        self.assertEqual(ids, [(1,), (2,), (3,), (4,)])
        self.assertEqual(metadata, (2024, True))
        self.assertEqual(result["prediction_rows"], 4)

    def test_all_season_preparation_includes_latest_and_preserves_season_tables(self):
        database = self.root / "all-seasons.ddb"
        earlier = self.rows.iloc[:1].copy()
        latest = self.rows.iloc[1:].copy()
        progress = []
        with patch.object(prepare_dataset, "get_league_details", return_value=([2023, 2024], "Test")), \
             patch.object(prepare_dataset, "join_tables", side_effect=[earlier, latest]):
            result = prepare_dataset.prepare_all_seasons_dataset(
                1, dataset_db_path=database,
                progress_callback=lambda done, total, message: progress.append((done, total)),
            )
        with duckdb.connect(str(database), read_only=True) as connection:
            ids = connection.execute('SELECT fixture_id FROM "1-training-dataset" ORDER BY fixture_id').fetchall()
            recent = connection.execute('SELECT fixture_id FROM "1-2024-prediction-dataset" ORDER BY fixture_id').fetchall()
            metadata = connection.execute('SELECT all_seasons FROM "1-dataset-metadata"').fetchone()
            self.assertTrue(server._has_all_season_dataset(connection, 1, [2023, 2024]))
            self.assertFalse(server._has_all_season_dataset(connection, 1, [2023, 2024, 2025]))
        self.assertEqual(ids, [(1,), (2,), (3,), (4,)])
        self.assertEqual(recent, [(2,), (3,), (4,)])
        self.assertEqual(metadata, (True,))
        self.assertEqual(result["training_seasons"], [2023, 2024])
        self.assertEqual(progress[-1], (7, 7))

    def test_all_season_preparation_rejects_duplicate_fixture_ids_without_replacing_tables(self):
        database = self.root / "all-seasons.ddb"
        with duckdb.connect(str(database)) as connection:
            connection.execute('CREATE TABLE "1-training-dataset" AS SELECT 999 AS fixture_id')
        with patch.object(prepare_dataset, "get_league_details", return_value=([2023, 2024], "Test")), \
             patch.object(prepare_dataset, "join_tables", side_effect=[self.rows.iloc[:1], self.rows.iloc[:1]]):
            with self.assertRaisesRegex(ValueError, "unique"):
                prepare_dataset.prepare_all_seasons_dataset(1, dataset_db_path=database)
        with duckdb.connect(str(database), read_only=True) as connection:
            self.assertEqual(connection.execute('SELECT fixture_id FROM "1-training-dataset"').fetchall(), [(999,)])

    def test_sequence_prediction_histories_do_not_include_target_result(self):
        from training.deep_learning.train_sequence_models import _build_sequences

        fixtures = pd.DataFrame([
            {"fixture_id": 1, "date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B", "status": "FT", "home_goals": 2, "away_goals": 0},
            {"fixture_id": 2, "date": pd.Timestamp("2020-01-02"), "home_team": "A", "away_team": "B", "status": "FT", "home_goals": 0, "away_goals": 3},
            {"fixture_id": 3, "date": pd.Timestamp("2020-01-02"), "home_team": "A", "away_team": "B", "status": "FT", "home_goals": 1, "away_goals": 1},
        ])
        _, _, _, _, targets, sequences = _build_sequences(fixtures, 3, {2, 3})
        self.assertEqual(targets["fixture_id"].tolist(), [2, 3])
        np.testing.assert_array_equal(sequences["home_history"][0], sequences["home_history"][1])
        self.assertEqual(sequences["home_history"][0, :, -1].sum(), 1)

    def test_all_model_inference_includes_played_and_undated_fixtures(self):
        from training import ensemble_predict

        classifier_dir = self.root / "classifier"
        regressor_dir = self.root / "regressor"
        mlp_dir = self.root / "mlp"
        sequence_dir = self.root / "sequence"
        for directory in (classifier_dir, regressor_dir, mlp_dir, sequence_dir):
            directory.mkdir()
        (classifier_dir / "ensemble.json").write_text(json.dumps({
            "models": {"test": {"model_file": "test.joblib", "features": ["feature"]}},
        }), encoding="utf-8")
        (sequence_dir / "metadata.json").write_text(json.dumps({"history_length": 2}), encoding="utf-8")
        for name in ("1_cnn.keras", "1_rnn.keras"):
            (sequence_dir / name).touch()
        fixtures = pd.DataFrame([
            {"fixture_id": 1, "date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B", "status": "FT", "home_goals": 2, "away_goals": 0},
            {"fixture_id": 2, "date": pd.Timestamp("2020-01-02"), "home_team": "A", "away_team": "B", "status": "FT", "home_goals": 0, "away_goals": 1},
            {"fixture_id": 3, "date": pd.NaT, "home_team": "A", "away_team": "B", "status": "TBD", "home_goals": None, "away_goals": None},
        ])
        observed = []

        class FakeModel:
            classes_ = ["H", "D", "A"]

            def predict_proba(self, features):
                return np.tile([0.5, 0.3, 0.2], (len(features), 1))

            def predict(self, features):
                return np.tile([1.5, 1.0], (len(features), 1))

            def __call__(self, inputs, training=False):
                observed.append(inputs["home_history"].copy())
                count = len(inputs["home_history"])
                return {"outcome": np.tile([0.5, 0.3, 0.2], (count, 1)), "scores": np.tile([1.5, 1.0], (count, 1))}

        tensorflow = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=lambda *a, **k: FakeModel())))
        features = lambda frame: pd.DataFrame({"feature": [1] * len(frame)})
        with patch.object(ensemble_predict, "classification_features", side_effect=lambda frame: (features(frame), None)), \
             patch.object(ensemble_predict, "regression_features", side_effect=features), \
             patch.object(ensemble_predict, "_regressor_specs", return_value=[{"name": "regressor_test", "filename": "test", "features": ["feature"]}]), \
             patch.object(ensemble_predict, "_read_fixture_tables", return_value=fixtures.iloc[:2]), \
             patch.object(ensemble_predict.joblib, "load", return_value=FakeModel()), \
             patch.object(ensemble_predict, "_load_tensorflow", return_value=tensorflow), \
             patch.dict(sys.modules, {"training.deep_learning.masked_layers": SimpleNamespace(MaskAwareGRU=object, MaskAwareLSTM=object)}):
            predictions = ensemble_predict._upcoming_predictions(
                1, self.root / "country.ddb", self.root / "dataset.ddb", fixtures,
                classifier_dir, regressor_dir, mlp_dir, sequence_dir, include_all=True,
            )
        self.assertEqual(set(predictions), {"classifier_test", "regressor_test", "neural_mlp", "neural_cnn", "neural_rnn"})
        self.assertEqual(observed[0][0, :, -1].sum(), 0)
        self.assertEqual(observed[0][1, :, -1].sum(), 1)
        self.assertEqual(observed[0][2, :, -1].sum(), 0)

    def test_outcome_networks_join_existing_ensemble_with_score_predictions(self):
        from training import ensemble_predict

        directories = [self.root / name for name in ("classifier", "regressor", "mlp", "sequence")]
        for directory in directories:
            directory.mkdir()
        classifier, regressor, mlp, sequence = directories
        (classifier / "ensemble.json").write_text(json.dumps({
            "models": {"test": {"model_file": "test.joblib", "features": ["feature"]}},
        }), encoding="utf-8")
        (sequence / "metadata.json").write_text('{"history_length": 2}', encoding="utf-8")
        (sequence / "outcome_metadata.json").write_text(json.dumps({
            "history_length": 3, "models": {
                "lstm": {"file": "1_lstm.keras"},
                "autoencoder": {"file": "1_autoencoder.keras"},
            },
        }), encoding="utf-8")
        for name in ("cnn", "rnn", "lstm", "autoencoder"):
            (sequence / f"1_{name}.keras").touch()
        fixtures = pd.DataFrame([{
            "fixture_id": 1, "date": pd.Timestamp("2099-01-01"), "home_team": "A",
            "away_team": "B", "status": "NS", "home_goals": None, "away_goals": None,
        }])

        class FakeModel:
            classes_ = ["H", "D", "A"]

            def __init__(self, outcome_only=False):
                self.outcome_only = outcome_only

            def predict_proba(self, features):
                return np.tile([0.5, 0.3, 0.2], (len(features), 1))

            def predict(self, features):
                return np.tile([1.5, 1.0], (len(features), 1))

            def __call__(self, inputs, training=False):
                probabilities = np.tile([0.5, 0.3, 0.2], (len(inputs["home_history"]), 1))
                if self.outcome_only:
                    return {"outcome": probabilities, "scores": np.tile([2.0, 0.5], (len(probabilities), 1))}
                return {"outcome": probabilities, "scores": np.tile([1.5, 1.0], (len(probabilities), 1))}

        tensorflow = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(
            load_model=lambda path, **kwargs: FakeModel("lstm" in str(path) or "autoencoder" in str(path))
        )))
        features = lambda frame: pd.DataFrame({"feature": [1] * len(frame)})
        with patch.object(ensemble_predict, "classification_features", side_effect=lambda frame: (features(frame), None)), \
             patch.object(ensemble_predict, "regression_features", side_effect=features), \
             patch.object(ensemble_predict, "_regressor_specs", return_value=[{"name": "regressor_test", "filename": "test", "features": ["feature"]}]), \
             patch.object(ensemble_predict, "_read_fixture_tables", return_value=fixtures), \
             patch.object(ensemble_predict.joblib, "load", return_value=FakeModel()), \
             patch.object(ensemble_predict, "_load_tensorflow", return_value=tensorflow), \
             patch.dict(sys.modules, {"training.deep_learning.masked_layers": SimpleNamespace(MaskAwareGRU=object, MaskAwareLSTM=object)}):
            predictions = ensemble_predict._upcoming_predictions(
                1, self.root / "country.ddb", self.root / "dataset.ddb", fixtures,
                classifier, regressor, mlp, sequence, include_all=True,
            )
        for name in ("neural_lstm", "neural_autoencoder"):
            self.assertIn(name, predictions)
            self.assertEqual(predictions[name]["predicted_home_goals"].tolist(), [2.0])
            np.testing.assert_allclose(predictions[name][["probability_home", "probability_draw", "probability_away"]], [[0.5, 0.3, 0.2]])
        combined = ensemble_predict._ensemble_frame(fixtures, predictions)
        self.assertEqual(combined["fixture_id"].tolist(), [1])
        self.assertEqual(combined["predicted_correct_score"].tolist(), ["2-1"])

    def test_classifier_all_fixtures_and_existing_upcoming_mode(self):
        database = self.root / "datasets.ddb"
        model_dir = self.root / "models"
        model_dir.mkdir()
        (model_dir / "ensemble.json").write_text(json.dumps({
            "league_id": 1,
            "models": {"test": {"model_file": "model.joblib", "features": ["year"], "validation_accuracy": 1}},
        }), encoding="utf-8")
        (model_dir / "model.joblib").touch()
        with duckdb.connect(str(database)) as connection:
            connection.register("fixture_rows", self.rows)
            connection.execute('CREATE TABLE "fixtures" AS SELECT * FROM fixture_rows')
        with patch.object(train_ensemble.joblib, "load", return_value=object()), \
             patch.object(train_ensemble, "_ordered_probabilities", side_effect=lambda model, features: np.tile([0.2, 0.3, 0.5], (len(features), 1))):
            all_rows = train_ensemble.predict_upcoming_fixtures(
                database, "fixtures", 1, model_dir, persist=False, include_all=True
            )
            future = train_ensemble.predict_upcoming_fixtures(
                database, "fixtures", 1, model_dir, now=pd.Timestamp("2025-01-01"), persist=False
            )
            past_unstarted = train_ensemble.predict_upcoming_fixtures(
                database, "fixtures", 1, model_dir, persist=False, include_past=True
            )
        self.assertEqual(all_rows["fixture_id"].tolist(), [1, 2, 3, 4])
        self.assertEqual(future["fixture_id"].tolist(), [3])
        self.assertEqual(past_unstarted["fixture_id"].tolist(), [2, 3])

    def test_played_scores_do_not_enter_model_features(self):
        classifier, _ = classifier_features(self.rows)
        regression = regression_features(self.rows)
        for features in (classifier, regression):
            self.assertTrue({"home_goals", "away_goals", "result", "status"}.isdisjoint(features.columns))

    def test_screen_returns_all_statuses_in_chronological_order(self):
        database = self.root / "country.ddb"
        with duckdb.connect(str(database)) as connection:
            connection.register("fixture_rows", self.rows)
            connection.execute('CREATE TABLE "1-2024-fixtures" AS SELECT * FROM fixture_rows')
        predictions = {i: {"confidence": i / 10, "is_test": i == 1} for i in range(1, 5)}
        league = {"country_name": "Test", "league_name": "League", "seasons": [2024]}
        with patch.object(server, "_read_league", return_value=league), \
             patch.object(server, "_predict_fixture_rows", return_value=predictions), \
             patch.object(server, "get_country_database_path", return_value=database), \
             patch.object(server, "FIXTURE_FILES", self.root):
            result = server.get_upcoming_predictions(1, 2024, self.root / "catalog.ddb")
        self.assertEqual([row["fixture_id"] for row in result["fixtures"]], [2, 1, 3, 4])
        self.assertEqual(result["fixtures"][1]["home_goals"], 2)
        self.assertTrue(result["fixtures"][1]["prediction"]["is_test"])

    def test_prediction_detail_includes_test_membership(self):
        prediction = {"is_test": True}
        with patch.object(server, "get_fixture_details", return_value={"fixture": {}, "stats": []}), \
             patch.object(server, "_predict_fixture_rows", return_value={1: prediction}), \
             patch.object(server, "favorite_list_ids_for_fixture", return_value=[]):
            details = server.get_prediction_fixture_details(1, 2024, 1, self.root / "catalog.ddb")
        self.assertTrue(details["prediction"]["is_test"])

    def test_screen_inference_predicts_the_whole_season(self):
        country_database = self.root / "country.ddb"
        country_database.touch()
        dataset_database = self.root / "datasets.ddb"
        model_dir = self.root / "storage" / "trained_classifier_models" / "v3" / "1"
        model_dir.mkdir(parents=True)
        (model_dir / "ensemble.json").touch()
        regressor_dir = self.root / "storage" / "trained_regressor_models" / "v3" / "1"
        regressor_dir.mkdir(parents=True)
        (regressor_dir / "goals.joblib").touch()
        with duckdb.connect(str(dataset_database)) as connection:
            connection.register("fixture_rows", self.rows)
            connection.execute('CREATE TABLE "1-2024-prediction-dataset" AS SELECT * FROM fixture_rows')
            connection.execute('CREATE TABLE "1-training-dataset" AS SELECT * FROM fixture_rows')
            connection.execute('CREATE TABLE "1-dataset-metadata" AS SELECT 2024 AS prediction_season, TRUE AS full_season, TRUE AS all_seasons, [2024]::INTEGER[] AS training_seasons')
            connection.execute('CREATE TABLE "1-ensemble-classifier-test-predictions" AS SELECT 1 AS fixture_id')
            connection.execute('CREATE TABLE "1-goal-regression-test" AS SELECT 2 AS fixture_id')
            connection.execute('CREATE TABLE "1-deep-learning-test-predictions" AS SELECT 3 AS fixture_id')
            connection.execute('CREATE TABLE "1-sequence-model-test-predictions" AS SELECT 3 AS fixture_id')
            connection.execute(
                """CREATE TABLE "1-goal-regression-feature-group-models" AS
                   SELECT 'basic' AS feature_bundle, 'fake' AS model,
                   'goals.joblib' AS model_filename, '["year"]' AS features_json,
                   0.8 AS validation_outcome_accuracy"""
            )
        prediction_rows = pd.DataFrame({
            "fixture_id": [1, 2, 3, 4],
            "probability_home": [0.5] * 4,
            "probability_draw": [0.3] * 4,
            "probability_away": [0.2] * 4,
            "predicted_home_goals": [1.5] * 4,
            "predicted_away_goals": [1.0] * 4,
        })

        league = {"country_name": "Test", "league_name": "League", "seasons": [2024]}
        with patch.object(server, "PROJECT_ROOT", self.root), \
             patch.dict(server.PREDICTION_CACHE, {}, clear=True), \
             patch.object(server, "_read_league", return_value=league), \
             patch.object(server, "get_country_database_path", return_value=country_database), \
             patch.object(server, "get_dataset_database_path", return_value=dataset_database), \
             patch("training.ensemble_predict._upcoming_predictions", return_value={
                 "classifier_test": prediction_rows, "regressor_test": prediction_rows,
                 "neural_test": prediction_rows,
             }) as ensemble:
            predictions = server._predict_fixture_rows(1, 2024, self.root / "catalog.ddb")
        self.assertEqual(set(predictions), {1, 2, 3, 4})
        self.assertTrue(ensemble.call_args.kwargs["include_all"])
        self.assertEqual(predictions[1]["classification"]["predicted_outcome"], "H")
        self.assertTrue(predictions[1]["is_test"])
        self.assertTrue(predictions[2]["is_test"])
        self.assertTrue(predictions[3]["is_test"])
        self.assertFalse(predictions[4]["is_test"])

        with duckdb.connect(str(dataset_database)) as connection:
            connection.execute('DROP TABLE "1-deep-learning-test-predictions"')
        with patch.object(server, "PROJECT_ROOT", self.root), \
             patch.dict(server.PREDICTION_CACHE, {}, clear=True), \
             patch.object(server, "_read_league", return_value=league), \
             patch.object(server, "get_country_database_path", return_value=country_database), \
             patch.object(server, "get_dataset_database_path", return_value=dataset_database), \
             patch("training.ensemble_predict._upcoming_predictions", return_value={
                 "classifier_test": prediction_rows, "regressor_test": prediction_rows,
                 "neural_test": prediction_rows,
             }):
            predictions = server._predict_fixture_rows(1, 2024, self.root / "catalog.ddb")
        self.assertTrue(predictions[1]["is_test"])
        self.assertIsNone(predictions[4]["is_test"])


if __name__ == "__main__":
    unittest.main()
