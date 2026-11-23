import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import duckdb
import joblib
import numpy as np
import pandas as pd

from training import market_models as market


def fixture_data():
    rows = []
    scores = [(1, 0), (0, 0), (0, 2), (2, 1)]
    for i in range(36):
        home, away = scores[i % len(scores)]
        if i == 29:
            home, away = 5, 5
        rows.append({
            "league_id": 39, "fixture_id": i + 1,
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "round": f"Round {i}", "home_team": "A" if i % 2 else "B",
            "away_team": "B" if i % 2 else "A", "home_goals": home, "away_goals": away,
            "status": "FT", "season": 2020, "result": "H",
            "goals__x_team_goals_scored_per_match": np.nan if i == 0 else float(i % 4),
            "goals__y_team_goals_scored_per_match": float((i + 1) % 4),
            "goals__x_team_goals_scored": i,
            "outcome_probability__x_team_streak_outcome_before": None if i == 0 else "win",
        })
    return pd.DataFrame(rows)


def create_databases(root):
    frame = fixture_data()
    country = root / "country.ddb"
    dataset = root / "dataset.ddb"
    with duckdb.connect(str(country)) as con:
        con.register("rows", frame)
        con.execute('CREATE TABLE "39-2020-fixtures" AS SELECT * FROM rows')
    with duckdb.connect(str(dataset)) as con:
        con.execute('CREATE TABLE "39-dataset-metadata" AS SELECT \'chronological-splits-v1\' AS dataset_version')
        for name, part in zip(("training", "validation", "test"), (frame.iloc[:24], frame.iloc[24:30], frame.iloc[30:])):
            con.register("rows", part)
            con.execute(f'CREATE TABLE "39-{name}-dataset" AS SELECT * FROM rows')
            con.unregister("rows")
        con.register("rows", frame.iloc[30:])
        con.execute('CREATE TABLE "39-prediction-dataset" AS SELECT * FROM rows')
    return country, dataset


class MarketModelTests(unittest.TestCase):
    def test_team_difference_features_use_matching_scope_and_missing_values(self):
        frame = pd.DataFrame({
            "goals__x_team_home_goals_scored_per_match": [3.0, np.nan],
            "goals__y_team_home_goals_scored_per_match": [1.0, 2.0],
            "goals__y_team_away_goals_scored_per_match": [0.5, 1.0],
            "goals__x_team_id": [1, 2],
            "goals__y_team_id": [2, 1],
            "h2h__h2h_all_time_last_3_x_team_points": [5, 2],
            "h2h__h2h_all_time_last_3_y_team_points": [2, 4],
            "goals__x_team_away_goals": [1, 2],
            "outcome__x_team_streak_outcome": ["win", "loss"],
            "outcome__y_team_streak_outcome": ["loss", "win"],
        })
        result = market.add_team_differences(frame)
        name = "goals__team_diff_home_goals_scored_per_match"
        self.assertEqual(result.loc[0, name], 2.0)
        self.assertTrue(pd.isna(result.loc[1, name]))
        self.assertEqual(result.loc[0, "goals__home_away_diff_goals_scored_per_match"], 2.5)
        self.assertTrue(pd.isna(result.loc[1, "goals__home_away_diff_goals_scored_per_match"]))
        self.assertNotIn("goals__team_diff_id", result)
        self.assertEqual(result["h2h__h2h_all_time_last_3_team_diff_points"].tolist(), [3, -2])
        self.assertNotIn("goals__team_diff_away_goals", result)
        self.assertNotIn("outcome__team_diff_streak_outcome", result)

    def test_feature_selection_excludes_match_targets(self):
        frame = fixture_data()
        frame["goals__fixture_id"] = 1
        frame["goals__home_goals"] = 1
        frame["goals__team_id"] = 1
        bundles = market.feature_bundles(frame)
        self.assertIn("goals__x_team_goals_scored_per_match", bundles["ratios"])
        self.assertNotIn("goals__x_team_goals_scored", bundles["ratios"])
        self.assertIn("goals__x_team_goals_scored", bundles["aggregates"])
        for forbidden in ("home_team", "away_team", "fixture_id", "home_goals", "away_goals", "result", "venue"):
            self.assertNotIn(forbidden, bundles["aggregates"])
        self.assertNotIn("goals__fixture_id", bundles["aggregates"])
        self.assertNotIn("goals__home_goals", bundles["aggregates"])
        self.assertNotIn("goals__team_id", bundles["aggregates"])

    def test_unseen_class_has_finite_penalty(self):
        metrics = market._metrics(["known", "new"], np.array([[0.7, 0.3], [0.7, 0.3]]),
                                  True, np.array(["known", "other"]))
        self.assertEqual(metrics["unseen_target_count"], 1)
        self.assertGreater(metrics["log_loss"], 10)

    def test_sequence_history_excludes_same_kickoff_and_target(self):
        frame = fixture_data().iloc[:4].copy()
        frame.loc[1, "date"] = frame.loc[0, "date"]
        frame = frame.sort_values(["date", "fixture_id"])
        sequences, _, _, rows, _, _ = market._build_sequences(frame, 10, set())
        self.assertEqual(sequences["home_history"][0, :, -1].sum(), 0)
        self.assertEqual(sequences["home_history"][1, :, -1].sum(), 0)
        self.assertEqual(sequences["home_history"][2, :, -1].sum(), 2)
        self.assertEqual(rows["fixture_id"].tolist(), [1, 2, 3, 4])

    def test_network_architectures_train_and_reload(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as directory:
            for architecture in ("mlp", "dnn", "lstm", "gru", "tcn", "transformer"):
                with self.subTest(architecture=architecture):
                    sequence = architecture not in ("mlp", "dnn")
                    x = {"home_history": np.ones((4, 10, 6), dtype=np.float32),
                         "away_history": np.ones((4, 10, 6), dtype=np.float32)} if sequence else np.ones((4, 5), dtype=np.float32)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", message=".*was passed an input with a mask attached.*", category=UserWarning)
                        model = market.make_network(architecture, 6 if sequence else 5, 3, 8)
                        market._network_fit(model, x, np.array([0, 1, 2, 0]), 1)
                    before = market._network_predict(model, x)
                    path = Path(directory) / f"{architecture}.keras"
                    model.save(path)
                    restored = tf.keras.models.load_model(path)
                    np.testing.assert_allclose(before, market._network_predict(restored, x), rtol=1e-5, atol=1e-5)
                    tf.keras.backend.clear_session()

    def test_recurrent_models_ignore_empty_padding(self):
        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = market.make_network(architecture, 6, 3, 8)
                early = np.zeros((1, 10, 6), dtype=np.float32)
                late = early.copy()
                early[0, 0] = [1, 0, 1, 1, 1, 1]
                late[0, -1] = early[0, 0]
                early_result = market._network_predict(model, {"home_history": early, "away_history": early})
                late_result = market._network_predict(model, {"home_history": late, "away_history": late})
                np.testing.assert_allclose(early_result, late_result, atol=1e-6)

    def test_svm_calibration_preserves_singleton_classes(self):
        x = np.arange(24, dtype=float).reshape(12, 2)
        y = np.array([0] * 5 + [1] * 5 + [2] + [3])
        splitter = market.RareClassCalibrationSplit()
        for train, calibration in splitter.split(x, y):
            self.assertEqual(set(y[train]), {0, 1, 2, 3})
            self.assertFalse(set(y[calibration]) & {2, 3})
            self.assertTrue(set(train).isdisjoint(calibration))
        estimator = next(e for name, e, _ in market.traditional_grid(True) if name == "svm")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            estimator.fit(x, y)
        probabilities = estimator.predict_proba(x)
        self.assertEqual(probabilities.shape, (12, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_svm_calibration_rejects_all_singleton_classes(self):
        with self.assertRaisesRegex(ValueError, "at least two held-out folds"):
            list(market.RareClassCalibrationSplit().split(np.zeros((3, 1)), [0, 1, 2]))

    def test_nonconverged_candidate_is_recorded_and_not_selected(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country, dataset = create_databases(root)
            grid = [
                ("logistic_regression", LogisticRegression(max_iter=1), True),
                ("random_forest", RandomForestClassifier(n_estimators=3, random_state=42), False),
            ]
            messages = []
            with patch.object(market, "traditional_grid", return_value=grid), \
                 self.assertLogs(market.LOGGER, level="WARNING") as logs:
                result = market.train_market_models(
                    39, dataset, country, root / "models", epochs=1,
                    algorithms={"logistic_regression", "random_forest"}, target_names=("result",),
                    progress_callback=lambda done, total, message: messages.append(message),
                )
            self.assertEqual(result["models"][0]["algorithm"], "random_forest")
            self.assertTrue(any("Skipping non-converged logistic_regression" in line for line in logs.output))
            self.assertTrue(any("Skipped non-converged logistic_regression" in message for message in messages))
            with duckdb.connect(str(dataset), read_only=True) as con:
                rows = con.execute(
                    'SELECT validation_metrics_json FROM "39-market-model-candidates" '
                    "WHERE algorithm = 'logistic_regression'"
                ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(json.loads(row[0])["status"] == "not_converged" for row in rows))

    def test_traditional_candidates_for_both_task_types(self):
        from sklearn.base import clone

        frame = fixture_data().iloc[:24]
        features = market.feature_bundles(frame)["ratios"]
        x = market.feature_frame(frame, features)
        targets = market.build_market_targets(frame)
        for classification in (True, False):
            y = (targets["result"].map({"A": 0, "D": 1, "H": 2}).to_numpy()
                 if classification else targets["home_goals"].to_numpy())
            seen = set()
            for name, estimator, scale in market.traditional_grid(classification):
                if name in seen:
                    continue
                seen.add(name)
                with self.subTest(classification=classification, algorithm=name):
                    pipeline = market.Pipeline([
                        ("preprocessor", market.make_preprocessor(frame, features, scale)),
                        ("model", clone(estimator)),
                    ])
                    pipeline.fit(x, y)
                    prediction = pipeline.predict_proba(x) if classification else pipeline.predict(x)
                    self.assertEqual(len(prediction), len(frame))

    def test_selection_persists_best_models_and_records_unseen_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country, dataset = create_databases(root)
            selected = []
            with patch.object(market, "traditional_grid", wraps=market.traditional_grid):
                result = market.train_market_models(
                    39, dataset, country, root / "models",
                    progress_callback=lambda done, total, msg: selected.append((done, total)),
                    algorithms={"random_forest"}, epochs=1,
                    target_names=("result", "correct_score", "home_goals"),
                )
            self.assertEqual(selected[-1], (15, 15))
            self.assertEqual(len(selected), 15)
            with duckdb.connect(str(dataset), read_only=True) as con:
                rows = con.execute('SELECT target, features_json, validation_metrics_json, artifact_path FROM "39-market-model-runs"').fetchall()
                count = con.execute('SELECT count(*) FROM "39-market-model-candidates"').fetchone()[0]
            self.assertEqual(len(rows), 3)
            self.assertEqual(count, 12)
            for target, features, validation, path in rows:
                self.assertTrue(Path(path).is_file())
                self.assertNotIn("fixture_id", json.loads(features))
                if target == "correct_score":
                    self.assertEqual(json.loads(validation)["unseen_target_count"], 1)
                self.assertIsNotNone(joblib.load(path))
            self.assertEqual(len(result["models"]), 3)

    def test_sequence_alignment_rejects_stale_fixture_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country, dataset = create_databases(root)
            frames = market.load_splits(dataset, 39)
            frames[0].loc[0, "home_goals"] += 1
            with self.assertRaisesRegex(ValueError, "differ in home_goals"):
                market.sequence_splits(country, 39, frames)

    def test_aware_prediction_dates_match_normalized_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country, dataset = create_databases(root)
            for path, table in ((country, "39-2020-fixtures"), (dataset, "39-prediction-dataset")):
                with duckdb.connect(str(path)) as con:
                    con.execute(f'ALTER TABLE "{table}" ALTER COLUMN date '
                                "TYPE TIMESTAMPTZ USING timezone('UTC', date)")
            frames = market.load_splits(dataset, 39)
            self.assertEqual(len(market.sequence_splits(country, 39, frames)), 3)
            prediction, sequences = market._load_prediction_data(dataset, country, 39)
            self.assertEqual(len(prediction), 6)
            self.assertEqual(sequences["home_history"].shape[0], 6)

    def test_progressive_predictions_replacement_and_rollback(self):
        from app import server
        from sklearn.ensemble import RandomForestClassifier

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country, dataset = create_databases(root)
            model_root = root / "models"
            published = []
            progress = []
            grid = [("random_forest", RandomForestClassifier(n_estimators=3, random_state=42), False)]

            def callback(model, count):
                with patch.object(server, "_read_league", return_value={"country_name": "England"}), \
                     patch.object(server, "get_dataset_database_path", return_value=dataset):
                    rows = server.get_market_predictions(39)["predictions"]
                published.append((model["target"], count, rows))

            with patch.object(market, "traditional_grid", return_value=grid):
                result = market.train_market_models(
                    39, dataset, country, model_root, algorithms={"random_forest"},
                    target_names=("result", "btts"), market_callback=callback,
                    progress_callback=lambda done, total, msg: progress.append((done, total, msg)),
                )
                path = model_root / "39" / "result.joblib"
                self.assertEqual([item[0] for item in published], ["result", "btts"])
                self.assertEqual({row["target"] for row in published[0][2]}, {"result"})
                self.assertEqual({row["target"] for row in published[1][2]}, {"result", "btts"})
                self.assertEqual(published[0][1], 6)
                probabilities = published[0][2][0]["probabilities"]
                self.assertAlmostEqual(probabilities["1X"], probabilities["H"] + probabilities["D"])
                self.assertAlmostEqual(probabilities["X2"], probabilities["D"] + probabilities["A"])
                self.assertAlmostEqual(probabilities["12"], probabilities["H"] + probabilities["A"])
                self.assertEqual(market.load_market_artifact(path)["metadata"]["run_id"], result["run_id"])
                self.assertEqual(progress[-1][:2], (6, 6))
                self.assertIn("Published", progress[-1][2])
                prior = path.read_bytes()
                with patch.object(market, "_write_metadata", side_effect=RuntimeError("write failed")):
                    with self.assertRaisesRegex(RuntimeError, "write failed"):
                        market.train_market_models(39, dataset, country, model_root,
                                                  algorithms={"random_forest"}, target_names=("result",))
                self.assertEqual(path.read_bytes(), prior)
                with duckdb.connect(str(dataset)) as con:
                    old_id = con.execute('SELECT preparation_id FROM "39-dataset-metadata"').fetchone()[0]
                with self.assertRaisesRegex(ValueError, "Datasets changed during training"):
                    market._publish_market(path, "new model", None, {}, False, dataset, 39,
                                           ("run", "result", "test", "ratios", "[]", "{}", "{}", "{}", str(path), "[]"),
                                           [], [], "different-generation")
                self.assertEqual(path.read_bytes(), prior)
                with duckdb.connect(str(dataset)) as con:
                    self.assertEqual(con.execute('SELECT preparation_id FROM "39-dataset-metadata"').fetchone()[0], old_id)
                second = market.train_market_models(39, dataset, country, model_root,
                                                    algorithms={"random_forest"}, target_names=("result",))
                self.assertNotEqual(second["run_id"], result["run_id"])
                self.assertEqual(market.load_market_artifact(path)["metadata"]["run_id"], second["run_id"])
                self.assertEqual(sorted(p.name for p in (model_root / "39").iterdir()),
                                 ["btts.joblib", "result.joblib"])
            with duckdb.connect(str(dataset)) as con:
                con.execute('UPDATE "39-dataset-metadata" SET preparation_id = ?', ["new-generation"])
            with patch.object(server, "_read_league", return_value={"country_name": "England"}), \
                 patch.object(server, "get_dataset_database_path", return_value=dataset):
                self.assertEqual(server.get_market_predictions(39)["predictions"], [])

    def test_derived_under_and_missing_result_class(self):
        row = market._prediction_rows("run", "over_2_5", [1], np.array([[0.3, 0.7]]),
                                      np.array(["0", "1"]))[0]
        self.assertAlmostEqual(json.loads(row[4])["under_2_5"], 0.3)
        with self.assertRaisesRegex(ValueError, "home, draw and away"):
            market._prediction_rows("run", "result", [1], np.array([[0.3, 0.7]]),
                                    np.array(["H", "D"]))

    def test_selected_neural_models_persist_with_preprocessing(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country, dataset = create_databases(root)
            for algorithm in ("mlp", "lstm"):
                with self.subTest(algorithm=algorithm):
                    result = market.train_market_models(
                        39, dataset, country, root / "models", algorithms={algorithm},
                        epochs=1, target_names=("result",),
                    )
                    run_dir = root / "models" / "39"
                    self.assertTrue((run_dir / "result.joblib").is_file())
                    artifact = market.load_market_artifact(run_dir / "result.joblib")
                    self.assertEqual(artifact["metadata"]["run_id"], result["run_id"])
                    model = artifact["model"]
                    if algorithm == "mlp":
                        preprocessor = artifact["preprocessor"]
                        frames = [market.add_team_differences(f) for f in market.load_splits(dataset, 39)]
                        features = market.feature_bundles(frames[0])[result["models"][0]["feature_bundle"]]
                        x = preprocessor.transform(market.feature_frame(frames[2], features)).astype(np.float32)
                    else:
                        self.assertIsNone(artifact["preprocessor"])
                        frames = market.load_splits(dataset, 39)
                        x = market.sequence_splits(country, 39, frames)[2]
                    self.assertEqual(market._network_predict(model, x).shape, (6, 3))


if __name__ == "__main__":
    unittest.main()
