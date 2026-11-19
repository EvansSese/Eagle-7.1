import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from training.deep_learning import train_outcome_networks as outcome
from training.deep_learning.train_sequence_models import (
    _build_sequences, _chronological_split, _model_inputs,
)


def fixture_data():
    rows = []
    scores = [(2, 0), (1, 1), (0, 2)]
    for index in range(75):
        home, away = scores[index % 3]
        rows.append({
            "fixture_id": index + 1,
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index // 2),
            "home_team": "A" if index % 2 == 0 else "B",
            "away_team": "B" if index % 2 == 0 else "A",
            "home_goals": home,
            "away_goals": away,
            "status": "FT",
        })
    rows.append({
        "fixture_id": 1000, "date": pd.Timestamp("2099-01-01"),
        "home_team": "A", "away_team": "B",
        "home_goals": None, "away_goals": None, "status": "NS",
    })
    return pd.DataFrame(rows)


class OutcomeNetworkTests(unittest.TestCase):
    def test_chronological_splits_and_same_time_histories(self):
        data = fixture_data()
        sequences, labels, scores, rows, _, _ = _build_sequences(data, 4)
        rows["outcome_index"] = labels
        train, test = _chronological_split(rows, 0.2)
        fit, validation = outcome._validation_split(rows, train)
        self.assertLess(rows.iloc[fit]["date"].max(), rows.iloc[validation]["date"].min())
        self.assertLess(rows.iloc[validation]["date"].max(), rows.iloc[test]["date"].min())
        np.testing.assert_array_equal(sequences["home_history"][0], sequences["away_history"][1])
        self.assertEqual(sequences["home_history"][1].sum(), 0)

    def test_invalid_grid_rejected_before_training(self):
        with self.assertRaises(ValueError):
            outcome.run_experiment(1, "unused", "unused", "unused", grid={"units": [0], "dropout": [0]})

    def test_grid_selects_by_validation_and_never_fits_holdout(self):
        data = fixture_data()
        sequences, labels, scores, rows, _, _ = _build_sequences(data, 4)
        rows["outcome_index"] = labels
        train, test = _chronological_split(rows, 0.2)
        fit, validation = outcome._validation_split(rows, train)
        calls = []

        class FakeModel:
            def __init__(self, units):
                self.units = units

            def __call__(self, inputs, training=False):
                n = len(inputs["home_history"])
                probabilities = np.full((n, 3), 0.1 if self.units == 8 else 1 / 3)
                if self.units == 8:
                    probabilities[np.arange(n), labels[validation[:n]] if n == len(validation) else 0] = 0.8
                return {"outcome": probabilities, "scores": np.tile([1.5, 0.6], (n, 1))}

            def save(self, path):
                Path(path).touch()

        def fake_train(tf, architecture, history_length, parameters, inputs, targets, goal_targets,
                       epochs, batch_size, validation=None, reconstruction_epochs=None):
            calls.append((architecture, parameters.copy(), np.array(targets), np.array(goal_targets), validation))
            return FakeModel(parameters["units"]), 1

        tf = SimpleNamespace(keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: None),
            utils=SimpleNamespace(set_random_seed=lambda seed: None),
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(outcome, "_load_tensorflow", return_value=tf), \
                 patch.object(outcome, "_read_fixture_tables", return_value=data), \
                 patch.object(outcome, "_train", side_effect=fake_train), \
                 patch.object(outcome, "_write_table"):
                metrics, _ = outcome.run_experiment(
                    39, root / "country.ddb", root / "dataset.ddb", root / "models",
                    history_length=4, max_epochs=1,
                    grid={"units": [4, 8], "dropout": [0.0, 0.25]},
                )
            self.assertEqual(metrics["best_units"].tolist(), [8, 8])
            search_calls = [call for call in calls if call[4] is not None]
            self.assertEqual(len(search_calls), 8)
            for _, _, targets, goal_targets, validation_data in search_calls:
                np.testing.assert_array_equal(targets, labels[fit])
                np.testing.assert_array_equal(goal_targets, scores[fit])
                np.testing.assert_array_equal(validation_data[1]["outcome"], labels[validation])
                np.testing.assert_array_equal(validation_data[1]["scores"], scores[validation])
                self.assertEqual(len(targets) + len(validation_data[1]["outcome"]), len(train))
            self.assertEqual(len(test), 15)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_real_training_artifacts_and_reload(self):
        tf = outcome._load_tensorflow()
        from training.deep_learning.masked_layers import MaskAwareLSTM

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "dataset.ddb"
            models = root / "models"
            progress = []
            with patch.object(outcome, "_read_fixture_tables", return_value=fixture_data()):
                metrics, upcoming = outcome.run_experiment(
                    39, root / "fixtures.ddb", database, models,
                    history_length=4, max_epochs=1, batch_size=16,
                    grid={"units": [4, 8], "dropout": [0.0]},
                    progress_callback=lambda done, total, message: progress.append((done, total)),
                )
            self.assertEqual(set(metrics["model"]), {"lstm", "autoencoder"})
            self.assertEqual(set(upcoming["model"]), {"lstm", "autoencoder"})
            self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
            metadata = json.loads((models / "outcome_metadata.json").read_text())
            _, _, _, _, rows, sequences = _build_sequences(fixture_data(), 4)
            self.assertEqual(len(rows), 1)
            for name in ("lstm", "autoencoder"):
                model = tf.keras.models.load_model(
                    models / metadata["models"][name]["file"],
                    custom_objects={"MaskAwareLSTM": MaskAwareLSTM}, compile=False,
                )
                actual = model(_model_inputs(sequences, name), training=False)
                saved = upcoming[upcoming["model"] == name][
                    ["probability_home", "probability_draw", "probability_away"]
                ].to_numpy()
                np.testing.assert_allclose(actual["outcome"], saved, atol=1e-6)
                expected_scores = np.maximum(0, np.floor(np.asarray(actual["scores"]) + 0.5)).astype(int)
                np.testing.assert_array_equal(
                    expected_scores, upcoming[upcoming["model"] == name][
                        ["predicted_home_goals", "predicted_away_goals"]
                    ].to_numpy(),
                )
                self.assertIn(upcoming[upcoming["model"] == name]["predicted_double_chance"].iloc[0], ("1X", "X2", "12"))
            with duckdb.connect(str(database), read_only=True) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM "39-outcome-network-grid"').fetchone()[0], 4)
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM "39-outcome-network-test-predictions"').fetchone()[0], 30)
                self.assertEqual(connection.execute(
                    'SELECT COUNT(*) FROM "39-outcome-network-metrics" WHERE double_chance_accuracy IS NOT NULL AND correct_score_accuracy IS NOT NULL'
                ).fetchone()[0], 2)

    def test_double_chance_and_exact_score_decoding(self):
        rows = fixture_data().iloc[:3]
        frame, _ = outcome._prediction_frame(
            rows, np.array([0, 1, 2]),
            np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.5, 0.1, 0.4]]),
            np.array([[1.5, 0.49], [0.5, 1.49], [2.4, 2.5]]), "lstm",
        )
        self.assertEqual(frame["predicted_double_chance"].tolist(), ["1X", "X2", "12"])
        self.assertEqual(frame["predicted_correct_score"].tolist(), ["2-0", "1-1", "2-3"])
        np.testing.assert_allclose(frame["double_chance_confidence"], [0.8, 0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
