"""Tune LSTM and autoencoder outcome and score predictors on fixture histories."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from training.deep_learning.train_sequence_models import (
    HISTORY_FEATURES,
    OUTCOME_CODES,
    _build_sequences,
    _chronological_split,
    _evaluate,
    _load_tensorflow,
    _model_inputs,
    _prediction_frame as _sequence_prediction_frame,
    _read_fixture_tables,
    _write_table,
)

DEFAULT_GRID = {"units": (16, 32), "dropout": (0.0, 0.25)}
SCORE_LOSS_WEIGHT = 0.15


def _validation_split(rows: pd.DataFrame, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(rows.iloc[train_indices]["date"])
    cutoff_index = int(np.floor(len(train_indices) * 0.85))
    if cutoff_index <= 0 or cutoff_index >= len(train_indices):
        raise ValueError("Not enough fixtures for a chronological validation period")
    cutoff = dates.iloc[cutoff_index]
    fit = train_indices[(dates < cutoff).to_numpy()]
    validation = train_indices[(dates >= cutoff).to_numpy()]
    if not len(fit) or not len(validation):
        raise ValueError("Chronological validation split produced an empty partition")
    if set(rows.iloc[fit]["outcome_index"]) != {0, 1, 2}:
        raise ValueError("Training period before validation must include all three outcomes")
    return fit, validation


def _make_lstm(tf, history_length: int, units: int, dropout: float):
    from training.deep_learning.masked_layers import MaskAwareLSTM

    layers = tf.keras.layers
    inputs = {
        "home_history": layers.Input((history_length, len(HISTORY_FEATURES)), name="home_history"),
        "away_history": layers.Input((history_length, len(HISTORY_FEATURES)), name="away_history"),
        "home_history_mask": layers.Input((history_length,), dtype="bool", name="home_history_mask"),
        "away_history_mask": layers.Input((history_length,), dtype="bool", name="away_history_mask"),
    }
    encoder = MaskAwareLSTM(units, name="shared_lstm_encoder")
    home = encoder(inputs["home_history"], mask=inputs["home_history_mask"])
    away = encoder(inputs["away_history"], mask=inputs["away_history_mask"])
    combined = layers.Concatenate()([home, away])
    hidden = layers.Dense(units, activation="relu")(combined)
    hidden = layers.Dropout(dropout)(hidden)
    return tf.keras.Model(inputs, {
        "outcome": layers.Dense(3, activation="softmax", name="outcome")(hidden),
        "scores": layers.Dense(2, activation="softplus", name="scores")(hidden),
    })


def _make_autoencoder(tf, history_length: int, units: int, dropout: float):
    layers = tf.keras.layers
    shape = (history_length, len(HISTORY_FEATURES))
    inputs = {
        "home_history": layers.Input(shape, name="home_history"),
        "away_history": layers.Input(shape, name="away_history"),
    }
    flattened = layers.Concatenate()([
        layers.Flatten()(inputs["home_history"]),
        layers.Flatten()(inputs["away_history"]),
    ])
    encoded = layers.Dense(units, activation="relu", name="history_encoding")(flattened)
    reconstruction = layers.Dense(2 * history_length * len(HISTORY_FEATURES), name="reconstruction")(encoded)
    autoencoder = tf.keras.Model(inputs, reconstruction)
    hidden = layers.Dropout(dropout)(encoded)
    outcome = layers.Dense(3, activation="softmax", name="outcome")(hidden)
    scores = layers.Dense(2, activation="softplus", name="scores")(hidden)
    return autoencoder, tf.keras.Model(inputs, {"outcome": outcome, "scores": scores})


def _inputs(sequences, architecture, indices):
    return {name: values[indices] for name, values in _model_inputs(sequences, architecture).items()}


def _reconstruction_targets(inputs):
    return np.concatenate([
        inputs["home_history"].reshape(len(inputs["home_history"]), -1),
        inputs["away_history"].reshape(len(inputs["away_history"]), -1),
    ], axis=1)


def _train(tf, architecture, history_length, parameters, fit_inputs, fit_labels, fit_scores,
           epochs, batch_size, validation=None, reconstruction_epochs=None):
    if architecture == "lstm":
        model = _make_lstm(tf, history_length, **parameters)
    elif architecture == "autoencoder":
        autoencoder, model = _make_autoencoder(tf, history_length, **parameters)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        autoencoder.fit(
            fit_inputs, _reconstruction_targets(fit_inputs),
            epochs=reconstruction_epochs or min(epochs, 20), batch_size=batch_size,
            shuffle=True, verbose=0,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"outcome": "sparse_categorical_crossentropy", "scores": "mse"},
        loss_weights={"outcome": 1.0, "scores": SCORE_LOSS_WEIGHT},
    )
    callbacks = []
    if validation is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_outcome_loss", mode="min", patience=8, restore_best_weights=True
        ))
    history = model.fit(
        fit_inputs, {"outcome": fit_labels, "scores": fit_scores}, validation_data=validation,
        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=0,
    )
    best_epoch = (
        int(np.argmin(history.history["val_outcome_loss"])) + 1
        if validation is not None else epochs
    )
    return model, best_epoch


def _prediction_frame(rows, labels, probabilities, scores, architecture):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(rows), 3) or not np.isfinite(probabilities).all():
        raise ValueError(f"{architecture} returned invalid outcome probabilities")
    if (probabilities < 0).any() or (probabilities.sum(axis=1) <= 0).any():
        raise ValueError(f"{architecture} returned invalid outcome probabilities")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows), 2) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError(f"{architecture} returned invalid goal predictions")
    frame = _sequence_prediction_frame(rows, probabilities, scores, architecture)
    if labels is not None:
        frame["actual_outcome"] = np.asarray(OUTCOME_CODES)[labels]
    return frame, probabilities


def run_experiment(
    league_id: int,
    country_database: str | Path,
    dataset_database: str | Path,
    model_dir: str | Path,
    history_length: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    max_epochs: int = 100,
    batch_size: int = 64,
    grid: dict | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select parameters by validation log loss, then evaluate an untouched holdout."""
    if max_epochs < 1 or batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive")
    grid = DEFAULT_GRID if grid is None else grid
    if set(grid) != {"units", "dropout"} or not grid["units"] or not grid["dropout"]:
        raise ValueError("grid must contain nonempty units and dropout values")
    if any(not isinstance(value, int) or value < 1 for value in grid["units"]):
        raise ValueError("grid units must be positive integers")
    if any(not 0 <= value < 1 for value in grid["dropout"]):
        raise ValueError("grid dropout must be in [0, 1)")
    tf = _load_tensorflow()
    fixtures = _read_fixture_tables(Path(country_database), league_id)
    sequences, labels, scores, rows, upcoming_rows, upcoming_sequences = _build_sequences(fixtures, history_length)
    rows = rows.reset_index(drop=True)
    rows["outcome_index"] = labels
    train_indices, test_indices = _chronological_split(rows, test_size)
    fit_indices, validation_indices = _validation_split(rows, train_indices)
    candidates = [
        {"units": units, "dropout": dropout}
        for units, dropout in itertools.product(grid["units"], grid["dropout"])
    ]
    total = len(candidates) * 2
    completed = 0
    metrics, predictions, upcoming_predictions, grid_results = [], [], [], []
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "league_id": league_id, "history_length": history_length,
        "history_features": list(HISTORY_FEATURES), "outcome_codes": list(OUTCOME_CODES),
        "selection_metric": "validation_log_loss", "random_state": random_state,
        "outputs": ["outcome", "scores"], "score_loss_weight": SCORE_LOSS_WEIGHT,
        "train_fixture_count": len(fit_indices),
        "validation_fixture_count": len(validation_indices),
        "test_fixture_count": len(test_indices),
        "validation_start": rows.iloc[validation_indices[0]]["date"].isoformat(),
        "test_start": rows.iloc[test_indices[0]]["date"].isoformat(),
        "models": {},
    }

    for architecture in ("lstm", "autoencoder"):
        best = None
        for parameters in candidates:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(random_state)
            fit_inputs = _inputs(sequences, architecture, fit_indices)
            validation_inputs = _inputs(sequences, architecture, validation_indices)
            model, best_epoch = _train(
                tf, architecture, history_length, parameters, fit_inputs, labels[fit_indices],
                scores[fit_indices], max_epochs, batch_size,
                (validation_inputs, {"outcome": labels[validation_indices], "scores": scores[validation_indices]}),
            )
            validation_outputs = model(validation_inputs, training=False)
            validation_scores = np.asarray(validation_outputs["scores"])
            validation_frame, validation_probabilities = _prediction_frame(
                rows.iloc[validation_indices], labels[validation_indices],
                validation_outputs["outcome"], validation_scores, architecture,
            )
            validation_loss = float(log_loss(
                labels[validation_indices], validation_probabilities, labels=[0, 1, 2]
            ))
            if not np.isfinite(validation_loss):
                raise ValueError(f"{architecture} produced a non-finite validation loss")
            validation_metrics = _evaluate(
                rows.iloc[validation_indices].reset_index(drop=True), validation_frame,
                validation_probabilities, validation_scores, architecture, league_id,
            )
            grid_results.append({
                "model": architecture, **parameters, "best_epoch": best_epoch,
                "validation_log_loss": validation_loss,
                "validation_double_chance_accuracy": validation_metrics["double_chance_accuracy"],
                "validation_correct_score_accuracy": validation_metrics["correct_score_accuracy"],
                "validation_home_goal_mae": validation_metrics["home_goal_mae"],
                "validation_away_goal_mae": validation_metrics["away_goal_mae"],
            })
            if best is None or validation_loss < best["loss"]:
                best = {"loss": validation_loss, "parameters": parameters.copy(), "epoch": best_epoch, "model": model}
            completed += 1
            if progress_callback:
                progress_callback(completed, total, f"Tuned {architecture} configuration {completed}/{total}")

        test_inputs = _inputs(sequences, architecture, test_indices)
        test_outputs = best["model"](test_inputs, training=False)
        test_scores = np.asarray(test_outputs["scores"])
        test_frame, probabilities = _prediction_frame(
            rows.iloc[test_indices], labels[test_indices],
            test_outputs["outcome"], test_scores, architecture,
        )
        test_frame["actual_home_goals"] = scores[test_indices, 0].astype(int)
        test_frame["actual_away_goals"] = scores[test_indices, 1].astype(int)
        predictions.append(test_frame)
        metrics.append({
            **_evaluate(rows.iloc[test_indices].reset_index(drop=True), test_frame, probabilities, test_scores, architecture, league_id),
            "validation_log_loss": best["loss"],
            "best_units": best["parameters"]["units"], "best_dropout": best["parameters"]["dropout"],
            "best_epoch": best["epoch"],
        })
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state)
        final_model, _ = _train(
            tf, architecture, history_length, best["parameters"],
            _inputs(sequences, architecture, np.arange(len(labels))), labels, scores,
            best["epoch"], batch_size, reconstruction_epochs=min(max_epochs, 20),
        )
        filename = f"{league_id}_{architecture}.keras"
        final_model.save(model_dir / filename)
        metadata["models"][architecture] = {
            "file": filename, "parameters": best["parameters"], "best_epoch": best["epoch"],
        }
        if not upcoming_rows.empty:
            future_outputs = final_model(_model_inputs(upcoming_sequences, architecture), training=False)
            future, _ = _prediction_frame(
                upcoming_rows, None,
                future_outputs["outcome"], future_outputs["scores"], architecture,
            )
            future.insert(0, "league_id", league_id)
            upcoming_predictions.append(future)

    metrics_df = pd.DataFrame(metrics)
    test_df = pd.concat(predictions, ignore_index=True)
    upcoming_df = pd.concat(upcoming_predictions, ignore_index=True) if upcoming_predictions else test_df.iloc[:0].drop(columns=["actual_outcome", "actual_home_goals", "actual_away_goals"]).assign(league_id=pd.Series(dtype=int))
    _write_table(Path(dataset_database), f"{league_id}-outcome-network-metrics", metrics_df)
    _write_table(Path(dataset_database), f"{league_id}-outcome-network-grid", pd.DataFrame(grid_results))
    _write_table(Path(dataset_database), f"{league_id}-outcome-network-test-predictions", test_df)
    _write_table(Path(dataset_database), f"{league_id}-outcome-network-upcoming-predictions", upcoming_df)
    with (model_dir / "outcome_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metrics_df, upcoming_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--country-database", type=Path, required=True)
    parser.add_argument("--dataset-database", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    metrics, _ = run_experiment(
        args.league_id, args.country_database, args.dataset_database, args.model_dir,
        history_length=args.history_length, test_size=args.test_size,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
