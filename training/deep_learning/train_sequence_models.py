"""Train CNN and RNN match predictors from raw, chronological fixture tables."""

from __future__ import annotations

import argparse
import json
import logging
import re
from itertools import groupby
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from core.get_league_details import get_league_details
from core.database import connect_existing_database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLETED_STATUSES = {"FT", "AET", "PEN"}
UPCOMING_STATUSES = {"NS", "TBD"}
OUTCOME_CODES = ("H", "D", "A")
OUTCOME_TO_INDEX = {code: index for index, code in enumerate(OUTCOME_CODES)}
DOUBLE_CHANCE_OUTCOMES = {
    "1X": {"H", "D"},
    "X2": {"D", "A"},
    "12": {"H", "A"},
}
HISTORY_FEATURES = (
    "goals_for",
    "goals_against",
    "was_home",
    "points",
    "days_since_last_match",
    "has_match",
)
LOGGER = logging.getLogger(__name__)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for CNN/RNN training. Install the optional "
            "dependencies in training/deep_learning/requirements.txt using a "
            "Python version supported by TensorFlow."
        ) from error
    return tf


def _read_fixture_tables(country_database: Path, league_id: int) -> pd.DataFrame:
    if not country_database.is_file():
        raise FileNotFoundError(f"Country fixture database does not exist: {country_database}")
    table_pattern = re.compile(rf"^{league_id}-\d+-fixtures$")
    with connect_existing_database(country_database) as connection:
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name"
            ).fetchall()
            if table_pattern.fullmatch(row[0])
        ]
        if not table_names:
            raise ValueError(
                f"No raw fixture tables found for league {league_id} in {country_database}"
            )

        frames = []
        for table_name in table_names:
            columns = {
                row[0]
                for row in connection.execute(f'DESCRIBE "{table_name}"').fetchall()
            }
            required = {
                "fixture_id", "date", "home_team", "away_team", "home_goals",
                "away_goals", "status",
            }
            missing = required.difference(columns)
            if missing:
                raise ValueError(
                    f"Raw fixture table {table_name!r} is missing columns: "
                    f"{sorted(missing)}"
                )
            frame = connection.execute(
                f'''SELECT fixture_id, date, home_team, away_team,
                           home_goals, away_goals, status
                    FROM "{table_name}"'''
            ).df()
            frame["source_table"] = table_name
            frames.append(frame)

    fixtures = pd.concat(frames, ignore_index=True)
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"], errors="coerce")
    fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"], errors="coerce")
    fixtures = fixtures.loc[
        fixtures["date"].notna()
        & fixtures["home_team"].notna()
        & fixtures["away_team"].notna()
    ].copy()
    fixtures["home_team"] = fixtures["home_team"].astype(str).str.strip()
    fixtures["away_team"] = fixtures["away_team"].astype(str).str.strip()
    fixtures = fixtures.loc[
        fixtures["home_team"].ne("") & fixtures["away_team"].ne("")
    ].copy()
    return fixtures.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def _history_tensor(
    history: list[tuple[pd.Timestamp, float, float, float, float]],
    fixture_date: pd.Timestamp,
    history_length: int,
) -> np.ndarray:
    result = np.zeros((history_length, len(HISTORY_FEATURES)), dtype=np.float32)
    recent = history[-history_length:]
    offset = 0
    for position, (match_date, goals_for, goals_against, was_home, points) in enumerate(
        recent,
        start=offset,
    ):
        days = max(0, (fixture_date - match_date).total_seconds() / 86_400)
        result[position] = (
            min(goals_for, 10.0) / 5.0,
            min(goals_against, 10.0) / 5.0,
            was_home,
            points / 3.0,
            min(days, 60.0) / 30.0,
            1.0,
        )
    return result


def _build_sequences(
    fixtures: pd.DataFrame,
    history_length: int,
    prediction_fixture_ids: set[int] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    if history_length < 1:
        raise ValueError("history_length must be at least one")

    team_histories: dict[str, list[tuple[pd.Timestamp, float, float, float, float]]] = {}
    training_home_sequences = []
    training_away_sequences = []
    upcoming_home_sequences = []
    upcoming_away_sequences = []
    outcome_targets = []
    score_targets = []
    training_rows = []
    upcoming_rows = []
    fixtures = fixtures.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    # All fixtures at the same kickoff see only earlier matches, avoiding leakage.
    for _, date_rows_iter in groupby(
        fixtures.to_dict(orient="records"),
        key=lambda row: row["date"],
    ):
        date_rows = list(date_rows_iter)
        completed_rows = []
        for row in date_rows:
            is_completed = (
                row["status"] in COMPLETED_STATUSES
                and pd.notna(row["home_goals"])
                and pd.notna(row["away_goals"])
            )
            is_upcoming = (
                row["status"] in UPCOMING_STATUSES
                and row["date"] >= pd.Timestamp.now()
            )
            is_target = (
                int(row["fixture_id"]) in prediction_fixture_ids
                if prediction_fixture_ids is not None else is_upcoming
            )
            if not is_completed and not is_target:
                continue

            home = row["home_team"]
            away = row["away_team"]
            home_history = team_histories.get(home, [])
            away_history = team_histories.get(away, [])
            home_sequence = _history_tensor(home_history, row["date"], history_length)
            away_sequence = _history_tensor(away_history, row["date"], history_length)
            if is_target:
                upcoming_rows.append(row)
                upcoming_home_sequences.append(home_sequence)
                upcoming_away_sequences.append(away_sequence)
            if is_completed:
                home_goals = float(row["home_goals"])
                away_goals = float(row["away_goals"])
                if (
                    not np.isfinite([home_goals, away_goals]).all()
                    or home_goals < 0
                    or away_goals < 0
                    or home_goals != int(home_goals)
                    or away_goals != int(away_goals)
                ):
                    continue
                outcome = (
                    "H" if home_goals > away_goals
                    else "A" if away_goals > home_goals
                    else "D"
                )
                outcome_targets.append(OUTCOME_TO_INDEX[outcome])
                score_targets.append((home_goals, away_goals))
                training_rows.append(row)
                training_home_sequences.append(home_sequence)
                training_away_sequences.append(away_sequence)
                completed_rows.append((row, home_goals, away_goals))
        # Do not let another fixture at this kickoff use a same-time result.
        for row, home_goals, away_goals in completed_rows:
            home, away = row["home_team"], row["away_team"]
            home_points = (
                3.0 if home_goals > away_goals
                else 1.0 if home_goals == away_goals
                else 0.0
            )
            away_points = (
                3.0 if away_goals > home_goals
                else 1.0 if home_goals == away_goals
                else 0.0
            )
            team_histories.setdefault(home, []).append(
                (row["date"], home_goals, away_goals, 1.0, home_points)
            )
            team_histories.setdefault(away, []).append(
                (row["date"], away_goals, home_goals, 0.0, away_points)
            )
            team_histories[home] = team_histories[home][-history_length:]
            team_histories[away] = team_histories[away][-history_length:]

    sequences = {
        "home_history": np.asarray(training_home_sequences, dtype=np.float32),
        "away_history": np.asarray(training_away_sequences, dtype=np.float32),
    }
    labels = np.asarray(outcome_targets, dtype=np.int32)
    scores = np.asarray(score_targets, dtype=np.float32)
    if len(labels) != len(training_rows) or len(scores) != len(training_rows):
        raise RuntimeError("Internal sequence/target alignment error")

    prediction_sequences = {
        "home_history": np.asarray(upcoming_home_sequences, dtype=np.float32),
        "away_history": np.asarray(upcoming_away_sequences, dtype=np.float32),
    }
    prediction_rows = pd.DataFrame(upcoming_rows)
    if not prediction_rows.empty:
        prediction_rows = prediction_rows.reset_index(drop=True)
    return (
        sequences,
        labels,
        scores,
        pd.DataFrame(training_rows),
        prediction_rows,
        prediction_sequences,
    )


def _chronological_split(
    rows: pd.DataFrame,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(rows) < 10:
        raise ValueError("At least 10 completed fixtures with usable histories are required")
    split_position = int(np.floor(len(rows) * (1 - test_size)))
    if split_position <= 0 or split_position >= len(rows):
        raise ValueError("test_size does not leave both training and test fixtures")
    dates = pd.to_datetime(rows["date"])
    cutoff = dates.iloc[split_position]
    train_indices = np.flatnonzero((dates < cutoff).to_numpy())
    test_indices = np.flatnonzero((dates >= cutoff).to_numpy())
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Chronological split produced an empty training or test set")
    training_outcomes = set(
        np.unique(np.asarray(rows.loc[train_indices, "outcome_index"], dtype=int))
    )
    if training_outcomes != {0, 1, 2}:
        raise ValueError("Training period must include home wins, draws, and away wins")
    return train_indices, test_indices


def _make_model(tf, architecture: str, history_length: int):
    layers = tf.keras.layers
    if architecture == "rnn":
        from training.deep_learning.masked_layers import MaskAwareGRU

    home_input = layers.Input(
        shape=(history_length, len(HISTORY_FEATURES)),
        name="home_history",
    )
    away_input = layers.Input(
        shape=(history_length, len(HISTORY_FEATURES)),
        name="away_history",
    )

    if architecture == "cnn":
        encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(history_length, len(HISTORY_FEATURES))),
                layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
                layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
                layers.GlobalMaxPooling1D(),
            ],
            name="shared_cnn_encoder",
        )
    else:
        encoder = MaskAwareGRU(
            32,
            name="shared_rnn_encoder",
        )

        home_mask = layers.Input(
            shape=(history_length,),
            dtype="bool",
            name="home_history_mask",
        )
        away_mask = layers.Input(
            shape=(history_length,),
            dtype="bool",
            name="away_history_mask",
        )

    if architecture == "cnn":
        model_inputs = {
            "home_history": home_input,
            "away_history": away_input,
        }
        home_encoded = encoder(home_input)
        away_encoded = encoder(away_input)
    else:
        model_inputs = {
            "home_history": home_input,
            "away_history": away_input,
            "home_history_mask": home_mask,
            "away_history_mask": away_mask,
        }
        home_encoded = encoder(home_input, mask=home_mask)
        away_encoded = encoder(away_input, mask=away_mask)

    combined = layers.Concatenate()([home_encoded, away_encoded])
    shared = layers.Dense(64, activation="relu")(combined)
    shared = layers.Dropout(0.25)(shared)
    outcome = layers.Dense(3, activation="softmax", name="outcome")(shared)
    scores = layers.Dense(2, activation="softplus", name="scores")(shared)
    model = tf.keras.Model(
        inputs=model_inputs,
        outputs={"outcome": outcome, "scores": scores},
        name=f"{architecture}_match_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"outcome": "sparse_categorical_crossentropy", "scores": "mse"},
        loss_weights={"outcome": 1.0, "scores": 0.15},
        metrics={
            "outcome": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "scores": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model


def _model_inputs(
    sequences: dict[str, np.ndarray],
    architecture: str,
) -> dict[str, np.ndarray]:
    inputs = {
        "home_history": sequences["home_history"],
        "away_history": sequences["away_history"],
    }
    if architecture in ("rnn", "lstm"):
        inputs["home_history_mask"] = sequences["home_history"][:, :, -1] > 0
        inputs["away_history_mask"] = sequences["away_history"][:, :, -1] > 0
    return inputs


def _prediction_frame(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    score_values: np.ndarray,
    architecture: str,
) -> pd.DataFrame:
    ranked = np.argsort(-probabilities, axis=1, kind="stable")
    codes = np.asarray(OUTCOME_CODES)
    scores = np.maximum(0, np.floor(score_values + 0.5)).astype(int)
    double_chance = []
    for first, second in ranked[:, :2]:
        covered = frozenset((OUTCOME_CODES[int(first)], OUTCOME_CODES[int(second)]))
        double_chance.append(next(
            code for code, outcomes in DOUBLE_CHANCE_OUTCOMES.items()
            if frozenset(outcomes) == covered
        ))

    identity_columns = [
        column for column in ("fixture_id", "date", "home_team", "away_team", "status")
        if column in rows.columns
    ]
    result = rows.loc[:, identity_columns].copy().reset_index(drop=True)
    result["model"] = architecture
    result["predicted_outcome"] = codes[probabilities.argmax(axis=1)]
    result["probability_home"] = probabilities[:, 0]
    result["probability_draw"] = probabilities[:, 1]
    result["probability_away"] = probabilities[:, 2]
    result["predicted_double_chance"] = double_chance
    result["double_chance_confidence"] = np.sort(probabilities, axis=1)[:, -2:].sum(axis=1)
    result["predicted_home_goals"] = scores[:, 0]
    result["predicted_away_goals"] = scores[:, 1]
    result["predicted_correct_score"] = [
        f"{home}-{away}" for home, away in scores
    ]
    return result


def _evaluate(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
    scores: np.ndarray,
    architecture: str,
    league_id: int,
) -> dict:
    actual_scores = rows[["home_goals", "away_goals"]].to_numpy(dtype=int)
    actual_outcomes = np.where(
        actual_scores[:, 0] > actual_scores[:, 1],
        0,
        np.where(actual_scores[:, 0] < actual_scores[:, 1], 2, 1),
    )
    actual_codes = np.asarray(OUTCOME_CODES)[actual_outcomes]
    predicted_codes = predictions["predicted_outcome"].to_numpy()
    chance_correct = [
        actual in DOUBLE_CHANCE_OUTCOMES[choice]
        for actual, choice in zip(
            actual_codes, predictions["predicted_double_chance"]
        )
    ]
    rounded_scores = predictions[
        ["predicted_home_goals", "predicted_away_goals"]
    ].to_numpy(dtype=int)
    labels = [0, 1, 2]
    return {
        "league_id": league_id,
        "model": architecture,
        "test_fixture_count": len(rows),
        "outcome_accuracy": float(np.mean(predicted_codes == actual_codes)),
        "double_chance_accuracy": float(np.mean(chance_correct)),
        "correct_score_accuracy": float(np.mean(np.all(rounded_scores == actual_scores, axis=1))),
        "home_goal_mae": float(np.abs(scores[:, 0] - actual_scores[:, 0]).mean()),
        "away_goal_mae": float(np.abs(scores[:, 1] - actual_scores[:, 1]).mean()),
        "outcome_log_loss": float(log_loss(actual_outcomes, probabilities, labels=labels)),
    }


def _write_table(database: Path, table_name: str, frame: pd.DataFrame) -> None:
    with duckdb.connect(str(database)) as connection:
        connection.register("result_frame", frame)
        try:
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM result_frame'
            )
        finally:
            connection.unregister("result_frame")


def run_experiment(
    league_id: int,
    country_database: str | Path,
    dataset_database: str | Path,
    model_dir: str | Path,
    history_length: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    max_epochs: int = 150,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate the CNN and RNN using only raw fixture histories."""
    tf = _load_tensorflow()
    tf.keras.utils.set_random_seed(random_state)
    country_database = Path(country_database).expanduser().resolve()
    dataset_database = Path(dataset_database).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()

    fixtures = _read_fixture_tables(country_database, league_id)
    (
        sequences,
        labels,
        scores,
        training_rows,
        upcoming_rows,
        upcoming_sequences,
    ) = _build_sequences(
        fixtures,
        history_length,
    )
    training_rows = training_rows.reset_index(drop=True)
    if len(training_rows) != len(labels):
        raise RuntimeError("Raw fixture examples and training rows are misaligned")
    training_rows["outcome_index"] = labels
    train_indices, test_indices = _chronological_split(training_rows, test_size)
    test_inputs = {name: values[test_indices] for name, values in sequences.items()}

    test_metric_rows = []
    test_prediction_frames = []
    upcoming_prediction_frames = []
    model_dir.mkdir(parents=True, exist_ok=True)

    for architecture in ("cnn", "rnn"):
        tf.keras.utils.set_random_seed(random_state)
        model = _make_model(tf, architecture, history_length)
        model_train_inputs = _model_inputs(sequences, architecture)
        model_train_inputs = {
            name: values[train_indices]
            for name, values in model_train_inputs.items()
        }
        model_test_inputs = _model_inputs(test_inputs, architecture)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        )
        history = model.fit(
            model_train_inputs,
            {
                "outcome": labels[train_indices],
                "scores": scores[train_indices],
            },
            validation_split=0.15,
            epochs=max_epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )

        test_outputs = model(model_test_inputs, training=False)
        test_probabilities = np.asarray(test_outputs["outcome"])
        test_scores = np.asarray(test_outputs["scores"])
        test_predictions = _prediction_frame(
            training_rows.iloc[test_indices],
            test_probabilities,
            test_scores,
            architecture,
        )
        test_rows = training_rows.iloc[test_indices].reset_index(drop=True)
        test_metric_rows.append(
            _evaluate(
                test_rows,
                test_predictions,
                test_probabilities,
                test_scores,
                architecture,
                league_id,
            )
        )
        test_predictions["actual_outcome"] = np.asarray(OUTCOME_CODES)[
            labels[test_indices]
        ]
        test_predictions["actual_home_goals"] = scores[test_indices, 0].astype(int)
        test_predictions["actual_away_goals"] = scores[test_indices, 1].astype(int)
        test_prediction_frames.append(test_predictions)

        best_epoch = max(1, int(np.argmin(history.history["val_loss"])) + 1)
        final_model = _make_model(tf, architecture, history_length)
        model_all_inputs = _model_inputs(sequences, architecture)
        final_model.fit(
            model_all_inputs,
            {"outcome": labels, "scores": scores},
            epochs=best_epoch,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
        )
        model_path = model_dir / f"{league_id}_{architecture}.keras"
        final_model.save(model_path)
        if not upcoming_rows.empty:
            future_outputs = final_model(
                _model_inputs(upcoming_sequences, architecture),
                training=False,
            )
            future_predictions = _prediction_frame(
                upcoming_rows,
                np.asarray(future_outputs["outcome"]),
                np.asarray(future_outputs["scores"]),
                architecture,
            )
            future_predictions.insert(0, "league_id", league_id)
            upcoming_prediction_frames.append(future_predictions)
        LOGGER.info(
            "%s: holdout accuracy %.3f, double chance %.3f, exact score %.3f",
            architecture.upper(),
            test_metric_rows[-1]["outcome_accuracy"],
            test_metric_rows[-1]["double_chance_accuracy"],
            test_metric_rows[-1]["correct_score_accuracy"],
        )

    metrics = pd.DataFrame(test_metric_rows)
    test_predictions = pd.concat(test_prediction_frames, ignore_index=True)
    if upcoming_prediction_frames:
        upcoming_predictions = pd.concat(upcoming_prediction_frames, ignore_index=True)
    else:
        upcoming_predictions = pd.DataFrame(
            columns=[
                "league_id", "fixture_id", "date", "home_team", "away_team",
                "model", "predicted_outcome", "probability_home", "probability_draw",
                "probability_away", "predicted_double_chance",
                "double_chance_confidence", "predicted_home_goals",
                "predicted_away_goals", "predicted_correct_score",
            ]
        )

    metadata = {
        "league_id": league_id,
        "source_database": str(country_database),
        "history_length": history_length,
        "history_features": list(HISTORY_FEATURES),
        "outcome_codes": list(OUTCOME_CODES),
        "models": {
            "cnn": f"{league_id}_cnn.keras",
            "rnn": f"{league_id}_rnn.keras",
        },
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    _write_table(dataset_database, f"{league_id}-sequence-model-metrics", metrics)
    _write_table(
        dataset_database,
        f"{league_id}-sequence-model-test-predictions",
        test_predictions,
    )
    _write_table(
        dataset_database,
        f"{league_id}-sequence-model-upcoming-predictions",
        upcoming_predictions,
    )
    return metrics, upcoming_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, default=122)
    parser.add_argument("--country-database", type=Path)
    parser.add_argument("--dataset-database", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    try:
        _load_tensorflow()
    except RuntimeError as error:
        parser.error(str(error))

    catalog = get_league_details(
        str(PROJECT_ROOT / "storage" / "database" / "leagues.ddb"),
        args.league_id,
    )
    if not catalog["status"]:
        parser.error(f"League {args.league_id} was not found in the league catalog")
    country = catalog["league_details"]["country_name"]
    country_database = args.country_database or (
        PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country}.ddb"
    )
    dataset_database = args.dataset_database or (
        PROJECT_ROOT / "storage" / "database" / "datasets" / f"{country}.ddb"
    )
    model_dir = args.model_dir or (
        PROJECT_ROOT / "storage" / "trained_deep_learning_models" / "sequence"
        / str(args.league_id)
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    metrics, upcoming = run_experiment(
        league_id=args.league_id,
        country_database=country_database,
        dataset_database=dataset_database,
        model_dir=model_dir,
        history_length=args.history_length,
        test_size=args.test_size,
        random_state=args.random_state,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )
    print(metrics.to_string(index=False))
    print(f"\nUpcoming fixtures predicted per model: {len(upcoming) // 2}")


if __name__ == "__main__":
    main()
