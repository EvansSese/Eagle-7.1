"""Train neural networks for match outcomes and scorelines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from core.database import connect_existing_database
from core.fixture_dates import fixture_dates_utc, fixture_time_utc

from training.classification.preprocess_dataset import build_feature_preprocessor, engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLETED_STATUSES = {"FT", "AET", "PEN"}
OUTCOME_CODES = ("H", "D", "A")
DOUBLE_CHANCE_CODES = {
    ("D", "H"): "1X",
    ("A", "D"): "X2",
    ("A", "H"): "12",
}
LOGGER = logging.getLogger(__name__)


def _read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    with connect_existing_database(db_path) as connection:
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        if not exists:
            raise ValueError(f"Dataset table {table_name!r} does not exist in {db_path}")
        return connection.execute(f'SELECT * FROM "{table_name}"').df()


def _write_table(db_path: Path, table_name: str, frame: pd.DataFrame) -> None:
    with duckdb.connect(str(db_path)) as connection:
        connection.register("output_frame", frame)
        try:
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM output_frame'
            )
        finally:
            connection.unregister("output_frame")


def _load_completed_matches(db_path: Path, table_name: str) -> pd.DataFrame:
    frame = _read_table(db_path, table_name)
    required = {"date", "fixture_id", "status", "home_goals", "away_goals"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{table_name!r} is missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    completed = frame["status"].isin(COMPLETED_STATUSES)
    frame = frame.loc[
        completed & frame["home_goals"].notna() & frame["away_goals"].notna()
    ].copy()
    if frame.empty:
        raise ValueError(f"{table_name!r} contains no completed fixtures with scores")
    if frame["date"].isna().any():
        raise ValueError("Completed fixtures contain invalid or missing dates")
    if frame["fixture_id"].isna().any() or frame["fixture_id"].duplicated().any():
        raise ValueError("Training data contains missing or duplicate fixture IDs")

    goals = frame[["home_goals", "away_goals"]].to_numpy(dtype=float)
    if (
        not np.isfinite(goals).all()
        or (goals < 0).any()
        or not np.equal(goals, np.floor(goals)).all()
    ):
        raise ValueError("Completed fixtures must have finite, non-negative integer scores")
    frame[["home_goals", "away_goals"]] = goals.astype(int)
    return frame.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def _outcomes(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    return np.where(scores[:, 0] > scores[:, 1], "H",
                    np.where(scores[:, 0] < scores[:, 1], "A", "D"))


def _double_chance(probabilities: np.ndarray) -> tuple[list[str], np.ndarray]:
    ranked = np.argsort(-probabilities, axis=1, kind="stable")
    choices = []
    confidences = []
    for row, indices in zip(probabilities, ranked):
        codes = tuple(sorted(
            (OUTCOME_CODES[int(indices[0])], OUTCOME_CODES[int(indices[1])])
        ))
        choices.append(DOUBLE_CHANCE_CODES[codes])
        confidences.append(float(row[indices[0]] + row[indices[1]]))
    return choices, np.asarray(confidences)


def _make_pipeline(features: pd.DataFrame, model: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_feature_preprocessor(features, scale_numeric=True)),
            ("model", model),
        ]
    )


def _predict_frame(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    classifier: Pipeline,
    scorer: Pipeline,
) -> pd.DataFrame:
    model_features = engineer_features(raw)[0]
    missing = set(features.columns).difference(model_features.columns)
    if missing:
        raise ValueError(f"Prediction data is missing trained features: {sorted(missing)}")
    model_features = model_features.loc[:, features.columns]

    classifier_model = classifier.named_steps["model"]
    raw_probabilities = classifier.predict_proba(model_features)
    class_columns = {label: index for index, label in enumerate(classifier_model.classes_)}
    probabilities = raw_probabilities[:, [class_columns[code] for code in OUTCOME_CODES]]
    predicted_outcomes = np.asarray(OUTCOME_CODES)[probabilities.argmax(axis=1)]
    double_chance, double_chance_confidence = _double_chance(probabilities)
    predicted_scores = np.maximum(0, np.floor(scorer.predict(model_features) + 0.5)).astype(int)

    output_columns = [
        column for column in ("fixture_id", "date", "home_team", "away_team")
        if column in raw.columns
    ]
    result = raw.loc[:, output_columns].copy().reset_index(drop=True)
    result["predicted_outcome"] = predicted_outcomes
    result["probability_home"] = probabilities[:, 0]
    result["probability_draw"] = probabilities[:, 1]
    result["probability_away"] = probabilities[:, 2]
    result["predicted_double_chance"] = double_chance
    result["double_chance_confidence"] = double_chance_confidence
    result["predicted_home_goals"] = predicted_scores[:, 0]
    result["predicted_away_goals"] = predicted_scores[:, 1]
    result["predicted_correct_score"] = [
        f"{home}-{away}" for home, away in predicted_scores
    ]
    return result


def _chronological_split(
    matches: pd.DataFrame,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(matches) < 10:
        raise ValueError("At least 10 completed fixtures are required")
    split_position = int(np.floor(len(matches) * (1 - test_size)))
    if split_position <= 0 or split_position >= len(matches):
        raise ValueError("test_size does not leave both training and test fixtures")
    test_start = matches["date"].iloc[split_position]
    train = matches.loc[matches["date"] < test_start].copy()
    test = matches.loc[matches["date"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty training or test set")
    if set(_outcomes(train[["home_goals", "away_goals"]].to_numpy())) != set(OUTCOME_CODES):
        raise ValueError("Training period must contain home wins, draws, and away wins")
    return train, test


def run_experiment(
    db_path: str | Path,
    league_id: int,
    model_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    training_table: str | None = None,
    prediction_table: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, evaluate chronologically, save models, and predict upcoming matches."""
    db_path = Path(db_path).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    if league_id < 1:
        raise ValueError("league_id must be positive")
    if not db_path.is_file():
        raise FileNotFoundError(f"Dataset database does not exist: {db_path}")

    training_table = training_table or f"{league_id}-training-dataset"
    prediction_table = prediction_table or f"{league_id}-prediction-dataset"
    matches = _load_completed_matches(db_path, training_table)
    upcoming = _read_table(db_path, prediction_table)
    required = {"date", "status", "fixture_id"}
    missing = required.difference(upcoming.columns)
    if missing:
        raise ValueError(f"{prediction_table!r} is missing columns: {sorted(missing)}")
    upcoming["date"] = fixture_dates_utc(upcoming["date"], errors="coerce")
    upcoming = upcoming.loc[
        upcoming["status"].isin({"NS", "TBD"})
        & upcoming["date"].notna()
        & (upcoming["date"] >= fixture_time_utc())
    ].copy()

    train, test = _chronological_split(matches, test_size)
    X_all = engineer_features(matches)[0]
    X_train = X_all.loc[train.index]
    X_test = X_all.loc[test.index]

    classifier_template = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=random_state,
    )
    scorer_template = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=random_state,
    )
    y_train_outcomes = _outcomes(train[["home_goals", "away_goals"]].to_numpy())
    y_train_scores = train[["home_goals", "away_goals"]]

    classifier = _make_pipeline(X_train, classifier_template)
    scorer = _make_pipeline(X_train, scorer_template)
    classifier.fit(X_train, y_train_outcomes)
    scorer.fit(X_train, y_train_scores)
    test_predictions = _predict_frame(test, X_train, classifier, scorer)

    actual_scores = test[["home_goals", "away_goals"]].to_numpy(dtype=int)
    actual_outcomes = _outcomes(actual_scores)
    test_predictions["actual_outcome"] = actual_outcomes
    test_predictions["actual_home_goals"] = actual_scores[:, 0]
    test_predictions["actual_away_goals"] = actual_scores[:, 1]
    test_predictions["correct_outcome"] = (
        test_predictions["predicted_outcome"].to_numpy() == actual_outcomes
    )
    test_predictions["correct_double_chance"] = [
        outcome in {"1X": {"H", "D"}, "X2": {"D", "A"}, "12": {"H", "A"}}[choice]
        for outcome, choice in zip(
            actual_outcomes, test_predictions["predicted_double_chance"]
        )
    ]
    test_predictions["correct_score"] = (
        (test_predictions["predicted_home_goals"].to_numpy() == actual_scores[:, 0])
        & (test_predictions["predicted_away_goals"].to_numpy() == actual_scores[:, 1])
    )

    test_probabilities = classifier.predict_proba(X_test)
    class_columns = {
        label: index for index, label in enumerate(classifier.named_steps["model"].classes_)
    }
    ordered_labels = sorted(OUTCOME_CODES)
    test_probabilities = test_probabilities[
        :, [class_columns[code] for code in ordered_labels]
    ]
    predicted_scores = test_predictions[
        ["predicted_home_goals", "predicted_away_goals"]
    ].to_numpy()
    metrics = pd.DataFrame(
        [
            {
                "league_id": league_id,
                "test_fixture_count": len(test),
                "outcome_accuracy": float(test_predictions["correct_outcome"].mean()),
                "double_chance_accuracy": float(
                    test_predictions["correct_double_chance"].mean()
                ),
                "correct_score_accuracy": float(test_predictions["correct_score"].mean()),
                "home_goal_mae": float(
                    np.abs(predicted_scores[:, 0] - actual_scores[:, 0]).mean()
                ),
                "away_goal_mae": float(
                    np.abs(predicted_scores[:, 1] - actual_scores[:, 1]).mean()
                ),
                "outcome_log_loss": float(
                    log_loss(actual_outcomes, test_probabilities, labels=ordered_labels)
                ),
            }
        ]
    )

    LOGGER.info(
        "Chronological test: 3-way accuracy %.3f, double-chance accuracy %.3f, "
        "exact-score accuracy %.3f",
        metrics.loc[0, "outcome_accuracy"],
        metrics.loc[0, "double_chance_accuracy"],
        metrics.loc[0, "correct_score_accuracy"],
    )

    # Refit on every completed fixture after evaluation for upcoming predictions.
    all_outcomes = _outcomes(matches[["home_goals", "away_goals"]].to_numpy())
    classifier = _make_pipeline(X_all, classifier_template)
    scorer = _make_pipeline(X_all, scorer_template)
    classifier.fit(X_all, all_outcomes)
    scorer.fit(X_all, matches[["home_goals", "away_goals"]])
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = model_dir / f"{league_id}_outcome_mlp.joblib"
    scorer_path = model_dir / f"{league_id}_score_mlp.joblib"
    joblib.dump(classifier, classifier_path)
    joblib.dump(scorer, scorer_path)

    metadata = {
        "league_id": league_id,
        "training_table": training_table,
        "prediction_table": prediction_table,
        "feature_columns": list(X_all.columns),
        "outcome_classes": list(OUTCOME_CODES),
        "models": {
            "outcome": classifier_path.name,
            "score": scorer_path.name,
        },
        "network": {
            "hidden_layer_sizes": list(classifier_template.hidden_layer_sizes),
            "activation": classifier_template.activation,
            "max_iter": classifier_template.max_iter,
            "random_state": random_state,
        },
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    _write_table(db_path, f"{league_id}-deep-learning-metrics", metrics)
    _write_table(db_path, f"{league_id}-deep-learning-test-predictions", test_predictions)

    if not upcoming.empty:
        upcoming_predictions = _predict_frame(upcoming, X_all, classifier, scorer)
        upcoming_predictions.insert(0, "league_id", league_id)
    else:
        upcoming_predictions = pd.DataFrame(
            columns=[
                "league_id", "fixture_id", "date", "home_team", "away_team",
                "predicted_outcome", "probability_home", "probability_draw",
                "probability_away", "predicted_double_chance",
                "double_chance_confidence", "predicted_home_goals",
                "predicted_away_goals", "predicted_correct_score",
            ]
        )
    _write_table(
        db_path,
        f"{league_id}-deep-learning-upcoming-predictions",
        upcoming_predictions,
    )
    LOGGER.info("Saved predictions for %d upcoming fixtures", len(upcoming_predictions))
    return metrics, upcoming_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, default=39)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "storage" / "database" / "datasets" / "England.ddb",
        help="DuckDB database containing the prepared training and prediction tables",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Output directory (default: storage/trained_deep_learning_models/v1/<league-id>)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    model_dir = args.model_dir or (
        PROJECT_ROOT / "storage" / "trained_deep_learning_models" / "v1"
        / str(args.league_id)
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    metrics, upcoming = run_experiment(
        db_path=args.db_path,
        league_id=args.league_id,
        model_dir=model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(metrics.to_string(index=False))
    print(f"\nUpcoming fixtures predicted: {len(upcoming)}")


if __name__ == "__main__":
    main()
