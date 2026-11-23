"""Blend every saved classifier, goal model, and neural model for a league."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import duckdb
import joblib
import numpy as np
import pandas as pd
from core.database import connect_existing_database
from core.fixture_dates import fixture_dates_utc, fixture_time_utc
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import poisson

from training.classification.preprocess_dataset import engineer_features as classification_features
from training.deep_learning.train_sequence_models import (
    OUTCOME_CODES,
    _build_sequences,
    _load_tensorflow,
    _model_inputs,
    _read_fixture_tables,
)
from training.regression.train_goal_models import engineer_features as regression_features

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_ROOT = ROOT / "storage" / "trained_classifier_models" / "v3"
REGRESSOR_ROOT = ROOT / "storage" / "trained_regressor_models" / "v3"
DEEP_ROOT = ROOT / "storage" / "trained_deep_learning_models"
OUTCOME_INDEX = {code: index for index, code in enumerate(OUTCOME_CODES)}
DOUBLE_CHANCE = {
    "1X": {"H", "D"},
    "X2": {"D", "A"},
    "12": {"H", "A"},
}
LOGGER = logging.getLogger(__name__)
GOAL_FAMILIES = ("classifier", "regressor", "neural")


def _read_table(database: Path, table_name: str) -> pd.DataFrame:
    with connect_existing_database(database) as connection:
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        if not exists:
            raise ValueError(f"Required DuckDB table {table_name!r} is missing from {database}")
        return connection.execute(f'SELECT * FROM "{table_name}"').df()


def _write_table(database: Path, table_name: str, frame: pd.DataFrame) -> None:
    with duckdb.connect(str(database)) as connection:
        connection.register("output_frame", frame)
        try:
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM output_frame'
            )
        finally:
            connection.unregister("output_frame")


def _model_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = list(model.classes_)
    class_index = {}
    for index, label in enumerate(classes):
        if label in (0, 1, 2):
            code = ("A", "D", "H")[int(label)]
        else:
            code = str(label)
        class_index[code] = index
    if not all(code in class_index for code in OUTCOME_CODES):
        raise ValueError(f"Classifier is missing outcome classes: {classes}")
    probabilities = raw[:, [class_index[code] for code in OUTCOME_CODES]]
    if not np.isfinite(probabilities).all():
        raise ValueError("A classifier returned non-finite probabilities")
    totals = probabilities.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("A classifier returned invalid probability totals")
    return probabilities / totals


def _probabilities_from_saved_frame(frame: pd.DataFrame) -> np.ndarray:
    columns = ["probability_home", "probability_draw", "probability_away"]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Saved neural prediction table is missing {sorted(missing)}")
    probabilities = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Saved neural predictions contain non-finite probabilities")
    totals = probabilities.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Saved neural predictions contain invalid probability totals")
    return probabilities / totals


def _probability_frame(
    fixture_ids: pd.Series,
    probabilities: np.ndarray,
    scores: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "fixture_id": fixture_ids.to_numpy(),
            "probability_home": probabilities[:, 0],
            "probability_draw": probabilities[:, 1],
            "probability_away": probabilities[:, 2],
        }
    )
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(frame), 2) or not np.isfinite(scores).all():
            raise ValueError("A score model returned invalid home/away goal predictions")
        frame["predicted_home_goals"] = scores[:, 0]
        frame["predicted_away_goals"] = scores[:, 1]
    return frame


def _score_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("Goal model returned invalid expected-goal values")

    goal_rates = np.maximum(scores, 0)
    maximum_rate = float(goal_rates.max(initial=0))
    maximum_goals = max(10, int(np.ceil(maximum_rate + 8 * np.sqrt(maximum_rate))))
    if maximum_goals > 100:
        raise ValueError("Goal model returned an implausibly high expected-goal value")

    goal_values = np.arange(maximum_goals + 1)
    home_goal_probabilities = poisson.pmf(
        goal_values[None, :],
        goal_rates[:, 0, None],
    )
    away_goal_probabilities = poisson.pmf(
        goal_values[None, :],
        goal_rates[:, 1, None],
    )
    home_goal_probabilities /= home_goal_probabilities.sum(axis=1, keepdims=True)
    away_goal_probabilities /= away_goal_probabilities.sum(axis=1, keepdims=True)

    scoreline_probabilities = (
        home_goal_probabilities[:, :, None] * away_goal_probabilities[:, None, :]
    )
    home_goals = goal_values[:, None]
    away_goals = goal_values[None, :]
    return np.column_stack(
        (
            scoreline_probabilities[:, home_goals > away_goals].sum(axis=1),
            scoreline_probabilities[:, home_goals == away_goals].sum(axis=1),
            scoreline_probabilities[:, home_goals < away_goals].sum(axis=1),
        )
    )


def _model_family(name: str) -> str:
    for family in GOAL_FAMILIES:
        if name.startswith(f"{family}_"):
            return family
    raise ValueError(f"Unknown prediction model family: {name}")


def _family_probabilities(
    model_probabilities: dict[str, np.ndarray],
    excluded_model: str | None = None,
) -> dict[str, np.ndarray]:
    family_probabilities: dict[str, list[np.ndarray]] = {}
    for name, probabilities in model_probabilities.items():
        if name == excluded_model:
            continue
        family_probabilities.setdefault(_model_family(name), []).append(probabilities)
    if not family_probabilities:
        raise ValueError("No model families are available for the ensemble")
    return {
        family: np.mean(models, axis=0)
        for family, models in family_probabilities.items()
    }


def _family_ensemble(
    model_probabilities: dict[str, np.ndarray],
    excluded_model: str | None = None,
) -> np.ndarray:
    family_means = _family_probabilities(model_probabilities, excluded_model)
    return np.mean(list(family_means.values()), axis=0)


def _regressor_specs(
    league_id: int,
    dataset_database: Path,
) -> list[dict[str, Any]]:
    selected = _read_table(
        dataset_database,
        f"{league_id}-goal-regression-selected-features",
    )
    feature_groups = _read_table(
        dataset_database,
        f"{league_id}-goal-regression-feature-group-models",
    )
    specs = [
        {
            "name": f"regressor_{str(row['model']).lower()}",
            "filename": f"{league_id}_{str(row['model']).lower()}_goal_model.joblib",
            "features": json.loads(row["features_json"]),
        }
        for row in selected.to_dict(orient="records")
    ]
    specs.extend(
        {
            "name": f"regressor_{row['feature_bundle']}",
            "filename": row["model_filename"],
            "features": json.loads(row["features_json"]),
        }
        for row in feature_groups.to_dict(orient="records")
    )
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Regression metadata contains duplicate model names")
    return specs


def _add_model(
    predictions: dict[str, pd.DataFrame],
    name: str,
    frame: pd.DataFrame,
) -> None:
    if "fixture_id" not in frame or frame["fixture_id"].duplicated().any():
        raise ValueError(f"Predictions for {name} have missing or duplicate fixture IDs")
    predictions[name] = frame


def _validation_predictions(
    league_id: int,
    dataset_database: Path,
    training: pd.DataFrame,
    classifier_dir: Path,
    regressor_dir: Path,
    sequence_dir: Path,
) -> dict[str, pd.DataFrame]:
    classifier_test = _read_table(
        dataset_database,
        f"{league_id}-ensemble-classifier-test-predictions",
    )
    validation_ids = set(classifier_test["fixture_id"].astype(int))
    holdout = training.loc[
        training["fixture_id"].astype(int).isin(validation_ids)
    ].copy()
    if holdout.empty:
        raise ValueError("Classifier holdout fixtures are absent from the training table")

    classifier_metadata = json.loads(
        (classifier_dir / "ensemble.json").read_text(encoding="utf-8")
    )
    result: dict[str, pd.DataFrame] = {}
    holdout_features = classification_features(holdout)[0]
    for model_name, details in classifier_metadata["models"].items():
        model_path = classifier_dir / details["model_file"]
        model = joblib.load(model_path)
        probabilities = _model_probability(
            model,
            holdout_features[details["features"]],
        )
        _add_model(
            result,
            f"classifier_{model_name.lower()}",
            _probability_frame(holdout["fixture_id"], probabilities),
        )

    regression_X = regression_features(holdout)
    for spec in _regressor_specs(league_id, dataset_database):
        model_path = regressor_dir / spec["filename"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Goal regressor is missing: {model_path}")
        model_features = spec["features"]
        missing = set(model_features).difference(regression_X.columns)
        if missing:
            raise ValueError(
                f"Goal regressor {spec['name']} features are missing: "
                f"{sorted(missing)}"
            )
        scores = np.asarray(joblib.load(model_path).predict(regression_X[model_features]))
        probabilities = _score_probabilities(scores)
        _add_model(
            result,
            spec["name"],
            _probability_frame(holdout["fixture_id"], probabilities, scores),
        )

    mlp_test_frame = _read_table(
        dataset_database,
        f"{league_id}-deep-learning-test-predictions",
    )
    mlp_probabilities = _probabilities_from_saved_frame(mlp_test_frame)
    mlp_scores = mlp_test_frame[
        ["predicted_home_goals", "predicted_away_goals"]
    ].to_numpy(dtype=float)
    _add_model(
        result,
        "neural_mlp",
        _probability_frame(mlp_test_frame["fixture_id"], mlp_probabilities, mlp_scores),
    )

    sequence = _read_table(
        dataset_database,
        f"{league_id}-sequence-model-test-predictions",
    )
    for name, rows in sequence.groupby("model"):
        probabilities = _probabilities_from_saved_frame(rows)
        scores = rows[
            ["predicted_home_goals", "predicted_away_goals"]
        ].to_numpy(dtype=float)
        _add_model(
            result,
            f"neural_{str(name).lower()}",
            _probability_frame(rows["fixture_id"], probabilities, scores),
        )
    if (sequence_dir / "outcome_metadata.json").is_file():
        outcome_table = f"{league_id}-outcome-network-test-predictions"
        outcome_predictions = _read_table(dataset_database, outcome_table)
        for name, rows in outcome_predictions.groupby("model"):
            score_columns = ["predicted_home_goals", "predicted_away_goals"]
            scores = rows[score_columns].to_numpy(dtype=float) if set(score_columns).issubset(rows.columns) else None
            _add_model(
                result, f"neural_{str(name).lower()}",
                _probability_frame(rows["fixture_id"], _probabilities_from_saved_frame(rows), scores),
            )
    return result


def _upcoming_predictions(
    league_id: int,
    raw_database: Path,
    dataset_database: Path,
    upcoming: pd.DataFrame,
    classifier_dir: Path,
    regressor_dir: Path,
    mlp_dir: Path,
    sequence_dir: Path,
    include_all: bool = False,
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    classifier_metadata = json.loads(
        (classifier_dir / "ensemble.json").read_text(encoding="utf-8")
    )
    classifier_X = classification_features(upcoming)[0]
    for name, details in classifier_metadata["models"].items():
        model_path = classifier_dir / details["model_file"]
        model = joblib.load(model_path)
        feature_columns = details["features"]
        missing = set(feature_columns).difference(classifier_X.columns)
        if missing:
            raise ValueError(f"Classifier {name} features are missing: {sorted(missing)}")
        probabilities = _model_probability(model, classifier_X[feature_columns])
        _add_model(
            predictions,
            f"classifier_{name.lower()}",
            _probability_frame(upcoming["fixture_id"], probabilities),
        )

    regression_X = regression_features(upcoming)
    for spec in _regressor_specs(league_id, dataset_database):
        model_path = regressor_dir / spec["filename"]
        model = joblib.load(model_path)
        features = spec["features"]
        missing = set(features).difference(regression_X.columns)
        if missing:
            raise ValueError(
                f"Goal regressor {spec['name']} features are missing: "
                f"{sorted(missing)}"
            )
        scores = np.asarray(model.predict(regression_X[features]), dtype=float)
        _add_model(
            predictions,
            spec["name"],
            _probability_frame(
                upcoming["fixture_id"],
                _score_probabilities(scores),
                scores,
            ),
        )

    mlp_model_path = mlp_dir / f"{league_id}_outcome_mlp.joblib"
    mlp_score_path = mlp_dir / f"{league_id}_score_mlp.joblib"
    mlp_classifier = joblib.load(mlp_model_path)
    mlp_scorer = joblib.load(mlp_score_path)
    mlp_features = classification_features(upcoming)[0]
    mlp_probabilities = _model_probability(mlp_classifier, mlp_features)
    mlp_scores = np.asarray(mlp_scorer.predict(mlp_features), dtype=float)
    _add_model(
        predictions,
        "neural_mlp",
        _probability_frame(upcoming["fixture_id"], mlp_probabilities, mlp_scores),
    )
    fixtures = _read_fixture_tables(raw_database, league_id)
    sequence_metadata = json.loads(
        (sequence_dir / "metadata.json").read_text(encoding="utf-8")
    )
    def histories_for_length(history_length):
        _, _, _, _, raw_upcoming, upcoming_sequences = _build_sequences(
            fixtures, history_length,
            prediction_fixture_ids=set(upcoming["fixture_id"].astype(int)) if include_all else None,
        )
        raw_upcoming_ids = set(raw_upcoming["fixture_id"].astype(int)) if not raw_upcoming.empty else set()
        missing_ids = set(upcoming["fixture_id"].astype(int)) - raw_upcoming_ids
        undated = set(upcoming.loc[pd.to_datetime(upcoming["date"], errors="coerce").isna(), "fixture_id"].astype(int))
        if missing_ids - undated or (missing_ids and not include_all):
            raise ValueError("Sequence histories do not cover every prepared fixture")
        positions = {
            int(fixture_id): index
            for index, fixture_id in enumerate(raw_upcoming.get("fixture_id", []))
        }
        return {
            name: np.stack([
                values[positions[int(fixture_id)]]
                if int(fixture_id) in positions
                else np.zeros((history_length, 6), dtype=np.float32)
                for fixture_id in upcoming["fixture_id"]
            ])
            for name, values in upcoming_sequences.items()
        }

    history_length = int(sequence_metadata["history_length"])
    selected_sequences = histories_for_length(history_length)
    tf = _load_tensorflow()
    from training.deep_learning.masked_layers import MaskAwareGRU, MaskAwareLSTM

    custom_objects = {"MaskAwareGRU": MaskAwareGRU, "MaskAwareLSTM": MaskAwareLSTM}
    for architecture in ("cnn", "rnn"):
        model_path = sequence_dir / f"{league_id}_{architecture}.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Sequence model is missing: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        outputs = model(
            _model_inputs(selected_sequences, architecture),
            training=False,
        )
        probabilities = np.asarray(outputs["outcome"])
        scores = np.asarray(outputs["scores"])
        _add_model(
            predictions,
            f"neural_{architecture}",
            _probability_frame(
                upcoming["fixture_id"],
                probabilities,
                scores,
            ),
        )
    outcome_metadata_path = sequence_dir / "outcome_metadata.json"
    if outcome_metadata_path.is_file():
        outcome_metadata = json.loads(outcome_metadata_path.read_text(encoding="utf-8"))
        outcome_length = int(outcome_metadata["history_length"])
        outcome_sequences = (
            selected_sequences if outcome_length == history_length
            else histories_for_length(outcome_length)
        )
        for architecture, details in outcome_metadata["models"].items():
            if architecture not in ("lstm", "autoencoder"):
                raise ValueError(f"Unknown outcome model architecture: {architecture}")
            model_path = sequence_dir / details["file"]
            if not model_path.is_file():
                raise FileNotFoundError(f"Outcome model is missing: {model_path}")
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects, compile=False,
            )
            outputs = model(_model_inputs(outcome_sequences, architecture), training=False)
            if isinstance(outputs, dict):
                if "outcome" not in outputs or "scores" not in outputs:
                    raise ValueError(f"{architecture} model is missing outcome or score outputs")
                probabilities = np.asarray(outputs["outcome"])
                scores = np.asarray(outputs["scores"])
            else:
                probabilities = np.asarray(outputs)
                scores = None
            _add_model(
                predictions, f"neural_{architecture}",
                _probability_frame(upcoming["fixture_id"], probabilities, scores),
            )
    return predictions


def _ensemble_frame(
    fixtures: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    names = list(predictions)
    if not names:
        raise ValueError("No trained model predictions are available")
    model_probabilities = {}
    output = fixtures[
        [
            column for column in ("league_id", "fixture_id", "date", "home_team", "away_team")
            if column in fixtures.columns
        ]
    ].copy().reset_index(drop=True)
    for name, frame in predictions.items():
        aligned = output[["fixture_id"]].merge(
            frame,
            on="fixture_id",
            how="left",
            validate="one_to_one",
        )
        probabilities = aligned[
            ["probability_home", "probability_draw", "probability_away"]
        ].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Model {name} did not predict every fixture")
        model_probabilities[name] = probabilities
        labels = np.asarray(OUTCOME_CODES)[probabilities.argmax(axis=1)]
        output[f"predicted_{name}"] = labels
        output[f"confidence_{name}"] = probabilities.max(axis=1)

    family_probabilities = _family_probabilities(model_probabilities)
    combined = np.mean(list(family_probabilities.values()), axis=0)
    output["probability_home"] = combined[:, 0]
    output["probability_draw"] = combined[:, 1]
    output["probability_away"] = combined[:, 2]
    output["predicted_outcome"] = np.asarray(OUTCOME_CODES)[combined.argmax(axis=1)]
    output["outcome_confidence"] = combined.max(axis=1)
    sorted_indices = np.argsort(-combined, axis=1, kind="stable")[:, :2]
    output["predicted_double_chance"] = [
        next(
            code
            for code, covered in DOUBLE_CHANCE.items()
            if covered
            == {OUTCOME_CODES[int(indices[0])], OUTCOME_CODES[int(indices[1])]}
        )
        for indices in sorted_indices
    ]
    output["double_chance_confidence"] = np.take_along_axis(
        combined, sorted_indices, axis=1
    ).sum(axis=1)
    output["model_count"] = len(names)
    output["outcome_agreement_count"] = [
        sum(
            np.asarray(OUTCOME_CODES)[probability.argmax(axis=1)][row] == combined_label
            for probability in model_probabilities.values()
        )
        for row, combined_label in enumerate(output["predicted_outcome"])
    ]
    output["family_agreement_count"] = [
        sum(
            np.asarray(OUTCOME_CODES)[probability.argmax(axis=1)][row] == combined_label
            for probability in family_probabilities.values()
        )
        for row, combined_label in enumerate(output["predicted_outcome"])
    ]

    score_columns = [
        frame[["predicted_home_goals", "predicted_away_goals"]].to_numpy(dtype=float)
        for frame in predictions.values()
        if {"predicted_home_goals", "predicted_away_goals"}.issubset(frame.columns)
    ]
    if score_columns:
        average_scores = np.mean(score_columns, axis=0)
        rounded = np.maximum(0, np.floor(average_scores + 0.5)).astype(int)
        output["predicted_home_goals"] = rounded[:, 0]
        output["predicted_away_goals"] = rounded[:, 1]
        output["predicted_correct_score"] = [
            f"{home}-{away}" for home, away in rounded
        ]
    return output


def _importance_report(
    validation_predictions: dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_ids = set(actuals["fixture_id"].astype(int))
    for frame in validation_predictions.values():
        shared_ids.intersection_update(set(frame["fixture_id"].astype(int)))
    if not shared_ids:
        raise ValueError("Trained models have no common chronological holdout fixtures")
    actuals = actuals.loc[
        actuals["fixture_id"].astype(int).isin(shared_ids)
    ].copy()
    actuals = actuals.sort_values("fixture_id").reset_index(drop=True)
    actual_codes = actuals["actual_outcome"].to_numpy()
    matrices = {}
    rows = []
    for name, frame in validation_predictions.items():
        aligned = actuals[["fixture_id"]].merge(
            frame,
            on="fixture_id",
            how="left",
            validate="one_to_one",
        )
        probabilities = aligned[
            ["probability_home", "probability_draw", "probability_away"]
        ].to_numpy(dtype=float)
        matrices[name] = probabilities
        metrics = _outcome_metrics(actual_codes, probabilities)
        rows.append({"model": name, "holdout_fixtures": len(aligned), **metrics})

    ensemble = _family_ensemble(matrices)
    baseline_metrics = _outcome_metrics(actual_codes, ensemble)
    for row in rows:
        without = _family_ensemble(matrices, excluded_model=row["model"])
        without_metrics = _outcome_metrics(actual_codes, without)
        row["accuracy_impact_if_removed"] = float(
            baseline_metrics["outcome_accuracy"] - without_metrics["outcome_accuracy"]
        )
        row["log_loss_impact_if_removed"] = (
            baseline_metrics["outcome_log_loss"] - without_metrics["outcome_log_loss"]
        )
        row["draw_recall_impact_if_removed"] = (
            baseline_metrics["draw_recall"] - without_metrics["draw_recall"]
        )
        row["draw_precision_impact_if_removed"] = (
            baseline_metrics["draw_precision"] - without_metrics["draw_precision"]
        )

    importance = pd.DataFrame(rows)
    importance["_accuracy_impact_magnitude"] = importance[
        "accuracy_impact_if_removed"
    ].abs()
    importance["_log_loss_impact_magnitude"] = importance[
        "log_loss_impact_if_removed"
    ].abs()
    importance = importance.sort_values(
        ["_accuracy_impact_magnitude", "_log_loss_impact_magnitude"],
        ascending=[False, False],
    ).drop(columns=["_accuracy_impact_magnitude", "_log_loss_impact_magnitude"])
    ensemble_summary = pd.DataFrame(
        [
            {
                "model": "equal_weighted_families",
                "holdout_fixtures": len(actuals),
                **baseline_metrics,
                "accuracy_impact_if_removed": 0.0,
                "log_loss_impact_if_removed": 0.0,
                "draw_recall_impact_if_removed": 0.0,
                "draw_precision_impact_if_removed": 0.0,
            }
        ]
    )
    return pd.concat([ensemble_summary, importance], ignore_index=True), actuals


def _outcome_metrics(
    actual_codes: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    predicted_codes = np.asarray(OUTCOME_CODES)[probabilities.argmax(axis=1)]
    actual_draws = actual_codes == "D"
    predicted_draws = predicted_codes == "D"
    true_draws = int(np.sum(actual_draws & predicted_draws))
    predicted_draw_count = int(predicted_draws.sum())
    actual_draw_count = int(actual_draws.sum())
    draw_precision = true_draws / predicted_draw_count if predicted_draw_count else 0.0
    draw_recall = true_draws / actual_draw_count if actual_draw_count else 0.0
    draw_f1 = (
        2 * draw_precision * draw_recall / (draw_precision + draw_recall)
        if draw_precision + draw_recall
        else 0.0
    )
    return {
        "outcome_accuracy": float(accuracy_score(actual_codes, predicted_codes)),
        "outcome_log_loss": float(
            log_loss(
                [OUTCOME_INDEX[code] for code in actual_codes],
                probabilities,
                labels=[0, 1, 2],
            )
        ),
        "draw_precision": float(draw_precision),
        "draw_recall": float(draw_recall),
        "draw_f1": float(draw_f1),
        "draw_predictions": predicted_draw_count,
    }


def _catalog_country(leagues_database: Path, league_id: int) -> str:
    with connect_existing_database(leagues_database) as connection:
        row = connection.execute(
            "SELECT country_name FROM leagues WHERE league_id = ?",
            [league_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"League {league_id} is missing from {leagues_database}")
    return str(row[0])


def run_experiment(
    league_id: int,
    leagues_database: str | Path,
    raw_database: str | Path,
    dataset_database: str | Path,
    classifier_dir: str | Path | None = None,
    regressor_dir: str | Path | None = None,
    mlp_dir: str | Path | None = None,
    sequence_dir: str | Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    leagues_database = Path(leagues_database).expanduser().resolve()
    raw_database = Path(raw_database).expanduser().resolve()
    dataset_database = Path(dataset_database).expanduser().resolve()
    country = _catalog_country(leagues_database, league_id)
    classifier_dir = Path(classifier_dir or CLASSIFIER_ROOT / str(league_id))
    regressor_dir = Path(regressor_dir or REGRESSOR_ROOT / str(league_id))
    mlp_dir = Path(
        mlp_dir or DEEP_ROOT / "v1" / str(league_id)
    )
    sequence_dir = Path(
        sequence_dir or DEEP_ROOT / "sequence" / str(league_id)
    )

    training = _read_table(dataset_database, f"{league_id}-training-dataset")
    upcoming = _read_table(dataset_database, f"{league_id}-prediction-dataset")
    if "fixture_id" not in upcoming:
        raise ValueError("Prepared prediction table has no fixture_id")
    upcoming["date"] = fixture_dates_utc(upcoming["date"], errors="coerce")
    upcoming = upcoming.loc[
        upcoming["status"].isin({"NS", "TBD"})
        & upcoming["date"].notna()
        & (upcoming["date"] >= fixture_time_utc())
    ].copy()
    if upcoming.empty:
        raise ValueError("No future unplayed fixtures are available for this league")

    if progress_callback is not None:
        progress_callback(1, 4, "Prepared training and upcoming fixture data.")
    LOGGER.info("Generating predictions for %d upcoming fixtures", len(upcoming))
    upcoming_models = _upcoming_predictions(
        league_id,
        raw_database,
        dataset_database,
        upcoming,
        classifier_dir,
        regressor_dir,
        mlp_dir,
        sequence_dir,
    )
    final_predictions = _ensemble_frame(upcoming, upcoming_models)

    if progress_callback is not None:
        progress_callback(2, 4, "Predicted upcoming fixtures with all trained models.")
    validation_models = _validation_predictions(
        league_id,
        dataset_database,
        training,
        classifier_dir,
        regressor_dir,
        sequence_dir,
    )
    if progress_callback is not None:
        progress_callback(3, 4, "Measuring model impact on the shared chronological holdout.")
    outcome_map = np.where(
        training["home_goals"].to_numpy(dtype=float)
        > training["away_goals"].to_numpy(dtype=float),
        "H",
        np.where(
            training["home_goals"].to_numpy(dtype=float)
            < training["away_goals"].to_numpy(dtype=float),
            "A",
            "D",
        ),
    )
    actuals = pd.DataFrame(
        {
            "fixture_id": training["fixture_id"],
            "actual_outcome": outcome_map,
        }
    )
    importance, shared_holdout = _importance_report(validation_models, actuals)
    importance.insert(0, "league_id", league_id)
    importance["common_holdout_fixtures"] = len(shared_holdout)
    if "league_id" not in final_predictions:
        final_predictions.insert(0, "league_id", league_id)
    final_predictions["country_name"] = country

    _write_table(
        dataset_database,
        f"{league_id}-all-model-upcoming-predictions",
        final_predictions,
    )
    _write_table(
        dataset_database,
        f"{league_id}-all-model-importance",
        importance,
    )
    if progress_callback is not None:
        progress_callback(4, 4, "Saved upcoming predictions and model-importance results.")
    LOGGER.info(
        "Evaluated %d models on %d shared chronological holdout fixtures",
        len(validation_models),
        len(shared_holdout),
    )
    return final_predictions, importance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, default=122)
    parser.add_argument(
        "--leagues-database",
        type=Path,
        default=ROOT / "storage" / "database" / "leagues.ddb",
    )
    parser.add_argument("--raw-database", type=Path)
    parser.add_argument("--dataset-database", type=Path)
    parser.add_argument("--classifier-dir", type=Path)
    parser.add_argument("--regressor-dir", type=Path)
    parser.add_argument("--mlp-dir", type=Path)
    parser.add_argument("--sequence-dir", type=Path)
    args = parser.parse_args()
    country = _catalog_country(args.leagues_database, args.league_id)
    raw_database = args.raw_database or (
        ROOT / "storage" / "database" / "leagues" / f"{country}.ddb"
    )
    dataset_database = args.dataset_database or (
        ROOT / "storage" / "database" / "datasets" / f"{country}.ddb"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    predictions, importance = run_experiment(
        league_id=args.league_id,
        leagues_database=args.leagues_database,
        raw_database=raw_database,
        dataset_database=dataset_database,
        classifier_dir=args.classifier_dir,
        regressor_dir=args.regressor_dir,
        mlp_dir=args.mlp_dir,
        sequence_dir=args.sequence_dir,
    )
    print("\nModel importance on the shared chronological holdout:")
    print(importance.to_string(index=False))
    print(f"\nUpcoming fixtures predicted: {len(predictions)}")
    print(f"Predictions table: {args.league_id}-all-model-upcoming-predictions")


if __name__ == "__main__":
    main()
