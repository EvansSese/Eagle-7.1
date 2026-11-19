"""Train, evaluate, and run a validation-weighted 3-way classifier ensemble.

Run from the repository root with:
    python -m training.classification.train_ensemble
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from training.classification.preprocess_dataset import (
    LABEL_MAP,
    build_feature_preprocessor,
    chronological_split,
    engineer_features,
    load_data,
    prepare_target,
)

LOGGER = logging.getLogger("classifier_ensemble")
ROOT = Path(__file__).resolve().parents[2]
CLASS_ORDER = (LABEL_MAP["A"], LABEL_MAP["D"], LABEL_MAP["H"])
OUTCOME_CODES = {LABEL_MAP["A"]: "A", LABEL_MAP["D"]: "D", LABEL_MAP["H"]: "H"}
DOUBLE_CHANCE_CODES = {
    frozenset(("H", "D")): "1X",
    frozenset(("D", "A")): "X2",
    frozenset(("H", "A")): "12",
}

# Each candidate is fitted against the same validation window. The validation
# log loss selects the feature bundle and parameter set for each classifier.
MODEL_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "LR": (
        {"C": 0.3, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
    ),
    "RF": (
        {
            "n_estimators": 250,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        {
            "n_estimators": 350,
            "max_features": 0.7,
            "min_samples_leaf": 4,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
    ),
    "ET": (
        {
            "n_estimators": 250,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        {
            "n_estimators": 350,
            "max_features": 0.7,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    ),
    "GBC": (
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 160, "learning_rate": 0.05, "max_depth": 2},
    ),
    "HGB": (
        {"max_iter": 150, "learning_rate": 0.06, "max_leaf_nodes": 15},
        {"max_iter": 250, "learning_rate": 0.04, "max_leaf_nodes": 15},
    ),
    "KNN": (
        {"n_neighbors": 21, "weights": "distance", "p": 1, "n_jobs": -1},
        {"n_neighbors": 41, "weights": "distance", "p": 2, "n_jobs": -1},
    ),
    "SVM": (
        {"C": 0.5, "kernel": "rbf", "class_weight": "balanced", "cache_size": 512},
        {"C": 1.5, "kernel": "rbf", "class_weight": "balanced", "cache_size": 512},
    ),
    "DT": (
        {"max_depth": 8, "min_samples_leaf": 15, "class_weight": "balanced"},
        {"max_depth": 14, "min_samples_leaf": 30, "class_weight": "balanced"},
    ),
    "NB": (
        {"var_smoothing": 1e-8},
        {"var_smoothing": 1e-9},
    ),
}


def _feature_bundles(columns: list[str]) -> dict[str, list[str]]:
    all_features = list(columns)
    without_h2h = [
        column for column in all_features if "_h2h" not in column.lower()
    ]
    form_without_market = [
        column
        for column in without_h2h
        if not any(
            token in column.lower()
            for token in ("odds", "probability", "_cs")
        )
    ]
    bundles = {
        "all_features": all_features,
        "without_h2h": without_h2h,
        "form_without_market": form_without_market,
    }
    return {
        name: features
        for name, features in bundles.items()
        if features
    }


def _make_classifier(
    model_name: str,
    params: dict[str, Any],
    random_state: int,
):
    settings = dict(params)
    if model_name == "LR":
        return LogisticRegression(max_iter=2000, random_state=random_state, **settings)
    if model_name == "RF":
        return RandomForestClassifier(random_state=random_state, **settings)
    if model_name == "ET":
        return ExtraTreesClassifier(random_state=random_state, **settings)
    if model_name == "GBC":
        return GradientBoostingClassifier(random_state=random_state, **settings)
    if model_name == "HGB":
        return HistGradientBoostingClassifier(random_state=random_state, **settings)
    if model_name == "KNN":
        return KNeighborsClassifier(**settings)
    if model_name == "SVM":
        estimator = SVC(random_state=random_state, **settings)
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=3,
            ensemble=False,
        )
    if model_name == "DT":
        return DecisionTreeClassifier(random_state=random_state, **settings)
    if model_name == "NB":
        return GaussianNB(**settings)
    raise ValueError(f"unsupported classifier: {model_name!r}")


def _build_pipeline(
    model_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    random_state: int,
) -> Pipeline:
    scale_numeric = model_name in {"LR", "KNN", "SVM", "NB"}
    steps: list[tuple[str, Any]] = [
        ("preprocessor", build_feature_preprocessor(X, scale_numeric=scale_numeric))
    ]
    if scale_numeric:
        steps.append(
            (
                "dense_imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            )
        )
    steps.append(
        ("classifier", _make_classifier(model_name, params, random_state))
    )
    return Pipeline(steps)


def _ordered_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    missing = set(CLASS_ORDER).difference(classes.tolist())
    if missing:
        raise ValueError(f"fitted classifier does not contain classes: {sorted(missing)}")
    return raw[:, [int(np.flatnonzero(classes == label)[0]) for label in CLASS_ORDER]]


def _candidate_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    predictions = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    return (
        float(log_loss(y_true, probabilities, labels=list(CLASS_ORDER))),
        float(accuracy_score(y_true, predictions)),
    )


def _write_table(db_path: Path, table_name: str, frame: pd.DataFrame) -> None:
    with duckdb.connect(str(db_path)) as connection:
        connection.register("result_frame", frame)
        try:
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                "SELECT * FROM result_frame"
            )
        finally:
            connection.unregister("result_frame")


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _ensemble_frame(
    raw_fixtures: pd.DataFrame,
    model_probabilities: dict[str, np.ndarray],
    model_weights: dict[str, float],
) -> pd.DataFrame:
    if not model_probabilities:
        raise ValueError("at least one fitted classifier is required")
    if set(model_probabilities) != set(model_weights):
        raise ValueError("each classifier must have a validation accuracy weight")
    row_count = len(raw_fixtures)
    if any(values.shape != (row_count, len(CLASS_ORDER)) for values in model_probabilities.values()):
        raise ValueError("each classifier must provide one probability per class and fixture")
    if any(not np.isfinite(weight) or weight < 0 for weight in model_weights.values()):
        raise ValueError("classifier validation weights must be finite and non-negative")
    total_weight = float(sum(model_weights.values()))
    if total_weight <= 0:
        raise ValueError("classifier validation weights must sum to a positive value")

    weighted_probabilities = sum(
        model_probabilities[name] * model_weights[name]
        for name in model_probabilities
    ) / total_weight
    if not np.isfinite(weighted_probabilities).all():
        raise ValueError("ensemble produced non-finite class probabilities")
    row_sums = weighted_probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("ensemble produced invalid class probability totals")
    weighted_probabilities /= row_sums

    outcomes = np.asarray(CLASS_ORDER)[np.argmax(weighted_probabilities, axis=1)]
    outcome_codes = np.asarray([OUTCOME_CODES[int(value)] for value in outcomes])
    sorted_class_indices = np.argsort(
        -weighted_probabilities, axis=1, kind="stable"
    )
    double_chance_indices = sorted_class_indices[:, :2]
    double_chance = np.asarray(
        [
            DOUBLE_CHANCE_CODES[
                frozenset(
                    OUTCOME_CODES[int(CLASS_ORDER[class_index])]
                    for class_index in row
                )
            ]
            for row in double_chance_indices
        ]
    )
    covered_probabilities = np.take_along_axis(
        weighted_probabilities, double_chance_indices, axis=1
    ).sum(axis=1)

    metadata = [
        column
        for column in ("league_id", "date", "fixture_id", "home_team", "away_team")
        if column in raw_fixtures
    ]
    output = raw_fixtures[metadata].copy()
    output["predicted_outcome"] = outcome_codes
    output["confidence_outcome"] = weighted_probabilities.max(axis=1)
    output["predicted_double_chance"] = double_chance
    output["confidence_double_chance"] = covered_probabilities

    for column, label in enumerate(("away", "draw", "home")):
        output[f"probability_{label}"] = weighted_probabilities[:, column]

    vote_counts = np.zeros((row_count, len(CLASS_ORDER)), dtype=int)
    weighted_votes = np.zeros((row_count, len(CLASS_ORDER)), dtype=float)
    for model_name, probabilities in model_probabilities.items():
        class_predictions = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
        for class_index, label in enumerate(CLASS_ORDER):
            mask = class_predictions == label
            vote_counts[:, class_index] += mask
            weighted_votes[:, class_index] += mask * model_weights[model_name]
        output[f"predicted_{model_name.lower()}"] = [
            OUTCOME_CODES[int(value)] for value in class_predictions
        ]
        output[f"confidence_{model_name.lower()}"] = probabilities.max(axis=1)

    for class_index, label in enumerate(("away", "draw", "home")):
        output[f"{label}_model_votes"] = vote_counts[:, class_index]
        output[f"{label}_weighted_votes"] = weighted_votes[:, class_index]
    predicted_class_indices = np.argmax(weighted_probabilities, axis=1)
    output["outcome_model_votes"] = vote_counts[
        np.arange(row_count), predicted_class_indices
    ]
    output["outcome_weighted_votes"] = weighted_votes[
        np.arange(row_count), predicted_class_indices
    ]
    output["outcome_weighted_vote_share"] = (
        output["outcome_weighted_votes"] / total_weight
    )

    double_chance_votes = []
    double_chance_weighted_votes = []
    for row_index, market in enumerate(double_chance):
        covered = next(pair for pair, code in DOUBLE_CHANCE_CODES.items() if code == market)
        supporting_models = [
            name
            for name, probabilities in model_probabilities.items()
            if OUTCOME_CODES[
                int(CLASS_ORDER[np.argmax(probabilities[row_index])])
            ]
            in covered
        ]
        double_chance_votes.append(len(supporting_models))
        double_chance_weighted_votes.append(
            sum(model_weights[name] for name in supporting_models)
        )
    output["double_chance_model_votes"] = double_chance_votes
    output["double_chance_weighted_votes"] = double_chance_weighted_votes
    output["double_chance_weighted_vote_share"] = (
        np.asarray(double_chance_weighted_votes) / total_weight
    )
    output["predicted_at"] = pd.Timestamp.now()
    return output


def _classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(CLASS_ORDER))),
        "brier_score": float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - np.eye(len(CLASS_ORDER))[
                            [CLASS_ORDER.index(int(label)) for label in y_true]
                        ]
                    )
                    ** 2,
                    axis=1,
                )
            )
        ),
    }


def predict_upcoming_fixtures(
    db_path: Path,
    prediction_table: str,
    league_id: int,
    model_dir: Path,
    now: pd.Timestamp | None = None,
    output_table: str | None = None,
    persist: bool = True,
    include_past: bool = False,
    include_all: bool = False,
) -> pd.DataFrame:
    """Load saved ensemble artifacts and predict fixtures."""
    db_path = Path(db_path)
    model_dir = Path(model_dir)
    if not db_path.is_file():
        raise FileNotFoundError(f"dataset database does not exist: {db_path}")
    with (model_dir / "ensemble.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    if int(metadata.get("league_id", -1)) != int(league_id):
        raise ValueError(
            f"ensemble artifacts are for league {metadata.get('league_id')}, "
            f"not {league_id}"
        )
    selected_models = metadata["models"]
    if not selected_models:
        raise ValueError("ensemble metadata contains no selected models")
    raw = load_data(str(db_path), prediction_table)
    required = {"date", "status", "fixture_id"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{prediction_table!r} is missing columns: {sorted(missing)}")
    if raw["fixture_id"].duplicated().any():
        raise ValueError(f"{prediction_table!r} contains duplicate fixture IDs")
    if not include_all:
        raw = raw.loc[raw["status"].isin({"NS", "TBD"})].copy()
    original_dates = raw["date"]
    raw["date"] = pd.to_datetime(original_dates, errors="coerce")
    if (raw["date"].isna() & original_dates.notna()).any():
        raise ValueError(f"{prediction_table!r} contains invalid fixture dates")
    if not include_all:
        raw = raw.loc[raw["date"].notna()].copy()
        current_time = pd.Timestamp.now() if now is None else pd.Timestamp(now)
        if not include_past:
            raw = raw.loc[raw["date"] >= current_time].copy()
    if raw.empty:
        LOGGER.info("No future unplayed fixtures in %s", prediction_table)
        return pd.DataFrame()

    features, _ = engineer_features(raw)
    probabilities = {}
    weights = {}
    for name, details in selected_models.items():
        model_path = model_dir / details["model_file"]
        if not model_path.is_file():
            raise FileNotFoundError(f"trained classifier does not exist: {model_path}")
        model = joblib.load(model_path)
        columns = details["features"]
        missing_features = set(columns).difference(features.columns)
        if missing_features:
            raise ValueError(
                f"upcoming fixtures are missing features for {name}: "
                f"{sorted(missing_features)}"
            )
        probabilities[name] = _ordered_probabilities(model, features[columns])
        weights[name] = float(details["validation_accuracy"])

    result = _ensemble_frame(raw, probabilities, weights)
    if "league_id" in result:
        result["league_id"] = int(league_id)
    else:
        result.insert(0, "league_id", int(league_id))
    if persist:
        output_table = output_table or (
            f"{league_id}-ensemble-classifier-upcoming-predictions"
        )
        _write_table(db_path, output_table, result)
    return result


def train_ensemble(
    db_path: Path,
    league_id: int,
    training_table: str | None = None,
    prediction_table: str | None = None,
    model_dir: Path | None = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    model_candidates: dict[str, tuple[dict[str, Any], ...]] | None = None,
    predict_upcoming: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select and fit one model per family, then evaluate and predict fixtures."""
    db_path = Path(db_path)
    training_table = training_table or f"{league_id}-training-dataset"
    prediction_table = prediction_table or f"{league_id}-prediction-dataset"
    model_dir = model_dir or ROOT / "storage/trained_classifier_models/v3" / str(league_id)
    if not db_path.is_file():
        raise FileNotFoundError(f"dataset database does not exist: {db_path}")

    raw = prepare_target(load_data(str(db_path), training_table))
    X, y = engineer_features(raw)
    X_outer_train, X_test, y_outer_train, y_test = chronological_split(
        X, y, raw["date"], test_size
    )
    train_dates = raw.loc[X_outer_train.index, "date"]
    X_fit, X_validation, y_fit, y_validation = chronological_split(
        X_outer_train,
        y_outer_train,
        train_dates,
        validation_size,
    )
    bundles = _feature_bundles(X.columns.tolist())
    model_candidates = MODEL_CANDIDATES if model_candidates is None else model_candidates
    if not model_candidates:
        raise ValueError("at least one classifier must be configured")

    total_models = len(model_candidates)
    completed_models = 0
    if progress_callback is not None:
        progress_callback(
            completed_models,
            total_models,
            f"Training classifiers: {completed_models} of {total_models} models trained.",
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_models: dict[str, dict[str, Any]] = {}
    test_model_probabilities: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []

    for model_name, parameter_candidates in model_candidates.items():
        if model_name not in MODEL_CANDIDATES:
            raise ValueError(f"unsupported classifier: {model_name!r}")
        if not parameter_candidates:
            raise ValueError(f"classifier {model_name!r} has no parameter candidates")
        if progress_callback is not None:
            progress_callback(
                completed_models,
                total_models,
                f"Training classifier {model_name} "
                f"({completed_models} of {total_models} models trained)…",
            )
        best: tuple[
            tuple[float, float, str, str],
            str,
            dict[str, Any],
            float,
        ] | None = None
        for bundle_name, columns in bundles.items():
            for params in parameter_candidates:
                LOGGER.info(
                    "Validating %s with feature bundle %s and parameters %s",
                    model_name,
                    bundle_name,
                    params,
                )
                candidate = _build_pipeline(
                    model_name, params, X_fit[columns], random_state
                )
                candidate.fit(X_fit[columns], y_fit)
                probabilities = _ordered_probabilities(
                    candidate, X_validation[columns]
                )
                candidate_log_loss, candidate_accuracy = _candidate_score(
                    y_validation, probabilities
                )
                candidate_key = (
                    candidate_log_loss,
                    -candidate_accuracy,
                    bundle_name,
                    json.dumps(_safe_json(params), sort_keys=True),
                )
                if best is None or candidate_key < best[0]:
                    best = (
                        candidate_key,
                        bundle_name,
                        dict(params),
                        candidate_accuracy,
                    )
        if best is None:
            raise RuntimeError(f"no candidate was evaluated for {model_name}")

        validation_loss, _, bundle_name, _ = best[0]
        params = best[2]
        validation_accuracy = best[3]
        selected_columns = bundles[bundle_name]
        fitted = _build_pipeline(
            model_name, params, X_outer_train[selected_columns], random_state
        )
        fitted.fit(X_outer_train[selected_columns], y_outer_train)
        test_probabilities = _ordered_probabilities(
            fitted, X_test[selected_columns]
        )
        model_file = f"{model_name.lower()}.joblib"
        joblib.dump(fitted, model_dir / model_file)

        test_model_probabilities[model_name] = test_probabilities
        model_metrics = _classification_metrics(y_test, test_probabilities)
        metrics_rows.append(
            {
                "model_name": model_name,
                "feature_bundle": bundle_name,
                "validation_accuracy": validation_accuracy,
                "validation_log_loss": float(validation_loss),
                "test_samples": len(y_test),
                **model_metrics,
            }
        )
        params_rows.append(
            {
                "model_name": model_name,
                "feature_bundle": bundle_name,
                "validation_accuracy": validation_accuracy,
                "validation_log_loss": float(validation_loss),
                "hyperparameters_json": json.dumps(_safe_json(params), sort_keys=True),
                "features_json": json.dumps(selected_columns),
                "model_file": model_file,
            }
        )
        metadata_models[model_name] = {
            "feature_bundle": bundle_name,
            "validation_accuracy": validation_accuracy,
            "validation_log_loss": float(validation_loss),
            "hyperparameters": _safe_json(params),
            "features": selected_columns,
            "model_file": model_file,
        }
        completed_models += 1
        if progress_callback is not None:
            progress_callback(
                completed_models,
                total_models,
                f"Trained classifier {model_name} "
                f"({completed_models} of {total_models} models trained).",
            )

    model_weights = {
        name: details["validation_accuracy"]
        for name, details in metadata_models.items()
    }
    test_frame = _ensemble_frame(
        raw.loc[X_test.index],
        test_model_probabilities,
        model_weights,
    )
    test_frame["actual_outcome"] = [
        OUTCOME_CODES[int(value)] for value in y_test.to_numpy()
    ]
    test_frame["correct_outcome"] = (
        test_frame["predicted_outcome"] == test_frame["actual_outcome"]
    )
    double_chance_outcomes = {
        "1X": {"H", "D"},
        "X2": {"D", "A"},
        "12": {"H", "A"},
    }
    test_frame["correct_double_chance"] = [
        actual in double_chance_outcomes[prediction]
        for actual, prediction in zip(
            test_frame["actual_outcome"], test_frame["predicted_double_chance"]
        )
    ]
    test_probabilities = sum(
        test_model_probabilities[name] * model_weights[name]
        for name in test_model_probabilities
    ) / sum(model_weights.values())
    ensemble_metrics = _classification_metrics(y_test, test_probabilities)
    metrics_rows.append(
        {
            "model_name": "weighted_soft_vote",
            "feature_bundle": "selected_per_model",
            "validation_accuracy": None,
            "validation_log_loss": None,
            "test_samples": len(y_test),
            **ensemble_metrics,
        }
    )
    metrics_rows.append(
        {
            "model_name": "double_chance",
            "feature_bundle": "top_two_probability",
            "validation_accuracy": None,
            "validation_log_loss": None,
            "test_samples": len(y_test),
            "accuracy": float(test_frame["correct_double_chance"].mean()),
            "balanced_accuracy": None,
            "macro_f1": None,
            "log_loss": None,
            "brier_score": None,
        }
    )
    params_rows.append(
        {
            "model_name": "weighted_soft_vote",
            "feature_bundle": "validation_accuracy_weights",
            "validation_accuracy": None,
            "validation_log_loss": None,
            "hyperparameters_json": json.dumps(_safe_json(model_weights), sort_keys=True),
            "features_json": None,
            "model_file": "ensemble.json",
        }
    )

    metrics_frame = pd.DataFrame(metrics_rows)
    params_frame = pd.DataFrame(params_rows)
    if "league_id" in test_frame:
        test_frame["league_id"] = int(league_id)
    else:
        test_frame.insert(0, "league_id", int(league_id))
    prefix = f"{league_id}-ensemble-classifier"
    _write_table(db_path, f"{prefix}-model-metrics", metrics_frame)
    _write_table(db_path, f"{prefix}-model-params", params_frame)
    _write_table(db_path, f"{prefix}-test-predictions", test_frame)

    ensemble_metadata = {
        "league_id": int(league_id),
        "class_order": list(CLASS_ORDER),
        "outcome_codes": OUTCOME_CODES,
        "models": metadata_models,
    }
    with (model_dir / "ensemble.json").open("w", encoding="utf-8") as file:
        json.dump(_safe_json(ensemble_metadata), file, indent=2, sort_keys=True)

    upcoming = pd.DataFrame()
    if predict_upcoming:
        upcoming = predict_upcoming_fixtures(
            db_path, prediction_table, league_id, model_dir
        )
    LOGGER.info(
        "Evaluated %d models on %d chronological test fixtures; ensemble accuracy %.3f",
        len(metadata_models),
        len(y_test),
        ensemble_metrics["accuracy"],
    )
    return metrics_frame, upcoming


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LEAGUE_ID = 39
    DATABASE = ROOT / "storage/database/datasets/England.ddb"
    MODEL_DIRECTORY = ROOT / "storage/trained_classifier_models/v3" / str(LEAGUE_ID)
    metrics, upcoming_predictions = train_ensemble(
        db_path=DATABASE,
        league_id=LEAGUE_ID,
        model_dir=MODEL_DIRECTORY,
    )
    print(metrics.to_string(index=False))
    print(upcoming_predictions.head(20).to_string(index=False))
