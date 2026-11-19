"""Prepare chronological goal datasets and compare score-regression models.

Feature bundles are selected on a validation window within the training
period. The final chronological test period is held back for evaluation.
"""

from __future__ import annotations

import json
import logging
from math import lgamma
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from core.database import connect_existing_database


ROOT = Path(__file__).resolve().parents[2]
COMPLETED_STATUSES = {"FT", "AET", "PEN"}
TARGET_COLUMNS = ["home_goals", "away_goals"]
FORBIDDEN_FEATURES = {
    "fixture_id",
    "league_id",
    "date",
    "venue",
    "round",
    "home_goals",
    "away_goals",
    "result",
    "status",
}
BASE_CONTEXT_FEATURES = {
    "home_team", "away_team", "year", "month", "day_of_week",
    "month_sin", "month_cos",
}
LOGGER = logging.getLogger("train_goal_models")


def load_completed_matches(db_path: Path, table_name: str) -> pd.DataFrame:
    """Load scored completed fixtures and reject inconsistent result labels."""
    with connect_existing_database(db_path) as conn:
        df = conn.execute(f'SELECT * FROM "{table_name}"').df()

    required = {"date", "fixture_id", "status", *TARGET_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{table_name!r} is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[
        df["status"].isin(COMPLETED_STATUSES)
        & df["home_goals"].notna()
        & df["away_goals"].notna()
    ].copy()
    if df.empty:
        raise ValueError(f"{table_name!r} contains no completed fixtures with scores")
    if df["date"].isna().any():
        raise ValueError("completed fixtures contain invalid or missing dates")
    if df["fixture_id"].duplicated().any():
        raise ValueError("dataset contains duplicate fixture IDs")

    for column in TARGET_COLUMNS:
        scores = pd.to_numeric(df[column], errors="coerce")
        if (
            scores.isna().any()
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or not np.equal(scores, np.floor(scores)).all()
        ):
            raise ValueError(f"{column} must contain finite, non-negative integer scores")
        df[column] = scores.astype(int)

    expected_result = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ],
        ["H", "A"],
        default="D",
    )
    if "result" in df and not df["result"].isna().all():
        labeled = df["result"].notna()
        if not df.loc[labeled, "result"].eq(expected_result[labeled.to_numpy()]).all():
            raise ValueError("result labels disagree with completed fixture scores")

    return df.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split at a date boundary so fixtures with the same kickoff stay together."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(df) < 10:
        raise ValueError("at least 10 completed matches are required")
    split_position = int(np.floor(len(df) * (1 - test_size)))
    if split_position <= 0 or split_position >= len(df):
        raise ValueError("test_size does not leave both train and test data")
    test_start = df["date"].iloc[split_position]
    train = df.loc[df["date"] < test_start].copy()
    test = df.loc[df["date"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("chronological split produced an empty partition")
    return train, test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date features and remove identifiers, targets, and labels."""
    X = df.drop(columns=list(FORBIDDEN_FEATURES), errors="ignore").copy()
    dates = pd.to_datetime(df["date"])
    X["year"] = dates.dt.year
    X["month"] = dates.dt.month
    X["day_of_week"] = dates.dt.dayofweek
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X = X.drop(columns=["date"], errors="ignore")
    X = X.drop(
        columns=[
            column for column in X.columns
            if column.lower().startswith(("actual_", "pred_", "predicted_"))
        ],
        errors="ignore",
    )
    return X.replace([np.inf, -np.inf], np.nan)


def build_feature_bundles(columns: Iterable[str]) -> Dict[str, list[str]]:
    """Build practical candidate feature families from available columns."""
    columns = list(columns)
    context = [column for column in columns if column in BASE_CONTEXT_FEATURES]
    form = [
        column for column in columns
        if re.search(r"points|goals|matches_played", column, flags=re.IGNORECASE)
        and not re.search(r"xg|_as|_ds|probability|odds|correct_score", column, flags=re.IGNORECASE)
    ]
    xg = [
        column for column in columns
        if re.search(r"xg|_as|_ds|league_avg_goals", column, flags=re.IGNORECASE)
    ]
    market = [
        column for column in columns
        if re.search(r"probability|_odds|_cs($|_)", column, flags=re.IGNORECASE)
    ]
    h2h = [
        column for column in columns
        if "h2h" in column.lower()
    ]

    bundles = {
        "team_form": context + form,
        "form_xg_market": context + form + xg + market,
        "all_features": columns,
    }
    if h2h:
        bundles["form_xg_h2h"] = context + form + xg + h2h
    return {
        name: list(dict.fromkeys(column for column in bundle if column in columns))
        for name, bundle in bundles.items()
        if bundle
    }


def build_estimator(model_name: str, random_state: int, n_jobs: int):
    """Create a multi-output regressor with fixed, reproducible settings."""
    if model_name == "RandomForest":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=80, min_samples_leaf=3, max_features=0.8,
                random_state=random_state, n_jobs=n_jobs,
            ),
            n_jobs=1,
        )
    if model_name == "ExtraTrees":
        return MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=80, min_samples_leaf=3, max_features=0.8,
                random_state=random_state, n_jobs=n_jobs,
            ),
            n_jobs=1,
        )
    if model_name == "HistGradientBoosting":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=15, l2_regularization=1.0,
                random_state=random_state,
            ),
            n_jobs=1,
        )
    if model_name == "XGBoost":
        return MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror", n_estimators=100, max_depth=3,
                learning_rate=0.04, subsample=0.85, colsample_bytree=0.8,
                reg_lambda=2.0, random_state=random_state, n_jobs=n_jobs,
                verbosity=0,
            ),
            n_jobs=1,
        )
    if model_name == "SVR":
        return MultiOutputRegressor(SVR(C=10.0, epsilon=0.2, kernel="rbf"), n_jobs=1)
    if model_name == "Poisson":
        return MultiOutputRegressor(PoissonRegressor(alpha=1.0, max_iter=1000), n_jobs=1)
    if model_name == "MeanBaseline":
        return MultiOutputRegressor(DummyRegressor(strategy="mean"), n_jobs=1)
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    X: pd.DataFrame,
    model_name: str,
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if model_name in {"SVR", "Poisson"}:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))

    if not transformers:
        raise ValueError("no usable model features remain")
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", build_estimator(model_name, random_state, n_jobs)),
        ]
    )


def rounded_scores(predictions: np.ndarray) -> np.ndarray:
    """Convert regressed means into non-negative integer score predictions."""
    return np.maximum(0, np.floor(np.asarray(predictions) + 0.5)).astype(int)


def outcome_labels(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    return np.where(scores[:, 0] > scores[:, 1], "H",
                    np.where(scores[:, 0] < scores[:, 1], "A", "D"))


def aggregate_outcomes(
    group_outcomes: dict[str, np.ndarray],
    validation_accuracies: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Choose outcomes by validation-accuracy-weighted votes."""
    if not group_outcomes:
        raise ValueError("at least one feature-group outcome is required")
    if set(group_outcomes) != set(validation_accuracies):
        raise ValueError("each feature group must have a validation accuracy")
    if any(
        not np.isfinite(accuracy) or not 0 <= accuracy <= 1
        for accuracy in validation_accuracies.values()
    ):
        raise ValueError("validation accuracies must be finite values between 0 and 1")

    outcome_order = ("H", "D", "A")
    group_names = list(group_outcomes)
    outcome_matrix = np.column_stack(
        [group_outcomes[group_name] for group_name in group_names]
    )
    row_count = outcome_matrix.shape[0]
    vote_counts = {
        outcome: (outcome_matrix == outcome).sum(axis=1)
        for outcome in outcome_order
    }
    weighted_vote_totals = {
        outcome: sum(
            (group_outcomes[group_name] == outcome)
            * validation_accuracies[group_name]
            for group_name in group_names
        )
        for outcome in outcome_order
    }

    overall = []
    for row_index in range(row_count):
        max_votes = max(
            votes[row_index] for votes in weighted_vote_totals.values()
        )
        tied_outcomes = [
            outcome
            for outcome in outcome_order
            if np.isclose(weighted_vote_totals[outcome][row_index], max_votes)
        ]
        if len(tied_outcomes) == 1:
            overall.append(tied_outcomes[0])
            continue

        accuracy_by_outcome = {
            outcome: max(
                validation_accuracies[group_name]
                for group_name in group_names
                if group_outcomes[group_name][row_index] == outcome
            )
            for outcome in tied_outcomes
        }
        overall.append(
            min(
                tied_outcomes,
                key=lambda outcome: (
                    -accuracy_by_outcome[outcome],
                    outcome_order.index(outcome),
                ),
            )
        )
    return np.asarray(overall), vote_counts, weighted_vote_totals


def double_chance_labels(
    weighted_vote_totals: dict[str, np.ndarray],
    primary_outcomes: np.ndarray,
) -> np.ndarray:
    """Choose the best-supported double chance that covers the primary outcome."""
    outcome_order = ("H", "D", "A")
    if set(weighted_vote_totals) != set(outcome_order):
        raise ValueError("weighted vote totals must include H, D, and A")

    totals = np.column_stack(
        [weighted_vote_totals[outcome] for outcome in outcome_order]
    )
    primary_outcomes = np.asarray(primary_outcomes)
    if primary_outcomes.shape != (len(totals),) or not np.isin(
        primary_outcomes, outcome_order
    ).all():
        raise ValueError("primary outcomes must contain one H, D, or A per fixture")

    pair_labels = {
        frozenset(("H", "D")): "1X",
        frozenset(("D", "A")): "X2",
        frozenset(("H", "A")): "12",
    }
    labels = []
    for row_index, primary in enumerate(primary_outcomes):
        secondary = max(
            (outcome for outcome in outcome_order if outcome != primary),
            key=lambda outcome: (
                totals[row_index, outcome_order.index(outcome)],
                -outcome_order.index(outcome),
            ),
        )
        labels.append(pair_labels[frozenset((primary, secondary))])
    return np.asarray(labels)


def poisson_scoreline_predictions(
    group_goal_means: dict[str, np.ndarray],
    validation_accuracies: dict[str, float],
) -> np.ndarray:
    """Choose the most likely scoreline from a validation-weighted Poisson mixture."""
    if not group_goal_means:
        raise ValueError("at least one feature-group goal prediction is required")
    if set(group_goal_means) != set(validation_accuracies):
        raise ValueError("each feature group must have a validation accuracy")
    if any(
        not np.isfinite(accuracy) or not 0 <= accuracy <= 1
        for accuracy in validation_accuracies.values()
    ):
        raise ValueError("validation accuracies must be finite values between 0 and 1")
    total_weight = sum(validation_accuracies.values())
    if total_weight <= 0:
        raise ValueError("at least one validation accuracy must be positive")

    group_names = list(group_goal_means)
    means = {
        name: np.asarray(group_goal_means[name], dtype=float)
        for name in group_names
    }
    row_count = next(iter(means.values())).shape[0]
    if row_count == 0:
        raise ValueError("goal predictions must include at least one fixture")
    for name, scores in means.items():
        if scores.shape != (row_count, 2):
            raise ValueError(
                f"goal predictions for group {name!r} must have shape "
                f"({row_count}, 2)"
            )
        if not np.isfinite(scores).all():
            raise ValueError(f"goal predictions for group {name!r} must be finite")

    rates = {name: np.maximum(scores, 1e-12) for name, scores in means.items()}
    maximum_rate = max(float(scores.max()) for scores in rates.values())
    max_goals = max(10, int(np.ceil(maximum_rate + 8 * np.sqrt(maximum_rate + 1))))
    goal_values = np.arange(max_goals + 1)
    log_factorials = np.fromiter(
        (lgamma(int(goals) + 1) for goals in goal_values),
        dtype=float,
        count=len(goal_values),
    )
    scoreline_probabilities = np.zeros(
        (row_count, len(goal_values), len(goal_values)),
        dtype=float,
    )
    for name in group_names:
        weight = validation_accuracies[name] / total_weight
        group_rates = rates[name]
        home_log_probabilities = (
            goal_values[None, :] * np.log(group_rates[:, 0, None])
            - group_rates[:, 0, None]
            - log_factorials[None, :]
        )
        away_log_probabilities = (
            goal_values[None, :] * np.log(group_rates[:, 1, None])
            - group_rates[:, 1, None]
            - log_factorials[None, :]
        )
        home_probabilities = np.exp(home_log_probabilities)
        away_probabilities = np.exp(away_log_probabilities)
        home_probabilities /= home_probabilities.sum(axis=1, keepdims=True)
        away_probabilities /= away_probabilities.sum(axis=1, keepdims=True)
        scoreline_probabilities += weight * (
            home_probabilities[:, :, None] * away_probabilities[:, None, :]
        )

    most_likely = scoreline_probabilities.reshape(row_count, -1).argmax(axis=1)
    score_count = len(goal_values)
    return np.column_stack((most_likely // score_count, most_likely % score_count))


def poisson_over_under_predictions(
    group_goal_means: dict[str, np.ndarray],
    validation_accuracies: dict[str, float],
    thresholds: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5),
) -> dict[str, np.ndarray]:
    """Predict total-goal Over/Under markets from validation-weighted Poisson rates."""
    if not thresholds or any(
        not np.isfinite(threshold)
        or threshold < 0
        or threshold % 1 != 0.5
        for threshold in thresholds
    ):
        raise ValueError("thresholds must be non-negative half-goal lines")

    if not group_goal_means:
        raise ValueError("at least one feature-group goal prediction is required")
    if set(group_goal_means) != set(validation_accuracies):
        raise ValueError("each feature group must have a validation accuracy")
    if any(
        not np.isfinite(accuracy) or not 0 <= accuracy <= 1
        for accuracy in validation_accuracies.values()
    ):
        raise ValueError("validation accuracies must be finite values between 0 and 1")
    total_weight = sum(validation_accuracies.values())
    if total_weight <= 0:
        raise ValueError("at least one validation accuracy must be positive")

    means = {
        name: np.asarray(scores, dtype=float)
        for name, scores in group_goal_means.items()
    }
    row_count = next(iter(means.values())).shape[0]
    if row_count == 0:
        raise ValueError("goal predictions must include at least one fixture")
    for name, scores in means.items():
        if scores.shape != (row_count, 2):
            raise ValueError(
                f"goal predictions for group {name!r} must have shape "
                f"({row_count}, 2)"
            )
        if not np.isfinite(scores).all():
            raise ValueError(f"goal predictions for group {name!r} must be finite")

    total_rates = {
        name: np.maximum(scores, 0).sum(axis=1)
        for name, scores in means.items()
    }
    maximum_rate = max(float(rates.max()) for rates in total_rates.values())
    max_goals = max(10, int(np.ceil(maximum_rate + 8 * np.sqrt(maximum_rate + 1))))
    goal_values = np.arange(max_goals + 1)
    log_factorials = np.fromiter(
        (lgamma(int(goals) + 1) for goals in goal_values),
        dtype=float,
        count=len(goal_values),
    )
    over_probabilities = {
        float(threshold): np.zeros(row_count, dtype=float)
        for threshold in thresholds
    }
    for name, rates in total_rates.items():
        weight = validation_accuracies[name] / total_weight
        rates = np.maximum(rates, 1e-12)
        log_probabilities = (
            goal_values[None, :] * np.log(rates[:, None])
            - rates[:, None]
            - log_factorials[None, :]
        )
        probabilities = np.exp(log_probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for threshold in over_probabilities:
            first_over = int(np.floor(threshold)) + 1
            over_probabilities[threshold] += (
                weight * probabilities[:, first_over:].sum(axis=1)
            )

    predictions = {}
    for threshold, probability in over_probabilities.items():
        suffix = str(threshold).replace(".", "_")
        predictions[f"over_probability_{suffix}"] = probability
        predictions[f"predicted_over_under_{suffix}"] = np.where(
            probability > 0.5, "Over", "Under"
        )
    return predictions


def evaluate_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    actual = y_true[TARGET_COLUMNS].to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    rounded = rounded_scores(predicted)
    return {
        "mae_home": float(mean_absolute_error(actual[:, 0], predicted[:, 0])),
        "mae_away": float(mean_absolute_error(actual[:, 1], predicted[:, 1])),
        "mae_overall": float(mean_absolute_error(actual, predicted)),
        "rmse_overall": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2_overall": float(r2_score(actual, predicted)),
        "scoreline_exact_accuracy": float(np.all(rounded == actual, axis=1).mean()),
        "outcome_accuracy": float((outcome_labels(rounded) == outcome_labels(actual)).mean()),
    }


def split_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df).reindex(columns=X_train.columns)
    return X_train, X_test, train_df[TARGET_COLUMNS].copy(), test_df[TARGET_COLUMNS].copy()


def _write_table(db_path: Path, table_name: str, df: pd.DataFrame) -> None:
    """Replace a DuckDB table with an experiment artifact."""
    with duckdb.connect(str(db_path)) as conn:
        conn.register("artifact_frame", df)
        try:
            conn.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                "SELECT * FROM artifact_frame"
            )
        finally:
            conn.unregister("artifact_frame")


def _artifact_frame(
    source: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        column for column in ("fixture_id", "date", "home_team", "away_team")
        if column in source.columns
    ]
    metadata = source[metadata_columns].reset_index(drop=True)
    feature_values = features.drop(columns=metadata_columns, errors="ignore")
    return pd.concat(
        [
            metadata,
            feature_values.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )


def run_experiment(
    db_path: Path,
    table_name: str,
    model_dir: Path,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    models: Iterable[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Select models on validation and evaluate feature groups on the final holdout."""
    matches = load_completed_matches(db_path, table_name)
    outer_train, test_df = chronological_split(matches, test_size)
    fitting_df, validation_df = chronological_split(outer_train, validation_size)
    X_fit, X_validation, y_fit, y_validation = split_features(fitting_df, validation_df)
    X_train, X_test, y_train, y_test = split_features(outer_train, test_df)
    bundles = build_feature_bundles(X_train.columns)

    league_id = table_name.split("-", maxsplit=1)[0]
    if not league_id.isdigit():
        raise ValueError(
            f"training table name must start with a numeric league ID: {table_name!r}"
        )
    table_prefix = f"{league_id}-goal-regression"
    model_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(
        db_path,
        f"{table_prefix}-train",
        _artifact_frame(outer_train, X_train, y_train),
    )
    test_dataset = _artifact_frame(test_df, X_test, y_test)

    model_names = list(models) if models is not None else [
        "RandomForest", "ExtraTrees", "HistGradientBoosting", "XGBoost", "SVR",
        "Poisson", "MeanBaseline",
    ]
    expected_feature_groups = (
        list(bundles)
        if any(model_name != "MeanBaseline" for model_name in model_names)
        else (["all_features"] if "MeanBaseline" in model_names else [])
    )
    total_models = len(model_names) + len(expected_feature_groups)
    completed_models = 0
    if progress_callback is not None:
        progress_callback(
            completed_models,
            total_models,
            f"Training goal regressors: {completed_models} of {total_models} models trained.",
        )

    results = []
    selected_features = {}
    feature_group_candidates = {bundle_name: [] for bundle_name in bundles}
    for model_name in model_names:
        if progress_callback is not None:
            progress_callback(
                completed_models,
                total_models,
                f"Training goal regressor {model_name} "
                f"({completed_models} of {total_models} models trained)…",
            )
        candidate_results = []
        feature_bundles = {"all_features": bundles["all_features"]} if model_name == "MeanBaseline" else bundles
        for bundle_name, features in feature_bundles.items():
            model = build_pipeline(X_fit[features], model_name, random_state, n_jobs)
            model.fit(X_fit[features], y_fit)
            validation_prediction = model.predict(X_validation[features])
            metrics = evaluate_predictions(y_validation, validation_prediction)
            candidate_results.append((metrics["outcome_accuracy"], -metrics["mae_overall"], bundle_name, features))
            feature_group_candidates[bundle_name].append(
                (
                    metrics["outcome_accuracy"],
                    metrics["mae_overall"],
                    model_name,
                    features,
                )
            )
            LOGGER.info(
                "%s/%s validation: W/D/L accuracy %.3f, MAE %.3f",
                model_name, bundle_name, metrics["outcome_accuracy"], metrics["mae_overall"],
            )

        _, _, selected_bundle, chosen_features = max(candidate_results)
        selected_features[model_name] = {
            "feature_bundle": selected_bundle,
            "features": chosen_features,
        }
        best_model = build_pipeline(X_train[chosen_features], model_name, random_state, n_jobs)
        best_model.fit(X_train[chosen_features], y_train)
        test_prediction = best_model.predict(X_test[chosen_features])
        metrics = evaluate_predictions(y_test, test_prediction)
        result = {
            "model": model_name,
            "feature_bundle": selected_bundle,
            "feature_count": len(chosen_features),
            **metrics,
        }
        results.append(result)
        model_filename = f"{league_id}_{model_name.lower()}_goal_model.joblib"
        joblib.dump(best_model, model_dir / model_filename)

        scores = rounded_scores(test_prediction)
        prediction_df = test_df[["fixture_id", "date", "home_team", "away_team"]].copy()
        prediction_df["actual_home_goals"] = y_test["home_goals"].to_numpy()
        prediction_df["actual_away_goals"] = y_test["away_goals"].to_numpy()
        prediction_df["predicted_home_goals"] = scores[:, 0]
        prediction_df["predicted_away_goals"] = scores[:, 1]
        prediction_df["actual_outcome"] = outcome_labels(y_test[TARGET_COLUMNS].to_numpy())
        prediction_df["predicted_outcome"] = outcome_labels(scores)
        _write_table(
            db_path,
            f"{table_prefix}-test-predictions-{model_name.lower()}",
            prediction_df,
        )
        completed_models += 1
        if progress_callback is not None:
            progress_callback(
                completed_models,
                total_models,
                f"Trained goal regressor {model_name} "
                f"({completed_models} of {total_models} models trained).",
            )

    feature_group_models = []
    group_test_predictions = {}
    for bundle_name, candidates in feature_group_candidates.items():
        if not candidates:
            continue
        if progress_callback is not None:
            progress_callback(
                completed_models,
                total_models,
                f"Training feature-group regressor {bundle_name} "
                f"({completed_models} of {total_models} models trained)…",
            )
        validation_accuracy, validation_mae, model_name, features = min(
            candidates,
            key=lambda candidate: (
                -candidate[0],
                candidate[1],
                candidate[2],
            ),
        )
        best_model = build_pipeline(X_train[features], model_name, random_state, n_jobs)
        best_model.fit(X_train[features], y_train)
        test_prediction = best_model.predict(X_test[features])
        rounded_test_prediction = rounded_scores(test_prediction)
        predicted_outcomes = outcome_labels(rounded_test_prediction)
        test_metrics = evaluate_predictions(y_test, test_prediction)
        model_filename = (
            f"{league_id}_{bundle_name.lower()}_goal_model.joblib"
        )
        joblib.dump(best_model, model_dir / model_filename)
        feature_group_models.append(
            {
                "feature_bundle": bundle_name,
                "model": model_name,
                "model_filename": model_filename,
                "feature_count": len(features),
                "features_json": json.dumps(features),
                "validation_outcome_accuracy": validation_accuracy,
                "validation_mae_overall": validation_mae,
                "test_outcome_accuracy": test_metrics["outcome_accuracy"],
                "test_mae_overall": test_metrics["mae_overall"],
            }
        )
        group_test_predictions[bundle_name] = {
            "model": model_name,
            "raw_scores": test_prediction,
            "scores": rounded_test_prediction,
            "outcomes": predicted_outcomes,
            "validation_accuracy": validation_accuracy,
            "metrics": test_metrics,
        }
        completed_models += 1
        if progress_callback is not None:
            progress_callback(
                completed_models,
                total_models,
                f"Trained feature-group regressor {bundle_name} "
                f"({completed_models} of {total_models} models trained).",
            )
        suffix = bundle_name.lower()
        test_dataset[f"model_{suffix}"] = model_name
        test_dataset[f"predicted_home_goals_{suffix}"] = rounded_test_prediction[:, 0]
        test_dataset[f"predicted_away_goals_{suffix}"] = rounded_test_prediction[:, 1]
        test_dataset[f"predicted_home_goals_raw_{suffix}"] = test_prediction[:, 0]
        test_dataset[f"predicted_away_goals_raw_{suffix}"] = test_prediction[:, 1]
        test_dataset[f"predicted_outcome_{suffix}"] = predicted_outcomes
        test_dataset[f"outcome_correct_{suffix}"] = (
            predicted_outcomes == outcome_labels(y_test[TARGET_COLUMNS].to_numpy())
        )
        test_dataset[f"scoreline_correct_{suffix}"] = np.all(
            rounded_test_prediction == y_test[TARGET_COLUMNS].to_numpy(), axis=1
        )

    actual_outcomes = outcome_labels(y_test[TARGET_COLUMNS].to_numpy())
    test_dataset["actual_outcome"] = actual_outcomes
    probabilistic_scores = poisson_scoreline_predictions(
        {
            bundle_name: result["raw_scores"]
            for bundle_name, result in group_test_predictions.items()
        },
        {
            bundle_name: result["validation_accuracy"]
            for bundle_name, result in group_test_predictions.items()
        },
    )
    probabilistic_outcomes = outcome_labels(probabilistic_scores)
    over_under_predictions = poisson_over_under_predictions(
        {
            bundle_name: result["raw_scores"]
            for bundle_name, result in group_test_predictions.items()
        },
        {
            bundle_name: result["validation_accuracy"]
            for bundle_name, result in group_test_predictions.items()
        },
    )
    for column, values in over_under_predictions.items():
        test_dataset[column] = values
    test_dataset["predicted_home_goals_poisson"] = probabilistic_scores[:, 0]
    test_dataset["predicted_away_goals_poisson"] = probabilistic_scores[:, 1]
    test_dataset["predicted_outcome_poisson"] = probabilistic_outcomes
    overall_outcomes, vote_counts, weighted_vote_totals = aggregate_outcomes(
        {
            bundle_name: result["outcomes"]
            for bundle_name, result in group_test_predictions.items()
        },
        {
            bundle_name: result["validation_accuracy"]
            for bundle_name, result in group_test_predictions.items()
        },
    )
    mean_scores = rounded_scores(
        np.mean(
            [result["scores"] for result in group_test_predictions.values()],
            axis=0,
        )
    )
    mean_score_outcomes = outcome_labels(mean_scores)
    test_dataset["predicted_home_goals"] = mean_scores[:, 0]
    test_dataset["predicted_away_goals"] = mean_scores[:, 1]
    test_dataset["predicted_outcome"] = mean_score_outcomes
    test_dataset["predicted_outcome_weighted_vote"] = overall_outcomes
    test_dataset["home_outcome_votes"] = vote_counts["H"]
    test_dataset["draw_outcome_votes"] = vote_counts["D"]
    test_dataset["away_outcome_votes"] = vote_counts["A"]
    test_dataset["home_outcome_weighted_votes"] = weighted_vote_totals["H"]
    test_dataset["draw_outcome_weighted_votes"] = weighted_vote_totals["D"]
    test_dataset["away_outcome_weighted_votes"] = weighted_vote_totals["A"]
    test_dataset["predicted_double_chance"] = double_chance_labels(
        weighted_vote_totals,
        mean_score_outcomes,
    )
    test_dataset["outcome_correct"] = mean_score_outcomes == actual_outcomes
    test_dataset["outcome_correct_weighted_vote"] = (
        overall_outcomes == actual_outcomes
    )
    double_chance_outcomes = {
        "1X": {"H", "D"},
        "X2": {"D", "A"},
        "12": {"H", "A"},
    }
    test_dataset["double_chance_correct"] = [
        actual in double_chance_outcomes[prediction]
        for actual, prediction in zip(
            actual_outcomes, test_dataset["predicted_double_chance"]
        )
    ]
    test_dataset["scoreline_correct"] = np.all(
        mean_scores == y_test[TARGET_COLUMNS].to_numpy(), axis=1
    )
    actual_total_goals = y_test[TARGET_COLUMNS].sum(axis=1).to_numpy()
    for threshold in (0.5, 1.5, 2.5, 3.5):
        suffix = str(threshold).replace(".", "_")
        actual_market = np.where(actual_total_goals > threshold, "Over", "Under")
        test_dataset[f"actual_over_under_{suffix}"] = actual_market
        test_dataset[f"correct_over_under_{suffix}"] = (
            test_dataset[f"predicted_over_under_{suffix}"] == actual_market
        )
    _write_table(db_path, f"{table_prefix}-test", test_dataset)

    test_fixture_count = len(test_dataset)
    mean_score_metrics = evaluate_predictions(y_test, mean_scores)
    test_metrics_rows = [
        {
            "feature_bundle": bundle_name,
            "model": result["model"],
            "test_fixture_count": test_fixture_count,
            "correct_outcome_count": int(
                test_dataset[f"outcome_correct_{bundle_name.lower()}"].sum()
            ),
            "outcome_accuracy": result["metrics"]["outcome_accuracy"],
            "mae_overall": result["metrics"]["mae_overall"],
            "correct_scoreline_count": int(
                test_dataset[f"scoreline_correct_{bundle_name.lower()}"].sum()
            ),
            "scoreline_exact_accuracy": result["metrics"]["scoreline_exact_accuracy"],
        }
        for bundle_name, result in group_test_predictions.items()
    ]
    test_metrics_rows.append(
        {
            "feature_bundle": "weighted_vote_ensemble",
            "model": "validation_weighted_vote",
            "test_fixture_count": test_fixture_count,
            "correct_outcome_count": int(
                test_dataset["outcome_correct_weighted_vote"].sum()
            ),
            "outcome_accuracy": float(
                test_dataset["outcome_correct_weighted_vote"].mean()
            ),
            "mae_overall": None,
            "correct_scoreline_count": None,
            "scoreline_exact_accuracy": None,
        }
    )
    test_metrics_rows.append(
        {
            "feature_bundle": "mean_score_ensemble",
            "model": "rounded_mean",
            "test_fixture_count": test_fixture_count,
            "correct_outcome_count": int(
                (mean_score_outcomes == actual_outcomes).sum()
            ),
            "outcome_accuracy": mean_score_metrics["outcome_accuracy"],
            "mae_overall": mean_score_metrics["mae_overall"],
            "correct_scoreline_count": int(
                test_dataset["scoreline_correct"].sum()
            ),
            "scoreline_exact_accuracy": mean_score_metrics[
                "scoreline_exact_accuracy"
            ],
        }
    )
    probabilistic_metrics = evaluate_predictions(y_test, probabilistic_scores)
    test_metrics_rows.append(
        {
            "feature_bundle": "poisson_probability_ensemble",
            "model": "validation_weighted_poisson",
            "test_fixture_count": test_fixture_count,
            "correct_outcome_count": int(
                (probabilistic_outcomes == actual_outcomes).sum()
            ),
            "outcome_accuracy": probabilistic_metrics["outcome_accuracy"],
            "mae_overall": probabilistic_metrics["mae_overall"],
            "correct_scoreline_count": int(
                np.all(
                    probabilistic_scores == y_test[TARGET_COLUMNS].to_numpy(),
                    axis=1,
                ).sum()
            ),
            "scoreline_exact_accuracy": probabilistic_metrics[
                "scoreline_exact_accuracy"
            ],
        }
    )
    test_metrics_rows.append(
        {
            "feature_bundle": "double_chance",
            "model": "rounded_mean_with_weighted_support",
            "test_fixture_count": test_fixture_count,
            "correct_outcome_count": int(test_dataset["double_chance_correct"].sum()),
            "outcome_accuracy": float(
                test_dataset["double_chance_correct"].mean()
            ),
            "mae_overall": None,
            "correct_scoreline_count": None,
            "scoreline_exact_accuracy": None,
        }
    )
    for threshold in (0.5, 1.5, 2.5, 3.5):
        suffix = str(threshold).replace(".", "_")
        correct_column = f"correct_over_under_{suffix}"
        test_metrics_rows.append(
            {
                "feature_bundle": f"over_under_{suffix}",
                "model": "validation_weighted_poisson",
                "test_fixture_count": test_fixture_count,
                "correct_outcome_count": int(test_dataset[correct_column].sum()),
                "outcome_accuracy": float(test_dataset[correct_column].mean()),
                "mae_overall": None,
                "correct_scoreline_count": None,
                "scoreline_exact_accuracy": None,
            }
        )
    test_metrics_df = pd.DataFrame(test_metrics_rows)

    results_df = pd.DataFrame(results).sort_values(
        ["outcome_accuracy", "mae_overall"], ascending=[False, True]
    )
    _write_table(db_path, f"{table_prefix}-model-comparison", results_df)
    selected_features_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "feature_bundle": details["feature_bundle"],
                "feature_count": len(details["features"]),
                "features_json": json.dumps(details["features"]),
            }
            for model_name, details in selected_features.items()
        ]
    )
    _write_table(db_path, f"{table_prefix}-selected-features", selected_features_df)
    _write_table(
        db_path,
        f"{table_prefix}-feature-group-models",
        pd.DataFrame(feature_group_models),
    )
    _write_table(
        db_path,
        f"{table_prefix}-test-metrics",
        test_metrics_df,
    )
    for result in test_metrics_rows:
        if result["correct_scoreline_count"] is None:
            LOGGER.info(
                "Test accuracy %s/%s: %d/%d predictions correct (%.3f)",
                result["feature_bundle"],
                result["model"],
                result["correct_outcome_count"],
                result["test_fixture_count"],
                result["outcome_accuracy"],
            )
            continue
        LOGGER.info(
            "Test accuracy %s/%s: %d/%d outcomes correct (%.3f), "
            "%d exact scorelines (%.3f)",
            result["feature_bundle"],
            result["model"],
            result["correct_outcome_count"],
            result["test_fixture_count"],
            result["outcome_accuracy"],
            result["correct_scoreline_count"],
            result["scoreline_exact_accuracy"],
        )
    LOGGER.info(
        "Used %d completed fixtures: %d train, %d test. Best W/D/L test accuracy: %.3f",
        len(matches), len(outer_train), len(test_df), results_df.iloc[0]["outcome_accuracy"],
    )
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    LEAGUE_ID = 39
    DB_PATH = ROOT / "storage/database/datasets/England.ddb"
    TRAINING_TABLE = f"{LEAGUE_ID}-training-dataset"
    MODEL_DIR = ROOT / "storage/trained_regressor_models/v3" / str(LEAGUE_ID)
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    RANDOM_STATE = 42
    N_JOBS = -1
    MODELS = None

    comparison = run_experiment(
        db_path=DB_PATH,
        table_name=TRAINING_TABLE,
        model_dir=MODEL_DIR,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        models=MODELS,
    )
    print(comparison.to_string(index=False))
