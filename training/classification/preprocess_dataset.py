"""
preprocess_dataset.py
====================
Shared data-loading and feature-engineering module for the football match
classifier pipeline.

This module is the single source of truth for:
    1. Loading raw match data from DuckDB.
    2. Encoding the target label.
    3. Engineering model-ready features (date, differential, and odds
       features).
    4. Filtering invalid targets and making a chronological train/test split.
    5. Producing an identical, reproducible train/test split that every
       downstream model script consumes.
    6. A handful of small, model-agnostic utilities (metric computation,
       class-weight optimisation, and generic DuckDB persistence) that are
       identical across every model and would otherwise be duplicated in
       each training script.

`train_knn.py` and `train_gbc.py` both import `prepare_datasets()` (and the
shared utilities below) from this module so that every model is trained and
evaluated on exactly the same data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TARGET_COLUMN = "result"

LABEL_MAP: Dict[str, int] = {"A": 0, "D": 1, "H": 2}
REVERSE_LABEL_MAP: Dict[int, str] = {0: "Away Win", 1: "Draw", 2: "Home Win"}

LEAKAGE_COLUMNS: List[str] = ["home_goals", "away_goals", "result", "status"]
ID_COLUMNS: List[str] = ["fixture_id", "league_id", "round"]

EPSILON = 1e-6
COMPLETED_STATUSES = {"FT", "AET", "PEN"}


@dataclass
class DataConfig:
    """All parameters needed to load, engineer, and split the dataset.

    Kept as a single object so every model script builds its splits from
    identical settings, instead of passing around a long list of arguments.
    """

    db_path: str
    input_table: str
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class DatasetSplits:
    """Container for everything a model-training script needs."""

    raw_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    """Load a table from DuckDB and parse the `date` column if present."""

    with duckdb.connect(db_path) as conn:
        df = conn.execute(f'SELECT * FROM "{table_name}"').df()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("Loaded %d rows from '%s'", len(df), table_name)
    return df


def prepare_target(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    label_map: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Map the raw target labels (e.g. 'H'/'D'/'A') to integer classes."""

    required = {target, "status", "home_goals", "away_goals", "fixture_id", "date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    label_map = LABEL_MAP if label_map is None else label_map
    df = df.copy()
    completed = df["status"].isin(COMPLETED_STATUSES)
    missing_scores = completed & (df["home_goals"].isna() | df["away_goals"].isna())
    if missing_scores.any():
        raise ValueError(
            f"{int(missing_scores.sum())} completed fixtures have missing scores"
        )

    df = df.loc[completed].copy()
    if df.empty:
        raise ValueError("dataset contains no completed fixtures")

    expected_labels = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] < df["away_goals"],
            df["home_goals"] == df["away_goals"],
        ],
        ["H", "A", "D"],
        default="?",
    )
    invalid_scores = pd.isna(expected_labels)
    if invalid_scores.any():
        raise ValueError("completed fixtures contain invalid goal values")
    inconsistent = df[target].to_numpy() != expected_labels
    if inconsistent.any():
        raise ValueError(
            f"{int(inconsistent.sum())} target labels disagree with completed scores"
        )

    df[target] = df[target].map(label_map)
    if df[target].isna().any():
        raise ValueError("Unknown target values found after label mapping.")
    if df["fixture_id"].duplicated().any():
        raise ValueError("dataset contains duplicate fixture IDs")
    if df["date"].isna().any():
        raise ValueError("completed fixtures have invalid or missing dates")

    return df.sort_values(["date", "fixture_id"]).copy()


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================

def engineer_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the model-ready feature matrix `X` and target vector `y`.

    Adds date-cyclical features, home-vs-away differential/ratio features
    for any paired `x_*` / `y_*` columns, and derived odds features.
    """

    drop_columns = drop_columns or (LEAKAGE_COLUMNS + ID_COLUMNS)
    if target not in df:
        raise ValueError(f"dataset is missing target column {target!r}")

    X = df.drop(columns=drop_columns, errors="ignore").copy()
    y = df[target].copy()

    X = _add_date_features(X)
    #X = _add_differential_features(X)
    X = _add_odds_features(X)

    X = X.replace([np.inf, -np.inf], np.nan)
    for column in X.columns:
        if not X[column].isna().any():
            continue
        dtype = X[column].dtype
        if pd.api.types.is_extension_array_dtype(dtype):
            if pd.api.types.is_numeric_dtype(dtype):
                X[column] = X[column].to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
        elif pd.api.types.is_object_dtype(dtype):
            X[column] = X[column].where(X[column].notna(), np.nan)

    return X, y


def _add_date_features(X: pd.DataFrame) -> pd.DataFrame:
    if "date" not in X.columns:
        return X

    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day_of_week"] = X["date"].dt.dayofweek
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    return X.drop(columns=["date"])


def _add_differential_features(X: pd.DataFrame) -> pd.DataFrame:
    """For every numeric `x_<suffix>` column with a matching `y_<suffix>`
    column, add a difference, ratio, and log-ratio feature."""

    x_columns = [c for c in X.columns if c.startswith("x_")]
    new_columns: Dict[str, pd.Series] = {}

    for x_col in x_columns:
        suffix = x_col[2:]
        y_col = f"y_{suffix}"

        if y_col not in X.columns or not pd.api.types.is_numeric_dtype(X[x_col]):
            continue

        x_val = pd.to_numeric(X[x_col], errors="coerce")
        y_val = pd.to_numeric(X[y_col], errors="coerce")

        new_columns[f"diff_{suffix}"] = x_val - y_val
        new_columns[f"ratio_{suffix}"] = (
            (x_val + EPSILON) / (y_val + EPSILON)
        ).clip(0.05, 20.0)
        new_columns[f"logratio_{suffix}"] = np.log1p(np.maximum(x_val, 0)) - np.log1p(
            np.maximum(y_val, 0)
        )

    if not new_columns:
        return X.copy()

    return pd.concat([X, pd.DataFrame(new_columns, index=X.index)], axis=1)


def _add_odds_features(X: pd.DataFrame) -> pd.DataFrame:
    required = {"home_win_odds", "draw_odds", "away_win_odds"}
    if not required.issubset(X.columns):
        return X

    X = X.copy()
    home = pd.to_numeric(X["home_win_odds"], errors="coerce")
    draw = pd.to_numeric(X["draw_odds"], errors="coerce")
    away = pd.to_numeric(X["away_win_odds"], errors="coerce")

    X["odds_home_away_diff"] = home - away
    X["odds_home_away_ratio"] = ((home + EPSILON) / (away + EPSILON)).clip(0.05, 20.0)
    X["odds_home_draw_diff"] = home - draw
    X["odds_away_draw_diff"] = away - draw

    return X


def build_feature_preprocessor(
    X: pd.DataFrame, scale_numeric: bool = False
) -> ColumnTransformer:
    """Impute missing values and encode categories consistently for each model."""
    categorical = X.select_dtypes(
        include=["object", "str", "category", "string"]
    ).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    return ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
    )


# ==============================================================================
# 3. PUBLIC ENTRY POINT
# ==============================================================================

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split chronologically without splitting fixtures sharing a kickoff."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    ordered_dates = pd.to_datetime(dates, errors="coerce")
    if ordered_dates.isna().any():
        raise ValueError("fixture dates cannot be missing or invalid")
    if len(X) < 2:
        raise ValueError("at least two completed fixtures are required")

    split_position = int(np.floor(len(X) * (1 - test_size)))
    if split_position <= 0 or split_position >= len(X):
        raise ValueError("test_size does not leave both train and test fixtures")
    test_start = ordered_dates.sort_values().iloc[split_position]
    train_mask = ordered_dates < test_start
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("chronological split produced an empty train or test set")

    expected_classes = set(LABEL_MAP.values())
    if set(y.loc[train_mask]) != expected_classes:
        raise ValueError("chronological training split does not contain all outcome classes")
    if set(y.loc[test_mask]) != expected_classes:
        raise ValueError(
            "chronological test split does not contain all outcome classes; "
            "adjust test_size or include more seasons"
        )
    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask]

def prepare_datasets(config: DataConfig) -> DatasetSplits:
    """Run the full data-preparation pipeline and return a reproducible
    train/test split for consumption by any model-training script.

    Every model script MUST call this with the same `DataConfig` (same
    `random_state`) to guarantee they train and evaluate on identical data.
    """

    df = load_data(config.db_path, config.input_table)
    df = prepare_target(df)

    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = chronological_split(
        X, y, df["date"], test_size=config.test_size
    )

    return DatasetSplits(
        raw_df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


# ==============================================================================
# 4. SHARED MODEL-AGNOSTIC UTILITIES
# ==============================================================================

def optimize_class_weights(
    y_true: pd.Series,
    y_proba: np.ndarray,
    draw_range: Tuple[float, float, int] = (1.0, 2.5, 16),
    side_range: Tuple[float, float, int] = (0.8, 1.2, 5),
) -> np.ndarray:
    """Grid-search per-class prediction multipliers that maximise macro F1."""

    best_f1 = -1.0
    best_weights = np.array([1.0, 1.0, 1.0])

    for w_draw in np.linspace(*draw_range):
        for w_away in np.linspace(*side_range):
            for w_home in np.linspace(*side_range):
                weights = np.array([w_away, w_draw, w_home])
                preds = np.argmax(y_proba * weights, axis=1)
                score = f1_score(y_true, preds, average="macro", zero_division=0)

                if score > best_f1:
                    best_f1 = score
                    best_weights = weights

    return best_weights


def optimize_class_weights_oof(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> np.ndarray:
    """Tune probability multipliers on out-of-fold training predictions."""
    probabilities = cross_val_predict(
        estimator, X, y, cv=cv, method="predict_proba", n_jobs=1
    )
    return optimize_class_weights(y, probabilities)


def compute_classification_metrics(
    y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Compute the standard set of evaluation metrics used for every model."""

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "log_loss": log_loss(y_test, y_proba, labels=[0, 1, 2]),
        "roc_auc_ovr": roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        ),
    }


def compute_param_grid_size(param_grid: Dict[str, list]) -> int:
    """Total number of hyperparameter combinations a grid represents."""

    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def build_hyperparameter_search(
    pipeline,
    param_grid: Dict[str, list],
    cv,
    scoring: str,
    n_iter: int,
    random_state: int,
    n_jobs: int = -1,
) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Pick the appropriate hyperparameter search strategy for a grid."""

    grid_size = compute_param_grid_size(param_grid)

    if grid_size <= n_iter:
        logger.info(
            "Parameter grid has %d combination(s) (<= n_iter=%d); "
            "using exhaustive GridSearchCV.",
            grid_size,
            n_iter,
        )
        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )

    logger.info(
        "Parameter grid has %d combinations; using RandomizedSearchCV "
        "with n_iter=%d.",
        grid_size,
        n_iter,
    )
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def persist_dataframe(
    db_path: str,
    table_name: str,
    df: pd.DataFrame,
    if_not_exists: bool = True,
    append: bool = True,
) -> None:
    """Append `df` to a DuckDB table, creating it first if needed."""

    with duckdb.connect(db_path) as conn:
        if if_not_exists:
            conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" AS '
                "SELECT * FROM df WHERE 1=0"
            )
        if not append:
            conn.execute(f'TRUNCATE TABLE "{table_name}"')
            conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM df')
        else:
            conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM df')

    logger.info("Wrote %d row(s) to '%s'", len(df), table_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_config = DataConfig(
        db_path="../../storage/database/datasets/France.ddb",
        input_table="61-training-dataset",
    )

    splits = prepare_datasets(demo_config)

    logger.info("Training samples: %d", len(splits.X_train))
    logger.info("Test samples: %d", len(splits.X_test))
    logger.info("Feature count: %d", splits.X_train.shape[1])