"""
preprocess_dataset.py
====================
Shared data-loading and feature-engineering module for the football match
goal prediction regression pipeline.

This module is the single source of truth for:
    1. Loading raw match data from DuckDB.
    2. Engineering model-ready date features.
    3. Filtering algorithm-specific feature columns (`keep_columns`).
    4. Setting up home_goals and away_goals regression targets.
    5. Producing an identical, reproducible train/test split.
    6. Deterministic evaluation metrics (MAE, MSE, RMSE, R2) and model persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TARGET_COLUMNS: List[str] = ["home_goals", "away_goals"]
LEAKAGE_COLUMNS: List[str] = ["result", "status"]
ID_COLUMNS: List[str] = ["fixture_id", "league_id", "round", "venue"]

EPSILON = 1e-6


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
    # Specify columns to keep for specific algorithms.
    # If None or empty, all engineered/available feature columns are kept.
    keep_columns: Optional[List[str]] = field(default=None)


@dataclass
class DatasetSplits:
    """Container for everything a model-training script needs."""

    raw_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    """Load a table from DuckDB and parse the `date` column if present."""

    with duckdb.connect(db_path) as conn:
        df = conn.execute(f"SELECT * FROM '{table_name}'").df()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("Loaded %d rows from '%s'", len(df), table_name)
    return df


# ==============================================================================
# 2. FEATURE ENGINEERING & SELECTION
# ==============================================================================

def engineer_features(
    df: pd.DataFrame,
    targets: List[str] = TARGET_COLUMNS,
    drop_columns: Optional[List[str]] = None,
    keep_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the model-ready feature matrix `X` and continuous target DataFrame `y`.

    Adds date-cyclical features and filters features to `keep_columns` if provided.
    """

    drop_columns = drop_columns or (LEAKAGE_COLUMNS + ID_COLUMNS)
    all_drops = list(set(drop_columns + targets))

    X = df.drop(columns=all_drops, errors="ignore").copy()
    y = df[targets].copy()

    X = _add_date_features(X)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Filter to algorithm-specific columns if explicitly configured
    if keep_columns:
        valid_keep_cols = [col for col in keep_columns if col in X.columns]
        missing_cols = set(keep_columns) - set(valid_keep_cols)

        if missing_cols:
            logger.warning("Requested keep_columns not found in feature matrix: %s", missing_cols)

        logger.info("Filtering features to retain %d specified columns.", len(valid_keep_cols))
        X = X[valid_keep_cols]

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


# ==============================================================================
# 3. PUBLIC ENTRY POINT
# ==============================================================================

def prepare_datasets(config: DataConfig) -> DatasetSplits:
    """Run the data-preparation pipeline and return a reproducible train/test split
    for target variables `home_goals` and `away_goals`.
    """

    df = load_data(config.db_path, config.input_table)
    X, y = engineer_features(
        df,
        targets=TARGET_COLUMNS,
        keep_columns=config.keep_columns,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )

    logger.info(
        "Prepared dataset: Train shape %s, Test shape %s. Retained %d features.",
        X_train.shape,
        X_test.shape,
        X_train.shape[1],
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

def compute_regression_metrics(
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.DataFrame, np.ndarray],
) -> Dict[str, float]:
    """Compute deterministic regression evaluation metrics for goal predictions."""

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    mae_home = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_away = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

    mse_home = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_away = mean_squared_error(y_true[:, 1], y_pred[:, 1])

    return {
        "mae_overall": mean_absolute_error(y_true, y_pred),
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mse_overall": mean_squared_error(y_true, y_pred),
        "rmse_home": np.sqrt(mse_home),
        "rmse_away": np.sqrt(mse_away),
        "r2_overall": r2_score(y_true, y_pred),
        "r2_home": r2_score(y_true[:, 0], y_pred[:, 0]),
        "r2_away": r2_score(y_true[:, 1], y_pred[:, 1]),
        "explained_variance": explained_variance_score(y_true, y_pred),
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
    scoring: str = "neg_mean_squared_error",
    n_iter: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Pick hyperparameter search strategy using deterministic regression scoring."""

    grid_size = compute_param_grid_size(param_grid)

    if grid_size <= n_iter:
        logger.info(
            "Parameter grid has %d combination(s) (<= n_iter=%d); using GridSearchCV.",
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
        "Parameter grid has %d combinations; using RandomizedSearchCV with n_iter=%d.",
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
    """Append or overwrite `df` to a DuckDB table."""

    with duckdb.connect(db_path) as conn:
        if if_not_exists:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS '{table_name}' AS "
                "SELECT * FROM df WHERE 1=0"
            )
        if not append:
            conn.execute(f'TRUNCATE TABLE "{table_name}"')
            conn.execute(f"INSERT INTO '{table_name}' SELECT * FROM df")
        else:
            conn.execute(f"INSERT INTO '{table_name}' SELECT * FROM df")

    logger.info("Wrote %d row(s) to '%s'", len(df), table_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db_config = DataConfig(
        db_path="../../storage/database/datasets/France.ddb",
        input_table="61-training-dataset",
        keep_columns=[
            "home_team_rest_days",
            "away_team_rest_days",
            "month_sin",
            "month_cos",
        ],
    )

    splits = prepare_datasets(db_config)

    logger.info("Training samples: %d", len(splits.X_train))
    logger.info("Test samples: %d", len(splits.X_test))
    logger.info("Feature count: %d", splits.X_train.shape[1])
    logger.info("Target columns: %s", list(splits.y_train.columns))