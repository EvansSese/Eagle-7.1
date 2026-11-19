"""
train_random_forest.py
======================
Trains a Random Forest Multi-Output regressor for home_goals and away_goals predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder

from training.regression.preprocess_dataset import (
    DataConfig,
    build_hyperparameter_search,
    compute_regression_metrics,
    persist_dataframe,
    prepare_datasets,
)
from core.get_league_details import get_league_details

logger = logging.getLogger(__name__)

MODEL_NAME = "RandomForest"

RF_PARAM_GRID = {
    "model__estimator__n_estimators": [100, 200, 300],
    "model__estimator__criterion": ["squared_error", "poisson"],
    "model__estimator__max_depth": [4, 6, 8, 12, None],
    "model__estimator__min_samples_split": [2, 5, 10],
    "model__estimator__min_samples_leaf": [1, 2, 4],
    "model__estimator__max_features": ["sqrt", "log2", 1.0],
}


@dataclass
class TrainingConfig:
    data_config: DataConfig
    n_iter: int = 10
    random_state: int = 42
    model_dir: Path = Path("../../../storage/trained_models/trained_regressor_models/v1/")
    params_table: str = "model-best-params"
    metrics_table: str = "model-evaluation-metrics"
    predictions_table: str = "model-predictions"
    model_filename: str = "random_forest.pkl"


def build_rf_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_iter: int,
    random_state: int,
) -> Dict[str, Any]:
    y_train_str = y_train.astype(str).agg("-".join, axis=1)

    rus = RandomUnderSampler(random_state=random_state)
    _, _ = rus.fit_resample(X_train, y_train_str)

    resampled_indices = rus.sample_indices_
    X_train_resampled = X_train.iloc[resampled_indices]
    y_train_resampled = y_train.iloc[resampled_indices]

    preprocessor = build_rf_pipeline(X_train_resampled)
    rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("model", MultiOutputRegressor(rf_base)),
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    search = build_hyperparameter_search(
        pipeline=pipeline,
        param_grid=RF_PARAM_GRID,
        cv=kf,
        scoring="neg_mean_squared_error",
        n_iter=n_iter,
        random_state=random_state,
    )

    search.fit(X_train_resampled, y_train_resampled)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)

    return {
        "model_name": MODEL_NAME,
        "best_model": best_model,
        "best_params": search.best_params_,
        "best_cv_score": abs(search.best_score_),
        "y_pred": y_pred,
        "metrics": metrics,
    }


def save_params(db_path: str, table_name: str, result: Dict[str, Any]) -> None:
    clean_params = {
        k.replace("model__estimator__", ""): v for k, v in result["best_params"].items()
    }

    params_df = pd.DataFrame(
        [
            {
                "created_at": pd.Timestamp.now(),
                "model_name": result["model_name"],
                "best_cv_mse": float(result["best_cv_score"]),
                "hyperparameters_json": json.dumps(clean_params),
            }
        ]
    )
    persist_dataframe(db_path, table_name, params_df)


def save_metrics(db_path: str, table_name: str, result: Dict[str, Any]) -> None:
    metrics = result["metrics"]

    metrics_df = pd.DataFrame(
        [
            {
                "created_at": pd.Timestamp.now(),
                "model_name": result["model_name"],
                **{k: float(v) for k, v in metrics.items()},
                "best_cv_mse": float(result["best_cv_score"]),
                "test_samples": len(result["y_pred"]),
            }
        ]
    )
    persist_dataframe(db_path, table_name, metrics_df)


def save_predictions(
    db_path: str,
    table_name: str,
    raw_df: pd.DataFrame,
    test_index: pd.Index,
    y_test: pd.DataFrame,
    result: Dict[str, Any],
) -> None:
    preds_df = raw_df.loc[test_index, ["date", "home_team", "away_team"]].copy()
    table_name = f"{table_name}-{result['model_name']}"

    preds_df["actual_home_goals"] = y_test["home_goals"].values
    preds_df["actual_away_goals"] = y_test["away_goals"].values
    preds_df["pred_home_goals"] = result["y_pred"][:, 0]
    preds_df["pred_away_goals"] = result["y_pred"][:, 1]

    persist_dataframe(db_path, table_name, preds_df, append=False)


def run(config: TrainingConfig) -> Dict[str, Any]:
    logger.info("Preparing dataset...")
    splits = prepare_datasets(config.data_config)

    logger.info("Training %s model...", MODEL_NAME)
    result = train_and_evaluate(
        splits.X_train,
        splits.y_train,
        splits.X_test,
        splits.y_test,
        n_iter=config.n_iter,
        random_state=config.random_state,
    )

    logger.info("%s evaluation metrics:", MODEL_NAME)
    for metric, value in result["metrics"].items():
        logger.info("  %-20s: %.4f", metric, value)

    db_path = config.data_config.db_path
    save_params(db_path, config.params_table, result)
    save_metrics(db_path, config.metrics_table, result)
    save_predictions(
        db_path,
        config.predictions_table,
        splits.raw_df,
        splits.X_test.index,
        splits.y_test,
        result,
    )

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / config.model_filename
    joblib.dump(result["best_model"], model_path)
    logger.info("Saved trained %s model to %s", MODEL_NAME, model_path)

    return result


def main(db_path: str, input_table: str, league_id: int, model_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_config = DataConfig(
        db_path=db_path,
        input_table=input_table,
        test_size=0.15,
        random_state=42,
        keep_columns=[
        ],
    )

    training_config = TrainingConfig(
        data_config=data_config,
        n_iter=150,
        random_state=42,
        model_dir=model_path,
        params_table=f"{league_id}-regressor-model-best-params",
        metrics_table=f"{league_id}-regressor-model-evaluation-metrics",
        predictions_table=f"{league_id}-regressor-model-predictions",
        model_filename=f"{league_id}_random_forest.pkl",
    )

    run(training_config)


if __name__ == "__main__":
    league_id = 39
    league_details = get_league_details(
        leagues_db_path="../../../storage/database/leagues.ddb", league_id=league_id
    )
    if league_details["status"]:
        db_path = f"../../../storage/database/datasets/{league_details['league_details']['country_name']}.ddb"
        input_table = f"{league_id}-training-dataset"
        main(
            db_path=db_path,
            input_table=input_table,
            league_id=league_id,
            model_path=Path(f"../../../storage/trained_models/trained_regressor_models/v2/{league_id}"),
        )
    else:
        print(f"{league_id} League not found")