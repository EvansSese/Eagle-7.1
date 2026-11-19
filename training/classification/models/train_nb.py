"""
train_nb.py
===========
Trains a Gaussian Naive Bayes classifier on the football match dataset.

Consumes the shared `DataConfig` / `prepare_datasets()` from
`preprocess_dataset.py` so it trains and evaluates on the exact same
train/test split as other classification[cite: 4]. Saves:
    - The fitted pipeline (joblib)[cite: 4].
    - Best hyperparameters + optimized class multipliers (DuckDB)[cite: 4].
    - Evaluation metrics (DuckDB)[cite: 4].
    - Per-fixture test-set predictions (DuckDB)[cite: 4].
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

from classification.preprocess_dataset import (
    DataConfig,
    REVERSE_LABEL_MAP,
    build_feature_preprocessor,
    build_hyperparameter_search,
    compute_classification_metrics,
    optimize_class_weights_oof,
    persist_dataframe,
    prepare_datasets,
)
from core.get_league_details import get_league_details

logger = logging.getLogger(__name__)

MODEL_NAME = "NB"

NB_PARAM_GRID = {
    "model__var_smoothing": [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0],
}


@dataclass
class TrainingConfig:
    """Settings specific to the Naive Bayes training run."""

    data_config: DataConfig
    n_iter: int = 10
    random_state: int = 42
    model_dir: Path = Path("../../../storage/trained_models/trained_classifier_models/v1/")
    params_table: str = "model-best-params"
    metrics_table: str = "model-evaluation-metrics"
    predictions_table: str = "model-predictions"
    model_filename: str = "nb.pkl"


def build_nb_pipeline(X_train: pd.DataFrame, random_state: int) -> ImbPipeline:
    """Gaussian Naive Bayes pipeline with one-hot encoded categorical features."""

    preprocessor = build_feature_preprocessor(X_train)

    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", GaussianNB()),
        ]
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int,
    random_state: int,
) -> Dict[str, Any]:
    """Hyperparameter search, class-weight optimisation, and evaluation[cite: 4]."""

    pipeline = build_nb_pipeline(X_train, random_state)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = build_hyperparameter_search(
        pipeline=pipeline,
        param_grid=NB_PARAM_GRID,
        cv=skf,
        scoring="neg_log_loss",
        n_iter=n_iter,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    multipliers = optimize_class_weights_oof(best_model, X_train, y_train, skf)

    y_proba = best_model.predict_proba(X_test)
    y_pred = np.argmax(y_proba * multipliers, axis=1)

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    return {
        "model_name": MODEL_NAME,
        "best_model": best_model,
        "best_params": search.best_params_,
        "best_cv_score": abs(search.best_score_),
        "multipliers": multipliers,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def save_params(db_path: str, table_name: str, result: Dict[str, Any]) -> None:
    clean_params = {k.replace("model__", ""): v for k, v in result["best_params"].items()}

    params_df = pd.DataFrame(
        [
            {
                "created_at": pd.Timestamp.now(),
                "model_name": result["model_name"],
                "best_cv_log_loss": float(result["best_cv_score"]),
                "multiplier_away": float(result["multipliers"][0]),
                "multiplier_draw": float(result["multipliers"][1]),
                "multiplier_home": float(result["multipliers"][2]),
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
                "best_cv_log_loss": float(result["best_cv_score"]),
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
    y_test: pd.Series,
    result: Dict[str, Any],
) -> None:
    preds_df = raw_df.loc[test_index, ["date", "home_team", "away_team"]].copy()
    table_name = table_name + f"-{result['model_name']}"
    preds_df["actual"] = y_test.map(REVERSE_LABEL_MAP).values
    preds_df["predicted"] = [REVERSE_LABEL_MAP[v] for v in result["y_pred"]]
    preds_df["away_prob"] = result["y_proba"][:, 0]
    preds_df["draw_prob"] = result["y_proba"][:, 1]
    preds_df["home_prob"] = result["y_proba"][:, 2]
    preds_df["correct"] = result["y_pred"] == y_test.to_numpy()

    persist_dataframe(db_path, table_name, preds_df, append=False)


def run(config: TrainingConfig) -> Dict[str, Any]:
    logger.info("Preparing dataset...")
    splits = prepare_datasets(config.data_config)
    logger.info("Training samples: %d | Test samples: %d | Features: %d",
                len(splits.X_train), len(splits.X_test), splits.X_train.shape[1])

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
        db_path, config.predictions_table, splits.raw_df, splits.X_test.index,
        splits.y_test, result,
    )

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / config.model_filename
    joblib.dump(result["best_model"], model_path)
    logger.info("Saved trained %s model to %s", MODEL_NAME, model_path)

    return result


def main(db_path: str, input_table: str, league_id: int, model_path:Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_config = DataConfig(
        db_path=db_path,
        input_table=input_table,
        test_size=0.15,
        random_state=42,
        low_var_thresh=0.01,
        corr_thresh=0.95,
    )

    training_config = TrainingConfig(
        data_config=data_config,
        n_iter=100,
        random_state=42,
        model_dir=model_path,
        params_table=f"{league_id}-model-best-params",
        metrics_table=f"{league_id}-model-evaluation-metrics",
        predictions_table=f"{league_id}-model-predictions",
        model_filename=f"{league_id}_nb.pkl",
    )

    run(training_config)


if __name__ == "__main__":
    league_id = 39
    league_details = get_league_details(leagues_db_path="../../../storage/database/leagues.ddb", league_id=league_id)
    if league_details["status"]:
        db_path = f"../../storage/database/datasets/{league_details['league_details']['country_name']}.ddb"
        input_table = f"{league_id}-training-dataset"
        main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=Path(f"../../storage/trained_classifier_models/v2/{league_id}"))
    else:
        print(f"{league_id} League not found")