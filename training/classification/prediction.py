"""
predict.py
==========
Generates predictions for unseen match fixtures.

Processes features using `preprocess_dataset.py`, loads a saved joblib model
pipeline, fetches optimized class multipliers from DuckDB, and persists final
fixture predictions into DuckDB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import time

from classification.preprocess_dataset import (
    REVERSE_LABEL_MAP,
    engineer_features,
    load_data,
    persist_dataframe,
)
from core.get_league_details import get_league_details

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """Settings specific to the inference pipeline."""

    db_path: str
    prediction_table: str
    model_path: Path
    params_table: str
    output_table: str
    model_name: str


def load_class_multipliers(db_path: str, params_table: str, model_name: str) -> np.ndarray:
    """Fetch latest class multipliers for class weight balancing from DuckDB."""
    default_multipliers = np.array([1.0, 1.0, 1.0])
    try:
        with duckdb.connect(db_path) as conn:
            query = f"""
                SELECT multiplier_away, multiplier_draw, multiplier_home
                FROM '{params_table}'
                WHERE UPPER(model_name) = UPPER(?)
                ORDER BY created_at DESC
                LIMIT 1
            """
            res = conn.execute(query, [model_name]).fetchone()
            if res:
                logger.info("Loaded class multipliers for %s: %s", model_name, res)
                return np.array(res, dtype=float)
    except Exception as exc:
        logger.warning(
            "Could not retrieve multipliers from '%s' (%s). Using default [1.0, 1.0, 1.0].",
            params_table,
            exc,
        )

    return default_multipliers


def prepare_prediction_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering on raw prediction fixtures."""
    df = raw_df.copy()
    # Provide dummy target if result is unlabelled in prediction dataset
    if "result" not in df.columns:
        df["result"] = 0

    X, _ = engineer_features(df)
    return X


def predict_fixtures(
    raw_df: pd.DataFrame,
    model: Any,
    multipliers: np.ndarray,
) -> pd.DataFrame:
    """Apply feature engineering, generate predictions, and format output DataFrame."""
    X = prepare_prediction_features(raw_df)

    y_proba = model.predict_proba(X)
    adjusted_proba = y_proba * multipliers
    y_pred = np.argmax(adjusted_proba, axis=1)

    meta_cols = [c for c in ["date", "home_team", "away_team", "fixture_id"] if c in raw_df.columns]
    preds_df = raw_df[meta_cols].copy() if meta_cols else pd.DataFrame(index=raw_df.index)

    preds_df["predicted"] = [REVERSE_LABEL_MAP[v] for v in y_pred]
    preds_df["away_prob"] = y_proba[:, 0]
    preds_df["draw_prob"] = y_proba[:, 1]
    preds_df["home_prob"] = y_proba[:, 2]
    preds_df["predicted_at"] = pd.Timestamp.now()

    return preds_df


def run_prediction(config: PredictionConfig) -> pd.DataFrame:
    """Execute prediction pipeline end-to-end."""
    logger.info("Loading prediction dataset from '%s'...", config.prediction_table)
    raw_df = load_data(config.db_path, config.prediction_table)
    if raw_df.empty:
        logger.warning("No records found in table '%s'.", config.prediction_table)
        return pd.DataFrame()
    upcoming_statuses = {"NS", "TBD"}
    raw_df = raw_df[
        raw_df["status"].isin(upcoming_statuses)
        & (raw_df["date"] >= pd.Timestamp.now())
    ].copy()
    if raw_df.empty:
        logger.info("No upcoming fixtures found in '%s'.", config.prediction_table)
        return pd.DataFrame()

    logger.info("Loading trained model pipeline from '%s'...", config.model_path)
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model file not found at {config.model_path}")
    model = joblib.load(config.model_path)

    multipliers = load_class_multipliers(
        db_path=config.db_path,
        params_table=config.params_table,
        model_name=config.model_name,
    )

    logger.info("Predicting outcomes for %d fixtures...", len(raw_df))
    preds_df = predict_fixtures(raw_df, model, multipliers)

    logger.info("Saving predictions to DuckDB table '%s'...", config.output_table)
    persist_dataframe(
        db_path=config.db_path,
        table_name=config.output_table,
        df=preds_df,
        append=False,
    )

    logger.info("Successfully generated and stored %d predictions.", len(preds_df))
    return preds_df

def main(db_path: str, league_id: int, model_name: str = "GBC") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    prediction_table = f"{league_id}-prediction-dataset"
    model_name_clean = model_name.upper()
    model_filename = f"{league_id}_{model_name.lower()}.pkl"

    config = PredictionConfig(
        db_path=db_path,
        prediction_table=prediction_table,
        model_path=Path(f"../../storage/trained_classifier_models/v2/{league_id}") / model_filename,
        params_table=f"{league_id}-model-best-params",
        output_table=f"{league_id}-upcoming-predictions-{model_name_clean}",
        model_name=model_name_clean,
    )

    run_prediction(config)


def combine_model_predictions(league_id: str | int, db_path: str, models: list[str] = None):
    """LEFT JOINs specific model prediction tables using fixture_id and date,

    then computes average probabilities into a combined table in DuckDB.
    """
    if models is None:
        models = ["KNN", "DT", "GBC", "NB", "LR", "SVM", "RF"]

    target_table = f"{league_id}-upcoming-predictions-combined"
    tables = [f"{league_id}-upcoming-predictions-{m.upper()}" for m in models]

    # 1. Base metadata from the first model table (t0)
    base_cols = [
        "t0.fixture_id",
        "t0.date",
        "t0.home_team",
        "t0.away_team"
    ]

    # 2. Probability arrays for NULL-safe average computation
    away_prob_list = [f"t{i}.away_prob" for i in range(len(tables))]
    draw_prob_list = [f"t{i}.draw_prob" for i in range(len(tables))]
    home_prob_list = [f"t{i}.home_prob" for i in range(len(tables))]

    avg_cols = [
        f"list_avg([{', '.join(away_prob_list)}]) AS away_prob_avg",
        f"list_avg([{', '.join(draw_prob_list)}]) AS draw_prob_avg",
        f"list_avg([{', '.join(home_prob_list)}]) AS home_prob_avg",
    ]

    select_clause = ",\n    ".join(base_cols + avg_cols)

    # 3. Construct base table and LEFT JOIN clauses
    from_clause = f'"{tables[0]}" t0'
    join_clauses = [
        f'LEFT JOIN "{tables[i]}" t{i} ON t0.fixture_id = t{i}.fixture_id AND t0.date = t{i}.date'
        for i in range(1, len(tables))
    ]
    joins_str = "\n".join(join_clauses)

    # 4. Create target table
    sql_query = f"""
        CREATE OR REPLACE TABLE "{target_table}" AS
        WITH averages AS (
            SELECT 
                {select_clause}
            FROM {from_clause}
            {joins_str}
        )
        SELECT 
            *,
            CASE 
                WHEN home_prob_avg >= away_prob_avg AND home_prob_avg >= draw_prob_avg THEN 'Home Win'
                WHEN away_prob_avg >= home_prob_avg AND away_prob_avg >= draw_prob_avg THEN 'Away Win'
                ELSE 'Draw'
            END AS combined_prediction
        FROM averages;
        """

    with duckdb.connect(db_path) as conn:
        conn.execute(sql_query)
    print(f"Successfully generated table '{target_table}'.")


if __name__ == "__main__":
    league_ids = [89] # 119,207,208,61,62,89,140,141,135,136,78,357,110,39,113,94,122
    model_types = ["gbc", "knn", "rf", "dt", "nb", "lr", "svm"]  # Options: "gbc", "knn", "rf", "dt", "nb", "lr", "svm"

    for league_id in league_ids:
        league_details = get_league_details(leagues_db_path="../../storage/database/leagues.ddb", league_id=league_id)
        if league_details["status"]:
            db_path = f"../storage/database/datasets/{league_details['league_details']['country_name']}.ddb"
            for model_type in model_types:
                main(db_path=db_path, league_id=league_id, model_name=model_type)
                time.sleep(15)
            combine_model_predictions(db_path=db_path, league_id=league_id, models=model_types)
        else:
            print(f"League {league_id} not found.")