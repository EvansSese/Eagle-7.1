"""Predict upcoming fixture scores with feature-group-specific models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import duckdb
import joblib
import pandas as pd
from core.database import connect_existing_database

from training.regression.train_goal_models import (
    ROOT,
    aggregate_outcomes,
    double_chance_labels,
    engineer_features,
    outcome_labels,
    poisson_over_under_predictions,
    poisson_scoreline_predictions,
    rounded_scores,
)

LOGGER = logging.getLogger("predict_upcoming_fixtures")


def _read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    with connect_existing_database(db_path) as conn:
        return conn.execute(f'SELECT * FROM "{table_name}"').df()


def _write_table(db_path: Path, table_name: str, df: pd.DataFrame) -> None:
    with duckdb.connect(str(db_path)) as conn:
        conn.register("prediction_frame", df)
        try:
            conn.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                "SELECT * FROM prediction_frame"
            )
        finally:
            conn.unregister("prediction_frame")


def _select_feature_group_models(
    db_path: Path,
    league_id: int,
) -> list[dict]:
    prefix = f"{league_id}-goal-regression"
    table_name = f"{prefix}-feature-group-models"
    selected = _read_table(db_path, table_name)
    required = {
        "feature_bundle",
        "model",
        "features_json",
        "validation_outcome_accuracy",
    }
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(
            f"{table_name!r} is missing columns: {sorted(missing)}"
        )
    if selected.empty:
        raise ValueError(f"{table_name!r} contains no selected feature-group models")
    if selected["feature_bundle"].duplicated().any():
        raise ValueError(f"{table_name!r} contains duplicate feature groups")

    models = []
    for row in selected.to_dict(orient="records"):
        try:
            features = json.loads(row["features_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid selected features for group {row['feature_bundle']!r}"
            ) from exc
        if not isinstance(features, list) or not features or not all(
            isinstance(feature, str) for feature in features
        ):
            raise ValueError(
                f"selected features for group {row['feature_bundle']!r} are invalid"
            )
        models.append(
            {
                "feature_bundle": str(row["feature_bundle"]),
                "model_name": str(row["model"]),
                "features": features,
                "validation_accuracy": float(row["validation_outcome_accuracy"]),
            }
        )
    return models


def predict_upcoming_fixtures(
    db_path: Path,
    league_id: int,
    season: int,
    input_table: Optional[str] = None,
    model_dir: Optional[Path] = None,
    now: Optional[pd.Timestamp] = None,
    fixtures_db_path: Optional[Path] = None,
    fixtures_table: Optional[str] = None,
) -> pd.DataFrame:
    """Predict future NS fixtures with each feature group's best model."""
    input_table = input_table or f"{league_id}-prediction-dataset"
    fixtures_db_path = fixtures_db_path or (
        db_path.parent.parent / "leagues" / db_path.name
    )
    fixtures_table = fixtures_table or f"{league_id}-{season}-fixtures"
    output_table = f"{league_id}-{season}-upcoming_fixtures_predictions"
    model_dir = model_dir or ROOT / "storage/trained_regressor_models/v3" / str(league_id)
    if not db_path.is_file():
        raise FileNotFoundError(f"dataset database does not exist: {db_path}")
    if not fixtures_db_path.is_file():
        raise FileNotFoundError(f"fixtures database does not exist: {fixtures_db_path}")

    fixture_records = _read_table(fixtures_db_path, fixtures_table)
    required = {"fixture_id", "date", "status", "home_team", "away_team"}
    missing = required.difference(fixture_records.columns)
    if missing:
        raise ValueError(
            f"{fixtures_table!r} is missing columns: {sorted(missing)}"
        )
    fixture_records["date"] = pd.to_datetime(
        fixture_records["date"], errors="coerce"
    )
    if fixture_records["date"].isna().any():
        raise ValueError(f"{fixtures_table!r} contains fixtures with invalid dates")
    if fixture_records["fixture_id"].duplicated().any():
        raise ValueError(f"{fixtures_table!r} contains duplicate fixture IDs")

    current_time = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    upcoming = fixture_records.loc[
        fixture_records["status"].eq("NS")
        & (fixture_records["date"] >= current_time)
    ].copy()
    if upcoming.empty:
        raise ValueError(f"{fixtures_table!r} contains no future NS fixtures")

    feature_rows = _read_table(db_path, input_table)
    if "fixture_id" not in feature_rows.columns:
        raise ValueError(f"{input_table!r} is missing required column 'fixture_id'")
    if feature_rows["fixture_id"].duplicated().any():
        raise ValueError(f"{input_table!r} contains duplicate fixture IDs")

    metadata_columns = {
        "league_id", "fixture_id", "date", "venue", "round", "home_team",
        "away_team", "home_goals", "away_goals", "result", "status",
    }
    feature_columns = [
        column for column in feature_rows.columns
        if column not in metadata_columns
    ]
    missing_features = set(upcoming["fixture_id"]).difference(
        feature_rows["fixture_id"]
    )
    if missing_features:
        raise ValueError(
            f"{input_table!r} is missing feature rows for "
            f"{len(missing_features)} future NS fixtures"
        )
    upcoming = upcoming.merge(
        feature_rows[["fixture_id", *feature_columns]],
        on="fixture_id",
        how="left",
        validate="one_to_one",
    )

    X = engineer_features(upcoming)
    group_models = _select_feature_group_models(db_path, league_id)
    group_predictions = {}
    for group_model in group_models:
        bundle_name = group_model["feature_bundle"]
        features = group_model["features"]
        missing_model_features = set(features).difference(X.columns)
        if missing_model_features:
            raise ValueError(
                f"feature group {bundle_name!r} requires missing features: "
                f"{sorted(missing_model_features)}"
            )
        model_path = model_dir / (
            f"{league_id}_{bundle_name.lower()}_goal_model.joblib"
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"trained model for feature group {bundle_name!r} "
                f"does not exist: {model_path}"
            )
        model = joblib.load(model_path)
        raw_scores = model.predict(X[features])
        rounded = rounded_scores(raw_scores)
        group_predictions[bundle_name] = {
            "model_name": group_model["model_name"],
            "validation_accuracy": group_model["validation_accuracy"],
            "raw_scores": raw_scores,
            "scores": rounded,
            "outcomes": outcome_labels(rounded),
        }

    predictions = upcoming[
        ["fixture_id", "date", "home_team", "away_team", "status"]
    ].copy()
    predictions.insert(0, "league_id", int(league_id))
    predictions.insert(1, "season", int(season))
    group_outcomes = {}
    validation_accuracies = {}
    for bundle_name, result in group_predictions.items():
        suffix = bundle_name.lower()
        predictions[f"model_{suffix}"] = result["model_name"]
        predictions[f"predicted_home_goals_{suffix}"] = result["scores"][:, 0]
        predictions[f"predicted_away_goals_{suffix}"] = result["scores"][:, 1]
        predictions[f"predicted_home_goals_raw_{suffix}"] = result["raw_scores"][:, 0]
        predictions[f"predicted_away_goals_raw_{suffix}"] = result["raw_scores"][:, 1]
        predictions[f"predicted_outcome_{suffix}"] = result["outcomes"]
        group_outcomes[bundle_name] = result["outcomes"]
        validation_accuracies[bundle_name] = result["validation_accuracy"]

    all_scores = [result["scores"] for result in group_predictions.values()]
    all_raw_scores = [result["raw_scores"] for result in group_predictions.values()]
    mean_scores = rounded_scores(sum(all_scores) / len(all_scores))
    predictions["predicted_home_goals"] = mean_scores[:, 0]
    predictions["predicted_away_goals"] = mean_scores[:, 1]
    predictions["predicted_home_goals_raw"] = (
        sum(all_raw_scores) / len(all_raw_scores)
    )[:, 0]
    predictions["predicted_away_goals_raw"] = (
        sum(all_raw_scores) / len(all_raw_scores)
    )[:, 1]
    poisson_scores = poisson_scoreline_predictions(
        {
            bundle_name: result["raw_scores"]
            for bundle_name, result in group_predictions.items()
        },
        validation_accuracies,
    )
    predictions["predicted_home_goals_poisson"] = poisson_scores[:, 0]
    predictions["predicted_away_goals_poisson"] = poisson_scores[:, 1]
    predictions["predicted_outcome_poisson"] = outcome_labels(poisson_scores)
    for column, values in poisson_over_under_predictions(
        {
            bundle_name: result["raw_scores"]
            for bundle_name, result in group_predictions.items()
        },
        validation_accuracies,
    ).items():
        predictions[column] = values

    overall_outcomes, vote_counts, weighted_vote_totals = aggregate_outcomes(
        group_outcomes,
        validation_accuracies,
    )
    predictions["home_outcome_votes"] = vote_counts["H"]
    predictions["draw_outcome_votes"] = vote_counts["D"]
    predictions["away_outcome_votes"] = vote_counts["A"]
    predictions["home_outcome_weighted_votes"] = weighted_vote_totals["H"]
    predictions["draw_outcome_weighted_votes"] = weighted_vote_totals["D"]
    predictions["away_outcome_weighted_votes"] = weighted_vote_totals["A"]
    mean_score_outcomes = outcome_labels(mean_scores)
    predictions["predicted_outcome"] = mean_score_outcomes
    predictions["predicted_outcome_weighted_vote"] = overall_outcomes
    predictions["predicted_double_chance"] = double_chance_labels(
        weighted_vote_totals,
        mean_score_outcomes,
    )
    predictions["predicted_at"] = pd.Timestamp.now()

    _write_table(db_path, output_table, predictions)
    LOGGER.info(
        "Saved %d upcoming fixture predictions to %s in %s",
        len(predictions),
        output_table,
        db_path,
    )
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    LEAGUE_ID = 39
    SEASON = 2026
    DB_PATH = ROOT / "storage/database/datasets/England.ddb"
    FIXTURES_DB_PATH = ROOT / "storage/database/leagues/England.ddb"
    INPUT_TABLE = f"{LEAGUE_ID}-prediction-dataset"
    FIXTURES_TABLE = f"{LEAGUE_ID}-{SEASON}-fixtures"
    MODEL_DIR = ROOT / "storage/trained_regressor_models/v3" / str(LEAGUE_ID)

    predictions = predict_upcoming_fixtures(
        db_path=DB_PATH,
        league_id=LEAGUE_ID,
        season=SEASON,
        input_table=INPUT_TABLE,
        fixtures_db_path=FIXTURES_DB_PATH,
        fixtures_table=FIXTURES_TABLE,
        model_dir=MODEL_DIR,
    )
    print(predictions.to_string(index=False))
