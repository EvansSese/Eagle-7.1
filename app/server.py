import json
import math
import os
import re
import threading
import uuid
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import duckdb

from core.database import connect_existing_database
from api.fetch.fixtures import fetch_fixtures
from app.features.leagues import controller as leagues_controller
from app.features.leagues.model import get_leagues as _get_leagues
from app.features.predictions.holdouts import read_test_fixture_ids
from storage.import_fixtures import import_fixtures


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = PROJECT_ROOT / "storage" / "database" / "leagues.ddb"
FIXTURE_FILES = PROJECT_ROOT / "api" / "json" / "fixtures"
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
FETCH_JOBS: dict[str, dict] = {}
FETCH_LOCK = threading.Lock()
FETCH_JOBS_LOCK = threading.Lock()
FAVORITES_LOCK = threading.RLock()
MODEL_JOBS: dict[str, dict] = {}
MODEL_LOCK = FETCH_LOCK
MODEL_JOBS_LOCK = threading.Lock()
MODEL_WORKFLOW_ACTIONS = (
    "fetch", "process", "prepare", "train-classifier", "train-regressor",
    "train-neural",
)
PREDICTION_CACHE: dict[tuple[int, int], tuple[tuple, dict[int, dict]]] = {}
FAVORITES_TABLE = "eagle_favorite_fixtures"
FAVORITE_LISTS_TABLE = "eagle_favorite_lists"
FAVORITE_LIST_FIXTURES_TABLE = "eagle_favorite_list_fixtures"
FAVORITE_MIGRATIONS_TABLE = "eagle_favorite_migrations"


def get_database_path() -> Path:
    return Path(os.environ.get("EAGLE_LEAGUES_DB", DEFAULT_DATABASE)).expanduser()


def get_country_database_path(country_name: str, database_path: Path) -> Path:
    country_directory = Path(
        os.environ.get("EAGLE_COUNTRY_DATABASES", database_path.parent / "leagues")
    ).expanduser().resolve()
    country_database = (country_directory / f"{country_name}.ddb").resolve()
    if country_database.parent != country_directory:
        raise ValueError("Invalid country database path")
    return country_database


def get_dataset_database_path(country_name: str, database_path: Path) -> Path:
    dataset_directory = Path(
        os.environ.get(
            "EAGLE_DATASET_DATABASES",
            database_path.parent / "datasets",
        )
    ).expanduser().resolve()
    dataset_database = (dataset_directory / f"{country_name}.ddb").resolve()
    if dataset_database.parent != dataset_directory:
        raise ValueError("Invalid dataset database path")
    return dataset_database


def _read_league(league_id: int, database_path: Path = DEFAULT_DATABASE) -> dict | None:
    with duckdb.connect(str(database_path)) as connection:
        row = connection.execute(
            """
            SELECT league_id, league_name, country_name, seasons
            FROM leagues
            WHERE league_id = ?
            """,
            [league_id],
        ).fetchone()
    if row is None:
        return None
    return {
        "league_id": row[0],
        "league_name": row[1],
        "country_name": row[2],
        "seasons": row[3] or [],
    }


def _fixture_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()[0] > 0


def get_seasons(league_id: int, database_path: Path = DEFAULT_DATABASE) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")

    country_database = get_country_database_path(league["country_name"], database_path)
    fetched = set()
    if country_database.is_file():
        with connect_existing_database(country_database) as connection:
            for season in league["seasons"]:
                table_name = f"{league_id}-{int(season)}-fixtures"
                if _fixture_table(connection, table_name) and connection.execute(
                    f'SELECT count(*) FROM "{table_name}"'
                ).fetchone()[0] > 0:
                    fetched.add(int(season))

    seasons = [
        {"season": int(season), "fetched": int(season) in fetched}
        for season in sorted(league["seasons"])
    ]
    return {"league_id": league_id, "seasons": seasons}


def get_rounds(league_id: int, season: int, database_path: Path = DEFAULT_DATABASE) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")

    table_name = f"{league_id}-{season}-fixtures"
    country_database = get_country_database_path(league["country_name"], database_path)
    if not country_database.is_file():
        return {"available": False, "rounds": []}

    with connect_existing_database(country_database) as connection:
        if not _fixture_table(connection, table_name):
            return {"available": False, "rounds": []}
        rounds = connection.execute(
            f"""
            SELECT round
            FROM "{table_name}"
            WHERE round IS NOT NULL
            GROUP BY round
            ORDER BY min(date), lower(round)
            """
        ).fetchall()
    return {"available": True, "rounds": [row[0] for row in rounds]}


def get_fixtures(
    league_id: int,
    season: int,
    round_name: str,
    database_path: Path = DEFAULT_DATABASE,
) -> list[dict]:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")

    table_name = f"{league_id}-{season}-fixtures"
    country_database = get_country_database_path(league["country_name"], database_path)
    if not country_database.is_file():
        return []

    with connect_existing_database(country_database) as connection:
        if not _fixture_table(connection, table_name):
            return []
        column_names = {
            row[0]
            for row in connection.execute(f"DESCRIBE \"{table_name}\"").fetchall()
        }
        home_logo = "home_logo" if "home_logo" in column_names else "NULL AS home_logo"
        away_logo = "away_logo" if "away_logo" in column_names else "NULL AS away_logo"
        cursor = connection.execute(
            f"""
            SELECT fixture_id, date, venue, round, home_team, away_team,
                   home_goals, away_goals, status, {home_logo}, {away_logo}
            FROM "{table_name}"
            WHERE round = ?
            ORDER BY date, fixture_id
            """,
            [round_name],
        )
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def get_season_fixture_groups(
    league_id: int,
    season: int,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")

    table_name = f"{league_id}-{season}-fixtures"
    country_database = get_country_database_path(league["country_name"], database_path)
    if not country_database.is_file():
        return {"available": False, "rounds": []}

    with connect_existing_database(country_database) as connection:
        if not _fixture_table(connection, table_name):
            return {"available": False, "rounds": []}
        column_names = {
            row[0]
            for row in connection.execute(f"DESCRIBE \"{table_name}\"").fetchall()
        }
        home_logo = "home_logo" if "home_logo" in column_names else "NULL AS home_logo"
        away_logo = "away_logo" if "away_logo" in column_names else "NULL AS away_logo"
        cursor = connection.execute(
            f"""
            SELECT fixture_id, date, venue, round, home_team, away_team,
                   home_goals, away_goals, status, {home_logo}, {away_logo}
            FROM "{table_name}"
            ORDER BY date, fixture_id
            """
        )
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()

    grouped_rounds: dict[str, list[dict]] = {}
    for row in rows:
        fixture = dict(zip(columns, row))
        grouped_rounds.setdefault(fixture["round"] or "Other", []).append(fixture)
    return {
        "available": bool(rows),
        "rounds": [
            {"name": name, "fixtures": fixtures}
            for name, fixtures in grouped_rounds.items()
        ],
    }


def get_fixture_details(
    league_id: int,
    season: int,
    fixture_id: int,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")

    prefix = f"{league_id}-{season}"
    fixture_table = f"{prefix}-fixtures"
    country_database = get_country_database_path(league["country_name"], database_path)
    if not country_database.is_file():
        raise ValueError("Fixtures have not been fetched for this season")

    with connect_existing_database(country_database) as connection:
        if not _fixture_table(connection, fixture_table):
            raise ValueError("Fixtures have not been fetched for this season")
        cursor = connection.execute(
            f'SELECT * FROM "{fixture_table}" WHERE fixture_id = ?',
            [fixture_id],
        )
        columns = [column[0] for column in cursor.description]
        row = cursor.fetchone()
        if row is None:
            raise ValueError("Fixture not found")
        fixture = dict(zip(columns, row))

        if not fixture.get("home_logo") or not fixture.get("away_logo"):
            fixture_file = FIXTURE_FILES / str(league_id) / f"{league_id}-{season}-fixtures.json"
            if fixture_file.is_file():
                with fixture_file.open("r", encoding="utf-8") as file:
                    api_fixtures = json.load(file).get("response", [])
                api_fixture = next(
                    (
                        item
                        for item in api_fixtures
                        if item.get("fixture", {}).get("id") == fixture_id
                    ),
                    None,
                )
                if api_fixture:
                    for side in ("home", "away"):
                        logo_column = f"{side}_logo"
                        fixture[logo_column] = fixture.get(logo_column) or (
                            api_fixture.get("teams", {}).get(side, {}).get("logo")
                        )

        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE ? AND table_name != ?
            ORDER BY table_name
            """,
            [f"{prefix}-%", fixture_table],
        ).fetchall()
        stats = []
        for (table_name,) in tables:
            if not re.fullmatch(rf"{re.escape(prefix)}-[a-z0-9-]+", table_name):
                continue
            cursor = connection.execute(
                f'SELECT * FROM "{table_name}" WHERE fixture_id = ?',
                [fixture_id],
            )
            metric_columns = [
                column[0]
                for column in cursor.description
                if column[0] not in {"fixture_id", "league_id", "date", "venue", "round", "home_team", "away_team", "home_goals", "away_goals", "status", "season", "source_table"}
            ]
            metric_row = cursor.fetchone()
            if metric_row is None:
                continue
            metrics = [
                {"name": name, "value": value}
                for name, value in zip(
                    [column[0] for column in cursor.description],
                    metric_row,
                )
                if name in metric_columns and value is not None
            ]
            if metrics:
                stats.append(
                    {
                        "name": table_name.removeprefix(prefix + "-"),
                        "metrics": metrics,
                    }
                )
    return {"fixture": fixture, "stats": stats}


def _prediction_signature(
    dataset_database: Path,
    classifier_model_dir: Path,
    regressor_model_dir: Path,
) -> tuple:
    paths = [
        dataset_database,
        classifier_model_dir / "ensemble.json",
        *regressor_model_dir.glob("*_goal_model.joblib"),
    ]
    signature = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            signature.append((str(path), None))
        else:
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _predict_fixture_rows(
    league_id: int,
    season: int,
    database_path: Path,
) -> dict[int, dict]:
    import pandas as pd

    from training.ensemble_predict import _ensemble_frame, _upcoming_predictions

    league = _read_league(league_id, database_path)
    if league is None or season not in league["seasons"]:
        raise ValueError("Season is not available for this league")
    country_database = get_country_database_path(league["country_name"], database_path)
    dataset_database = get_dataset_database_path(league["country_name"], database_path)
    if not country_database.is_file() or not dataset_database.is_file():
        raise ValueError("Prepare the league dataset and train all models first")
    classifier_dir = PROJECT_ROOT / "storage" / "trained_classifier_models" / "v3" / str(league_id)
    regressor_dir = PROJECT_ROOT / "storage" / "trained_regressor_models" / "v3" / str(league_id)
    mlp_dir = PROJECT_ROOT / "storage" / "trained_deep_learning_models" / "v1" / str(league_id)
    sequence_dir = PROJECT_ROOT / "storage" / "trained_deep_learning_models" / "sequence" / str(league_id)
    paths = [dataset_database, country_database]
    for directory in (classifier_dir, regressor_dir, mlp_dir, sequence_dir):
        paths.extend(sorted(path for path in directory.glob("*") if path.is_file()))
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    cache_key = (league_id, season)
    cached = PREDICTION_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    prediction_table = f"{league_id}-{season}-prediction-dataset"
    with connect_existing_database(dataset_database) as connection:
        if not _has_all_season_dataset(connection, league_id, league["seasons"]) or not _fixture_table(connection, prediction_table):
            raise ValueError("Prepare all seasons in Model Lab before requesting predictions")
        fixtures = connection.execute(f'SELECT * FROM "{prediction_table}"').df()
        test_fixture_ids, missing_test_tables = read_test_fixture_ids(
            connection, league_id, (sequence_dir / "outcome_metadata.json").is_file()
        )
    if fixtures.empty:
        return {}
    if not {"date", "fixture_id", "status"}.issubset(fixtures.columns):
        raise ValueError("The prediction dataset is missing fixture identifiers, dates or statuses")
    dates = pd.to_datetime(fixtures["date"], errors="coerce")
    if (dates.isna() & fixtures["date"].notna()).any():
        raise ValueError("The prediction dataset contains invalid fixture dates")
    try:
        models = _upcoming_predictions(
            league_id, country_database, dataset_database, fixtures,
            classifier_dir, regressor_dir, mlp_dir, sequence_dir, include_all=True,
        )
    except FileNotFoundError as error:
        raise ValueError(f"A trained model artifact is missing; train all models in Model Lab. {error}") from error
    if not all(any(name.startswith(f"{family}_") for name in models) for family in ("classifier", "regressor", "neural")):
        raise ValueError("Train classifiers, regressors, and neural networks before predicting")
    combined = _ensemble_frame(fixtures, models)
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    pairs = {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}
    results = {}
    for row in combined.to_dict(orient="records"):
        probabilities = {code: float(row[f"probability_{key}"]) for code, key in (
            ("H", "home"), ("D", "draw"), ("A", "away"),
        )}
        outcomes = sorted(
            ({"code": code, "label": names[code], "confidence": value}
             for code, value in probabilities.items()),
            key=lambda item: (-item["confidence"], item["code"]),
        )
        chances = sorted(
            ({"code": code, "label": label, "confidence": sum(probabilities[x] for x in covered)}
             for code, label, covered in (
                 ("1X", "Home or draw", pairs["1X"]),
                 ("X2", "Draw or away", pairs["X2"]),
                 ("12", "Home or away", pairs["12"]),
             )),
            key=lambda item: (-item["confidence"], item["code"]),
        )
        home, away = int(row["predicted_home_goals"]), int(row["predicted_away_goals"])
        results[int(row["fixture_id"])] = {
            "summary": f"{outcomes[0]['label']} most likely (all-model ensemble)",
            "confidence": chances[0]["confidence"],
            "is_test": (
                True if int(row["fixture_id"]) in test_fixture_ids
                else None if missing_test_tables else False
            ),
            "classification": {
                "outcomes": outcomes, "double_chance": chances,
                "predicted_outcome": outcomes[0]["code"],
                "predicted_double_chance": chances[0]["code"],
            },
            "regression": {
                "score": f"{home}-{away}", "home_goals": home, "away_goals": away,
                "confidence": None, "models": [],
                "predicted_outcome": "H" if home > away else "A" if away > home else "D",
            },
        }
    if len(PREDICTION_CACHE) >= 8:
        PREDICTION_CACHE.pop(next(iter(PREDICTION_CACHE)))
    PREDICTION_CACHE[cache_key] = (signature, results)
    return results


def _legacy_predict_fixture_rows(
    league_id: int,
    season: int,
    database_path: Path,
) -> dict[int, dict]:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")

    country_database = get_country_database_path(league["country_name"], database_path)
    dataset_database = get_dataset_database_path(league["country_name"], database_path)
    if not country_database.is_file() or not dataset_database.is_file():
        raise ValueError("Prepare the league dataset and train both model types first")

    prediction_table = f"{league_id}-prediction-dataset"
    dataset_metadata_table = f"{league_id}-dataset-metadata"
    regression_metadata_table = f"{league_id}-goal-regression-feature-group-models"
    classifier_model_dir = (
        PROJECT_ROOT / "storage" / "trained_classifier_models" / "v3" / str(league_id)
    )
    regressor_model_dir = (
        PROJECT_ROOT / "storage" / "trained_regressor_models" / "v3" / str(league_id)
    )
    classifier_metadata = classifier_model_dir / "ensemble.json"
    if not classifier_metadata.is_file():
        raise ValueError("Train the classifiers for this league before requesting predictions")

    signature = _prediction_signature(
        dataset_database,
        classifier_model_dir,
        regressor_model_dir,
    )
    cache_key = (league_id, season)
    cached = PREDICTION_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    import joblib
    import numpy as np
    import pandas as pd

    from training.classification.train_ensemble import predict_upcoming_fixtures
    from training.regression.train_goal_models import (
        engineer_features,
        poisson_scoreline_predictions,
        rounded_scores,
    )

    with connect_existing_database(dataset_database) as connection:
        if not _fixture_table(connection, prediction_table):
            raise ValueError("Prepare a prediction dataset for this league and season first")
        if not _fixture_table(connection, dataset_metadata_table):
            raise ValueError("Prepare a prediction dataset for this league and season first")
        metadata_columns = {
            row[0] for row in connection.execute(f'DESCRIBE "{dataset_metadata_table}"').fetchall()
        }
        if "full_season" not in metadata_columns:
            raise ValueError("Reprepare the dataset in Model lab to include all season fixtures")
        prepared_season = connection.execute(
            f'SELECT prediction_season, full_season FROM "{dataset_metadata_table}" LIMIT 1'
        ).fetchone()
        if prepared_season is None or int(prepared_season[0]) != season:
            raise ValueError("Prepare a prediction dataset for the selected league and season first")
        if not prepared_season[1]:
            raise ValueError("Reprepare the dataset in Model lab to include all season fixtures")
        if not _fixture_table(connection, regression_metadata_table):
            raise ValueError("Train the goal regressors for this league before requesting predictions")
        fixtures = connection.execute(f'SELECT * FROM "{prediction_table}"').df()
        regression_models = connection.execute(
            f'SELECT feature_bundle, model, model_filename, features_json, '
            f'validation_outcome_accuracy FROM "{regression_metadata_table}"'
        ).fetchdf()

    if fixtures.empty:
        return {}
    if "status" not in fixtures or "fixture_id" not in fixtures or "date" not in fixtures:
        raise ValueError("The prediction dataset is missing fixture identifiers or dates")
    original_dates = fixtures["date"]
    fixtures["date"] = pd.to_datetime(original_dates, errors="coerce")
    if (fixtures["date"].isna() & original_dates.notna()).any():
        raise ValueError("The prediction dataset contains invalid fixture dates")

    classifier_predictions = predict_upcoming_fixtures(
        db_path=dataset_database,
        prediction_table=prediction_table,
        league_id=league_id,
        model_dir=classifier_model_dir,
        persist=False,
        include_all=True,
    )
    classifier_by_fixture = {
        int(row["fixture_id"]): row
        for row in classifier_predictions.to_dict(orient="records")
    }

    if regression_models.empty:
        raise ValueError("No trained feature-group regressors are available")
    features = engineer_features(fixtures)
    group_goal_means: dict[str, np.ndarray] = {}
    group_weights: dict[str, float] = {}
    group_details: dict[str, dict] = {}
    for model_row in regression_models.to_dict(orient="records"):
        model_path = regressor_model_dir / model_row["model_filename"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained goal regressor does not exist: {model_path}")
        model_features = json.loads(model_row["features_json"])
        missing_features = set(model_features).difference(features.columns)
        if missing_features:
            raise ValueError(
                f"Prediction data is missing features for {model_row['feature_bundle']}: "
                f"{sorted(missing_features)}"
            )
        model = joblib.load(model_path)
        scores = np.asarray(model.predict(features[model_features]), dtype=float)
        if scores.shape != (len(fixtures), 2) or not np.isfinite(scores).all():
            raise ValueError(f"Regressor {model_row['feature_bundle']} returned invalid scores")
        bundle = model_row["feature_bundle"]
        group_goal_means[bundle] = scores
        group_weights[bundle] = float(model_row["validation_outcome_accuracy"])
        group_details[bundle] = {
            "model": model_row["model"],
            "validation_accuracy": group_weights[bundle],
        }

    scorelines = poisson_scoreline_predictions(group_goal_means, group_weights)
    total_weight = sum(group_weights.values())
    prediction_rows: dict[int, dict] = {}
    outcome_names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    double_chance_groups = {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}
    for index, fixture in enumerate(fixtures.to_dict(orient="records")):
        fixture_id = int(fixture["fixture_id"])
        classifier = classifier_by_fixture.get(fixture_id)
        if classifier is None:
            continue
        outcome_predictions = [
            {
                "code": code,
                "label": outcome_names[code],
                "confidence": float(classifier[f"probability_{name}"]),
            }
            for code, name in (("H", "home"), ("D", "draw"), ("A", "away"))
        ]
        outcome_predictions.sort(key=lambda item: (-item["confidence"], item["code"]))
        probabilities = {
            item["code"]: item["confidence"] for item in outcome_predictions
        }
        double_chance_predictions = [
            {
                "code": code,
                "label": {
                    "1X": "Home or draw",
                    "X2": "Draw or away",
                    "12": "Home or away",
                }[code],
                "confidence": sum(probabilities[outcome] for outcome in outcomes),
            }
            for code, outcomes in double_chance_groups.items()
        ]
        double_chance_predictions.sort(
            key=lambda item: (-item["confidence"], item["code"])
        )

        score = scorelines[index]
        score_home, score_away = int(score[0]), int(score[1])
        score_confidence = 0.0
        if total_weight > 0:
            for bundle, means in group_goal_means.items():
                home_rate, away_rate = np.maximum(means[index], 1e-12)
                home_probability = math.exp(
                    score_home * math.log(home_rate)
                    - home_rate
                    - math.lgamma(score_home + 1)
                )
                away_probability = math.exp(
                    score_away * math.log(away_rate)
                    - away_rate
                    - math.lgamma(score_away + 1)
                )
                score_confidence += (
                    group_weights[bundle] / total_weight
                ) * home_probability * away_probability

        home_team = str(fixture.get("home_team") or "Home")
        away_team = str(fixture.get("away_team") or "Away")
        top_outcome = outcome_predictions[0]
        top_double_chance = double_chance_predictions[0]
        regression_outcome = (
            "H" if score_home > score_away
            else "A" if score_away > score_home
            else "D"
        )
        summary = f"{top_outcome['label']} most likely"
        rounded_model_scores = {
            bundle: rounded_scores(means[index:index + 1])[0]
            for bundle, means in group_goal_means.items()
        }
        confidence = max(
            top_outcome["confidence"],
            top_double_chance["confidence"],
            score_confidence,
        )
        prediction_rows[fixture_id] = {
            "summary": summary,
            "confidence": confidence,
            "classification": {
                "outcomes": outcome_predictions,
                "double_chance": double_chance_predictions,
                "predicted_outcome": top_outcome["code"],
                "predicted_double_chance": top_double_chance["code"],
                "model_votes": {
                    outcome_names[code]: int(classifier.get(f"{name}_model_votes", 0))
                    for code, name in (("H", "home"), ("D", "draw"), ("A", "away"))
                },
            },
            "regression": {
                "score": f"{score_home}-{score_away}",
                "home_goals": score_home,
                "away_goals": score_away,
                "confidence": score_confidence,
                "predicted_outcome": regression_outcome,
                "models": [
                    {
                        "feature_bundle": bundle,
                        "model": details["model"],
                        "validation_accuracy": details["validation_accuracy"],
                        "score": f"{rounded_model_scores[bundle][0]}-{rounded_model_scores[bundle][1]}",
                    }
                    for bundle, details in sorted(
                        group_details.items(),
                        key=lambda item: (-item[1]["validation_accuracy"], item[0]),
                    )
                ],
            },
        }

    if len(PREDICTION_CACHE) >= 8:
        PREDICTION_CACHE.pop(next(iter(PREDICTION_CACHE)))
    PREDICTION_CACHE[cache_key] = (signature, prediction_rows)
    return prediction_rows


def get_upcoming_predictions(
    league_id: int,
    season: int,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")
    predictions = _predict_fixture_rows(league_id, season, database_path)
    country_database = get_country_database_path(league["country_name"], database_path)
    with connect_existing_database(country_database) as connection:
        fixture_table = f"{league_id}-{season}-fixtures"
        if not _fixture_table(connection, fixture_table):
            return {"fixtures": []}
        fixture_columns = {
            row[0] for row in connection.execute(f'DESCRIBE "{fixture_table}"').fetchall()
        }
        home_logo = "home_logo" if "home_logo" in fixture_columns else "NULL AS home_logo"
        away_logo = "away_logo" if "away_logo" in fixture_columns else "NULL AS away_logo"
        rows = connection.execute(
            f"""SELECT fixture_id, date, venue, round, home_team, away_team,
                       home_goals, away_goals, status, {home_logo}, {away_logo}
                FROM "{fixture_table}"
                ORDER BY date NULLS LAST, fixture_id"""
        ).fetchall()
        columns = [column[0] for column in connection.description]
    fixtures = []
    for values in rows:
        fixture = dict(zip(columns, values))
        prediction = predictions.get(int(fixture["fixture_id"]))
        if prediction is None:
            raise ValueError(
                "Prediction data is missing season fixtures; reprepare the dataset in Model lab"
            )
        fixture["prediction"] = prediction
        fixtures.append(fixture)
    if any(not fixture["home_logo"] or not fixture["away_logo"] for fixture in fixtures):
        fixture_file = FIXTURE_FILES / str(league_id) / f"{league_id}-{season}-fixtures.json"
        if fixture_file.is_file():
            with fixture_file.open("r", encoding="utf-8") as file:
                api_fixtures = json.load(file).get("response", [])
            logos_by_id = {
                item.get("fixture", {}).get("id"): item.get("teams", {})
                for item in api_fixtures
            }
            for fixture in fixtures:
                logos = logos_by_id.get(fixture["fixture_id"], {})
                fixture["home_logo"] = fixture["home_logo"] or (
                    logos.get("home", {}).get("logo")
                )
                fixture["away_logo"] = fixture["away_logo"] or (
                    logos.get("away", {}).get("logo")
                )
    return {
        "fixtures": fixtures,
        "country_name": league["country_name"],
        "league_name": league["league_name"],
        "season": season,
    }


def get_prediction_fixture_details(
    league_id: int,
    season: int,
    fixture_id: int,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    details = get_fixture_details(league_id, season, fixture_id, database_path)
    prediction_rows = _predict_fixture_rows(league_id, season, database_path)
    details["prediction"] = prediction_rows.get(fixture_id)
    details["favorite_list_ids"] = favorite_list_ids_for_fixture(
        league_id, season, fixture_id, database_path
    )
    details["is_favorite"] = bool(details["favorite_list_ids"])
    return details


def _ensure_favorite_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS "{FAVORITE_LISTS_TABLE}" (
            list_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )"""
    )
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS "{FAVORITE_LIST_FIXTURES_TABLE}" (
            list_id VARCHAR NOT NULL,
            league_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            fixture_id BIGINT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            PRIMARY KEY (list_id, league_id, season, fixture_id)
        )"""
    )
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS "{FAVORITE_MIGRATIONS_TABLE}" (
            migration_name VARCHAR PRIMARY KEY,
            completed_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )"""
    )


def _migrate_legacy_favorites(database_path: Path) -> None:
    with FAVORITES_LOCK:
        _migrate_legacy_favorites_locked(database_path)


def _migrate_legacy_favorites_locked(database_path: Path) -> None:
    leagues = get_leagues(database_path)
    with duckdb.connect(str(database_path)) as connection:
        _ensure_favorite_tables(connection)
        migrated = connection.execute(
            f"""SELECT count(*) FROM "{FAVORITE_MIGRATIONS_TABLE}"
                WHERE migration_name = 'country-favorites-v1'"""
        ).fetchone()[0]
        if migrated:
            return

        legacy_favorites = []
        for league in leagues:
            country_database = get_country_database_path(league["country_name"], database_path)
            if not country_database.is_file():
                continue
            with connect_existing_database(country_database) as country_connection:
                if not _fixture_table(country_connection, FAVORITES_TABLE):
                    continue
                rows = country_connection.execute(
                    f"""SELECT league_id, season, fixture_id, created_at
                        FROM "{FAVORITES_TABLE}" WHERE league_id = ?""",
                    [league["league_id"]],
                ).fetchall()
                legacy_favorites.extend(rows)

        if legacy_favorites:
            list_id = uuid.uuid4().hex
            connection.execute(
                f'INSERT INTO "{FAVORITE_LISTS_TABLE}" (list_id, name) VALUES (?, ?)',
                [list_id, "Favorites"],
            )
            connection.executemany(
                f"""INSERT INTO "{FAVORITE_LIST_FIXTURES_TABLE}"
                    (list_id, league_id, season, fixture_id, created_at)
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                [(list_id, *favorite) for favorite in legacy_favorites],
            )
        connection.execute(
            f"""INSERT INTO "{FAVORITE_MIGRATIONS_TABLE}" (migration_name)
                VALUES ('country-favorites-v1') ON CONFLICT DO NOTHING"""
        )


def get_favorite_lists(database_path: Path = DEFAULT_DATABASE) -> list[dict]:
    _migrate_legacy_favorites(database_path)
    with duckdb.connect(str(database_path)) as connection:
        lists = connection.execute(
            f"""SELECT list_id, name FROM "{FAVORITE_LISTS_TABLE}"
                ORDER BY lower(name), created_at"""
        ).fetchall()
        memberships = connection.execute(
            f"""SELECT list_id, league_id, season, fixture_id
                FROM "{FAVORITE_LIST_FIXTURES_TABLE}"
                ORDER BY created_at DESC"""
        ).fetchall()

    league_by_id = {league["league_id"]: league for league in get_leagues(database_path)}
    fixtures_by_list: dict[str, list[dict]] = {list_id: [] for list_id, _ in lists}
    country_connections: dict[str, duckdb.DuckDBPyConnection] = {}
    try:
        for list_id, league_id, season, fixture_id in memberships:
            league = league_by_id.get(league_id)
            if league is None:
                fixtures_by_list[list_id].append(
                    {
                        "league_id": league_id,
                        "league_name": "Unavailable league",
                        "country_name": "Unavailable country",
                        "country_flag": None,
                        "season": season,
                        "fixture_id": fixture_id,
                        "home_team": "Fixture unavailable",
                        "away_team": "",
                        "date": None,
                        "round": None,
                        "status": None,
                        "fixture_available": False,
                    }
                )
                continue
            country_database = get_country_database_path(league["country_name"], database_path)
            if not country_database.is_file():
                row = None
            else:
                country_key = str(country_database)
                if country_key not in country_connections:
                    country_connections[country_key] = connect_existing_database(country_key)
                country_connection = country_connections[country_key]
                table_name = f"{league_id}-{season}-fixtures"
                row = None
                if _fixture_table(country_connection, table_name):
                    row = country_connection.execute(
                        f"""SELECT fixture_id, date, round, home_team, away_team, status
                            FROM "{table_name}" WHERE fixture_id = ?""",
                        [fixture_id],
                    ).fetchone()
            if row is None:
                fixtures_by_list[list_id].append(
                    {
                        "league_id": league_id,
                        "league_name": league["league_name"],
                        "country_name": league["country_name"],
                        "country_flag": league["country_flag"],
                        "season": season,
                        "fixture_id": fixture_id,
                        "home_team": "Fixture unavailable",
                        "away_team": "",
                        "date": None,
                        "round": None,
                        "status": None,
                        "fixture_available": False,
                    }
                )
                continue
            fixtures_by_list[list_id].append(
                {
                    "league_id": league_id,
                    "league_name": league["league_name"],
                    "country_name": league["country_name"],
                    "country_flag": league["country_flag"],
                    "season": season,
                    "fixture_id": row[0],
                    "date": row[1],
                    "round": row[2],
                    "home_team": row[3],
                    "away_team": row[4],
                    "status": row[5],
                    "fixture_available": True,
                }
            )
    finally:
        for country_connection in country_connections.values():
            country_connection.close()
    return [
        {"list_id": list_id, "name": name, "fixtures": fixtures_by_list[list_id]}
        for list_id, name in lists
    ]


def _favorite_list_id(connection: duckdb.DuckDBPyConnection, list_id: str) -> str:
    result = connection.execute(
        f'SELECT list_id FROM "{FAVORITE_LISTS_TABLE}" WHERE list_id = ?',
        [list_id],
    ).fetchone()
    if result is None:
        raise ValueError("Favorite list not found")
    return result[0]


def create_favorite_list(name: str, database_path: Path = DEFAULT_DATABASE) -> dict:
    if not isinstance(name, str):
        raise ValueError("List name must be between 1 and 80 characters")
    normalized_name = name.strip()
    if not normalized_name or len(normalized_name) > 80:
        raise ValueError("List name must be between 1 and 80 characters")
    _migrate_legacy_favorites(database_path)
    list_id = uuid.uuid4().hex
    try:
        with FAVORITES_LOCK:
            with duckdb.connect(str(database_path)) as connection:
                connection.execute(
                    f'INSERT INTO "{FAVORITE_LISTS_TABLE}" (list_id, name) VALUES (?, ?)',
                    [list_id, normalized_name],
                )
    except duckdb.ConstraintException:
        raise ValueError("A favorite list with that name already exists") from None
    return {"list_id": list_id, "name": normalized_name, "fixtures": []}


def rename_favorite_list(
    list_id: str, name: str, database_path: Path = DEFAULT_DATABASE
) -> None:
    if not isinstance(name, str):
        raise ValueError("List name must be between 1 and 80 characters")
    normalized_name = name.strip()
    if not normalized_name or len(normalized_name) > 80:
        raise ValueError("List name must be between 1 and 80 characters")
    _migrate_legacy_favorites(database_path)
    try:
        with FAVORITES_LOCK:
            with duckdb.connect(str(database_path)) as connection:
                _favorite_list_id(connection, list_id)
                connection.execute(
                    f'UPDATE "{FAVORITE_LISTS_TABLE}" SET name = ? WHERE list_id = ?',
                    [normalized_name, list_id],
                )
    except duckdb.ConstraintException:
        raise ValueError("A favorite list with that name already exists") from None


def delete_favorite_list(list_id: str, database_path: Path = DEFAULT_DATABASE) -> None:
    _migrate_legacy_favorites(database_path)
    with FAVORITES_LOCK:
        with duckdb.connect(str(database_path)) as connection:
            _favorite_list_id(connection, list_id)
            connection.execute(
                f'DELETE FROM "{FAVORITE_LIST_FIXTURES_TABLE}" WHERE list_id = ?',
                [list_id],
            )
            connection.execute(
                f'DELETE FROM "{FAVORITE_LISTS_TABLE}" WHERE list_id = ?',
                [list_id],
            )


def add_fixture_to_favorite_lists(
    league_id: int,
    season: int,
    fixture_id: int,
    list_ids: list[str],
    database_path: Path = DEFAULT_DATABASE,
) -> None:
    if not list_ids:
        raise ValueError("Activate at least one favorite list before saving a fixture")
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")
    country_database = get_country_database_path(league["country_name"], database_path)
    if not country_database.is_file():
        raise ValueError("Fixtures have not been fetched for this season")
    fixture_table = f"{league_id}-{season}-fixtures"
    with connect_existing_database(country_database) as connection:
        if not _fixture_table(connection, fixture_table):
            raise ValueError("Fixtures have not been fetched for this season")
        exists = connection.execute(
            f'SELECT count(*) FROM "{fixture_table}" WHERE fixture_id = ?',
            [fixture_id],
        ).fetchone()[0]
        if not exists:
            raise ValueError("Fixture not found")

    _migrate_legacy_favorites(database_path)
    unique_list_ids = list(dict.fromkeys(list_ids))
    with FAVORITES_LOCK:
        with duckdb.connect(str(database_path)) as connection:
            for list_id in unique_list_ids:
                _favorite_list_id(connection, list_id)
            connection.executemany(
                f"""INSERT INTO "{FAVORITE_LIST_FIXTURES_TABLE}"
                    (list_id, league_id, season, fixture_id)
                    VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                [
                    (list_id, league_id, season, fixture_id)
                    for list_id in unique_list_ids
                ],
            )


def remove_fixture_from_favorite_list(
    list_id: str,
    league_id: int,
    season: int,
    fixture_id: int,
    database_path: Path = DEFAULT_DATABASE,
) -> None:
    _migrate_legacy_favorites(database_path)
    with FAVORITES_LOCK:
        with duckdb.connect(str(database_path)) as connection:
            _favorite_list_id(connection, list_id)
            connection.execute(
                f"""DELETE FROM "{FAVORITE_LIST_FIXTURES_TABLE}"
                    WHERE list_id = ? AND league_id = ? AND season = ? AND fixture_id = ?""",
                [list_id, league_id, season, fixture_id],
            )


def favorite_list_ids_for_fixture(
    league_id: int,
    season: int,
    fixture_id: int,
    database_path: Path = DEFAULT_DATABASE,
) -> list[str]:
    _migrate_legacy_favorites(database_path)
    with duckdb.connect(str(database_path)) as connection:
        return [
            row[0]
            for row in connection.execute(
                f"""SELECT list_id FROM "{FAVORITE_LIST_FIXTURES_TABLE}"
                    WHERE league_id = ? AND season = ? AND fixture_id = ?""",
                [league_id, season, fixture_id],
            ).fetchall()
        ]


def _update_fetch_job(job_id: str, **updates: object) -> None:
    with FETCH_JOBS_LOCK:
        FETCH_JOBS[job_id].update(updates)


def _fetch_seasons(league_id: int, seasons: list[int], update_job) -> None:
    if not seasons:
        raise ValueError("No seasons are available to fetch")
    season_indexes = {season: index for index, season in enumerate(seasons)}

    def on_progress(season: int, status: str) -> None:
        index = season_indexes[season]
        completed = index
        if status == "fetching":
            update_job(
                current_season=season,
                message=f"Fetching {season} fixtures…",
                progress=round((completed + 0.05) * 100 / len(seasons)),
            )
            return

        update_job(
            current_season=season,
            message=f"Importing {season} fixtures into DuckDB…",
            progress=round((completed + 0.65) * 100 / len(seasons)),
        )
        fixture_file = FIXTURE_FILES / str(league_id) / f"{league_id}-{season}-fixtures.json"
        import_fixtures(fixture_file.name, str(fixture_file))
        completed += 1
        update_job(
            completed=completed,
            progress=round(completed * 100 / len(seasons)),
            message=f"Imported {season} fixtures into DuckDB.",
        )

    fetch_fixtures(league_id, seasons, progress_callback=on_progress)


def _run_fetch_job(job_id: str, league_id: int, seasons: list[int]) -> None:
    try:
        database_path = get_database_path()
        league = _read_league(league_id, database_path)
        if league is None:
            raise ValueError("League not found")

        _fetch_seasons(league_id, seasons, lambda **updates: _update_fetch_job(job_id, **updates))
        _update_fetch_job(job_id, status="completed", progress=100, message="Fixture fetch complete.")
    except Exception as error:
        print(f"Fixture fetch job {job_id} failed: {error}")
        _update_fetch_job(job_id, status="failed", message=str(error))


def get_leagues(database_path: Path = DEFAULT_DATABASE) -> list[dict]:
    """Return the league catalogue, preserving the original server API."""
    return _get_leagues(database_path)


def _has_all_season_dataset(connection, league_id: int, seasons: list[int]) -> bool:
    metadata_table = f"{league_id}-dataset-metadata"
    if not _fixture_table(connection, f"{league_id}-training-dataset") or not _fixture_table(connection, metadata_table):
        return False
    columns = {row[0] for row in connection.execute(f'DESCRIBE "{metadata_table}"').fetchall()}
    if not {"all_seasons", "training_seasons"}.issubset(columns):
        return False
    metadata = connection.execute(
        f'SELECT all_seasons, training_seasons FROM "{metadata_table}" LIMIT 1'
    ).fetchone()
    return bool(metadata and metadata[0] and metadata[1] is not None
                and sorted(metadata[1]) == sorted(set(seasons)))


def get_model_lab_status(
    league_id: int,
    season: int,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")
    if season not in league["seasons"]:
        raise ValueError("Season is not available for this league")

    country_database = get_country_database_path(league["country_name"], database_path)
    dataset_database = get_dataset_database_path(league["country_name"], database_path)
    expected_features = [
        "points", "points-l3", "points-l5", "points-l10",
        "goals", "goals-l3", "goals-l5", "goals-l10",
        "xg", "xg-l3", "xg-l5", "xg-l10",
        "odds", "odds-l3", "odds-l5", "odds-l10", "h2h",
    ]
    feature_tables = [
        f"{league_id}-{item}-{suffix}"
        for item in league["seasons"] for suffix in expected_features
    ]
    processed = []
    if country_database.is_file():
        with connect_existing_database(country_database) as connection:
            for table in feature_tables:
                if _fixture_table(connection, table):
                    processed.append(table)

    datasets = {"training": False, "prediction": False}
    if dataset_database.is_file():
        with connect_existing_database(dataset_database) as connection:
            datasets["training"] = _has_all_season_dataset(connection, league_id, league["seasons"])
            datasets["prediction"] = datasets["training"] and all(
                _fixture_table(connection, f"{league_id}-{item}-prediction-dataset")
                for item in league["seasons"]
            )

    model_dir = PROJECT_ROOT / "storage" / "trained_classifier_models" / "v3" / str(league_id)
    regression_model_dir = PROJECT_ROOT / "storage" / "trained_regressor_models" / "v3" / str(league_id)
    neural_model_dir = (
        PROJECT_ROOT / "storage" / "trained_deep_learning_models" / "v1" / str(league_id)
    )
    sequence_model_dir = (
        PROJECT_ROOT
        / "storage"
        / "trained_deep_learning_models"
        / "sequence"
        / str(league_id)
    )
    return {
        "league_id": league_id,
        "season": season,
        "feature_tables": {"available": len(processed), "total": len(feature_tables)},
        "datasets": datasets,
        "trained": {
            "classifier": (model_dir / "ensemble.json").is_file(),
            "regression": any(regression_model_dir.glob("*_goal_model.joblib")),
            "neural": all(
                path.is_file()
                for path in (
                    neural_model_dir / f"{league_id}_outcome_mlp.joblib",
                    neural_model_dir / f"{league_id}_score_mlp.joblib",
                    sequence_model_dir / f"{league_id}_cnn.keras",
                    sequence_model_dir / f"{league_id}_rnn.keras",
                    sequence_model_dir / f"{league_id}_lstm.keras",
                    sequence_model_dir / f"{league_id}_autoencoder.keras",
                    sequence_model_dir / "outcome_metadata.json",
                )
            ),
        },
    }


def get_model_lab_metrics(
    league_id: int,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    league = _read_league(league_id, database_path)
    if league is None:
        raise ValueError("League not found")

    dataset_database = get_dataset_database_path(league["country_name"], database_path)
    classifier_model_dir = (
        PROJECT_ROOT / "storage" / "trained_classifier_models" / "v3" / str(league_id)
    )
    regressor_model_dir = (
        PROJECT_ROOT / "storage" / "trained_regressor_models" / "v3" / str(league_id)
    )
    deep_learning_root = PROJECT_ROOT / "storage" / "trained_deep_learning_models"
    groups = []

    if dataset_database.is_file():
        with connect_existing_database(dataset_database) as connection:
            classifier_metrics = {}
            classifier_table = f"{league_id}-ensemble-classifier-model-metrics"
            if _fixture_table(connection, classifier_table):
                classifier_metrics = {
                    row[0]: {
                        "holdout_accuracy": row[1],
                        "holdout_log_loss": row[2],
                        "holdout_samples": row[3],
                    }
                    for row in connection.execute(
                        f"""SELECT model_name, accuracy, log_loss, test_samples
                            FROM "{classifier_table}" """
                    ).fetchall()
                }

            classifier_manifest = classifier_model_dir / "ensemble.json"
            if classifier_manifest.is_file():
                with classifier_manifest.open("r", encoding="utf-8") as file:
                    manifest = json.load(file)
                classifier_rows = []
                for model_name, details in manifest.get("models", {}).items():
                    model_file = details.get("model_file")
                    if not model_file or not (classifier_model_dir / model_file).is_file():
                        continue
                    metrics = classifier_metrics.get(model_name, {})
                    classifier_rows.append(
                        {
                            "model": model_name,
                            "feature_bundle": details.get("feature_bundle"),
                            "validation_accuracy": details.get("validation_accuracy"),
                            "holdout_accuracy": metrics.get("holdout_accuracy"),
                            "holdout_log_loss": metrics.get("holdout_log_loss"),
                            "holdout_samples": metrics.get("holdout_samples"),
                        }
                    )
                if classifier_rows:
                    groups.append({"name": "Classifiers", "rows": classifier_rows})

            regression_rows = []
            regression_metadata = f"{league_id}-goal-regression-feature-group-models"
            regression_comparison = f"{league_id}-goal-regression-model-comparison"
            comparison_metrics = {}
            metadata_metrics = {}
            if _fixture_table(connection, regression_comparison):
                comparison_metrics = {
                    (row[0], row[1]): {
                        "holdout_accuracy": row[2],
                        "holdout_goal_mae": row[3],
                        "exact_score_accuracy": row[4],
                    }
                    for row in connection.execute(
                        f"""SELECT model, feature_bundle, outcome_accuracy,
                                   mae_overall, scoreline_exact_accuracy
                            FROM "{regression_comparison}" """
                    ).fetchall()
                }
            if _fixture_table(connection, regression_metadata):
                for row in connection.execute(
                    f"""SELECT feature_bundle, model, model_filename,
                               validation_outcome_accuracy, test_outcome_accuracy,
                               validation_mae_overall, test_mae_overall
                        FROM "{regression_metadata}" """
                ).fetchall():
                    feature_bundle, model_name = row[:2]
                    metadata_metrics[(model_name, feature_bundle)] = {
                        "model_filename": row[2],
                        "validation_accuracy": row[3],
                        "fallback_holdout_accuracy": row[4],
                        "validation_goal_mae": row[5],
                        "fallback_holdout_goal_mae": row[6],
                    }

            for model_name, feature_bundle in comparison_metrics:
                details = metadata_metrics.get((model_name, feature_bundle), {})
                model_filename = details.get("model_filename") or (
                    f"{league_id}_{str(model_name).lower()}_goal_model.joblib"
                )
                if not (regressor_model_dir / model_filename).is_file():
                    continue
                comparison = comparison_metrics[(model_name, feature_bundle)]
                regression_rows.append(
                    {
                        "model": model_name,
                        "feature_bundle": feature_bundle,
                        "validation_accuracy": details.get("validation_accuracy"),
                        "holdout_accuracy": comparison["holdout_accuracy"],
                        "validation_goal_mae": details.get("validation_goal_mae"),
                        "holdout_goal_mae": comparison["holdout_goal_mae"],
                        "exact_score_accuracy": comparison["exact_score_accuracy"],
                    }
                )

            for (model_name, feature_bundle), details in metadata_metrics.items():
                if (model_name, feature_bundle) in comparison_metrics:
                    continue
                model_filename = details["model_filename"]
                if not model_filename or not (regressor_model_dir / model_filename).is_file():
                    continue
                regression_rows.append(
                    {
                        "model": model_name,
                        "feature_bundle": feature_bundle,
                        "validation_accuracy": details["validation_accuracy"],
                        "holdout_accuracy": details["fallback_holdout_accuracy"],
                        "validation_goal_mae": details["validation_goal_mae"],
                        "holdout_goal_mae": details["fallback_holdout_goal_mae"],
                        "exact_score_accuracy": None,
                    }
                )
            if regression_rows:
                groups.append({"name": "Goal regressors", "rows": regression_rows})

            neural_rows = []
            deep_learning_metrics = f"{league_id}-deep-learning-metrics"
            mlp_model = deep_learning_root / "v1" / str(league_id) / f"{league_id}_outcome_mlp.joblib"
            if _fixture_table(connection, deep_learning_metrics) and mlp_model.is_file():
                row = connection.execute(
                    f"""SELECT test_fixture_count, outcome_accuracy,
                               double_chance_accuracy, correct_score_accuracy
                        FROM "{deep_learning_metrics}" LIMIT 1"""
                ).fetchone()
                if row is not None:
                    neural_rows.append(
                        {
                            "model": "MLP",
                            "holdout_samples": row[0],
                            "holdout_accuracy": row[1],
                            "double_chance_accuracy": row[2],
                            "exact_score_accuracy": row[3],
                        }
                    )

            sequence_metrics = f"{league_id}-sequence-model-metrics"
            if _fixture_table(connection, sequence_metrics):
                for row in connection.execute(
                    f"""SELECT model, test_fixture_count, outcome_accuracy,
                               double_chance_accuracy, correct_score_accuracy
                        FROM "{sequence_metrics}" """
                ).fetchall():
                    model_name = str(row[0]).upper()
                    model_file = (
                        deep_learning_root
                        / "sequence"
                        / str(league_id)
                        / f"{league_id}_{str(row[0]).lower()}.keras"
                    )
                    if not model_file.is_file():
                        continue
                    neural_rows.append(
                        {
                            "model": model_name,
                            "holdout_samples": row[1],
                            "holdout_accuracy": row[2],
                            "double_chance_accuracy": row[3],
                            "exact_score_accuracy": row[4],
                        }
                    )
            outcome_metrics = f"{league_id}-outcome-network-metrics"
            if _fixture_table(connection, outcome_metrics):
                columns = {
                    row[0] for row in connection.execute(f'DESCRIBE "{outcome_metrics}"').fetchall()
                }
                double_chance = "double_chance_accuracy" if "double_chance_accuracy" in columns else "NULL"
                correct_score = "correct_score_accuracy" if "correct_score_accuracy" in columns else "NULL"
                for model, count, accuracy, loss, dc_accuracy, score_accuracy in connection.execute(
                    f'SELECT model, test_fixture_count, outcome_accuracy, outcome_log_loss, '
                    f'{double_chance}, {correct_score} FROM "{outcome_metrics}"'
                ).fetchall():
                    model_file = deep_learning_root / "sequence" / str(league_id) / f"{league_id}_{model}.keras"
                    if model_file.is_file():
                        neural_rows.append({
                            "model": str(model).upper(), "holdout_samples": count,
                            "holdout_accuracy": accuracy, "holdout_log_loss": loss,
                            "double_chance_accuracy": dc_accuracy,
                            "exact_score_accuracy": score_accuracy,
                        })
            if neural_rows:
                groups.append({"name": "Neural networks", "rows": neural_rows})

    return {"league_id": league_id, "groups": groups}


def _update_model_job(job_id: str, **updates: object) -> None:
    with MODEL_JOBS_LOCK:
        MODEL_JOBS[job_id].update(updates)


def _report_model_job_progress(
    job_id: str, completed: int, total: int, message: str
) -> None:
    progress = round(completed * 100 / total) if total else 0
    _update_model_job(
        job_id,
        completed=completed,
        total=total,
        progress=progress,
        message=message,
    )


def _run_model_job(
    job_id: str,
    action: str,
    league_id: int,
    season: int,
    process_all_seasons: bool = False,
    fetch_all_seasons: bool = False,
    batch: bool = False,
) -> bool:
    try:
        database_path = get_database_path()
        league = _read_league(league_id, database_path)
        if league is None or season not in league["seasons"]:
            raise ValueError("League or season is no longer available")
        country_database = get_country_database_path(league["country_name"], database_path)
        dataset_database = get_dataset_database_path(league["country_name"], database_path)
        if action == "fetch":
            seasons_to_fetch = (
                sorted(league["seasons"])
                if fetch_all_seasons
                else [season]
            )
            _update_model_job(
                job_id,
                total=len(seasons_to_fetch),
                completed=0,
                message=f"Fetching fixtures for {len(seasons_to_fetch)} seasons…",
            )
            _fetch_seasons(
                league_id,
                seasons_to_fetch,
                lambda **updates: _update_model_job(job_id, **updates),
            )
            result = {
                "processed_seasons": seasons_to_fetch,
                "message": (
                    f"Fetched fixtures for {len(seasons_to_fetch)} seasons."
                ),
            }
        elif action == "process":
            from processing.process_seasons_data import process_seasons_data

            seasons_to_process = [
                available_season
                for available_season in league["seasons"]
                if process_all_seasons or available_season <= season
            ]
            process_seasons_data(
                league_id,
                seasons_to_process,
                leagues_db_path=database_path,
                country_db_path=country_database,
                delay_seconds=0,
                progress_callback=lambda completed, total, message: (
                    _report_model_job_progress(job_id, completed, total, message)
                ),
            )
            result = {
                "season": season,
                "processed_seasons": seasons_to_process,
                "message": (
                    f"Processed all {len(seasons_to_process)} available seasons."
                    if process_all_seasons
                    else (
                        f"Processed feature data for {len(seasons_to_process)} seasons "
                        f"through {season}."
                    )
                ),
            }
        elif action == "prepare":
            from training.prepare_dataset import prepare_all_seasons_dataset

            result = prepare_all_seasons_dataset(
                league_id,
                leagues_db_path=database_path,
                country_db_path=country_database,
                dataset_db_path=dataset_database,
                progress_callback=lambda completed, total, message: (
                    _report_model_job_progress(job_id, completed, total, message)
                ),
            )
        elif action in {"train-classifier", "train-regressor"}:
            if not dataset_database.is_file():
                raise ValueError("Prepare the all-season dataset for this league first")
            with connect_existing_database(dataset_database) as connection:
                prepared = _has_all_season_dataset(connection, league_id, league["seasons"])
            if not prepared:
                raise ValueError("Prepare the all-season dataset for this league first")

            if action == "train-classifier":
                from training.classification.train_ensemble import train_ensemble

                metrics, _ = train_ensemble(
                    db_path=dataset_database,
                    league_id=league_id,
                    model_dir=PROJECT_ROOT / "storage" / "trained_classifier_models" / "v3" / str(league_id),
                    predict_upcoming=False,
                    progress_callback=lambda completed, total, message: (
                        _report_model_job_progress(job_id, completed, total, message)
                    ),
                )
                result = {
                    "model_type": "classifier",
                    "metrics": json.loads(metrics.to_json(orient="records")),
                }
            else:
                from training.regression.train_goal_models import run_experiment

                metrics = run_experiment(
                    db_path=dataset_database,
                    table_name=f"{league_id}-training-dataset",
                    model_dir=PROJECT_ROOT / "storage" / "trained_regressor_models" / "v3" / str(league_id),
                    progress_callback=lambda completed, total, message: (
                        _report_model_job_progress(job_id, completed, total, message)
                    ),
                )
                result = {
                    "model_type": "regression",
                    "metrics": json.loads(metrics.to_json(orient="records")),
                }
        elif action == "train-neural":
            if not dataset_database.is_file():
                raise ValueError("Prepare the all-season dataset for this league first")
            with connect_existing_database(dataset_database) as connection:
                prepared = _has_all_season_dataset(connection, league_id, league["seasons"])
            if not prepared:
                raise ValueError("Prepare the all-season dataset for this league first")

            from training.deep_learning.train_match_networks import (
                run_experiment as train_match_networks,
            )
            from training.deep_learning.train_sequence_models import (
                run_experiment as train_sequence_models,
            )
            from training.deep_learning.train_outcome_networks import (
                run_experiment as train_outcome_networks,
            )

            _update_model_job(
                job_id,
                total=4,
                completed=0,
                progress=0,
                message="Training the MLP on prepared match features...",
            )
            mlp_metrics, _ = train_match_networks(
                db_path=dataset_database,
                league_id=league_id,
                model_dir=(
                    PROJECT_ROOT
                    / "storage"
                    / "trained_deep_learning_models"
                    / "v1"
                    / str(league_id)
                ),
            )
            _report_model_job_progress(
                job_id,
                1,
                4,
                "MLP complete. Training CNN and RNN on raw team histories...",
            )
            sequence_metrics, _ = train_sequence_models(
                league_id=league_id,
                country_database=country_database,
                dataset_database=dataset_database,
                model_dir=(
                    PROJECT_ROOT
                    / "storage"
                    / "trained_deep_learning_models"
                    / "sequence"
                    / str(league_id)
                ),
            )
            _report_model_job_progress(
                job_id,
                2,
                4,
                "CNN and RNN complete. Tuning LSTM and autoencoder outcome models...",
            )
            outcome_metrics, _ = train_outcome_networks(
                league_id=league_id,
                country_database=country_database,
                dataset_database=dataset_database,
                model_dir=(
                    PROJECT_ROOT / "storage" / "trained_deep_learning_models"
                    / "sequence" / str(league_id)
                ),
                progress_callback=lambda completed, total, message: _update_model_job(
                    job_id, message=f"{message} ({completed}/{total})"
                ),
            )
            _report_model_job_progress(job_id, 4, 4, "Neural network training and evaluation completed.")
            metrics_rows = [{
                "model": "MLP",
                **json.loads(mlp_metrics.to_json(orient="records"))[0],
            }]
            metrics_rows.extend(
                {
                    **row,
                    "model": str(row["model"]).upper(),
                }
                for row in json.loads(sequence_metrics.to_json(orient="records"))
            )
            metrics_rows.extend(
                {**row, "model": str(row["model"]).upper()}
                for row in json.loads(outcome_metrics.to_json(orient="records"))
            )
            result = {
                "model_type": "neural",
                "metrics": metrics_rows,
                "message": "Trained the MLP, CNN, RNN, LSTM, and autoencoder.",
            }
        elif action == "predict-all-models":
            if not dataset_database.is_file():
                raise ValueError("Prepare a dataset for the selected league and season first")
            with connect_existing_database(dataset_database) as connection:
                metadata_table = f"{league_id}-dataset-metadata"
                if not _fixture_table(connection, metadata_table):
                    raise ValueError("Prepare a dataset for the selected league and season first")
                prepared_season = connection.execute(
                    f'SELECT prediction_season FROM "{metadata_table}" LIMIT 1'
                ).fetchone()
            if prepared_season is None or int(prepared_season[0]) != season:
                raise ValueError("Prepare a dataset for the selected league and season first")

            from training.ensemble_predict import run_experiment as predict_with_all_models

            _update_model_job(
                job_id,
                total=4,
                completed=0,
                message="Starting all-model upcoming predictions...",
            )
            predictions, importance = predict_with_all_models(
                league_id=league_id,
                leagues_database=database_path,
                raw_database=country_database,
                dataset_database=dataset_database,
                progress_callback=lambda completed, total, message: (
                    _report_model_job_progress(job_id, completed, total, message)
                ),
            )
            prediction_columns = [
                column
                for column in (
                    "date",
                    "home_team",
                    "away_team",
                    "predicted_outcome",
                    "outcome_confidence",
                    "predicted_double_chance",
                    "double_chance_confidence",
                    "predicted_correct_score",
                    "model_count",
                    "family_agreement_count",
                )
                if column in predictions.columns
            ]
            importance_columns = [
                column
                for column in (
                    "model",
                    "holdout_fixtures",
                    "outcome_accuracy",
                    "outcome_log_loss",
                    "draw_precision",
                    "draw_recall",
                    "draw_f1",
                    "draw_predictions",
                    "accuracy_impact_if_removed",
                    "log_loss_impact_if_removed",
                    "draw_precision_impact_if_removed",
                    "draw_recall_impact_if_removed",
                )
                if column in importance.columns
            ]
            result = {
                "model_type": "all-model-predictions",
                "message": (
                    f"Predicted {len(predictions)} upcoming fixtures using "
                    f"{int(predictions['model_count'].iloc[0])} trained models."
                ),
                "prediction_count": len(predictions),
                "models_used": int(predictions["model_count"].iloc[0]),
                "holdout_fixture_count": int(importance["holdout_fixtures"].iloc[0]),
                "predictions": json.loads(
                    predictions[prediction_columns].to_json(
                        orient="records",
                        date_format="iso",
                    )
                ),
                "importance": json.loads(
                    importance[importance_columns].to_json(orient="records")
                ),
            }
        else:
            raise ValueError("Unsupported Model Lab action")
        _update_model_job(
            job_id,
            status="running" if batch else "completed",
            progress=100,
            message="Operation completed.",
            result=result,
        )
        return True
    except Exception as error:
        print(f"Model Lab job {job_id} failed: {error}")
        if batch:
            _update_model_job(job_id, message=str(error))
        else:
            _update_model_job(job_id, status="failed", message=str(error))
        return False


def _run_model_workflow_batch(
    job_id: str,
    actions: list[str],
    league_id: int,
    season: int,
    process_all_seasons: bool,
    fetch_all_seasons: bool,
) -> None:
    for index, action in enumerate(actions):
        with MODEL_JOBS_LOCK:
            job = MODEL_JOBS[job_id]
            job.update(
                active_action=action,
                current_step=index + 1,
                progress=0,
                completed=0,
                total=0,
                message="Preparing workflow...",
            )
            job["workflow_states"][action] = "running"
        succeeded = _run_model_job(
            job_id, action, league_id, season,
            process_all_seasons=process_all_seasons,
            fetch_all_seasons=fetch_all_seasons,
            batch=True,
        )
        with MODEL_JOBS_LOCK:
            job = MODEL_JOBS[job_id]
            job["workflow_states"][action] = "completed" if succeeded else "failed"
            if not succeeded:
                job["status"] = "failed"
                return
            job["workflow_results"][action] = job["result"]
    _update_model_job(job_id, status="completed", progress=100, message="Selected workflows completed.")


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path == "/":
            self._serve_file(WEB_ROOT / "templates" / "index.html", "text/html; charset=utf-8")
        elif request_path == "/api/leagues":
            self._serve_leagues()
        elif request_path == "/api/seasons":
            self._serve_seasons(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/rounds":
            self._serve_rounds(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/season-fixtures":
            self._serve_season_fixtures(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/fixtures":
            self._serve_fixtures(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/fixture":
            self._serve_fixture(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/prediction-fixtures":
            self._serve_prediction_fixtures(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/prediction-fixture":
            self._serve_prediction_fixture(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/favorites":
            self._serve_favorites()
        elif request_path == "/api/fetch-status":
            self._serve_fetch_status(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/model-lab-status":
            self._serve_model_lab_status(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/model-lab-metrics":
            self._serve_model_lab_metrics(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/model-lab-job-status":
            self._serve_model_job_status(parse_qs(urlsplit(self.path).query))
        elif request_path == "/api/model-lab-workflows":
            self._serve_model_workflows(parse_qs(urlsplit(self.path).query))
        elif request_path.startswith("/static/"):
            self._serve_static(request_path.removeprefix("/static/"))
        else:
            self.send_error(404, "Not found")

    def _serve_leagues(self) -> None:
        leagues_controller.serve_leagues(self, get_database_path(), get_leagues)

    def _query_int(self, query: dict, key: str) -> int:
        try:
            value = int(query[key][0])
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"Invalid {key}") from None
        if value < 1:
            raise ValueError(f"Invalid {key}")
        return value

    def _serve_seasons(self, query: dict) -> None:
        try:
            self._send_json(get_seasons(self._query_int(query, "league_id"), get_database_path()))
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load seasons: %s", error)
            self._send_json({"error": "Unable to load seasons from DuckDB."}, status=500)

    def _serve_rounds(self, query: dict) -> None:
        try:
            result = get_rounds(
                self._query_int(query, "league_id"),
                self._query_int(query, "season"),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load rounds: %s", error)
            self._send_json({"error": "Unable to load rounds from DuckDB."}, status=500)
            return
        self._send_json(result)

    def _serve_season_fixtures(self, query: dict) -> None:
        try:
            result = get_season_fixture_groups(
                self._query_int(query, "league_id"),
                self._query_int(query, "season"),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError, json.JSONDecodeError) as error:
            self.log_error("Unable to load season fixtures: %s", error)
            self._send_json({"error": "Unable to load fixtures from DuckDB."}, status=500)
            return
        self._send_json(result)

    def _serve_fixtures(self, query: dict) -> None:
        try:
            fixtures = get_fixtures(
                self._query_int(query, "league_id"),
                self._query_int(query, "season"),
                query["round"][0],
                get_database_path(),
            )
        except (KeyError, IndexError):
            self._send_json({"error": "Invalid round"}, status=400)
            return
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load fixtures: %s", error)
            self._send_json({"error": "Unable to load fixtures from DuckDB."}, status=500)
            return
        self._send_json({"fixtures": fixtures})

    def _serve_fixture(self, query: dict) -> None:
        try:
            details = get_fixture_details(
                self._query_int(query, "league_id"),
                self._query_int(query, "season"),
                self._query_int(query, "fixture_id"),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load fixture details: %s", error)
            self._send_json({"error": "Unable to load fixture details from DuckDB."}, status=500)
            return
        self._send_json(details)

    def _serve_prediction_fixtures(self, query: dict) -> None:
        try:
            result = get_upcoming_predictions(
                self._query_int(query, "league_id"),
                self._query_int(query, "season"),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load model predictions: %s", error)
            self._send_json({"error": "Unable to load predictions from the trained models."}, status=500)
            return
        except Exception as error:
            self.log_error("Unable to calculate model predictions: %s", error)
            self._send_json({"error": f"Unable to calculate model predictions: {error}"}, status=500)
            return
        self._send_json(result)

    def _serve_prediction_fixture(self, query: dict) -> None:
        try:
            result = get_prediction_fixture_details(
                self._query_int(query, "league_id"),
                self._query_int(query, "season"),
                self._query_int(query, "fixture_id"),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load fixture prediction details: %s", error)
            self._send_json({"error": "Unable to load fixture predictions and details."}, status=500)
            return
        except Exception as error:
            self.log_error("Unable to calculate fixture prediction details: %s", error)
            self._send_json({"error": f"Unable to calculate fixture predictions and details: {error}"}, status=500)
            return
        self._send_json(result)

    def _serve_favorites(self) -> None:
        try:
            lists = get_favorite_lists(get_database_path())
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load favorite fixtures: %s", error)
            self._send_json({"error": "Unable to load favorite lists from DuckDB."}, status=500)
            return
        self._send_json({"lists": lists})

    def do_POST(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path == "/api/model-lab-jobs":
            self._start_model_lab_job()
            return
        if request_path == "/api/favorites":
            self._set_favorite()
            return
        if request_path != "/api/fetch-fixtures":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > 16_384:
                raise ValueError("Invalid request body")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict) or not isinstance(payload.get("seasons"), list):
                raise ValueError("Select one or more seasons")
            league_id = int(payload["league_id"])
            seasons = sorted({int(season) for season in payload["seasons"]})
            if league_id < 1 or not seasons or any(season < 1 for season in seasons):
                raise ValueError("Select one or more valid seasons")
            league = _read_league(league_id, get_database_path())
            if league is None:
                raise ValueError("League not found")
            if not set(seasons).issubset(set(league["seasons"])):
                raise ValueError("One or more seasons are not available for this league")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except duckdb.Error as error:
            self.log_error("Unable to validate fixture fetch request: %s", error)
            self._send_json({"error": "Unable to validate the request against DuckDB."}, status=500)
            return

        if not FETCH_LOCK.acquire(blocking=False):
            self._send_json({"error": "Another data operation is already running."}, status=409)
            return

        job_id = uuid.uuid4().hex
        with FETCH_JOBS_LOCK:
            FETCH_JOBS[job_id] = {
                "status": "running",
                "progress": 0,
                "completed": 0,
                "total": len(seasons),
                "current_season": None,
                "message": "Preparing fixture fetch…",
            }

        def run_and_release() -> None:
            try:
                _run_fetch_job(job_id, league_id, seasons)
            finally:
                FETCH_LOCK.release()

        threading.Thread(target=run_and_release, daemon=True).start()
        self._send_json({"job_id": job_id}, status=202)

    def _set_favorite(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > 16_384:
                raise ValueError("Invalid request body")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("Invalid request body")
            database_path = get_database_path()
            action = payload.get("action")
            if action == "create-list":
                result = create_favorite_list(payload.get("name", ""), database_path)
            elif action == "rename-list":
                result = rename_favorite_list(
                    str(payload.get("list_id", "")),
                    payload.get("name", ""),
                    database_path,
                )
            elif action == "delete-list":
                result = delete_favorite_list(
                    str(payload.get("list_id", "")), database_path
                )
            elif action == "add-fixture":
                list_ids = payload.get("list_ids")
                if (
                    not isinstance(list_ids, list)
                    or not list_ids
                    or any(not isinstance(list_id, str) or not list_id for list_id in list_ids)
                ):
                    raise ValueError("Activate at least one favorite list before saving a fixture")
                add_fixture_to_favorite_lists(
                    int(payload["league_id"]),
                    int(payload["season"]),
                    int(payload["fixture_id"]),
                    list_ids,
                    database_path,
                )
                result = {"added_to": list(dict.fromkeys(list_ids))}
            elif action == "remove-fixture":
                remove_fixture_from_favorite_list(
                    str(payload.get("list_id", "")),
                    int(payload["league_id"]),
                    int(payload["season"]),
                    int(payload["fixture_id"]),
                    database_path,
                )
                result = {"removed": True}
            else:
                raise ValueError("Select a valid favorite-list action")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to update favorite lists: %s", error)
            self._send_json({"error": "Unable to update favorite lists in DuckDB."}, status=500)
            return
        self._send_json({"result": result})

    def _serve_fetch_status(self, query: dict) -> None:
        job_id = query.get("job_id", [""])[0]
        with FETCH_JOBS_LOCK:
            job = FETCH_JOBS.get(job_id)
        if job is None:
            self._send_json({"error": "Fetch job not found"}, status=404)
            return
        self._send_json(job)

    def _serve_model_lab_status(self, query: dict) -> None:
        try:
            league_id = self._query_int(query, "league_id")
            league = _read_league(league_id, get_database_path())
            if not league or not league["seasons"]:
                raise ValueError("No seasons are available for this league")
            status = get_model_lab_status(
                league_id,
                max(league["seasons"]),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError) as error:
            self.log_error("Unable to load Model Lab status: %s", error)
            self._send_json({"error": "Unable to load Model Lab status from DuckDB."}, status=500)
            return
        self._send_json(status)

    def _serve_model_lab_metrics(self, query: dict) -> None:
        try:
            metrics = get_model_lab_metrics(
                self._query_int(query, "league_id"),
                get_database_path(),
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except (duckdb.Error, OSError, json.JSONDecodeError) as error:
            self.log_error("Unable to load Model Lab model metrics: %s", error)
            self._send_json(
                {"error": "Unable to load Model Lab model metrics."},
                status=500,
            )
            return
        self._send_json(metrics)

    def _serve_model_job_status(self, query: dict) -> None:
        job_id = query.get("job_id", [""])[0]
        with MODEL_JOBS_LOCK:
            job = MODEL_JOBS.get(job_id)
            job = deepcopy(job) if job is not None else None
        if job is None:
            self._send_json({"error": "Model Lab job not found"}, status=404)
            return
        self._send_json(job)

    def _serve_model_workflows(self, query: dict) -> None:
        try:
            league_id = self._query_int(query, "league_id")
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        states = {}
        active_job_id = None
        with MODEL_JOBS_LOCK:
            for job_id, job in MODEL_JOBS.items():
                if job["league_id"] != league_id:
                    continue
                if job["action"] == "batch":
                    states.update({
                        action: state for action, state in job["workflow_states"].items()
                        if state != "queued"
                    })
                else:
                    states[job["action"]] = job["status"]
                if job["status"] == "running":
                    active_job_id = job_id
        self._send_json({"states": states, "active_job_id": active_job_id})

    def _start_model_lab_job(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > 16_384:
                raise ValueError("Invalid request body")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("Invalid request body")
            action = payload.get("action")
            if action not in (*MODEL_WORKFLOW_ACTIONS, "batch"):
                raise ValueError("Select a valid Model Lab operation")
            actions = payload.get("actions", []) if action == "batch" else [action]
            if (
                not isinstance(actions, list)
                or not actions
                or any(not isinstance(item, str) or item not in MODEL_WORKFLOW_ACTIONS for item in actions)
                or len(set(actions)) != len(actions)
            ):
                raise ValueError("Select one or more valid workflows")
            actions = [item for item in MODEL_WORKFLOW_ACTIONS if item in actions]
            process_all_seasons = True
            fetch_all_seasons = True
            league_id = int(payload["league_id"])
            if league_id < 1:
                raise ValueError("Select a valid league")
            league = _read_league(league_id, get_database_path())
            if league is None or not league["seasons"]:
                raise ValueError("No seasons are available for this league")
            season = max(league["seasons"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except duckdb.Error as error:
            self.log_error("Unable to validate Model Lab request: %s", error)
            self._send_json({"error": "Unable to validate the request against DuckDB."}, status=500)
            return

        if not MODEL_LOCK.acquire(blocking=False):
            self._send_json({"error": "Another data operation is already running."}, status=409)
            return

        job_id = uuid.uuid4().hex
        with MODEL_JOBS_LOCK:
            MODEL_JOBS[job_id] = {
                "status": "running",
                "progress": 0,
                "completed": 0,
                "total": 0,
                "message": "Preparing Model Lab operation…",
                "action": action,
                "league_id": league_id,
                "season": season,
                "actions": actions,
                "workflow_states": {item: "queued" for item in actions},
                "workflow_results": {},
                "active_action": actions[0],
                "current_step": 1,
            }

        def run_and_release() -> None:
            try:
                if action == "batch":
                    _run_model_workflow_batch(
                        job_id, actions, league_id, season,
                        process_all_seasons, fetch_all_seasons,
                    )
                else:
                    _run_model_job(
                        job_id,
                        action,
                        league_id,
                        season,
                        process_all_seasons=process_all_seasons,
                        fetch_all_seasons=fetch_all_seasons,
                    )
            except Exception as error:
                print(f"Model Lab job {job_id} failed unexpectedly: {error}")
                with MODEL_JOBS_LOCK:
                    job = MODEL_JOBS[job_id]
                    job.update(status="failed", message=str(error))
                    if action == "batch":
                        job["workflow_states"][job["active_action"]] = "failed"
            finally:
                MODEL_LOCK.release()

        threading.Thread(target=run_and_release, daemon=True).start()
        self._send_json({"job_id": job_id}, status=202)

    def _serve_static(self, requested_path: str) -> None:
        file_path = (STATIC_ROOT / requested_path).resolve()
        if not file_path.is_relative_to(STATIC_ROOT.resolve()) or not file_path.is_file():
            self.send_error(404, "Not found")
            return

        content_types = {
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }
        self._serve_file(file_path, content_types.get(file_path.suffix, "application/octet-stream"))

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        try:
            content = file_path.read_bytes()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        content = json.dumps(
            payload,
            default=lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    host = os.environ.get("EAGLE_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("EAGLE_DASHBOARD_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Eagle dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Eagle dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
