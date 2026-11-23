"""Prepare fixture-keyed feature datasets and chronological historical splits."""

import logging
import re
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from core.database import connect_existing_database
from core.fixture_dates import fixture_dates_utc


LOGGER = logging.getLogger(__name__)
COMPLETED_STATUSES = frozenset({"FT", "AET", "PEN"})
FEATURE_SUFFIXES = (
    "goals", "goals-l3", "goals-l5", "goals-l10",
    "points", "points-l3", "points-l5", "points-l10",
    "xg", "xg-l3", "xg-l5", "xg-l10",
    "rho", "odds", "odds-l3", "odds-l5", "odds-l10",
    "outcome-probability", "power-ratings", "h2h",
)
FIXTURE_COLUMNS = {
    "league_id", "fixture_id", "date", "venue", "round", "home_team", "away_team",
    "home_goals", "away_goals", "status", "season", "source_table", "result",
    "home_logo", "away_logo",
}


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def available_seasons(connection, league_id):
    pattern = re.compile(rf"^{int(league_id)}-(\d+)-fixtures$")
    seasons = []
    for (name,) in connection.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall():
        match = pattern.fullmatch(name)
        if match:
            seasons.append(int(match.group(1)))
    return sorted(set(seasons))


def join_season(connection, league_id, season):
    """Left join all feature columns, qualifying each by its source table."""
    league_id, season = int(league_id), int(season)
    fixture_table = f"{league_id}-{season}-fixtures"
    base = _quote(fixture_table)
    fixture_count, fixture_unique = connection.execute(
        f"SELECT count(*), count(DISTINCT fixture_id) FROM {base}"
    ).fetchone()
    if not fixture_count or fixture_count != fixture_unique:
        raise ValueError(f"{fixture_table!r} must contain unique, non-null fixture IDs and at least one row")
    projection = [
        "f.league_id", "f.fixture_id", "f.date", "f.venue", "f.round",
        "f.home_team", "f.away_team", "f.home_goals", "f.away_goals", "f.status",
        f"{season}::INTEGER AS season",
        """CASE WHEN f.status IN ('FT', 'AET', 'PEN')
                  AND f.home_goals IS NOT NULL AND f.away_goals IS NOT NULL
             THEN CASE WHEN f.home_goals > f.away_goals THEN 'H'
                       WHEN f.home_goals < f.away_goals THEN 'A' ELSE 'D' END
             ELSE NULL END AS result""",
    ]
    joins = []
    output_names = set()
    for index, suffix in enumerate(FEATURE_SUFFIXES):
        table = f"{league_id}-{season}-{suffix}"
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            raise ValueError(f"Missing feature table {table!r}; process this season first")
        quoted = _quote(table)
        count, unique = connection.execute(
            f"SELECT count(*), count(DISTINCT fixture_id) FROM {quoted}"
        ).fetchone()
        if count != unique:
            raise ValueError(f"{table!r} contains duplicate or null fixture IDs")
        covered = connection.execute(
            f"SELECT count(*) FROM {base} f JOIN {quoted} t USING (fixture_id)"
        ).fetchone()[0]
        if covered != fixture_count:
            LOGGER.warning("%s is missing %d fixture rows; joined features will be NULL",
                           table, fixture_count - covered)
        if count != covered:
            LOGGER.warning("%s has %d rows outside the fixture table", table, count - covered)
        alias = f"t{index}"
        columns = [row[0] for row in connection.execute(f"DESCRIBE {quoted}").fetchall()]
        for column in columns:
            if column in FIXTURE_COLUMNS:
                continue
            name = f"{suffix.replace('-', '_')}__{column}"
            if name in output_names:
                raise ValueError(f"Duplicate feature name {name!r}")
            output_names.add(name)
            projection.append(f"{alias}.{_quote(column)} AS {_quote(name)}")
        joins.append(f"LEFT JOIN {quoted} {alias} ON f.fixture_id = {alias}.fixture_id")
    query = "SELECT " + ", ".join(projection) + f" FROM {base} f " + " ".join(joins)
    result = connection.execute(query).df()
    result["date"] = fixture_dates_utc(result["date"])
    if len(result) != fixture_count or result["fixture_id"].duplicated().any():
        raise ValueError("Feature joins changed fixture count or duplicated fixture IDs")
    return result


def prepare_datasets(league_id, country_db_path, dataset_db_path, progress_callback=None):
    """Write train/validation/test splits and the latest season's fixtures."""
    league_id = int(league_id)
    with connect_existing_database(country_db_path) as source:
        seasons = available_seasons(source, league_id)
        if len(seasons) < 2:
            raise ValueError("At least two available fixture seasons are required")
        total = len(seasons) + 1
        if progress_callback:
            progress_callback(0, total, "Preparing fixture datasets.")
        frames = []
        expected_columns = None
        for index, season in enumerate(seasons, start=1):
            frame = join_season(source, league_id, season)
            if expected_columns is None:
                expected_columns = set(frame.columns)
            elif set(frame.columns) != expected_columns:
                missing = sorted(expected_columns - set(frame.columns))
                extra = sorted(set(frame.columns) - expected_columns)
                raise ValueError(f"Season {season} feature schema differs: missing={missing}, extra={extra}")
            frames.append(frame)
            if progress_callback:
                progress_callback(index, total, f"Joined season {season}.")
    combined = pd.concat(frames, ignore_index=True)
    if combined["fixture_id"].duplicated().any():
        raise ValueError("Fixtures must be unique across all seasons")
    latest = seasons[-1]
    historical = combined[combined["season"] != latest].copy()
    missing_scores = historical["home_goals"].isna() | historical["away_goals"].isna()
    completed = historical["status"].isin(COMPLETED_STATUSES)
    for column in ("home_goals", "away_goals"):
        numeric = pd.to_numeric(historical[column], errors="coerce")
        invalid = completed & (~numeric.notna() | (numeric < 0) | (numeric % 1 != 0))
        invalid &= ~missing_scores
        if invalid.any():
            raise ValueError(f"Historical completed fixtures contain invalid {column}")
    excluded = int((~completed | missing_scores).sum())
    historical = historical[completed & ~missing_scores].copy()
    historical["date"] = pd.to_datetime(historical["date"])
    if historical["date"].isna().any():
        raise ValueError("Historical completed fixtures must have kickoff dates")
    historical = historical.sort_values(["date", "fixture_id"], kind="stable").reset_index(drop=True)
    n = len(historical)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    if train_end == 0 or validation_end <= train_end or validation_end >= n:
        raise ValueError("At least seven completed historical fixtures are required for non-empty splits")
    upcoming = combined[combined["season"] == latest].sort_values(
        ["date", "fixture_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    frames_to_save = {
        f"{league_id}-training-dataset": historical.iloc[:train_end],
        f"{league_id}-validation-dataset": historical.iloc[train_end:validation_end],
        f"{league_id}-test-dataset": historical.iloc[validation_end:],
        f"{league_id}-prediction-dataset": upcoming,
    }
    feature_columns = [column for column in historical if "__" in column]
    missing = historical[feature_columns].isna().sum()
    for column, count in missing[missing > 0].items():
        LOGGER.warning("Historical feature %s is missing for %d of %d fixtures; retained as NULL", column, count, n)
    if excluded:
        LOGGER.warning("Excluded %d unfinished or scoreless historical fixtures", excluded)
    target = Path(dataset_db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(target)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for name, frame in frames_to_save.items():
                connection.register("prepared_frame", frame)
                try:
                    connection.execute(f"CREATE OR REPLACE TABLE {_quote(name)} AS SELECT * FROM prepared_frame")
                    connection.execute(f"CREATE UNIQUE INDEX {_quote(name + '-fixture-id')} ON {_quote(name)} (fixture_id)")
                finally:
                    connection.unregister("prepared_frame")
            connection.execute(
                f"CREATE OR REPLACE TABLE {_quote(str(league_id) + '-dataset-metadata')} AS "
                "SELECT 'chronological-splits-v1' AS dataset_version, ?::INTEGER AS prediction_season, "
                "?::INTEGER[] AS historical_seasons, ?::INTEGER AS excluded_historical_rows, "
                "?::VARCHAR AS preparation_id",
                [latest, seasons[:-1], excluded, uuid.uuid4().hex],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    if progress_callback:
        progress_callback(total, total, "Saved chronological datasets.")
    return {
        "training_rows": train_end,
        "validation_rows": validation_end - train_end,
        "test_rows": n - validation_end,
        "prediction_rows": len(upcoming),
        "prediction_season": latest,
        "training_seasons": seasons[:-1],
        "excluded_historical_rows": excluded,
        "message": f"Prepared {n} historical fixtures and {len(upcoming)} fixtures for season {latest}; excluded {excluded} historical fixtures.",
    }
