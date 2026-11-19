"""Pre-match football power ratings and scoped extensions."""

from collections import defaultdict
from itertools import groupby
import math
from numbers import Real

import pandas as pd

from processing.core import COMPLETED_STATUSES, database_connection, iter_fixture_kickoffs, write_fixture_table


INITIAL_RATING = 10.0
ADJUSTER = 0.25
HOME_ADVANTAGE = 0.5
WINDOWS = (3, 5, 10)
SCOPES = ("overall", "home_only", "away_only", "last_3", "last_5", "last_10")


def load_fixtures(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Load fixtures from a DuckDB table."""
    with database_connection(db_path, connection, read_only=True) as con:
        fixtures = con.execute(f'SELECT * FROM "{table_name}"').df()
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    return fixtures


def _change(team_rating, opponent_rating, goal_difference, venue_advantage):
    return ADJUSTER * (goal_difference - (team_rating - opponent_rating) - venue_advantage)


def _replay(history):
    """Replay a team's selected matches using recorded opponent ratings."""
    rating = INITIAL_RATING
    for _, matches in groupby(history, key=lambda match: match[0]):
        changes = [
            _change(rating, opponent, goals, advantage)
            for _, opponent, goals, advantage in matches
        ]
        rating += sum(changes)
    return rating


def add_power_ratings_before_match(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Compute each fixture's ratings from earlier completed matches only."""
    required = {
        "fixture_id", "date", "home_team", "away_team",
        "home_goals", "away_goals", "status",
    }
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"fixtures are missing required columns: {sorted(missing)}")
    df = fixtures.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")
    if df["fixture_id"].isna().any() or df["fixture_id"].duplicated().any():
        raise ValueError("fixture IDs must be present and unique")
    for column in ("season", "league_id"):
        if column in df and df[column].isna().any():
            raise ValueError(f"fixture {column} values cannot be missing")
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    seasons = df["season"].astype(str).tolist() if "season" in df else [""] * len(df)
    leagues = df["league_id"].astype(str).tolist() if "league_id" in df else [""] * len(df)

    histories = defaultdict(list)
    records = []
    position = 0
    for kickoff_date, kickoff in iter_fixture_kickoffs(df):
        finished = []
        for row in kickoff:
            group = (leagues[position], seasons[position])
            position += 1
            home, away = row.home_team, row.away_team
            if (not isinstance(home, str) or not home.strip()
                    or not isinstance(away, str) or not away.strip()):
                raise ValueError("fixture team names cannot be empty")
            if home == away:
                raise ValueError("fixture teams must be different")

            ratings = {}
            record = {}
            for side, team in (("x", home), ("y", away)):
                for scope in SCOPES:
                    history = histories[(group, team, scope)]
                    selected = history[-int(scope[5:]):] if scope.startswith("last_") else history
                    ratings[(side, scope)] = _replay(selected)
                    record[f"{side}_team_{scope}_power_rating_before"] = ratings[(side, scope)]
                    record[f"{side}_team_{scope}_matches_before"] = len(selected)
            for scope in SCOPES:
                record[f"{scope}_match_power_rating_before"] = (
                    ratings[("x", scope)] - ratings[("y", scope)]
                )
                record[f"{scope}_expected_goal_margin_before"] = (
                    record[f"{scope}_match_power_rating_before"] + HOME_ADVANTAGE
                )
            record["venue_match_power_rating_before"] = (
                ratings[("x", "home_only")] - ratings[("y", "away_only")]
            )
            record["venue_expected_goal_margin_before"] = (
                record["venue_match_power_rating_before"] + HOME_ADVANTAGE
            )
            record["both_teams_at_least_5_matches_before"] = (
                record["x_team_overall_matches_before"] >= 5
                and record["y_team_overall_matches_before"] >= 5
            )
            records.append(record)

            if row.status not in COMPLETED_STATUSES:
                continue
            for goals in (row.home_goals, row.away_goals):
                if (isinstance(goals, bool) or not isinstance(goals, Real)
                        or not math.isfinite(goals) or goals < 0 or goals != int(goals)):
                    raise ValueError("completed fixtures must include finite, non-negative integer goals")
            finished.append((group, home, away, row.home_goals - row.away_goals, ratings))

        for group, home, away, goal_difference, ratings in finished:
            for scope in SCOPES:
                if scope != "away_only":
                    histories[(group, home, scope)].append(
                        (kickoff_date, ratings[("y", scope)], goal_difference, HOME_ADVANTAGE)
                    )
                if scope != "home_only":
                    histories[(group, away, scope)].append(
                        (kickoff_date, ratings[("x", scope)], -goal_difference, -HOME_ADVANTAGE)
                    )

    columns = [
        column
        for side in ("x", "y")
        for scope in SCOPES
        for column in (
            f"{side}_team_{scope}_power_rating_before",
            f"{side}_team_{scope}_matches_before",
        )
    ] + [
        column for scope in SCOPES for column in (
            f"{scope}_match_power_rating_before", f"{scope}_expected_goal_margin_before",
        )
    ] + [
        "venue_match_power_rating_before",
        "venue_expected_goal_margin_before",
        "both_teams_at_least_5_matches_before",
    ]
    features = pd.DataFrame.from_records(records, columns=columns, index=df.index)
    for column in columns:
        if column.endswith("_matches_before"):
            features[column] = features[column].astype("int64")
        elif column == "both_teams_at_least_5_matches_before":
            features[column] = features[column].astype(bool)
        else:
            features[column] = features[column].astype(float)
    return pd.concat([df, features], axis=1)


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save ratings in a fixture-keyed DuckDB table."""
    write_fixture_table(df, db_path, table_name, connection)
