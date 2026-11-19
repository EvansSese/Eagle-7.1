"""
goals.py

Cumulative goal metrics for every fixture, computed from the matches played
BEFORE it within the supplied fixtures (round 1 is therefore all zeros).
Covers overall / home-only / away-only goals scored and conceded, per-match
rates (2 d.p.), goal frequency (1, 2, 3, above-3 goals) and clean sheets.
The calculations live in goal_metrics.py; see its docstring for the columns.

Output table: "<league>-<season>-goals", keyed by fixture_id.
"""

import pandas as pd
from processing.core import database_connection, write_fixture_table
from processing.goals.goal_metrics import build_goal_features


def load_fixtures(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Load a fixtures table with `date` parsed as a datetime."""
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_goals_before_match(df: pd.DataFrame, history: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add cumulative goal metrics (all matches before each fixture) to `df`.
    Optional `history` (earlier fixtures) seeds the totals instead of
    starting every team from zero.
    """
    return build_goal_features(df, window=None, history=history)


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save the result as a table with fixture_id as its unique key."""
    write_fixture_table(df, db_path, table_name, connection)


if __name__ == "__main__":
    INPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    INPUT_TABLE = "39-2025-fixtures"
    OUTPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    OUTPUT_TABLE = "39-2025-goals"

    fixtures = load_fixtures(INPUT_DB_PATH, INPUT_TABLE)
    result = add_goals_before_match(fixtures)
    write_output(result, OUTPUT_DB_PATH, OUTPUT_TABLE)

    print(f"Wrote {len(result)} rows to '{OUTPUT_TABLE}' in {OUTPUT_DB_PATH}")
    print(result[["fixture_id", "round", "home_team", "away_team",
                  "x_team_goals_scored", "x_team_goals_conceded", "x_team_goals_scored_per_match",
                  "x_team_clean_sheets", "y_team_goals_scored", "y_team_goals_conceded",
                  "y_team_goals_scored_per_match", "y_team_clean_sheets"]].head(10).to_string(index=False))
