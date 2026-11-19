"""
expected_goals.py

Pre-match expected goals (xG) for the home team (x_team) and away team
(y_team) from a league-relative attack/defence-strength model. Only
home-only data is used for the home team and away-only data for the away
team, and only matches finished before each fixture count. The formulas and
all shared calculations live in xg_metrics.py; see its docstring.

Output table: "<league>-<season>-xg", keyed by fixture_id. Columns:
  home_league_avg_goals, away_league_avg_goals,
  x_team_home_matches_played, x_team_home_goals_scored_per_match,
  x_team_home_goals_conceded_per_match, x_team_home_attacking_strength,
  x_team_home_defense_strength, x_team_home_expected_goals,
  y_team_away_matches_played, y_team_away_goals_scored_per_match,
  y_team_away_goals_conceded_per_match, y_team_away_attacking_strength,
  y_team_away_defense_strength, y_team_away_expected_goals

Requires: duckdb, pandas
"""

import pandas as pd
from processing.core import database_connection, write_fixture_table
from processing.xG.xg_metrics import (  # noqa: F401  (re-exported for other modules)
    add_expected_goals_columns,
    add_league_rolling_averages,
    determine_round_order,
)

# League averages come from the single previous round only.
PREVIOUS_ROUNDS = 1

OUTPUT_COLUMNS = [
    "league_id", "fixture_id", "date", "round", "home_team", "away_team",
    "home_goals", "away_goals", "status",
    "home_league_avg_goals", "away_league_avg_goals",
    "x_team_home_matches_played",
    "x_team_home_goals_scored_per_match", "x_team_home_goals_conceded_per_match",
    "x_team_home_attacking_strength", "x_team_home_defense_strength",
    "x_team_home_expected_goals",
    "y_team_away_matches_played",
    "y_team_away_goals_scored_per_match", "y_team_away_goals_conceded_per_match",
    "y_team_away_attacking_strength", "y_team_away_defense_strength",
    "y_team_away_expected_goals",
]


def load_fixtures(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Load a fixtures table with `date` parsed as a datetime."""
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_expected_goals_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full pipeline and return the output columns, sorted by date, fixture_id."""
    df = add_league_rolling_averages(df, window=PREVIOUS_ROUNDS)
    df = add_expected_goals_columns(df)
    return df[[c for c in OUTPUT_COLUMNS if c in df.columns]]


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save the result as a table with fixture_id as its unique key."""
    write_fixture_table(df, db_path, table_name, connection)


if __name__ == "__main__":
    INPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    INPUT_TABLE = "39-2025-fixtures"
    OUTPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    OUTPUT_TABLE = "39-2025-xg"

    fixtures = load_fixtures(db_path=INPUT_DB_PATH, table_name=INPUT_TABLE)
    result = build_expected_goals_table(fixtures)
    write_output(result, db_path=OUTPUT_DB_PATH, table_name=OUTPUT_TABLE)

    print(f"Wrote {len(result)} rows to '{OUTPUT_TABLE}' in {OUTPUT_DB_PATH}")
    print(result[["fixture_id", "round", "home_team", "away_team",
                  "x_team_home_expected_goals", "y_team_away_expected_goals"]].head(15).to_string(index=False))
