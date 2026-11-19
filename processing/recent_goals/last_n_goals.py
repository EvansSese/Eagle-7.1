"""
last_n_goals.py

Goal metrics over only each team's last `window` matches (3, 5 or 10)
before the fixture. Same metrics as goals.py - overall, home-only and
away-only goals, per-match rates, goal frequency and clean sheets - but
home-only/away-only windows hold the team's last N home/away matches.
Columns end in `_last{window}`; calculations live in
processing/goals/goal_metrics.py.

Output table: "<league>-<season>-goals-l<window>", keyed by fixture_id.
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


def add_goals_last_n_matches(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add goal metrics over each team's last `window` matches before each fixture."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    return build_goal_features(df, window=window)


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save the result as a table with fixture_id as its unique key."""
    write_fixture_table(df, db_path, table_name, connection)


if __name__ == "__main__":
    FORM_WINDOWS = [3, 5, 10]
    DB_PATH = "../../storage/database/leagues/England.ddb"
    INPUT_TABLE = "39-2025-fixtures"

    fixtures = load_fixtures(DB_PATH, INPUT_TABLE)
    for window in FORM_WINDOWS:
        output_table = f"39-2025-goals-l{window}"
        result = add_goals_last_n_matches(fixtures, window=window)
        write_output(result, DB_PATH, output_table)
        print(f"Wrote {len(result)} rows to '{output_table}' in {DB_PATH}")
