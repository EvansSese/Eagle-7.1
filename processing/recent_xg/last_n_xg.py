"""
last_n_expected_goals.py

Same idea as expected_goals.py, but every rolling average involved —
both the league-wide baseline and each team's home/away scoring and
conceding rates — is computed over only the last `window` matches
instead of accumulating from the start of the season. This makes the
resulting xG more sensitive to recent form and less anchored to
early-season results that may no longer reflect a team's (or the
league's) current level.

Two rolling windows are involved, and they're windowed differently
because they operate over different units:

  1. League-wide averages (x_league_avg_goals, y_league_avg_goals):
     windowed over the last `window` ROUNDS (not individual matches),
     since a league average is inherently a property of a group of
     matches happening around the same time — pooling goals scored
     across the last `window` rounds' worth of fixtures, using only
     rounds EARLIER than the current one.

  2. Each team's home/away scoring and conceding rates
     (x_avg_home_scored / x_avg_home_conceded / y_avg_away_scored /
     y_avg_away_conceded): windowed over that team's own last `window`
     matches at that specific venue (home-only history for the home
     team, away-only history for the away team) — same venue-specific
     split as the original model, just capped to a rolling window
     instead of unbounded.

Attacking strength (AS), defensive strength (DS), and expected goals
(xG) are computed with the exact same formulas as expected_goals.py,
just fed by these windowed rates instead of season-cumulative ones:
       - x_team_as = x_avg_home_scored / x_league_avg_goals
       - x_team_ds = x_avg_home_conceded / y_league_avg_goals
       - y_team_as = y_avg_away_scored / y_league_avg_goals
       - y_team_ds = y_avg_away_conceded / x_league_avg_goals
       - x_team_xg = x_team_as * y_team_ds * x_league_avg_goals
       - y_team_xg = y_team_as * x_team_ds * y_league_avg_goals

Everything is computed using only data available BEFORE the match in
question — no lookahead.

Requires: duckdb, pandas
"""

from collections import deque

import duckdb
import numpy as np
import pandas as pd
from processing.core import COMPLETED_STATUSES, database_connection, iter_fixture_kickoffs
from processing.xG.expected_goals import add_league_rolling_averages

PRIOR_MATCHES = 5
DEFAULT_HOME_GOALS = 1.4
DEFAULT_AWAY_GOALS = 1.1


def safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising/NaN when denominator is 0."""
    return numerator / denominator if denominator else 0.0


def load_fixtures(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def determine_round_order(df: pd.DataFrame) -> dict:
    """
    Maps each unique round label to a chronological rank, based on the
    earliest match date within that round. Works for numbered rounds,
    named knockout rounds (e.g. 'Regular Season - Final'), or a mix.
    """
    round_first_date = df.groupby("round")["date"].min().sort_values()
    return {round_label: rank for rank, round_label in enumerate(round_first_date.index, start=1)}


def add_rolling_league_averages(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Adds x_league_avg_goals and y_league_avg_goals: for every match in a
    given round, both columns hold the league-wide average goals scored
    by home teams (x) and away teams (y), pooled across FT matches from
    only the last `window` EARLIER rounds, using only matches with an
    earlier kickoff. Rounds older than the window do not contribute.
    """
    return add_league_rolling_averages(df, window=window)


def add_team_home_away_rolling_averages(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Adds, using only each team's own last `window` matches at that
    specific venue:
      - x_avg_home_scored / x_avg_home_conceded: the home team's
        (x_team) average goals scored/conceded across its last `window`
        HOME matches only.
      - y_avg_away_scored / y_avg_away_conceded: the away team's
        (y_team) average goals scored/conceded across its last `window`
        AWAY matches only.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True).copy()
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")

    # team -> deque of (goals_scored, goals_conceded) for their last
    # `window` matches at that specific venue
    home_history = {}
    away_history = {}

    x_avg_home_scored_col = []
    x_avg_home_conceded_col = []
    y_avg_away_scored_col = []
    y_avg_away_conceded_col = []
    x_home_matches_played_col = []
    y_away_matches_played_col = []

    for _, kickoff in iter_fixture_kickoffs(df):
        finished_results = []
        for row in kickoff:
            x_team, y_team = row.home_team, row.away_team

            x_hist = home_history.setdefault(x_team, deque(maxlen=window))
            y_hist = away_history.setdefault(y_team, deque(maxlen=window))

            x_scored = sum(scored for scored, _ in x_hist)
            x_conceded = sum(conceded for _, conceded in x_hist)
            y_scored = sum(scored for scored, _ in y_hist)
            y_conceded = sum(conceded for _, conceded in y_hist)
            x_mp = len(x_hist)
            y_mp = len(y_hist)

            x_avg_home_scored_col.append(safe_div(x_scored, x_mp))
            x_avg_home_conceded_col.append(safe_div(x_conceded, x_mp))
            y_avg_away_scored_col.append(safe_div(y_scored, y_mp))
            y_avg_away_conceded_col.append(safe_div(y_conceded, y_mp))
            x_home_matches_played_col.append(x_mp)
            y_away_matches_played_col.append(y_mp)

            if row.status in COMPLETED_STATUSES:
                if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                    raise ValueError("completed fixtures must include both teams' goals")
                finished_results.append(
                    (x_hist, y_hist, row.home_goals, row.away_goals)
                )

        for x_hist, y_hist, home_goals, away_goals in finished_results:
            x_hist.append((home_goals, away_goals))
            y_hist.append((away_goals, home_goals))

    df["x_avg_home_scored"] = x_avg_home_scored_col
    df["x_avg_home_conceded"] = x_avg_home_conceded_col
    df["y_avg_away_scored"] = y_avg_away_scored_col
    df["y_avg_away_conceded"] = y_avg_away_conceded_col
    df["x_home_matches_played"] = x_home_matches_played_col
    df["y_away_matches_played"] = y_away_matches_played_col
    return df


def add_strength_and_expected_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds attacking strength (AS), defensive strength (DS), and expected
    goals (xG) for both the home team (x_team) and away team (y_team),
    relative to the rolling league averages already computed.
    """
    df = df.copy()
    df["x_league_avg_goals"] = df["x_league_avg_goals"].replace(
        0, DEFAULT_HOME_GOALS
    )
    df["y_league_avg_goals"] = df["y_league_avg_goals"].replace(
        0, DEFAULT_AWAY_GOALS
    )
    x_home_weight = df["x_home_matches_played"] / (
        df["x_home_matches_played"] + PRIOR_MATCHES
    )
    y_away_weight = df["y_away_matches_played"] / (
        df["y_away_matches_played"] + PRIOR_MATCHES
    )
    x_home_scored = (
        x_home_weight * df["x_avg_home_scored"]
        + (1 - x_home_weight) * df["x_league_avg_goals"]
    )
    x_home_conceded = (
        x_home_weight * df["x_avg_home_conceded"]
        + (1 - x_home_weight) * df["y_league_avg_goals"]
    )
    y_away_scored = (
        y_away_weight * df["y_avg_away_scored"]
        + (1 - y_away_weight) * df["y_league_avg_goals"]
    )
    y_away_conceded = (
        y_away_weight * df["y_avg_away_conceded"]
        + (1 - y_away_weight) * df["x_league_avg_goals"]
    )

    df["x_team_as"] = np.divide(
        x_home_scored,
        df["x_league_avg_goals"],
        out=np.zeros(len(df), dtype=float),
        where=df["x_league_avg_goals"].to_numpy() != 0,
    )
    df["x_team_ds"] = np.divide(
        x_home_conceded,
        df["y_league_avg_goals"],
        out=np.zeros(len(df), dtype=float),
        where=df["y_league_avg_goals"].to_numpy() != 0,
    )
    df["y_team_as"] = np.divide(
        y_away_scored,
        df["y_league_avg_goals"],
        out=np.zeros(len(df), dtype=float),
        where=df["y_league_avg_goals"].to_numpy() != 0,
    )
    df["y_team_ds"] = np.divide(
        y_away_conceded,
        df["x_league_avg_goals"],
        out=np.zeros(len(df), dtype=float),
        where=df["x_league_avg_goals"].to_numpy() != 0,
    )

    df["x_team_xg"] = df["x_team_as"] * df["y_team_ds"] * df["x_league_avg_goals"]
    df["y_team_xg"] = df["y_team_as"] * df["x_team_ds"] * df["y_league_avg_goals"]

    return df


def build_expected_goals_table(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Runs the full rolling pipeline and returns just the columns needed for the output table."""
    df = add_rolling_league_averages(df, window=window)
    df = add_team_home_away_rolling_averages(df, window=window)
    df = add_strength_and_expected_goals(df)

    return df[[
        "league_id","fixture_id","date", "round", "home_team", "away_team", "home_goals", "away_goals", "status",
        "x_team_as", "y_team_as", "x_team_ds", "y_team_ds",
        "x_team_xg", "y_team_xg",
        "x_league_avg_goals", "y_league_avg_goals",
    ]]


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    with database_connection(db_path, connection) as con:
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        con.register("df_view", df)
        try:
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_view')
        finally:
            con.unregister("df_view")


if __name__ == "__main__":
    FORM_WINDOW = [3,5,10]  # using last 3 matches for testing

    INPUT_DB_PATH = "../../storage/database/leagues/England.ddb"  # path to the duckdb file with raw fixtures
    OUTPUT_DB_PATH = "../../storage/database/leagues/England.ddb"  # output duckdb database file

    for window in FORM_WINDOW:
        INPUT_TABLE = "39-2025-fixtures"  # name of the fixtures table inside that db
        OUTPUT_TABLE = f"39-2025-xg-l{window}"  # table name to create in the output db

        fixtures = load_fixtures(db_path=INPUT_DB_PATH, table_name=INPUT_TABLE)
        result = build_expected_goals_table(fixtures, window=window)

        write_output(result, db_path=OUTPUT_DB_PATH, table_name=OUTPUT_TABLE)

        print(f"Wrote {len(result)} rows to '{OUTPUT_TABLE}' in {OUTPUT_DB_PATH}")
        print(result.head(15).to_string(index=False))