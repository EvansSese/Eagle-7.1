"""
xg_metrics.py

Shared, reusable building blocks for the attack/defence-strength expected
goals (xG) model. Every value for a fixture uses ONLY matches that finished
before it (no lookahead).

Model (home scope for the fixture's home team x, away scope for its away
team y - there is no overall scope):
  home_league_avg = goals scored by home teams in the previous round / matches in that round
  away_league_avg = goals scored by away teams in the previous round / matches in that round

  x_team_home_attacking_strength = x home goals scored per match   / home_league_avg
  x_team_home_defense_strength   = x home goals conceded per match / away_league_avg
  y_team_away_attacking_strength = y away goals scored per match   / away_league_avg
  y_team_away_defense_strength   = y away goals conceded per match / home_league_avg

  x_team_home_expected_goals = x_attack * y_defense * home_league_avg
  y_team_away_expected_goals = y_attack * x_defense * away_league_avg

A division whose denominator is 0 (e.g. round 1, nothing played yet) yields
0. Intermediate values keep full precision; every stored value is rounded to
2 decimals.

Efficiency: everything is vectorised (group cumulative sums), so the cost is
O(n log n) for the sort and O(n) for the rest, with no per-fixture Python
loop and no per-team history stored.
"""

import numpy as np
import pandas as pd

from processing.core import COMPLETED_STATUSES

DECIMALS = 2


def safe_ratio(numerator, denominator) -> np.ndarray:
    """Element-wise numerator / denominator, with 0.0 wherever the denominator is 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros(numerator.shape), where=denominator != 0)


def determine_round_order(df: pd.DataFrame) -> dict:
    """
    Map each round label to a chronological rank using the earliest match
    date in that round. Unlike parsing a number from the label, this works
    for any naming scheme (numbered rounds, knockout rounds, or a mix).
    """
    round_first_date = df.groupby("round")["date"].min().sort_values()
    return {round_label: rank for rank, round_label in enumerate(round_first_date.index, start=1)}


def add_league_rolling_averages(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Add x_league_avg_goals / y_league_avg_goals: for each fixture, the average
    goals scored by home (x) and away (y) teams across all finished matches in
    EARLIER rounds. All fixtures of a round share the value. With `window`,
    only that many earlier rounds contribute (window=1: previous round only,
    as used by the xG model). Returns rows sorted by
    round_number, date, fixture_id (round_number is added as a column).
    """
    if window is not None and window <= 0:
        raise ValueError("window must be a positive integer")
    df = df.copy()
    df["round_number"] = df["round"].map(determine_round_order(df))
    df = df.sort_values(["round_number", "date", "fixture_id"]).reset_index(drop=True)

    finished = df["status"].isin(COMPLETED_STATUSES)
    totals = pd.DataFrame({
        "home": df["home_goals"].where(finished, 0),
        "away": df["away_goals"].where(finished, 0),
        "matches": finished.astype(int),
    }).groupby(df["round_number"], sort=True).sum()
    if window is None:
        previous = totals.cumsum().shift(1, fill_value=0)
    else:
        previous = totals.rolling(window, min_periods=1).sum().shift(1, fill_value=0)
    counts = previous["matches"].replace(0, np.nan)
    df["x_league_avg_goals"] = df["round_number"].map((previous["home"] / counts).fillna(0.0)).fillna(0.0)
    df["y_league_avg_goals"] = df["round_number"].map((previous["away"] / counts).fillna(0.0)).fillna(0.0)
    return df


def prior_venue_totals(team: pd.Series, scored: pd.Series, conceded: pd.Series, finished: pd.Series) -> tuple:
    """
    For rows already in chronological order, return (matches, goals scored,
    goals conceded) the row's team accumulated at this venue type in its
    earlier finished matches. `team` is the home_team (home scope) or the
    away_team (away scope) of each row. A team never appears twice in one
    kickoff, so the running sum minus the row's own contribution is exactly
    the total before this match.
    """
    own = pd.DataFrame({
        "n": finished.astype(np.int64),
        "scored": scored.where(finished, 0).astype(np.float64),
        "conceded": conceded.where(finished, 0).astype(np.float64),
    })
    running = own.groupby(team.to_numpy(), sort=False).cumsum()
    before = running - own
    return before["n"].to_numpy(), before["scored"].to_numpy(), before["conceded"].to_numpy()


def add_expected_goals_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add league averages, home/away per-match rates, strengths and expected
    goals (see module docstring) to `df`, which must already carry
    x_league_avg_goals / y_league_avg_goals from add_league_rolling_averages.
    Returns rows sorted by date, fixture_id.
    """
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")
    finished = df["status"].isin(COMPLETED_STATUSES)
    if (finished & (df["home_goals"].isna() | df["away_goals"].isna())).any():
        raise ValueError("completed fixtures must include both teams' goals")

    hn, hs, hc = prior_venue_totals(df["home_team"], df["home_goals"], df["away_goals"], finished)
    an, a_s, ac = prior_venue_totals(df["away_team"], df["away_goals"], df["home_goals"], finished)

    home_avg = df["x_league_avg_goals"].to_numpy()
    away_avg = df["y_league_avg_goals"].to_numpy()
    x_scored_pm, x_conceded_pm = safe_ratio(hs, hn), safe_ratio(hc, hn)
    y_scored_pm, y_conceded_pm = safe_ratio(a_s, an), safe_ratio(ac, an)

    x_attack, x_defense = safe_ratio(x_scored_pm, home_avg), safe_ratio(x_conceded_pm, away_avg)
    y_attack, y_defense = safe_ratio(y_scored_pm, away_avg), safe_ratio(y_conceded_pm, home_avg)

    values = {
        "home_league_avg_goals": home_avg,
        "away_league_avg_goals": away_avg,
        "x_team_home_matches_played": hn,
        "y_team_away_matches_played": an,
        "x_team_home_goals_scored_per_match": x_scored_pm,
        "x_team_home_goals_conceded_per_match": x_conceded_pm,
        "y_team_away_goals_scored_per_match": y_scored_pm,
        "y_team_away_goals_conceded_per_match": y_conceded_pm,
        "x_team_home_attacking_strength": x_attack,
        "x_team_home_defense_strength": x_defense,
        "y_team_away_attacking_strength": y_attack,
        "y_team_away_defense_strength": y_defense,
        "x_team_home_expected_goals": x_attack * y_defense * home_avg,
        "y_team_away_expected_goals": y_attack * x_defense * away_avg,
    }
    for name, column in values.items():
        df[name] = column if name.endswith("matches_played") else np.round(column, DECIMALS)
    return df
