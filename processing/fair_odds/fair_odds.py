"""
fair_odds.py

Dixon-Coles Poisson fair odds from the pre-match expected goals (xG) table.

Steps per match:
  1. Take the model's expected goals: x_team_home_expected_goals (home side)
     and y_team_away_expected_goals (away side).
  2. Scale them by each team's recent FORM (team_form.TeamFormTracker): the
     recency-weighted ratio of what the team actually scored/conceded in its
     last matches to what the xG model expected, so recent games weigh more:
         x_team_adjusted_xg = x_xg * x_attack_form * y_defense_form
         y_team_adjusted_xg = y_xg * y_attack_form * x_defense_form
     Only matches that kicked off earlier than the fixture feed the form.
  3. Use the adjusted values as the means (lambda, mu) of independent Poisson
     goal distributions and apply the Dixon-Coles low-score correction
     (dixon_coles.py) to get the score grid. rho is league/season specific:
     it is read from the "<league>-<season>-rho" table (rho_estimation.py,
     estimated from earlier rounds only) when available.
  4. Derive win / draw / win probabilities, the most likely correct score and
     decimal odds (1 / probability, 2 dp).

Output table "<league>-<season>-odds" is keyed by fixture_id.

Requires: duckdb, pandas, numpy
"""

import numpy as np
import pandas as pd

from processing.core import COMPLETED_STATUSES, database_connection, write_fixture_table
from processing.fair_odds.dixon_coles import (  # noqa: F401  (re-exported for other modules)
    MAX_GOALS,
    RHO,
    bounded_rho,
    build_score_matrix,
    dixon_coles_tau,
    most_likely_score,
    outcome_probabilities,
    poisson_pmf,
    probability_to_decimal_odds,
)
from processing.fair_odds.team_form import TeamFormTracker

BASE_COLUMNS = [
    "league_id", "fixture_id", "date", "round", "home_team", "away_team",
    "home_goals", "away_goals", "status",
]
ODDS_COLUMNS = [
    "x_team_win_probability", "draw_probability", "y_team_win_probabilty",
    "x_team_cs", "y_team_cs",
    "home_win_odds", "draw_odds", "away_win_odds",
]
FORM_COLUMNS = [
    "x_team_attack_form", "x_team_defense_form",
    "y_team_attack_form", "y_team_defense_form",
    "x_team_adjusted_xg", "y_team_adjusted_xg",
]


def load_expected_goals(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Load the xG table with `date` parsed as a datetime."""
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_team_form(df: pd.DataFrame, tracker: TeamFormTracker = None) -> pd.DataFrame:
    """
    Add the form factors and form-adjusted expected goals (FORM_COLUMNS).

    Fixtures are walked in kickoff order. A finished match is recorded only
    after every fixture with the same kickoff has been read, so a match never
    sees itself or a simultaneous match. Matches where either side's xG is 0
    (no model expectation yet, e.g. round 1) are not recorded, since their
    "expected" side would be meaningless. One pass: O(n) with O(1) work per
    fixture because each team's history is a short bounded deque.
    """
    tracker = tracker or TeamFormTracker()
    df = df.copy()
    n = len(df)
    columns = {name: [1.0] * n for name in FORM_COLUMNS}

    order = df.sort_values(["date", "fixture_id"], kind="stable").index
    positions = {label: pos for pos, label in enumerate(df.index)}
    pending, pending_date = [], None

    def flush():
        for home, away, when, hg, ag, lam, mu in pending:
            tracker.record(home, when, hg, ag, lam, mu)
            tracker.record(away, when, ag, hg, mu, lam)
        pending.clear()

    for label in order:
        row = df.loc[label]
        when = row["date"]
        if pending_date is not None and when != pending_date:
            flush()
        pending_date = when
        pos = positions[label]

        lam, mu = float(row["x_team_home_expected_goals"]), float(row["y_team_away_expected_goals"])
        x_attack, x_defense = tracker.form(row["home_team"], when)
        y_attack, y_defense = tracker.form(row["away_team"], when)
        columns["x_team_attack_form"][pos] = x_attack
        columns["x_team_defense_form"][pos] = x_defense
        columns["y_team_attack_form"][pos] = y_attack
        columns["y_team_defense_form"][pos] = y_defense
        columns["x_team_adjusted_xg"][pos] = lam * x_attack * y_defense
        columns["y_team_adjusted_xg"][pos] = mu * y_attack * x_defense

        if row["status"] in COMPLETED_STATUSES and lam > 0 and mu > 0 \
                and pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            pending.append((row["home_team"], row["away_team"], when,
                            float(row["home_goals"]), float(row["away_goals"]), lam, mu))
    flush()

    for name, values in columns.items():
        df[name] = values
    return df


def compute_match_probabilities(df: pd.DataFrame, rho: float = RHO, max_goals: int = MAX_GOALS,
                                use_form: bool = True, tracker: TeamFormTracker = None,
                                rho_table: pd.DataFrame = None) -> pd.DataFrame:
    """
    Fair odds for every match. `rho_table` (fixture_id, rho; see
    rho_estimation.py) gives each fixture its own league/season rho, which is
    also saved in a `rho` column; fixtures missing from it, or no table at
    all, use the constant `rho`. With use_form (default) the xG means are
    first adjusted by recency-weighted team form and the form columns are
    included; with use_form=False the raw xG values are used and the output
    keeps just the base and odds columns.
    """
    if use_form:
        df = add_team_form(df, tracker)
        lam_col, mu_col = "x_team_adjusted_xg", "y_team_adjusted_xg"
    else:
        df = df.copy()
        lam_col, mu_col = "x_team_home_expected_goals", "y_team_away_expected_goals"

    if rho_table is not None:
        rho_by_fixture = rho_table.drop_duplicates("fixture_id").set_index("fixture_id")["rho"]
        rhos = df["fixture_id"].map(rho_by_fixture).fillna(rho).to_numpy(float)
        df["rho"] = rhos
    else:
        rhos = np.full(len(df), rho, dtype=float)

    x_win, draw, y_win, x_cs, y_cs = [], [], [], [], []
    for lam, mu, match_rho in zip(df[lam_col].to_numpy(float), df[mu_col].to_numpy(float), rhos):
        matrix = build_score_matrix(lam, mu, rho=match_rho, max_goals=max_goals)
        outcomes = outcome_probabilities(matrix)
        home_score, away_score = most_likely_score(matrix)
        x_win.append(outcomes["home_win"])
        draw.append(outcomes["draw"])
        y_win.append(outcomes["away_win"])
        x_cs.append(home_score)
        y_cs.append(away_score)

    df["x_team_win_probability"] = x_win
    df["draw_probability"] = draw
    df["y_team_win_probabilty"] = y_win
    df["x_team_cs"] = x_cs
    df["y_team_cs"] = y_cs
    df["home_win_odds"] = [probability_to_decimal_odds(p) for p in x_win]
    df["draw_odds"] = [probability_to_decimal_odds(p) for p in draw]
    df["away_win_odds"] = [probability_to_decimal_odds(p) for p in y_win]

    columns = BASE_COLUMNS + (["rho"] if rho_table is not None else []) + ODDS_COLUMNS \
        + (FORM_COLUMNS if use_form else [])
    result = df[columns]
    if use_form:
        result = result.round({name: 2 for name in FORM_COLUMNS})
    return result


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save the odds as a table with fixture_id as its unique key."""
    write_fixture_table(df, db_path, table_name, connection)


if __name__ == "__main__":
    INPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    INPUT_TABLE = "39-2025-xg"
    OUTPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    OUTPUT_TABLE = "39-2025-odds"

    expected_goals = load_expected_goals(db_path=INPUT_DB_PATH, table_name=INPUT_TABLE)
    result = compute_match_probabilities(expected_goals)
    write_output(result, db_path=OUTPUT_DB_PATH, table_name=OUTPUT_TABLE)

    print(f"Wrote {len(result)} rows to '{OUTPUT_TABLE}' in {OUTPUT_DB_PATH}")
    print(result.head(15).to_string(index=False))
