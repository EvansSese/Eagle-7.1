"""
last_n_fair_odds.py

Same model as fair_odds.py — Poisson goal distributions with the
improved (bounded-rho) Dixon-Coles low-score correction — but reads its
expected goals from the ROLLING last-N-matches xG table (produced by
last_n_expected_goals.py) instead of the season-cumulative one. The
Dixon-Coles math itself is identical; only the source of x_team_xg /
y_team_xg changes, so a match's fair odds here reflect each team's
recent form over their last `window` matches rather than their whole
season so far.

From the resulting joint probability grid we derive:
  - x_team_win_probability: P(home goals > away goals)
  - draw_probability:       P(home goals == away goals)
  - y_team_win_probabilty:  P(home goals < away goals)
  - x_team_cs / y_team_cs:  the single most probable exact scoreline
                             (the "correct score"), home and away goals
                             respectively

Odds are computed as decimal (European) odds = 1 / probability, via
`probability_to_decimal_odds`, and included as home_win_odds, draw_odds,
and away_win_odds.

Requires: duckdb, pandas, numpy
"""

import math
import duckdb
import pandas as pd
from processing.core import database_connection
import numpy as np

# Dixon-Coles low-score correlation parameter. The original 1997 paper
# estimated rho around -0.13 for English football; small negative values
# are typical. Adjust if you fit rho to your own data.
RHO = -0.05

# How many goals (0..MAX_GOALS) to include per side when building the
# score grid. Probability mass beyond this is negligible for football.
MAX_GOALS = 10
# ----------------------------------------------------


def load_expected_goals(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for a Poisson distribution with mean lam. Handles lam == 0 (all mass at k=0)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def bounded_rho(lam: float, mu: float, rho: float) -> float:
    """
    Improved Dixon-Coles refinement: clamps rho so that all four
    adjusted low-score cells (0-0, 1-0, 0-1, 1-1) stay mathematically
    valid (non-negative) for this specific match's lambda/mu, instead
    of relying on the original 1997 model's fixed rho plus a post-hoc
    max(p, 0.0) clip.

    The four cells require:
      1 - lambda*mu*rho >= 0  ->  rho <= 1 / (lambda*mu)
      1 + lambda*rho    >= 0  ->  rho >= -1 / lambda
      1 + mu*rho        >= 0  ->  rho >= -1 / mu
      1 - rho           >= 0  ->  rho <= 1

    Combining these gives a valid [lower, upper] range for rho; the
    requested rho is clamped into that range.
    """
    if lam <= 0 or mu <= 0:
        return 0.0

    upper_bound = min(1.0, 1.0 / (lam * mu))
    lower_bound = max(-1.0, -1.0 / lam, -1.0 / mu)

    return max(lower_bound, min(rho, upper_bound))


def dixon_coles_tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """
    The Dixon-Coles correction factor tau(i,j), applied only to the four
    low-scoring cells where the independent-Poisson assumption is known
    to break down slightly. All other scorelines get a factor of 1.0
    (no adjustment). Callers should pass an already-bounded rho (see
    bounded_rho) so every returned factor is non-negative by construction.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - (lam * mu * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (lam * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (mu * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lam: float, mu: float, rho: float = RHO, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Builds a (max_goals+1) x (max_goals+1) grid where cell [i, j] holds
    the improved-Dixon-Coles-adjusted probability of a final score of
    i-j (home-away).

    rho is bounded per match via bounded_rho() before being applied, so
    the four adjusted low-score cells are guaranteed non-negative for
    this match's own lambda/mu.

    The grid is renormalized to sum to 1, which corrects for truncation
    at max_goals.
    """
    effective_rho = bounded_rho(lam, mu, rho)

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    home_pmf = [poisson_pmf(i, lam) for i in range(max_goals + 1)]
    away_pmf = [poisson_pmf(j, mu) for j in range(max_goals + 1)]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_pmf[i] * away_pmf[j] * dixon_coles_tau(i, j, lam, mu, effective_rho)
            matrix[i, j] = max(p, 0.0)  # defensive only; bounded_rho should make this a no-op

    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


def outcome_probabilities(matrix: np.ndarray) -> dict:
    """
    Sums the score grid into home win / draw / away win probabilities.
    Row index = home goals, column index = away goals.
    """
    home_win = float(np.tril(matrix, k=-1).sum())   # home goals > away goals
    draw = float(np.trace(matrix))                  # home goals == away goals
    away_win = float(np.triu(matrix, k=1).sum())     # home goals < away goals
    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def most_likely_score(matrix: np.ndarray) -> tuple:
    """Returns (home_goals, away_goals) for the single most probable scoreline."""
    home_goals, away_goals = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(home_goals), int(away_goals)


def probability_to_decimal_odds(probability: float) -> float:
    """Converts a probability into decimal (European) odds: 1 / p. Returns inf for p == 0."""
    return round(1 / probability, 2) if probability > 0 else float(1000000000)


def compute_match_probabilities(df: pd.DataFrame, rho: float = RHO, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """
    For every match, builds the Dixon-Coles score grid from x_team_xg /
    y_team_xg (here: rolling last-N-matches xG), then derives win/draw/
    win probabilities and the most likely correct score.
    """
    df = df.copy()

    x_win_prob, draw_prob, y_win_prob = [], [], []
    x_cs, y_cs = [], []

    for _, row in df.iterrows():
        lam = row["x_team_xg"]
        mu = row["y_team_xg"]

        matrix = build_score_matrix(lam, mu, rho=rho, max_goals=max_goals)
        outcomes = outcome_probabilities(matrix)
        home_score, away_score = most_likely_score(matrix)

        x_win_prob.append(outcomes["home_win"])
        draw_prob.append(outcomes["draw"])
        y_win_prob.append(outcomes["away_win"])
        x_cs.append(home_score)
        y_cs.append(away_score)

    df["x_team_win_probability"] = x_win_prob
    df["draw_probability"] = draw_prob
    df["y_team_win_probabilty"] = y_win_prob
    df["x_team_cs"] = x_cs
    df["y_team_cs"] = y_cs
    df["home_win_odds"] = [probability_to_decimal_odds(p) for p in x_win_prob]
    df["draw_odds"] = [probability_to_decimal_odds(p) for p in draw_prob]
    df["away_win_odds"] = [probability_to_decimal_odds(p) for p in y_win_prob]

    return df[[
        "league_id","fixture_id","date", "round", "home_team", "away_team", "home_goals", "away_goals", "status",
        "x_team_win_probability", "draw_probability", "y_team_win_probabilty",
        "x_team_cs", "y_team_cs",
        "home_win_odds", "draw_odds", "away_win_odds",
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
    FORM_WINDOW = [3,5,10]

    INPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    OUTPUT_DB_PATH = "../../storage/database/leagues/England.ddb"
    for window in FORM_WINDOW:
        INPUT_TABLE = f"39-2025-xg-l{window}"
        OUTPUT_TABLE = f"39-2025-odds-l{window}"

        expected_goals = load_expected_goals(db_path=INPUT_DB_PATH, table_name=INPUT_TABLE)
        result = compute_match_probabilities(expected_goals)

        write_output(result, db_path=OUTPUT_DB_PATH, table_name=OUTPUT_TABLE)

        print(f"Wrote {len(result)} rows to '{OUTPUT_TABLE}' in {OUTPUT_DB_PATH}")
        print(result.head(15).to_string(index=False))