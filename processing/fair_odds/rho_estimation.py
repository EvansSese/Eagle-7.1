"""
rho_estimation.py

Estimates the Dixon-Coles low-score correlation parameter rho for a league
season from the season's own results, so it no longer has to be a constant.

Method (maximum likelihood): with the match means (lambda, mu) fixed, rho
only enters the likelihood through the correction factor tau of the four
low-scoring cells, so
    log L(rho) = sum over finished matches ending 0-0, 1-0, 0-1 or 1-1 of
                 log tau(cell; lambda, mu, rho)
(tau = 1 for every other scoreline, i.e. log tau = 0). We evaluate that on a
fixed rho grid (cheap and robust; invalid rho with tau <= 0 gets -inf) and
add a weak Gaussian prior centred on the default RHO with std PRIOR_SD, so
with little data rho stays near the default instead of swinging.

No lookahead: the rho used for a fixture's round is estimated ONLY from
matches in strictly earlier rounds, via one cumulative sum over rounds
(O(low-score matches x grid size) time, small memory). The first round
therefore gets the default RHO. Means are the pre-match expected goals from
the xG table; matches whose xG is 0 (no model expectation) are skipped.

Output table "<league>-<season>-rho" is keyed by fixture_id.
"""

import numpy as np
import pandas as pd

from processing.core import COMPLETED_STATUSES, database_connection, write_fixture_table
from processing.fair_odds.dixon_coles import RHO
from processing.xG.xg_metrics import determine_round_order

RHO_GRID = np.round(np.arange(-0.5, 0.5 + 1e-9, 0.005), 3)
PRIOR_SD = 0.1
RHO_DECIMALS = 3
OUTPUT_COLUMNS = ["league_id", "fixture_id", "round", "rho", "rho_matches_used"]


def low_score_log_tau(home_goals, away_goals, lam, mu, grid: np.ndarray = RHO_GRID) -> np.ndarray:
    """
    Matrix (matches x grid) of log tau for matches that ended 0-0, 0-1, 1-0
    or 1-1 (inputs must already be restricted to those). tau <= 0 gives -inf.
    """
    h = np.asarray(home_goals)[:, None]
    a = np.asarray(away_goals)[:, None]
    lam = np.asarray(lam, dtype=float)[:, None]
    mu = np.asarray(mu, dtype=float)[:, None]
    r = grid[None, :]
    tau = np.where(
        (h == 0) & (a == 0), 1 - lam * mu * r,
        np.where((h == 0) & (a == 1), 1 + lam * r,
                 np.where((h == 1) & (a == 0), 1 + mu * r, 1 - r)),
    )
    return np.where(tau > 0, np.log(np.where(tau > 0, tau, 1.0)), -np.inf)


def estimate_rho_by_round(df: pd.DataFrame, grid: np.ndarray = RHO_GRID,
                          prior_rho: float = RHO, prior_sd: float = PRIOR_SD) -> pd.DataFrame:
    """
    Per round, the MAP rho estimated from earlier rounds only.
    Returns one row per round: round, round_number, rho, rho_matches_used.
    `df` needs round, date, status, home_goals, away_goals and the
    x_team_home_expected_goals / y_team_away_expected_goals columns.
    """
    order = determine_round_order(df)
    rounds = sorted(order, key=order.get)
    round_pos = {label: i for i, label in enumerate(rounds)}
    n_rounds = len(rounds)

    usable = (
        df["status"].isin(COMPLETED_STATUSES)
        & df["home_goals"].notna() & df["away_goals"].notna()
        & (df["x_team_home_expected_goals"] > 0) & (df["y_team_away_expected_goals"] > 0)
    )
    used = df[usable]
    positions = used["round"].map(round_pos).to_numpy(dtype=int)
    matches_per_round = np.bincount(positions, minlength=n_rounds)

    low = ((used["home_goals"] <= 1) & (used["away_goals"] <= 1)).to_numpy()
    low_rows = used[low]
    log_lik_per_round = np.zeros((n_rounds, len(grid)))
    if len(low_rows):
        contributions = low_score_log_tau(
            low_rows["home_goals"].to_numpy(), low_rows["away_goals"].to_numpy(),
            low_rows["x_team_home_expected_goals"].to_numpy(), low_rows["y_team_away_expected_goals"].to_numpy(),
            grid,
        )
        np.add.at(log_lik_per_round, positions[low], contributions)

    # Exclusive cumulative sums: round i sees rounds 0..i-1 only
    before_ll = np.vstack([np.zeros(len(grid)), np.cumsum(log_lik_per_round, axis=0)[:-1]])
    matches_before = np.concatenate([[0], np.cumsum(matches_per_round)[:-1]])

    log_prior = -0.5 * ((grid - prior_rho) / prior_sd) ** 2
    best = np.argmax(before_ll + log_prior[None, :], axis=1)
    return pd.DataFrame({
        "round": rounds,
        "round_number": range(1, n_rounds + 1),
        "rho": np.round(grid[best], RHO_DECIMALS),
        "rho_matches_used": matches_before,
    })


def build_rho_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture rho table (OUTPUT_COLUMNS), keyed by fixture_id."""
    by_round = estimate_rho_by_round(df)
    merged = df[["league_id", "fixture_id", "round"]].merge(
        by_round[["round", "rho", "rho_matches_used"]], on="round", how="left"
    )
    return merged[OUTPUT_COLUMNS].sort_values("fixture_id").reset_index(drop=True)


def load_rho_table(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Read a stored rho table."""
    with database_connection(db_path, connection, read_only=True) as con:
        return con.execute(f'SELECT * FROM "{table_name}"').df()


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save the rho table with fixture_id as its unique key."""
    write_fixture_table(df, db_path, table_name, connection)
