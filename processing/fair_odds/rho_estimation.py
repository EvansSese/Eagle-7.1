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

The rho used for a fixture is estimated from earlier kickoffs in strictly
earlier rounds. It may therefore vary within a round when fixtures overlap.
The first round gets the default RHO. Means are pre-match expected goals;
matches whose xG is 0 (no model expectation) are skipped.

Output table "<league>-<season>-rho" is keyed by fixture_id.
"""

import numpy as np
import pandas as pd
from itertools import groupby

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
    Per round, the MAP rho at the round's first kickoff.
    Returns one row per round: round, round_number, rho, rho_matches_used.
    `df` needs round, date, status, home_goals, away_goals and the
    x_team_home_expected_goals / y_team_away_expected_goals columns.
    """
    df = df.copy()
    if "league_id" not in df:
        df["league_id"] = 0
    if "fixture_id" not in df:
        df["fixture_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"])
    order = determine_round_order(df)
    estimates = _rho_per_fixture(df, grid, prior_rho, prior_sd)
    estimates["round_number"] = estimates["round"].map(order)
    return estimates.drop_duplicates("round")[["round", "round_number", "rho", "rho_matches_used"]].sort_values("round_number").reset_index(drop=True)


def _rho_per_fixture(df, grid, prior_rho, prior_sd):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any() or df["round"].isna().any():
        raise ValueError("fixture dates and rounds cannot be missing")
    order = determine_round_order(df)
    df = df.sort_values(["date", "fixture_id"], kind="stable")
    log_likelihood = np.zeros((len(order) + 1, len(grid)))
    counts = np.zeros(len(order) + 1, dtype=int)
    log_prior = -0.5 * ((grid - prior_rho) / prior_sd) ** 2
    records = []
    for _, rows in groupby(df.itertuples(index=False), key=lambda row: row.date):
        kickoff = list(rows)
        for row in kickoff:
            rank = order[row.round]
            likelihood = log_likelihood[:rank].sum(axis=0)
            best = np.argmax(likelihood + log_prior)
            records.append({
                "league_id": row.league_id, "fixture_id": row.fixture_id,
                "round": row.round, "rho": round(float(grid[best]), RHO_DECIMALS),
                "rho_matches_used": int(counts[:rank].sum()),
            })
        for row in kickoff:
            if (row.status not in COMPLETED_STATUSES or pd.isna(row.home_goals)
                    or pd.isna(row.away_goals)
                    or not row.x_team_home_expected_goals > 0
                    or not row.y_team_away_expected_goals > 0):
                continue
            rank = order[row.round]
            counts[rank] += 1
            if row.home_goals <= 1 and row.away_goals <= 1:
                log_likelihood[rank] += low_score_log_tau(
                    [row.home_goals], [row.away_goals],
                    [row.x_team_home_expected_goals], [row.y_team_away_expected_goals], grid,
                )[0]
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def build_rho_table(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate rho from earlier-round results available before each kickoff."""
    return _rho_per_fixture(df, RHO_GRID, RHO, PRIOR_SD).sort_values("fixture_id").reset_index(drop=True)


def load_rho_table(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Read a stored rho table."""
    with database_connection(db_path, connection, read_only=True) as con:
        return con.execute(f'SELECT * FROM "{table_name}"').df()


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Save the rho table with fixture_id as its unique key."""
    write_fixture_table(df, db_path, table_name, connection)
