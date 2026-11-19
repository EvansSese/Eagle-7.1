"""
dixon_coles.py

Reusable Dixon-Coles / Poisson building blocks for turning a pair of
expected-goal means (lambda for the home side, mu for the away side) into a
score grid, outcome probabilities, a correct score and decimal odds.

The Dixon-Coles model starts from two independent Poisson distributions and
multiplies the four low-scoring cells (0-0, 1-0, 0-1, 1-1) by a correction
factor tau controlled by one parameter rho, because independent Poisson
under/over-states those scorelines. `bounded_rho` clamps rho per match so
every tau stays non-negative by construction.

Requires: numpy
"""

import math

import numpy as np

# Dixon-Coles low-score correlation parameter (the 1997 paper found about
# -0.13 for English football; small negative values are typical).
RHO = -0.05

# Goals (0..MAX_GOALS) per side kept in the score grid; mass beyond is negligible.
MAX_GOALS = 10


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for a Poisson distribution with mean lam. lam <= 0 puts all mass at k = 0."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def bounded_rho(lam: float, mu: float, rho: float) -> float:
    """
    Clamp rho into the range where all four adjusted cells stay >= 0 for this
    match's lambda/mu:
      1 - lambda*mu*rho >= 0  ->  rho <= 1 / (lambda*mu)
      1 + lambda*rho    >= 0  ->  rho >= -1 / lambda
      1 + mu*rho        >= 0  ->  rho >= -1 / mu
      1 - rho           >= 0  ->  rho <= 1
    """
    if lam <= 0 or mu <= 0:
        return 0.0
    upper_bound = min(1.0, 1.0 / (lam * mu))
    lower_bound = max(-1.0, -1.0 / lam, -1.0 / mu)
    return max(lower_bound, min(rho, upper_bound))


def dixon_coles_tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor tau(i, j); 1.0 for every cell outside the low-score block."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (lam * mu * rho)
    if home_goals == 0 and away_goals == 1:
        return 1 + (lam * rho)
    if home_goals == 1 and away_goals == 0:
        return 1 + (mu * rho)
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lam: float, mu: float, rho: float = RHO, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    (max_goals+1) x (max_goals+1) grid; cell [i, j] is the Dixon-Coles
    probability of a final score of i-j (home-away). The Poisson outer product
    is built in one step and only the 2x2 low-score block is corrected, then
    the grid is renormalised to sum to 1 (corrects truncation at max_goals).
    """
    effective_rho = bounded_rho(lam, mu, rho)
    home_pmf = np.array([poisson_pmf(i, lam) for i in range(max_goals + 1)])
    away_pmf = np.array([poisson_pmf(j, mu) for j in range(max_goals + 1)])
    matrix = np.outer(home_pmf, away_pmf)

    if max_goals >= 1:
        for i in (0, 1):
            for j in (0, 1):
                matrix[i, j] *= dixon_coles_tau(i, j, lam, mu, effective_rho)
    np.maximum(matrix, 0.0, out=matrix)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def outcome_probabilities(matrix: np.ndarray) -> dict:
    """Sum the grid into home win / draw / away win probabilities (rows = home goals)."""
    return {
        "home_win": float(np.tril(matrix, k=-1).sum()),
        "draw": float(np.trace(matrix)),
        "away_win": float(np.triu(matrix, k=1).sum()),
    }


def most_likely_score(matrix: np.ndarray) -> tuple:
    """(home_goals, away_goals) of the single most probable scoreline."""
    home_goals, away_goals = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(home_goals), int(away_goals)


def probability_to_decimal_odds(probability: float) -> float:
    """Decimal (European) odds = 1 / p, rounded to 2 dp; 1e9 when p == 0."""
    return round(1 / probability, 2) if probability > 0 else float(1000000000)
