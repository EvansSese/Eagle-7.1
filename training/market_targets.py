"""Targets and probability relationships for football prediction markets."""

import numpy as np
import pandas as pd


OVER_THRESHOLDS = (0.5, 1.5, 2.5, 3.5)


def build_market_targets(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Derive supervised targets from completed fixtures, never as input features."""
    required = {"home_goals", "away_goals"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    scores = fixtures[["home_goals", "away_goals"]].apply(pd.to_numeric, errors="coerce")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Targets require finite, non-negative integer scores")
    scores = scores.astype(int)
    home, away = scores["home_goals"], scores["away_goals"]
    total = home + away
    result = pd.DataFrame(index=fixtures.index)
    result["result"] = np.where(home > away, "H", np.where(home < away, "A", "D"))
    result["btts"] = ((home > 0) & (away > 0)).astype(int)
    for threshold in OVER_THRESHOLDS:
        result[f"over_{str(threshold).replace('.', '_')}"] = (total > threshold).astype(int)
    result["home_goals"] = home
    result["away_goals"] = away
    result["total_goals"] = total
    result["correct_score"] = home.astype(str) + "-" + away.astype(str)
    return result


def derived_market_probabilities(
    result_probabilities: pd.DataFrame, over_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Derive double-chance and under probabilities from matching model outputs."""
    if not result_probabilities.index.equals(over_probabilities.index):
        raise ValueError("Market probabilities must refer to the same fixtures in the same order")
    result_columns = ["H", "D", "A"]
    over_columns = [f"over_{str(t).replace('.', '_')}" for t in OVER_THRESHOLDS]
    missing = (set(result_columns) - set(result_probabilities.columns)) | (
        set(over_columns) - set(over_probabilities.columns)
    )
    if missing:
        raise ValueError(f"Missing market probabilities: {sorted(missing)}")
    outcome = result_probabilities[result_columns].to_numpy(dtype=float)
    over = over_probabilities[over_columns].to_numpy(dtype=float)
    if (not np.isfinite(outcome).all() or not np.isfinite(over).all()
            or (outcome < 0).any() or (outcome > 1).any()
            or (over < 0).any() or (over > 1).any()
            or not np.allclose(outcome.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Market probabilities must be finite, valid probabilities; 3-way probabilities must sum to one")
    derived = pd.DataFrame(index=result_probabilities.index)
    derived["1X"] = outcome[:, 0] + outcome[:, 1]
    derived["X2"] = outcome[:, 1] + outcome[:, 2]
    derived["12"] = outcome[:, 0] + outcome[:, 2]
    for index, threshold in enumerate(OVER_THRESHOLDS):
        derived[f"under_{str(threshold).replace('.', '_')}"] = 1.0 - over[:, index]
    return derived
