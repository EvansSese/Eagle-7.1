"""Analyze pre-match team outcome probabilities against completed fixtures."""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from processing.core import COMPLETED_STATUSES


OUTCOMES = ("win", "draw", "loss")
PROBABILITY_COLUMNS = {
    "home": {
        outcome: f"x_team_{outcome}_probability_before"
        for outcome in OUTCOMES
    },
    "away": {
        outcome: f"y_team_{outcome}_probability_before"
        for outcome in OUTCOMES
    },
}
COMBINED_PROBABILITY_COLUMNS = {
    "win": "combined_home_win_probability_before",
    "draw": "combined_draw_probability_before",
    "loss": "combined_away_win_probability_before",
}


def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    """Load generated probabilities from DuckDB, keeping completed matches."""
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()

    required_columns = {
        "status", "home_goals", "away_goals", "predicted_outcome",
        *PROBABILITY_COLUMNS["home"].values(),
        *PROBABILITY_COLUMNS["away"].values(),
        *COMBINED_PROBABILITY_COLUMNS.values(),
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"table is missing required columns: {sorted(missing_columns)}")
    return df[df["status"].isin(COMPLETED_STATUSES)].copy()


def count_predicted_outcome_matches(df: pd.DataFrame) -> dict:
    """Count fixtures where the stored predicted outcome matches the result."""
    required_columns = {"status", "home_goals", "away_goals", "predicted_outcome"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"data is missing required columns: {sorted(missing_columns)}")

    completed = df[df["status"].isin(COMPLETED_STATUSES)].dropna(
        subset=["home_goals", "away_goals", "predicted_outcome"]
    )
    if completed.empty:
        raise ValueError("no completed fixtures with scores are available to compare")
    if not completed["predicted_outcome"].isin(("home_win", "draw", "away_win")).all():
        raise ValueError("predicted_outcome must be 'home_win', 'draw', or 'away_win'")

    actual_outcome = pd.Series(
        "draw",
        index=completed.index,
        dtype="object",
    )
    actual_outcome.loc[completed["home_goals"] > completed["away_goals"]] = "home_win"
    actual_outcome.loc[completed["home_goals"] < completed["away_goals"]] = "away_win"
    matches = completed["predicted_outcome"] == actual_outcome
    return {
        "matches": int(matches.sum()),
        "total": int(len(completed)),
        "accuracy": float(matches.mean()),
    }


def analyze_outcome_probabilities(
    df: pd.DataFrame,
    probability_bins: int = 5,
) -> dict:
    """Compare pre-match probabilities with results and return calibration metrics.

    Each completed fixture contributes one observation for each team. The
    returned calibration table groups predicted probabilities into bins and
    compares their average with the observed frequency of that outcome.
    """
    if isinstance(probability_bins, bool) or not isinstance(probability_bins, int) or probability_bins < 1:
        raise ValueError("probability_bins must be a positive integer")

    required_columns = {
        "status", "home_goals", "away_goals", "predicted_outcome",
        *PROBABILITY_COLUMNS["home"].values(),
        *PROBABILITY_COLUMNS["away"].values(),
        *COMBINED_PROBABILITY_COLUMNS.values(),
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"data is missing required columns: {sorted(missing_columns)}")

    completed = df[df["status"].isin(COMPLETED_STATUSES)].copy()
    completed = completed.dropna(subset=["home_goals", "away_goals"])
    if completed.empty:
        raise ValueError("no completed fixtures with scores are available to analyze")

    if (completed[["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError("goals must be non-negative")

    home_actual = pd.DataFrame({
        outcome: (
            completed["home_goals"] > completed["away_goals"]
            if outcome == "win"
            else completed["home_goals"] == completed["away_goals"]
            if outcome == "draw"
            else completed["home_goals"] < completed["away_goals"]
        ).astype(float)
        for outcome in OUTCOMES
    })
    away_actual = pd.DataFrame({
        outcome: (
            completed["away_goals"] > completed["home_goals"]
            if outcome == "win"
            else completed["away_goals"] == completed["home_goals"]
            if outcome == "draw"
            else completed["away_goals"] < completed["home_goals"]
        ).astype(float)
        for outcome in OUTCOMES
    })

    home_probabilities = completed[list(PROBABILITY_COLUMNS["home"].values())].copy()
    home_probabilities.columns = OUTCOMES
    away_probabilities = completed[list(PROBABILITY_COLUMNS["away"].values())].copy()
    away_probabilities.columns = OUTCOMES
    probabilities = pd.concat([home_probabilities, away_probabilities], ignore_index=True)
    actual = pd.concat([home_actual, away_actual], ignore_index=True)

    if probabilities.isna().any().any():
        raise ValueError("outcome probabilities cannot be missing for completed fixtures")
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError("outcome probabilities must be between 0 and 1")
    if not probabilities.sum(axis=1).between(0.985, 1.015).all():
        raise ValueError("rounded win, draw, and loss probabilities must sum approximately to 1")

    combined_probabilities = completed[list(COMBINED_PROBABILITY_COLUMNS.values())].copy()
    combined_probabilities.columns = OUTCOMES
    combined_probabilities = combined_probabilities.reset_index(drop=True)
    home_actual = home_actual.reset_index(drop=True)
    if combined_probabilities.isna().any().any():
        raise ValueError("combined outcome probabilities cannot be missing")
    if ((combined_probabilities < 0) | (combined_probabilities > 1)).any().any():
        raise ValueError("combined outcome probabilities must be between 0 and 1")
    if not combined_probabilities.sum(axis=1).between(0.985, 1.015).all():
        raise ValueError("rounded combined outcome probabilities must sum approximately to 1")

    predicted_outcomes = probabilities.idxmax(axis=1)
    actual_outcomes = actual.idxmax(axis=1)
    prediction_accuracy = float((predicted_outcomes == actual_outcomes).mean())
    brier_score = float(((probabilities - actual) ** 2).sum(axis=1).mean())
    combined_prediction_accuracy = float(
        (combined_probabilities.idxmax(axis=1) == home_actual.idxmax(axis=1)).mean()
    )
    combined_brier_score = float(
        ((combined_probabilities - home_actual) ** 2).sum(axis=1).mean()
    )
    prediction_match_summary = count_predicted_outcome_matches(completed)

    calibration_rows = []
    for level, level_probabilities, level_actual in (
        ("team", probabilities, actual),
        ("combined", combined_probabilities, home_actual.reset_index(drop=True)),
    ):
        for outcome in OUTCOMES:
            outcome_data = pd.DataFrame({
                "probability": level_probabilities[outcome],
                "occurred": level_actual[outcome],
            })
            outcome_data["probability_bin"] = pd.cut(
                outcome_data["probability"],
                bins=probability_bins,
                include_lowest=True,
            )
            grouped = outcome_data.groupby("probability_bin", observed=False)
            for probability_bin, group in grouped:
                if group.empty:
                    continue
                calibration_rows.append({
                    "level": level,
                    "outcome": outcome,
                    "probability_bin": probability_bin,
                    "matches": len(group),
                    "mean_predicted_probability": float(group["probability"].mean()),
                    "observed_frequency": float(group["occurred"].mean()),
                })

    return {
        "fixtures_analyzed": int(len(completed)),
        "team_outcomes_analyzed": int(len(probabilities)),
        "prediction_accuracy": prediction_accuracy,
        "multiclass_brier_score": brier_score,
        "combined_prediction_accuracy": combined_prediction_accuracy,
        "combined_multiclass_brier_score": combined_brier_score,
        "predicted_outcome_matches": prediction_match_summary["matches"],
        "predicted_outcome_total": prediction_match_summary["total"],
        "predicted_outcome_accuracy": prediction_match_summary["accuracy"],
        "calibration": pd.DataFrame(calibration_rows),
    }


def plot_summary(summary: dict, output_path: str, show: bool = True) -> None:
    """Plot predicted probability against observed outcome frequency."""
    calibration = summary["calibration"]
    fig, axes = plt.subplots(2, len(OUTCOMES), figsize=(15, 9), sharex=True, sharey=True)

    for row_index, level in enumerate(("team", "combined")):
        for column_index, outcome in enumerate(OUTCOMES):
            axis = axes[row_index, column_index]
            outcome_data = calibration[
                (calibration["level"] == level) & (calibration["outcome"] == outcome)
            ]
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
            if not outcome_data.empty:
                axis.scatter(
                    outcome_data["mean_predicted_probability"],
                    outcome_data["observed_frequency"],
                    s=30 + 4 * outcome_data["matches"],
                    alpha=0.75,
                    label="Probability bins",
                )
            axis.set_title(f"{level.capitalize()} {outcome.capitalize()}")
            axis.set_xlabel("Mean predicted probability")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Observed frequency")
    axes[1, 0].set_ylabel("Observed frequency")
    axes[1, -1].legend(loc="best")
    fig.suptitle(
        f"Pre-match outcome probability calibration "
        f"(fixtures={summary['fixtures_analyzed']}, "
        f"combined accuracy={summary['combined_prediction_accuracy']:.1%}, "
        f"combined Brier score={summary['combined_multiclass_brier_score']:.3f})"
    )
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved outcome probability chart to {output}")
    if show:
        plt.show()
    plt.close(fig)


def run_analysis(
    db_path: str,
    table_name: str,
    output_path: str,
    probability_bins: int = 5,
    show: bool = True,
) -> dict:
    """Load generated probabilities, analyze them, and display/save charts."""
    data = load_data(db_path, table_name)
    summary = analyze_outcome_probabilities(data, probability_bins=probability_bins)
    print(f"Completed fixtures analyzed: {summary['fixtures_analyzed']}")
    print(f"Team outcomes analyzed: {summary['team_outcomes_analyzed']}")
    print(f"Most-likely outcome accuracy: {summary['prediction_accuracy']:.1%}")
    print(f"Multiclass Brier score: {summary['multiclass_brier_score']:.4f}")
    print(f"Combined outcome accuracy: {summary['combined_prediction_accuracy']:.1%}")
    print(f"Combined multiclass Brier score: {summary['combined_multiclass_brier_score']:.4f}")
    print(
        "Least-probability predicted outcome matched actual: "
        f"{summary['predicted_outcome_matches']}/{summary['predicted_outcome_total']} "
        f"({summary['predicted_outcome_accuracy']:.1%})"
    )
    print("\nCalibration by outcome and probability bin:")
    print(summary["calibration"].to_string(index=False))
    plot_summary(summary, output_path=output_path, show=show)
    return summary


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/England.ddb"
    TABLE_NAME = "39-2025-outcome-probability"
    OUTPUT_CHART_PATH = "charts/39-2025-outcome_probability_analysis.png"

    run_analysis(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        output_path=OUTPUT_CHART_PATH,
    )