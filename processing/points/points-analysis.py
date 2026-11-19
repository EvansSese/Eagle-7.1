"""
points-analysis.py

Reads the `points_before_match` table (produced by points_before_duckdb.py)
and analyses how pre-match cumulative points / points-per-match relate to
match outcomes.

Requires: duckdb, pandas, matplotlib
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from processing.core import COMPLETED_STATUSES

def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    # Only completed matches have a meaningful result to analyse
    return df[df["status"].isin(COMPLETED_STATUSES)].copy()


def _is_home_win(df: pd.DataFrame) -> pd.Series:
    return df["home_goals"] > df["away_goals"]


def _is_away_win(df: pd.DataFrame) -> pd.Series:
    return df["away_goals"] > df["home_goals"]


def _is_draw(df: pd.DataFrame) -> pd.Series:
    return df["home_goals"] == df["away_goals"]


def home_win_when_home_favored(df: pd.DataFrame) -> dict:
    """
    Q1: Matches that ended in a home win where the home team's
    points_before was higher than the away team's, and — independently —
    matches where the home team's points_per_match_before was higher.
    The two conditions are counted separately, not combined.
    """
    home_win = _is_home_win(df)
    points_mask = home_win & (df["x_team_points_before"] > df["y_team_points_before"])
    ppm_mask = home_win & (df["x_team_points_per_match_before"] > df["y_team_points_per_match_before"])
    return {"points": int(points_mask.sum()), "points_per_match": int(ppm_mask.sum())}


def home_win_when_home_underdog(df: pd.DataFrame) -> dict:
    """
    Q2: Matches that ended in a home win where the home team's
    points_before was lower than the away team's, and — independently —
    matches where the home team's points_per_match_before was lower.
    """
    home_win = _is_home_win(df)
    points_mask = home_win & (df["x_team_points_before"] < df["y_team_points_before"])
    ppm_mask = home_win & (df["x_team_points_per_match_before"] < df["y_team_points_per_match_before"])
    return {"points": int(points_mask.sum()), "points_per_match": int(ppm_mask.sum())}


def away_win_when_away_favored(df: pd.DataFrame) -> dict:
    """
    Q3: Matches that ended in an away win where the away team's
    points_before was higher than the home team's, and — independently —
    matches where the away team's points_per_match_before was higher.
    """
    away_win = _is_away_win(df)
    points_mask = away_win & (df["y_team_points_before"] > df["x_team_points_before"])
    ppm_mask = away_win & (df["y_team_points_per_match_before"] > df["x_team_points_per_match_before"])
    return {"points": int(points_mask.sum()), "points_per_match": int(ppm_mask.sum())}


def away_win_when_away_underdog(df: pd.DataFrame) -> dict:
    """
    Q4: Matches that ended in an away win where the away team's
    points_before was lower than the home team's, and — independently —
    matches where the away team's points_per_match_before was lower.
    """
    away_win = _is_away_win(df)
    points_mask = away_win & (df["y_team_points_before"] < df["x_team_points_before"])
    ppm_mask = away_win & (df["y_team_points_per_match_before"] < df["x_team_points_per_match_before"])
    return {"points": int(points_mask.sum()), "points_per_match": int(ppm_mask.sum())}


def draws_by_favorite(df: pd.DataFrame) -> dict:
    """
    Q5: Matches that ended in a draw, categorized by which side was
    favored pre-match — home favored, away favored, or both equal —
    computed independently for points_before and points_per_match_before.
    """
    draws = df[_is_draw(df)].copy()

    def bucket_counts(home_val: pd.Series, away_val: pd.Series) -> dict:
        return {
            "home_favored": int((home_val > away_val).sum()),
            "away_favored": int((home_val < away_val).sum()),
            "equal": int((home_val == away_val).sum()),
        }

    return {
        "points": bucket_counts(draws["x_team_points_before"], draws["y_team_points_before"]),
        "points_per_match": bucket_counts(
            draws["x_team_points_per_match_before"], draws["y_team_points_per_match_before"]
        ),
        "total_draws": int(len(draws)),
    }


def summarize(df: pd.DataFrame) -> dict:
    return {
        "home_win_when_home_favored": home_win_when_home_favored(df),
        "home_win_when_home_underdog": home_win_when_home_underdog(df),
        "away_win_when_away_favored": away_win_when_away_favored(df),
        "away_win_when_away_underdog": away_win_when_away_underdog(df),
        "draws_by_favorite": draws_by_favorite(df),
    }


def plot_summary(summary: dict, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Chart 1: win outcomes vs favorite/underdog status (points vs PPM, independent) ---
    win_labels = [
        "Home win\n(home favored)",
        "Home win\n(home underdog)",
        "Away win\n(away favored)",
        "Away win\n(away underdog)",
    ]
    points_values = [
        summary["home_win_when_home_favored"]["points"],
        summary["home_win_when_home_underdog"]["points"],
        summary["away_win_when_away_favored"]["points"],
        summary["away_win_when_away_underdog"]["points"],
    ]
    ppm_values = [
        summary["home_win_when_home_favored"]["points_per_match"],
        summary["home_win_when_home_underdog"]["points_per_match"],
        summary["away_win_when_away_favored"]["points_per_match"],
        summary["away_win_when_away_underdog"]["points_per_match"],
    ]
    x = range(len(win_labels))
    width = 0.35
    b1 = axes[0].bar([i - width / 2 for i in x], points_values, width, label="Points", color="#2a9d8f")
    b2 = axes[0].bar([i + width / 2 for i in x], ppm_values, width, label="Points per match", color="#e76f51")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(win_labels)
    axes[0].set_title("Match wins vs. pre-match standing (points & PPM independent)")
    axes[0].set_ylabel("Number of matches")
    axes[0].legend()
    axes[0].bar_label(b1, padding=3)
    axes[0].bar_label(b2, padding=3)

    # --- Chart 2: draws bucketed by which side was favored ---
    draw_data = summary["draws_by_favorite"]
    bucket_labels = ["Home favored", "Away favored", "Equal"]
    points_vals = [
        draw_data["points"]["home_favored"],
        draw_data["points"]["away_favored"],
        draw_data["points"]["equal"],
    ]
    ppm_vals = [
        draw_data["points_per_match"]["home_favored"],
        draw_data["points_per_match"]["away_favored"],
        draw_data["points_per_match"]["equal"],
    ]

    x = range(len(bucket_labels))
    width = 0.35
    bars1 = axes[1].bar([i - width / 2 for i in x], points_vals, width, label="Points", color="#457b9d")
    bars2 = axes[1].bar([i + width / 2 for i in x], ppm_vals, width, label="Points per match", color="#f4a261")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(bucket_labels)
    axes[1].set_title(f"Draws by pre-match favorite (n={draw_data['total_draws']})")
    axes[1].set_ylabel("Number of matches")
    axes[1].legend()
    axes[1].bar_label(bars1, padding=3)
    axes[1].bar_label(bars2, padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def print_summary(summary: dict) -> None:
    print("=== Points-Before Match Outcome Analysis ===\n")
    q1, q2 = summary["home_win_when_home_favored"], summary["home_win_when_home_underdog"]
    q3, q4 = summary["away_win_when_away_favored"], summary["away_win_when_away_underdog"]
    print(f"1. Home win with home favored -> points: {q1['points']}, points_per_match: {q1['points_per_match']}")
    print(f"2. Home win with home underdog -> points: {q2['points']}, points_per_match: {q2['points_per_match']}")
    print(f"3. Away win with away favored -> points: {q3['points']}, points_per_match: {q3['points_per_match']}")
    print(f"4. Away win with away underdog -> points: {q4['points']}, points_per_match: {q4['points_per_match']}")
    print("5. Draws by pre-match favorite (home vs. away):")
    dd = summary["draws_by_favorite"]
    print(f"   Points          -> home_favored: {dd['points']['home_favored']}, "
          f"away_favored: {dd['points']['away_favored']}, "
          f"equal: {dd['points']['equal']}")
    print(f"   Points_per_match -> home_favored: {dd['points_per_match']['home_favored']}, "
          f"away_favored: {dd['points_per_match']['away_favored']}, "
          f"equal: {dd['points_per_match']['equal']}")
    print(f"   Total draws: {dd['total_draws']}")


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/Kenya.ddb"
    TABLE_NAME = "276-2025-points"
    OUTPUT_CHART_PATH = "charts/276-2025-points_analysis_charts.png"
    df = load_data(db_path=DB_PATH, table_name=TABLE_NAME)
    summary = summarize(df)
    print_summary(summary)
    plot_summary(summary, output_path=OUTPUT_CHART_PATH)