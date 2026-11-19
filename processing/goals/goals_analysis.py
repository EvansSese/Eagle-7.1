"""
goals_analysis.py

Reads the `goals_before_match` table (produced by goals_before_match.py)
and analyses how pre-match goal-scoring form relates to match outcomes.

"Favored" here is judged on attacking output — goals_scored (cumulative)
and goals_scored_per_match (rate) — compared independently, the same way
points and points_per_match were treated independently in points-analysis.py.

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


def _bucket(outcome_mask: pd.Series, home_val: pd.Series, away_val: pd.Series) -> dict:
    """Split an outcome mask into home_favored / away_favored / equal, by comparing home_val vs away_val."""
    return {
        "home_favored": int((outcome_mask & (home_val > away_val)).sum()),
        "away_favored": int((outcome_mask & (home_val < away_val)).sum()),
        "equal": int((outcome_mask & (home_val == away_val)).sum()),
    }


def home_wins_by_favorite(df: pd.DataFrame) -> dict:
    """
    Q1: Matches that ended in a home win, categorized by which side had
    the stronger pre-match attacking form — home favored, away favored
    (i.e. home was the underdog), or equal — computed independently for
    goals_scored and goals_scored_per_match.
    """
    hw = _is_home_win(df)
    return {
        "goals_scored": _bucket(hw, df["x_team_goals_scored"], df["y_team_goals_scored"]),
        "goals_scored_per_match": _bucket(hw, df["x_team_goals_scored_per_match"], df["y_team_goals_scored_per_match"]),
        "total_home_wins": int(hw.sum()),
    }


def draws_by_favorite(df: pd.DataFrame) -> dict:
    """
    Q2: Matches that ended in a draw, categorized by which side had the
    stronger pre-match attacking form — home favored, away favored, or
    equal — computed independently for goals_scored and
    goals_scored_per_match.
    """
    d = _is_draw(df)
    return {
        "goals_scored": _bucket(d, df["x_team_goals_scored"], df["y_team_goals_scored"]),
        "goals_scored_per_match": _bucket(d, df["x_team_goals_scored_per_match"], df["y_team_goals_scored_per_match"]),
        "total_draws": int(d.sum()),
    }


def away_wins_by_favorite(df: pd.DataFrame) -> dict:
    """
    Q3: Matches that ended in an away win, categorized by which side had
    the stronger pre-match attacking form — home favored (away underdog),
    away favored, or equal — computed independently for goals_scored and
    goals_scored_per_match.
    """
    aw = _is_away_win(df)
    return {
        "goals_scored": _bucket(aw, df["x_team_goals_scored"], df["y_team_goals_scored"]),
        "goals_scored_per_match": _bucket(aw, df["x_team_goals_scored_per_match"], df["y_team_goals_scored_per_match"]),
        "total_away_wins": int(aw.sum()),
    }


def summarize(df: pd.DataFrame) -> dict:
    return {
        "home_wins": home_wins_by_favorite(df),
        "draws": draws_by_favorite(df),
        "away_wins": away_wins_by_favorite(df),
    }


def _grouped_bar(ax, data: dict, title: str) -> None:
    labels = ["Home favored", "Away favored", "Equal"]
    gs_vals = [data["goals_scored"]["home_favored"], data["goals_scored"]["away_favored"], data["goals_scored"]["equal"]]
    gspm_vals = [
        data["goals_scored_per_match"]["home_favored"],
        data["goals_scored_per_match"]["away_favored"],
        data["goals_scored_per_match"]["equal"],
    ]
    x = range(len(labels))
    width = 0.35
    b1 = ax.bar([i - width / 2 for i in x], gs_vals, width, label="Goals scored", color="#2a9d8f")
    b2 = ax.bar([i + width / 2 for i in x], gspm_vals, width, label="Goals scored / match", color="#e76f51")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Number of matches")
    ax.legend()
    ax.bar_label(b1, padding=2)
    ax.bar_label(b2, padding=2)


def _pie(ax, data: dict, title: str) -> None:
    # Uses goals_scored_per_match as the representative metric for the pie split
    gspm = data["goals_scored_per_match"]
    labels = ["Home favored", "Away favored", "Equal"]
    values = [gspm["home_favored"], gspm["away_favored"], gspm["equal"]]
    colors = ["#264653", "#e9c46a", "#a8dadc"]
    ax.pie(values, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
    ax.set_title(title)


def plot_summary(summary: dict, output_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    _grouped_bar(axes[0][0], summary["home_wins"], f"Home wins (n={summary['home_wins']['total_home_wins']})")
    _grouped_bar(axes[0][1], summary["draws"], f"Draws (n={summary['draws']['total_draws']})")
    _grouped_bar(axes[0][2], summary["away_wins"], f"Away wins (n={summary['away_wins']['total_away_wins']})")

    _pie(axes[1][0], summary["home_wins"], "Home wins by favorite (goals/match)")
    _pie(axes[1][1], summary["draws"], "Draws by favorite (goals/match)")
    _pie(axes[1][2], summary["away_wins"], "Away wins by favorite (goals/match)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def print_summary(summary: dict) -> None:
    print("=== Goals-Before Match Outcome Analysis ===\n")

    hw = summary["home_wins"]
    print(f"1. Home wins (total {hw['total_home_wins']}):")
    print(f"   goals_scored           -> home_favored: {hw['goals_scored']['home_favored']}, "
          f"away_favored: {hw['goals_scored']['away_favored']}, equal: {hw['goals_scored']['equal']}")
    print(f"   goals_scored_per_match  -> home_favored: {hw['goals_scored_per_match']['home_favored']}, "
          f"away_favored: {hw['goals_scored_per_match']['away_favored']}, equal: {hw['goals_scored_per_match']['equal']}")

    d = summary["draws"]
    print(f"\n2. Draws (total {d['total_draws']}):")
    print(f"   goals_scored           -> home_favored: {d['goals_scored']['home_favored']}, "
          f"away_favored: {d['goals_scored']['away_favored']}, equal: {d['goals_scored']['equal']}")
    print(f"   goals_scored_per_match  -> home_favored: {d['goals_scored_per_match']['home_favored']}, "
          f"away_favored: {d['goals_scored_per_match']['away_favored']}, equal: {d['goals_scored_per_match']['equal']}")

    aw = summary["away_wins"]
    print(f"\n3. Away wins (total {aw['total_away_wins']}):")
    print(f"   goals_scored           -> home_favored: {aw['goals_scored']['home_favored']}, "
          f"away_favored: {aw['goals_scored']['away_favored']}, equal: {aw['goals_scored']['equal']}")
    print(f"   goals_scored_per_match  -> home_favored: {aw['goals_scored_per_match']['home_favored']}, "
          f"away_favored: {aw['goals_scored_per_match']['away_favored']}, equal: {aw['goals_scored_per_match']['equal']}")


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/England.ddb"
    TABLE_NAME = "40-2025-goals"
    OUTPUT_CHART_PATH = "charts/40-2025-goals_analysis_charts.png"

    df = load_data(db_path=DB_PATH, table_name=TABLE_NAME)
    summary = summarize(df)
    print_summary(summary)
    plot_summary(summary, output_path=OUTPUT_CHART_PATH)