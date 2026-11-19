"""
expected_goals_analysis.py

Reads the `expected_goals` table (produced by expected_goals.py) and
compares the pre-match xG "favorite" (whichever of x_team_home_expected_goals / y_team_away_expected_goals
was higher) against the actual match outcome.

Definitions:
  - favored side  = the team (home/x_team or away/y_team) with the
                     HIGHER expected goals going into the match.
  - underdog       = the team with the LOWER expected goals.
  - equal          = x_team_home_expected_goals == y_team_away_expected_goals (no favorite either way).

Decisive matches (home win or away win) are split into three mutually
exclusive buckets:
  1. favored_won    -> the favored side's actual result was a win
                        (equivalently, the underdog lost)
  2. underdog_won    -> the underdog's actual result was a win
                        (equivalently, the favored side lost — an upset)
  3. equal_decisive  -> xG was tied pre-match, but the match still had
                        a winner

Draws are split into three buckets based on which side (if any) was
favored pre-match:
  - home_favored, away_favored, equal

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


def classify_favorite(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of 'home_favored' / 'away_favored' / 'equal' based on
    comparing x_team_home_expected_goals (home) to y_team_away_expected_goals (away) for each match.
    """
    def label(row):
        if row["x_team_home_expected_goals"] > row["y_team_away_expected_goals"]:
            return "home_favored"
        elif row["x_team_home_expected_goals"] < row["y_team_away_expected_goals"]:
            return "away_favored"
        return "equal"

    return df.apply(label, axis=1)


def win_outcomes_by_prediction(df: pd.DataFrame) -> dict:
    """
    Q1 & Q2 combined: among decisive matches (home win or away win),
    classify each into:
      - favored_won:   the pre-match xG favorite actually won
      - underdog_won:  the pre-match xG underdog actually won (an upset)
      - equal_decisive: xG was tied pre-match, but the match had a winner
    """
    favorite = classify_favorite(df)
    home_win = _is_home_win(df)
    away_win = _is_away_win(df)
    decisive = home_win | away_win

    favored_won = (
        (favorite == "home_favored") & home_win
    ) | (
        (favorite == "away_favored") & away_win
    )
    underdog_won = (
        (favorite == "home_favored") & away_win
    ) | (
        (favorite == "away_favored") & home_win
    )
    equal_decisive = (favorite == "equal") & decisive

    return {
        "favored_won": int(favored_won.sum()),
        "underdog_won": int(underdog_won.sum()),
        "equal_decisive": int(equal_decisive.sum()),
        "total_decisive": int(decisive.sum()),
    }


def draws_by_favorite(df: pd.DataFrame) -> dict:
    """
    Q3: among drawn matches, classify each by which side (if any) was
    favored pre-match by xG: home_favored, away_favored, or equal.
    """
    favorite = classify_favorite(df)
    draw = _is_draw(df)

    return {
        "home_favored": int(((favorite == "home_favored") & draw).sum()),
        "away_favored": int(((favorite == "away_favored") & draw).sum()),
        "equal": int(((favorite == "equal") & draw).sum()),
        "total_draws": int(draw.sum()),
    }


def summarize(df: pd.DataFrame) -> dict:
    return {
        "win_outcomes": win_outcomes_by_prediction(df),
        "draws": draws_by_favorite(df),
    }


def plot_summary(summary: dict, output_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- Chart 1: bar chart of decisive-match prediction outcomes ---
    wo = summary["win_outcomes"]
    labels1 = ["Favored won", "Underdog won\n(upset)", "Equal xG\n(still decisive)"]
    values1 = [wo["favored_won"], wo["underdog_won"], wo["equal_decisive"]]
    colors1 = ["#2a9d8f", "#e76f51", "#a8dadc"]
    bars1 = axes[0][0].bar(labels1, values1, color=colors1)
    axes[0][0].set_title(f"Decisive matches vs. xG prediction (n={wo['total_decisive']})")
    axes[0][0].set_ylabel("Number of matches")
    axes[0][0].bar_label(bars1, padding=3)

    # --- Chart 2: pie chart, same data ---
    axes[0][1].pie(values1, labels=labels1, autopct="%1.0f%%", colors=colors1, startangle=90)
    axes[0][1].set_title("Share of decisive matches by prediction outcome")

    # --- Chart 3: bar chart of draws by favorite ---
    dd = summary["draws"]
    labels2 = ["Home favored", "Away favored", "Equal"]
    values2 = [dd["home_favored"], dd["away_favored"], dd["equal"]]
    colors2 = ["#264653", "#e9c46a", "#f4a261"]
    bars2 = axes[1][0].bar(labels2, values2, color=colors2)
    axes[1][0].set_title(f"Draws vs. xG favorite (n={dd['total_draws']})")
    axes[1][0].set_ylabel("Number of matches")
    axes[1][0].bar_label(bars2, padding=3)

    # --- Chart 4: pie chart, same data ---
    axes[1][1].pie(values2, labels=labels2, autopct="%1.0f%%", colors=colors2, startangle=90)
    axes[1][1].set_title("Share of draws by xG favorite")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def print_summary(summary: dict) -> None:
    print("=== Expected Goals (xG) Prediction Analysis ===\n")

    wo = summary["win_outcomes"]
    print(f"Decisive matches (total {wo['total_decisive']}):")
    print(f"1. Win for favored side (loss for underdog): {wo['favored_won']}")
    print(f"2. Loss for favored side (win for underdog / upset): {wo['underdog_won']}")
    print(f"   Equal xG, still decisive: {wo['equal_decisive']}")

    dd = summary["draws"]
    print(f"\n3. Draws (total {dd['total_draws']}):")
    print(f"   Home favored: {dd['home_favored']}")
    print(f"   Away favored: {dd['away_favored']}")
    print(f"   Equal: {dd['equal']}")


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/England.ddb"
    TABLE_NAME = "40-2025-xg"
    OUTPUT_CHART_PATH = "charts/40-2025-xg_analysis_charts.png"

    df = load_data(db_path=DB_PATH, table_name=TABLE_NAME)
    summary = summarize(df)
    print_summary(summary)
    plot_summary(summary, output_path=OUTPUT_CHART_PATH)