"""
l3_points_analysis.py

Reads the `39-2025-points-l3` table (produced by l3-points.py) and checks
how well each team's RECENT FORM — points per match over their last 3
matches — predicted actual match outcomes.

Definitions:
  - favored side = the team (home/x_team or away/y_team) with the
                    HIGHER points_per_match_last3 going into the match.
  - underdog      = the team with the LOWER points_per_match_last3.
  - equal         = both teams' recent-form rate was exactly tied
                     (including 0-0 early in a team's fixture list,
                     before either side has 3 match's worth of history).

Every match falls into exactly one of three primary buckets:
  1. Favored win  -> the team with better recent form actually won
  2. Draw         -> the match ended level, regardless of who was favored
  3. Underdog win -> the team with worse recent form actually won (upset)

(Matches where recent form was exactly tied and still produced a winner
are tracked separately as a 4th, usually small, bucket for completeness.)

A heatmap then compares the model's single predicted outcome (Home Win /
Draw / Away Win, based on which side's recent form was better, or Draw
if tied) against the actual outcome.

Requires: duckdb, pandas, numpy, matplotlib
"""
import duckdb
import pandas as pd
import numpy as np
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


def classify_favorite(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Returns 'home_favored' / 'away_favored' / 'equal' per match, based on
    comparing x_team_points_per_match_last{window} to
    y_team_points_per_match_last{window}.
    """
    def label(row):
        if row[f"x_team_points_per_match_last{window}"] > row[f"y_team_points_per_match_last{window}"]:
            return "home_favored"
        elif row[f"x_team_points_per_match_last{window}"] < row[f"y_team_points_per_match_last{window}"]:
            return "away_favored"
        return "equal"

    return df.apply(label, axis=1)


def win_draw_underdog_breakdown(df: pd.DataFrame, window: int) -> dict:
    """
    Q1: Every match falls into one of three primary buckets:
      - favored_win: the team with better recent form won
      - draw: the match ended level
      - underdog_win: the team with worse recent form won (an upset)
    Matches where recent form was exactly tied but still decisive are
    tracked separately as equal_form_decisive.
    """
    favorite = classify_favorite(df, window=window)
    home_win = _is_home_win(df)
    away_win = _is_away_win(df)
    draw = _is_draw(df)

    favored_win = ((favorite == "home_favored") & home_win) | ((favorite == "away_favored") & away_win)
    underdog_win = ((favorite == "home_favored") & away_win) | ((favorite == "away_favored") & home_win)
    equal_form_decisive = (favorite == "equal") & (home_win | away_win)

    return {
        "favored_win": int(favored_win.sum()),
        "draw": int(draw.sum()),
        "underdog_win": int(underdog_win.sum()),
        "equal_form_decisive": int(equal_form_decisive.sum()),
        "total_matches": int(len(df)),
    }


def _predicted_outcome_label(row, window: int) -> str:
    home_form = row[f"x_team_points_per_match_last{window}"]
    away_form = row[f"y_team_points_per_match_last{window}"]
    if home_form > away_form:
        return "Home Win"
    elif home_form < away_form:
        return "Away Win"
    return "Draw"


def _actual_outcome_label(row) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "Home Win"
    elif row["home_goals"] < row["away_goals"]:
        return "Away Win"
    return "Draw"


def prediction_confusion_matrix(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Q2: A 3x3 table of predicted outcome (based on which side had better
    recent form over their last `window` matches, or Draw if tied) vs.
    actual outcome.
    """
    order = ["Home Win", "Draw", "Away Win"]
    predicted = df.apply(lambda row: _predicted_outcome_label(row, window), axis=1).rename("Predicted")
    actual = df.apply(_actual_outcome_label, axis=1).rename("Actual")
    matrix = pd.crosstab(predicted, actual).reindex(index=order, columns=order, fill_value=0)
    return matrix


def summarize(df: pd.DataFrame, window: int) -> dict:
    return {
        "breakdown": win_draw_underdog_breakdown(df, window=window),
        "confusion_matrix": prediction_confusion_matrix(df, window=window),
    }


def plot_summary(summary: dict, output_path: str, window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- Chart 1: bar chart of favored win / draw / underdog win ---
    b = summary["breakdown"]
    labels = ["Favored win", "Draw", "Underdog win\n(upset)"]
    values = [b["favored_win"], b["draw"], b["underdog_win"]]
    colors = ["#2a9d8f", "#457b9d", "#e76f51"]
    if b["equal_form_decisive"] > 0:
        labels.append("Equal form,\nstill decisive")
        values.append(b["equal_form_decisive"])
        colors.append("#a8dadc")

    bars = axes[0][0].bar(labels, values, color=colors)
    axes[0][0].set_title(f"Match outcomes vs. last-{window}-games form (n={b['total_matches']})")
    axes[0][0].set_ylabel("Number of matches")
    axes[0][0].bar_label(bars, padding=3)

    # --- Chart 2: pie chart, same data ---
    axes[0][1].pie(values, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
    axes[0][1].set_title("Share of matches by form-based prediction outcome")

    # --- Chart 3: heatmap of predicted vs actual outcome ---
    cm = summary["confusion_matrix"]
    im = axes[1][0].imshow(cm.values, cmap="Blues")
    axes[1][0].set_xticks(range(len(cm.columns)))
    axes[1][0].set_xticklabels(cm.columns)
    axes[1][0].set_yticks(range(len(cm.index)))
    axes[1][0].set_yticklabels(cm.index)
    axes[1][0].set_xlabel("Actual outcome")
    axes[1][0].set_ylabel(f"Predicted outcome (better last-{window} form)")
    axes[1][0].set_title("Recent-form prediction accuracy heatmap")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm.values[i, j]
            text_color = "white" if value > cm.values.max() / 2 else "black"
            axes[1][0].text(j, i, str(value), ha="center", va="center", color=text_color)
    fig.colorbar(im, ax=axes[1][0], fraction=0.046, pad=0.04)

    # --- Panel 4: summary text ---
    axes[1][1].axis("off")
    accuracy = np.trace(cm.values) / cm.values.sum() if cm.values.sum() > 0 else 0.0
    summary_text = (
        f"Total matches: {b['total_matches']}\n\n"
        f"Favored win (better L{window} form): {b['favored_win']}\n"
        f"Draw: {b['draw']}\n"
        f"Underdog win (upset): {b['underdog_win']}\n"
        f"Equal form, still decisive: {b['equal_form_decisive']}\n\n"
        f"Overall 1X2 prediction accuracy: {accuracy:.1%}"
    )
    axes[1][1].text(0.05, 0.95, summary_text, transform=axes[1][1].transAxes,
                     va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def print_summary(summary: dict, window: int) -> None:
    print(f"=== Last-{window}-Games Form Prediction Analysis ===\n")

    b = summary["breakdown"]
    print(f"Total matches: {b['total_matches']}")
    print(f"1. Favored win (better recent form): {b['favored_win']}")
    print(f"   Draw: {b['draw']}")
    print(f"   Underdog win (upset): {b['underdog_win']}")
    print(f"   Equal form, still decisive: {b['equal_form_decisive']}")

    cm = summary["confusion_matrix"]
    print("\n2. Predicted vs. actual outcome (counts):")
    print(cm.to_string())


if __name__ == "__main__":
    FORM_WINDOW = [3,5,10]
    DB_PATH = "../../storage/database/leagues/England.ddb"

    for window in FORM_WINDOW:
        TABLE_NAME = f"39-2025-points-l{window}"
        OUTPUT_CHART_PATH = f"charts/39-2025-l{window}_points_analysis_charts.png"

        df = load_data(db_path=DB_PATH, table_name=TABLE_NAME)
        summary = summarize(df, window=window)
        print_summary(summary, window=window)
        plot_summary(summary, output_path=OUTPUT_CHART_PATH, window=window)