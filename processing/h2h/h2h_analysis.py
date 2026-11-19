"""
head_to_head_analysis.py

Reads the `{league_id}-{season}-h2h` table (produced by h2h.py)
and checks how well the head-to-head fair-odds win probabilities
(x_team_win_probability / draw_probability / y_team_win_probabilty —
computed from ONLY these two teams' past meetings) predicted actual
results.

This mirrors fair_odds_analysis.py's structure exactly, with one
addition specific to head-to-head data: a fixture with ZERO prior
meetings between the two teams gets a trivial, uninformative
prediction (expected_goals collapses to 0-0, so the model just predicts
a deterministic draw) — that's not a real prediction, just the
"no data yet" default. This script reports overall accuracy alongside
accuracy split by whether the two teams had ANY head-to-head history at
all, so a pile of first-ever meetings can't quietly make the H2H model
look better or worse than it actually is on fixtures it had real
history to work with.

Definitions:
  - favored side = the team (home/x_team or away/y_team) with the
                    HIGHER head-to-head win probability going into the
                    match.
  - underdog      = the team with the LOWER head-to-head win probability.
  - equal         = the two win probabilities were exactly tied
                     (guaranteed for any fixture with 0 prior meetings,
                     since both default to the same trivial values).

Requires: duckdb, pandas, matplotlib, numpy
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from processing.core import COMPLETED_STATUSES


def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    return df[df["status"].isin(COMPLETED_STATUSES)].copy()


def _is_home_win(df: pd.DataFrame) -> pd.Series:
    return df["home_goals"] > df["away_goals"]


def _is_away_win(df: pd.DataFrame) -> pd.Series:
    return df["away_goals"] > df["home_goals"]


def _is_draw(df: pd.DataFrame) -> pd.Series:
    return df["home_goals"] == df["away_goals"]


def classify_favorite(df: pd.DataFrame) -> pd.Series:
    def label(row):
        if row["x_team_win_probability"] > row["y_team_win_probabilty"]:
            return "home_favored"
        elif row["x_team_win_probability"] < row["y_team_win_probabilty"]:
            return "away_favored"
        return "equal"

    return df.apply(label, axis=1)


def win_outcomes_by_favorite(df: pd.DataFrame) -> dict:
    favorite = classify_favorite(df)
    home_win = _is_home_win(df)
    away_win = _is_away_win(df)
    decisive = home_win | away_win

    favored_won = ((favorite == "home_favored") & home_win) | ((favorite == "away_favored") & away_win)
    underdog_won = ((favorite == "home_favored") & away_win) | ((favorite == "away_favored") & home_win)
    equal_decisive = (favorite == "equal") & decisive

    return {
        "favored_won": int(favored_won.sum()),
        "underdog_won": int(underdog_won.sum()),
        "equal_decisive": int(equal_decisive.sum()),
        "total_decisive": int(decisive.sum()),
    }


def draws_by_favorite(df: pd.DataFrame) -> dict:
    favorite = classify_favorite(df)
    draw = _is_draw(df)

    return {
        "home_favored": int(((favorite == "home_favored") & draw).sum()),
        "away_favored": int(((favorite == "away_favored") & draw).sum()),
        "equal": int(((favorite == "equal") & draw).sum()),
        "total_draws": int(draw.sum()),
    }


def _predicted_outcome_label(row) -> str:
    probs = {
        "Home Win": row["x_team_win_probability"],
        "Draw": row["draw_probability"],
        "Away Win": row["y_team_win_probabilty"],
    }
    return max(probs, key=probs.get)


def _actual_outcome_label(row) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "Home Win"
    elif row["home_goals"] < row["away_goals"]:
        return "Away Win"
    return "Draw"


def prediction_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Home Win", "Draw", "Away Win"]
    predicted = df.apply(_predicted_outcome_label, axis=1).rename("Predicted")
    actual = df.apply(_actual_outcome_label, axis=1).rename("Actual")
    matrix = pd.crosstab(predicted, actual).reindex(index=order, columns=order, fill_value=0)
    return matrix


def meetings_distribution(df: pd.DataFrame) -> dict:
    buckets = df["h2h_previous_meetings"].clip(upper=4)
    counts = buckets.value_counts().sort_index()
    return {
        "0": int(counts.get(0, 0)),
        "1": int(counts.get(1, 0)),
        "2": int(counts.get(2, 0)),
        "3": int(counts.get(3, 0)),
        "4+": int(counts.get(4, 0)),
        "total": int(len(df)),
    }


def accuracy_by_history(df: pd.DataFrame) -> dict:
    def _accuracy(subset: pd.DataFrame) -> dict:
        if len(subset) == 0:
            return {"correct": 0, "total": 0, "accuracy": 0.0}
        predicted = subset.apply(_predicted_outcome_label, axis=1)
        actual = subset.apply(_actual_outcome_label, axis=1)
        correct = int((predicted == actual).sum())
        return {"correct": correct, "total": len(subset), "accuracy": correct / len(subset)}

    no_history = df[df["h2h_previous_meetings"] == 0]
    has_history = df[df["h2h_previous_meetings"] > 0]

    return {
        "no_history": _accuracy(no_history),
        "has_history": _accuracy(has_history),
        "overall": _accuracy(df),
    }


def summarize(df: pd.DataFrame) -> dict:
    return {
        "win_outcomes": win_outcomes_by_favorite(df),
        "draws": draws_by_favorite(df),
        "confusion_matrix": prediction_confusion_matrix(df),
        "meetings_distribution": meetings_distribution(df),
        "accuracy_by_history": accuracy_by_history(df),
    }


def plot_summary(summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(18, 16))
    axes = [
        fig.add_subplot(3, 3, 1), fig.add_subplot(3, 3, 2), fig.add_subplot(3, 3, 3),
        fig.add_subplot(3, 3, 4), fig.add_subplot(3, 3, 5), fig.add_subplot(3, 3, 6),
        fig.add_subplot(3, 3, 7),
    ]
    summary_ax = fig.add_subplot(3, 3, (8, 9))

    wo = summary["win_outcomes"]
    labels1 = ["Favored won", "Underdog won\n(upset)", "Equal prob.\n(still decisive)"]
    values1 = [wo["favored_won"], wo["underdog_won"], wo["equal_decisive"]]
    colors1 = ["#2a9d8f", "#e76f51", "#a8dadc"]
    bars1 = axes[0].bar(labels1, values1, color=colors1)
    axes[0].set_title(f"Decisive matches vs. H2H win-probability favorite (n={wo['total_decisive']})")
    axes[0].set_ylabel("Number of matches")
    axes[0].bar_label(bars1, padding=3)

    axes[1].pie(values1, labels=labels1, autopct="%1.0f%%", colors=colors1, startangle=90)
    axes[1].set_title("Share of decisive matches by H2H prediction outcome")

    dd = summary["draws"]
    labels2 = ["Home favored", "Away favored", "Equal"]
    values2 = [dd["home_favored"], dd["away_favored"], dd["equal"]]
    colors2 = ["#264653", "#e9c46a", "#f4a261"]
    bars2 = axes[2].bar(labels2, values2, color=colors2)
    axes[2].set_title(f"Draws vs. H2H win-probability favorite (n={dd['total_draws']})")
    axes[2].set_ylabel("Number of matches")
    axes[2].bar_label(bars2, padding=3)

    cm = summary["confusion_matrix"]
    im = axes[3].imshow(cm.values, cmap="Blues")
    axes[3].set_xticks(range(len(cm.columns)))
    axes[3].set_xticklabels(cm.columns)
    axes[3].set_yticks(range(len(cm.index)))
    axes[3].set_yticklabels(cm.index)
    axes[3].set_xlabel("Actual outcome")
    axes[3].set_ylabel("Predicted outcome (highest H2H prob.)")
    axes[3].set_title("H2H prediction accuracy heatmap")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm.values[i, j]
            text_color = "white" if value > cm.values.max() / 2 else "black"
            axes[3].text(j, i, str(value), ha="center", va="center", color=text_color)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    md = summary["meetings_distribution"]
    labels5 = ["0\n(first meeting)", "1", "2", "3", "4+"]
    values5 = [md["0"], md["1"], md["2"], md["3"], md["4+"]]
    bars5 = axes[4].bar(labels5, values5, color="#457b9d")
    axes[4].set_title(f"Matches by number of prior H2H meetings (n={md['total']})")
    axes[4].set_ylabel("Number of matches")
    axes[4].bar_label(bars5, padding=3)

    ah = summary["accuracy_by_history"]
    values6 = [ah["no_history"]["total"], ah["has_history"]["total"]]
    labels6 = [f"No prior meetings\n({ah['no_history']['total']})", f"Has prior meetings\n({ah['has_history']['total']})"]
    axes[5].pie(values6, labels=labels6, autopct="%1.0f%%", colors=["#a8dadc", "#2a9d8f"], startangle=90)
    axes[5].set_title("Share of matches with vs. without H2H history")

    labels7 = ["No prior\nmeetings", "Has prior\nmeetings", "Overall"]
    values7 = [ah["no_history"]["accuracy"] * 100, ah["has_history"]["accuracy"] * 100, ah["overall"]["accuracy"] * 100]
    bars7 = axes[6].bar(labels7, values7, color=["#e76f51", "#2a9d8f", "#264653"])
    axes[6].set_title("1X2 prediction accuracy: with vs. without H2H history")
    axes[6].set_ylabel("Accuracy (%)")
    axes[6].set_ylim(0, 100)
    axes[6].bar_label(bars7, fmt="%.1f%%", padding=3)

    summary_ax.axis("off")
    summary_text = (
        f"Total matches: {wo['total_decisive'] + dd['total_draws']}\n\n"
        f"Win outcomes:\n"
        f"  Favored won: {wo['favored_won']}\n"
        f"  Underdog won: {wo['underdog_won']}\n"
        f"  Equal prob.: {wo['equal_decisive']}\n\n"
        f"Draws:\n"
        f"  Home favored: {dd['home_favored']}\n"
        f"  Away favored: {dd['away_favored']}\n"
        f"  Equal: {dd['equal']}\n\n"
        f"Accuracy with NO prior H2H meetings: "
        f"{ah['no_history']['correct']}/{ah['no_history']['total']} "
        f"({ah['no_history']['accuracy']:.1%})\n"
        f"Accuracy WITH prior H2H meetings: "
        f"{ah['has_history']['correct']}/{ah['has_history']['total']} "
        f"({ah['has_history']['accuracy']:.1%})\n"
        f"Overall accuracy: {ah['overall']['correct']}/{ah['overall']['total']} "
        f"({ah['overall']['accuracy']:.1%})"
    )
    summary_ax.text(0.05, 0.95, summary_text, transform=summary_ax.transAxes,
                     va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def print_summary(summary: dict) -> None:
    print("=== Head-to-Head Prediction Analysis ===\n")

    wo = summary["win_outcomes"]
    print(f"Decisive matches (total {wo['total_decisive']}):")
    print(f"1. Win for favored team: {wo['favored_won']}")
    print(f"   Win for underdog (upset): {wo['underdog_won']}")
    print(f"   Equal win probability, still decisive: {wo['equal_decisive']}")

    dd = summary["draws"]
    print(f"\n2. Draws (total {dd['total_draws']}):")
    print(f"   Home favored: {dd['home_favored']}")
    print(f"   Away favored: {dd['away_favored']}")
    print(f"   Equal: {dd['equal']}")

    cm = summary["confusion_matrix"]
    print("\n3. Predicted vs. actual outcome (counts):")
    print(cm.to_string())

    md = summary["meetings_distribution"]
    print(f"\n4. Matches by number of prior H2H meetings (total {md['total']}):")
    print(f"   0 (first meeting): {md['0']}, 1: {md['1']}, 2: {md['2']}, 3: {md['3']}, 4+: {md['4+']}")

    ah = summary["accuracy_by_history"]
    print("\n5. Accuracy with vs. without H2H history:")
    print(f"   No prior meetings:   {ah['no_history']['correct']}/{ah['no_history']['total']} ({ah['no_history']['accuracy']:.1%})")
    print(f"   Has prior meetings:  {ah['has_history']['correct']}/{ah['has_history']['total']} ({ah['has_history']['accuracy']:.1%})")
    print(f"   Overall:             {ah['overall']['correct']}/{ah['overall']['total']} ({ah['overall']['accuracy']:.1%})")


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/England.ddb"
    LEAGUE_ID = "39"
    SEASON = "2025"
    TABLE_NAME = f"{LEAGUE_ID}-{SEASON}-h2h"
    OUTPUT_CHART_PATH = f"charts/{LEAGUE_ID}-{SEASON}-h2h_analysis_charts.png"

    df = load_data(db_path=DB_PATH, table_name=TABLE_NAME)
    summary = summarize(df)
    print_summary(summary)
    plot_summary(summary, output_path=OUTPUT_CHART_PATH)