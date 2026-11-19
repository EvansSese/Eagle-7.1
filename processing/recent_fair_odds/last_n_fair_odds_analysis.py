"""
last_n_fair_odds_analysis.py

Reads the `40-2025-odds-l{window}` table (produced by
last_n_fair_odds.py) and checks how well the model's rolling
last-N-matches win probabilities predicted actual results.

Definitions:
  - favored side = the team (home/x_team or away/y_team) with the
                    HIGHER win probability going into the match
                    (x_team_win_probability vs y_team_win_probabilty).
  - underdog      = the team with the LOWER win probability.
  - equal         = the two win probabilities were exactly tied
                     (common early on, before either team has `window`
                     matches of history).

Wins are split into:
  1. favored_won   -> the pre-match favorite actually won
  2. underdog_won   -> the pre-match underdog actually won (an upset)
  3. equal_decisive -> win probabilities were tied, but the match still
                        had a winner

Draws are split by which side (if any) was favored pre-match:
  - home_favored, away_favored, equal

A bonus 3x3 heatmap compares the model's single most-likely outcome
(whichever of home win / draw / away win had the highest probability)
against the actual outcome — a quick visual gut-check of overall model
calibration, beyond just the favored/underdog framing.

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
    Returns 'home_favored' / 'away_favored' / 'equal' per match, based on
    comparing x_team_win_probability to y_team_win_probabilty.
    """
    def label(row):
        if row["x_team_win_probability"] > row["y_team_win_probabilty"]:
            return "home_favored"
        elif row["x_team_win_probability"] < row["y_team_win_probabilty"]:
            return "away_favored"
        return "equal"

    return df.apply(label, axis=1)


def win_outcomes_by_favorite(df: pd.DataFrame) -> dict:
    """
    Among decisive matches (home win or away win), classify each into:
      - favored_won:   the pre-match favorite (by win probability) won
      - underdog_won:  the pre-match underdog won (an upset)
      - equal_decisive: win probabilities were tied, but there was a winner
    """
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
    """
    Among drawn matches, classify each by which side (if any) was favored
    pre-match by win probability: home_favored, away_favored, or equal.
    """
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
    """
    Builds a 3x3 table of predicted outcome (whichever of home win/draw/
    away win had the highest probability) vs. actual outcome, counting
    matches in each combination.
    """
    order = ["Home Win", "Draw", "Away Win"]
    predicted = df.apply(_predicted_outcome_label, axis=1).rename("Predicted")
    actual = df.apply(_actual_outcome_label, axis=1).rename("Actual")
    matrix = pd.crosstab(predicted, actual).reindex(index=order, columns=order, fill_value=0)
    return matrix


def predicted_vs_actual_totals(confusion_matrix: pd.DataFrame) -> dict:
    """
    Collapses the 3x3 confusion matrix into total counts per outcome
    category (Home Win / Draw / Away Win), once for predicted outcomes
    and once for actual outcomes — for a direct side-by-side comparison
    of how often the model called each outcome vs. how often it actually
    happened.
    """
    predicted_totals = confusion_matrix.sum(axis=1)  # row sums
    actual_totals = confusion_matrix.sum(axis=0)      # column sums
    return {"predicted": predicted_totals.to_dict(), "actual": actual_totals.to_dict()}


def summarize(df: pd.DataFrame) -> dict:
    confusion_matrix = prediction_confusion_matrix(df)
    return {
        "win_outcomes": win_outcomes_by_favorite(df),
        "draws": draws_by_favorite(df),
        "confusion_matrix": confusion_matrix,
        "predicted_vs_actual_totals": predicted_vs_actual_totals(confusion_matrix),
    }


def plot_summary(summary: dict, output_path: str, window: int) -> None:
    fig = plt.figure(figsize=(18, 16))
    axes = [
        fig.add_subplot(3, 3, 1), fig.add_subplot(3, 3, 2), fig.add_subplot(3, 3, 3),
        fig.add_subplot(3, 3, 4), fig.add_subplot(3, 3, 5), fig.add_subplot(3, 3, 6),
        fig.add_subplot(3, 3, 7),
    ]
    summary_ax = fig.add_subplot(3, 3, (8, 9))  # spans the last two cells

    # --- Chart 1: bar chart of win outcomes vs prediction ---
    wo = summary["win_outcomes"]
    labels1 = ["Favored won", "Underdog won\n(upset)", "Equal prob.\n(still decisive)"]
    values1 = [wo["favored_won"], wo["underdog_won"], wo["equal_decisive"]]
    colors1 = ["#2a9d8f", "#e76f51", "#a8dadc"]
    bars1 = axes[0].bar(labels1, values1, color=colors1)
    axes[0].set_title(f"Decisive matches vs. last-{window}-games win prob. favorite (n={wo['total_decisive']})")
    axes[0].set_ylabel("Number of matches")
    axes[0].bar_label(bars1, padding=3)

    # --- Chart 2: pie chart, same data ---
    axes[1].pie(values1, labels=labels1, autopct="%1.0f%%", colors=colors1, startangle=90)
    axes[1].set_title("Share of decisive matches by prediction outcome")

    # --- Chart 3: bar chart of draws by favorite ---
    dd = summary["draws"]
    labels2 = ["Home favored", "Away favored", "Equal"]
    values2 = [dd["home_favored"], dd["away_favored"], dd["equal"]]
    colors2 = ["#264653", "#e9c46a", "#f4a261"]
    bars2 = axes[2].bar(labels2, values2, color=colors2)
    axes[2].set_title(f"Draws vs. last-{window}-games win prob. favorite (n={dd['total_draws']})")
    axes[2].set_ylabel("Number of matches")
    axes[2].bar_label(bars2, padding=3)

    # --- Chart 4: pie chart, same data ---
    axes[3].pie(values2, labels=labels2, autopct="%1.0f%%", colors=colors2, startangle=90)
    axes[3].set_title("Share of draws by win-probability favorite")

    # --- Chart 5: heatmap of predicted vs actual outcome ---
    cm = summary["confusion_matrix"]
    im = axes[4].imshow(cm.values, cmap="Blues")
    axes[4].set_xticks(range(len(cm.columns)))
    axes[4].set_xticklabels(cm.columns)
    axes[4].set_yticks(range(len(cm.index)))
    axes[4].set_yticklabels(cm.index)
    axes[4].set_xlabel("Actual outcome")
    axes[4].set_ylabel(f"Predicted outcome (highest last-{window} prob.)")
    axes[4].set_title("Prediction accuracy heatmap")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm.values[i, j]
            text_color = "white" if value > cm.values.max() / 2 else "black"
            axes[4].text(j, i, str(value), ha="center", va="center", color=text_color)
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    # --- Chart 6: grouped bar chart, predicted totals vs actual totals ---
    pvat = summary["predicted_vs_actual_totals"]
    outcome_order = ["Home Win", "Draw", "Away Win"]
    predicted_vals = [pvat["predicted"][o] for o in outcome_order]
    actual_vals = [pvat["actual"][o] for o in outcome_order]
    x = range(len(outcome_order))
    width = 0.35
    b1 = axes[5].bar([i - width / 2 for i in x], predicted_vals, width, label="Predicted", color="#457b9d")
    b2 = axes[5].bar([i + width / 2 for i in x], actual_vals, width, label="Actual", color="#f4a261")
    axes[5].set_xticks(list(x))
    axes[5].set_xticklabels(outcome_order)
    axes[5].set_title("Predicted vs. actual outcome totals")
    axes[5].set_ylabel("Number of matches")
    axes[5].legend()
    axes[5].bar_label(b1, padding=3)
    axes[5].bar_label(b2, padding=3)

    # --- Chart 7: pie chart of actual outcome distribution, for reference ---
    axes[6].pie(actual_vals, labels=outcome_order, autopct="%1.0f%%",
                colors=["#2a9d8f", "#f4a261", "#e76f51"], startangle=90)
    axes[6].set_title("Actual outcome distribution")

    # --- Summary panel ---
    summary_ax.axis("off")
    accuracy = np.trace(cm.values) / cm.values.sum() if cm.values.sum() > 0 else 0.0
    summary_text = (
        f"Form window: last {window} matches\n"
        f"Total matches: {wo['total_decisive'] + dd['total_draws']}\n\n"
        f"Win outcomes:\n"
        f"  Favored won: {wo['favored_won']}\n"
        f"  Underdog won: {wo['underdog_won']}\n"
        f"  Equal prob.: {wo['equal_decisive']}\n\n"
        f"Draws:\n"
        f"  Home favored: {dd['home_favored']}\n"
        f"  Away favored: {dd['away_favored']}\n"
        f"  Equal: {dd['equal']}\n\n"
        f"Overall 1X2 prediction accuracy: {accuracy:.1%}"
    )
    summary_ax.text(0.05, 0.95, summary_text, transform=summary_ax.transAxes,
                     va="top", ha="left", fontsize=11, family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def print_summary(summary: dict, window: int) -> None:
    print(f"=== Last-{window}-Games Fair Odds / Win Probability Prediction Analysis ===\n")

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

    pvat = summary["predicted_vs_actual_totals"]
    print("\n4. Predicted vs. actual outcome totals:")
    for outcome in ["Home Win", "Draw", "Away Win"]:
        print(f"   {outcome}: predicted {pvat['predicted'][outcome]}, actual {pvat['actual'][outcome]}")


if __name__ == "__main__":
    FORM_WINDOW = [3,5,10]
    DB_PATH = "../../storage/database/leagues/England.ddb"

    for window in FORM_WINDOW:
        TABLE_NAME = f"39-2025-odds-l{window}"
        OUTPUT_CHART_PATH = f"charts/39-2025-l{window}_fair_odds_analysis_charts.png"

        df = load_data(db_path=DB_PATH, table_name=TABLE_NAME)
        summary = summarize(df)
        print_summary(summary, window=window)
        plot_summary(summary, output_path=OUTPUT_CHART_PATH, window=window)