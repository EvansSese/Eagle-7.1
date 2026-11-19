"""Season-local, pre-match outcome probabilities and consistency metrics."""

from collections import Counter, defaultdict
import math

import pandas as pd

from processing.core import COMPLETED_STATUSES, database_connection, iter_fixture_kickoffs


OUTCOMES = ("win", "draw", "loss")
SCOPES = ("overall", "home_only", "away_only", "last_3", "last_5", "last_10")
PRIOR_MATCHES = 3
STREAK_PRIOR_MATCHES = 3
TEAM_METRICS = (
    "matches", "win_probability", "draw_probability", "loss_probability",
    "consistency", "streak_outcome", "streak_length", "streak_observations",
)


def load_fixtures(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Load fixtures from a DuckDB table."""
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _metrics(results):
    """Estimate outcome probabilities from a sequence of prior team results."""
    count = len(results)
    counts = Counter(results)
    baseline = {
        outcome: (counts[outcome] + PRIOR_MATCHES / len(OUTCOMES)) / (count + PRIOR_MATCHES)
        for outcome in OUTCOMES
    }
    streak_outcome = results[-1] if count else None
    streak_length = 0
    for result in reversed(results):
        if result != streak_outcome:
            break
        streak_length += 1

    # Learn what followed the same outcome and streak length in this history.
    # The current streak itself has no observed successor and is excluded.
    successors = []
    run_length = 0
    previous = None
    for index, result in enumerate(results):
        if index and previous == streak_outcome and run_length >= streak_length:
            successors.append(result)
        run_length = run_length + 1 if result == previous else 1
        previous = result
    successor_counts = Counter(successors)
    probabilities = {
        outcome: (
            STREAK_PRIOR_MATCHES * baseline[outcome] + successor_counts[outcome]
        ) / (STREAK_PRIOR_MATCHES + len(successors))
        for outcome in OUTCOMES
    }
    consistency = (
        sum(a == b for a, b in zip(results, results[1:])) / (count - 1)
        if count >= 2 else None
    )
    return {
        "matches": count,
        **{f"{outcome}_probability": probabilities[outcome] for outcome in OUTCOMES},
        "consistency": consistency,
        "streak_outcome": streak_outcome,
        "streak_length": streak_length,
        "streak_observations": len(successors),
    }


def _scoped_results(history):
    results = [result for _, result in history]
    return {
        "overall": results,
        "home_only": [result for venue, result in history if venue == "home"],
        "away_only": [result for venue, result in history if venue == "away"],
        **{f"last_{window}": results[-window:] for window in (3, 5, 10)},
    }


def add_outcome_probabilities_before_match(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-local team metrics using only results before each kickoff."""
    required_columns = {
        "fixture_id", "date", "home_team", "away_team",
        "home_goals", "away_goals", "status",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"fixtures are missing required columns: {sorted(missing_columns)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")
    if "season" in df and df["season"].isna().any():
        raise ValueError("fixture seasons cannot be missing")
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    seasons = df["season"].astype(str).tolist() if "season" in df else [""] * len(df)
    # The earliest round in each season starts from the same uninformative prior,
    # even if a fixture from that round is played after later rounds.
    first_rounds = {}
    if "round" in df:
        for season, round_label in zip(seasons, df["round"]):
            if season not in first_rounds and pd.notna(round_label):
                first_rounds[season] = round_label

    histories = defaultdict(list)
    records = []
    position = 0
    for _, kickoff in iter_fixture_kickoffs(df):
        finished_results = []
        for row in kickoff:
            season = seasons[position]
            first_round = (
                "round" in df
                and season in first_rounds
                and pd.notna(df.at[position, "round"])
                and df.at[position, "round"] == first_rounds[season]
            )
            position += 1
            home, away = row.home_team, row.away_team
            if (not isinstance(home, str) or not home.strip()
                    or not isinstance(away, str) or not away.strip()):
                raise ValueError("fixture team names cannot be empty")
            if home == away:
                raise ValueError("fixture teams must be different")

            record = {}
            for side, team in (("x", home), ("y", away)):
                history = [] if first_round else histories[(season, team)]
                for scope, results in _scoped_results(history).items():
                    metrics = _metrics(results)
                    for metric, value in metrics.items():
                        record[f"{side}_team_{scope}_{metric}_before"] = value
                    if scope == "overall":
                        for outcome in OUTCOMES:
                            record[f"{side}_team_{outcome}_probability_before"] = metrics[f"{outcome}_probability"]

            for outcome, left, right in (
                ("home_win", "win", "loss"),
                ("draw", "draw", "draw"),
                ("away_win", "loss", "win"),
            ):
                record[f"combined_{outcome}_probability_before"] = (
                    record[f"x_team_{left}_probability_before"]
                    + record[f"y_team_{right}_probability_before"]
                ) / 2
            # Preserve the existing least-probable-event interpretation.
            record["predicted_outcome"] = min(
                ("home_win", "draw", "away_win"),
                key=lambda outcome: record[f"combined_{outcome}_probability_before"],
            )
            records.append(record)

            if row.status not in COMPLETED_STATUSES:
                continue
            if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                raise ValueError("completed fixtures must include both teams' goals")
            if (not math.isfinite(row.home_goals) or not math.isfinite(row.away_goals)
                    or row.home_goals < 0 or row.away_goals < 0):
                raise ValueError("completed fixture goals must be finite and non-negative")
            if row.home_goals > row.away_goals:
                home_result, away_result = "win", "loss"
            elif row.home_goals < row.away_goals:
                home_result, away_result = "loss", "win"
            else:
                home_result = away_result = "draw"
            finished_results.append((season, home, away, home_result, away_result))

        for season, home, away, home_result, away_result in finished_results:
            histories[(season, home)].append(("home", home_result))
            histories[(season, away)].append(("away", away_result))

    metric_columns = [
        f"{side}_team_{scope}_{metric}_before"
        for side in ("x", "y") for scope in SCOPES for metric in TEAM_METRICS
    ]
    legacy_columns = [
        f"{side}_team_{outcome}_probability_before"
        for side in ("x", "y") for outcome in OUTCOMES
    ] + [
        f"combined_{outcome}_probability_before"
        for outcome in ("home_win", "draw", "away_win")
    ] + ["predicted_outcome"]
    result = pd.DataFrame.from_records(records, columns=metric_columns + legacy_columns, index=df.index)
    decimal_columns = [
        column for column in result
        if "probability" in column or "consistency" in column
    ]
    result[decimal_columns] = result[decimal_columns].astype(float).round(2)
    text_columns = [column for column in result if column.endswith("_streak_outcome_before")]
    text_columns.append("predicted_outcome")
    result[text_columns] = result[text_columns].astype("string")
    count_columns = [
        f"{side}_team_{scope}_{metric}_before"
        for side in ("x", "y") for scope in SCOPES
        for metric in ("matches", "streak_length", "streak_observations")
    ]
    result[count_columns] = result[count_columns].astype("int64")
    return pd.concat([df, result], axis=1)


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Write processed fixture probabilities to a DuckDB table."""
    with database_connection(db_path, connection) as con:
        con.register("df_view", df)
        try:
            con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_view')
        finally:
            con.unregister("df_view")


if __name__ == "__main__":
    input_db_path = "../../storage/database/leagues/England.ddb"
    input_table = "39-2025-fixtures"
    output_table = "39-2025-outcome-probability"

    fixtures = load_fixtures(input_db_path, input_table)
    result = add_outcome_probabilities_before_match(fixtures)
    write_output(result, input_db_path, output_table)
    print(f"Wrote {len(result)} rows to '{output_table}' in {input_db_path}")
