import math
from typing import Dict, Mapping, Optional

import duckdb
import pandas as pd
from processing.core import COMPLETED_STATUSES, database_connection, iter_fixture_kickoffs


OUTCOMES = ("win", "draw", "loss")
DEFAULT_ADJUSTMENT_RATE = 0.1


def load_fixtures(db_path: str, table_name: str, connection=None) -> pd.DataFrame:
    """Load fixtures from a DuckDB table."""
    with database_connection(db_path, connection, read_only=True) as con:
        df = con.execute(f'SELECT * FROM "{table_name}"').df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def update_outcome_probabilities(
    outcome: Optional[str] = None,
    probabilities: Optional[Mapping[str, float]] = None,
    adjustment_rate: float = DEFAULT_ADJUSTMENT_RATE,
) -> Dict[str, float]:
    """Initialize equal outcome probabilities or adjust them after a result.

    The adjustment rate is the fraction of the observed outcome's probability
    transferred equally to the two outcomes that did not occur.
    """
    if not math.isfinite(adjustment_rate) or not 0 <= adjustment_rate <= 1:
        raise ValueError("adjustment_rate must be between 0 and 1")

    if probabilities is None:
        current = {result: 1 / len(OUTCOMES) for result in OUTCOMES}
    else:
        if set(probabilities) != set(OUTCOMES):
            raise ValueError("probabilities must contain win, draw, and loss")
        current = {}
        for result in OUTCOMES:
            value = probabilities[result]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("probabilities must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError("probabilities must be finite and non-negative")
            current[result] = float(value)
        if not math.isclose(sum(current.values()), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to 1")

    if outcome is None:
        return current
    if outcome not in OUTCOMES:
        raise ValueError("outcome must be 'win', 'draw', or 'loss'")

    transfer = current[outcome] * adjustment_rate
    current[outcome] -= transfer
    increase = transfer / (len(OUTCOMES) - 1)
    for result in OUTCOMES:
        if result != outcome:
            current[result] += increase
    return current


def add_outcome_probabilities_before_match(
    df: pd.DataFrame,
    adjustment_rate: float = DEFAULT_ADJUSTMENT_RATE,
) -> pd.DataFrame:
    """Add each team's pre-match win/draw/loss probabilities to its fixtures."""
    required_columns = {
        "fixture_id", "date", "home_team", "away_team",
        "home_goals", "away_goals", "status",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"fixtures are missing required columns: {sorted(missing_columns)}")
    initial_probabilities = update_outcome_probabilities(adjustment_rate=adjustment_rate)

    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True).copy()
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")
    team_probabilities: Dict[str, Dict[str, float]] = {}
    probability_columns = {
        "x": {result: [] for result in OUTCOMES},
        "y": {result: [] for result in OUTCOMES},
    }
    combined_probability_columns = {
        "home_win": [],
        "draw": [],
        "away_win": [],
    }

    for _, kickoff in iter_fixture_kickoffs(df):
        finished_results = []
        for row in kickoff:
            home, away = row.home_team, row.away_team
            if pd.isna(home) or pd.isna(away):
                raise ValueError("fixture team names cannot be empty")

            if home not in team_probabilities:
                team_probabilities[home] = initial_probabilities.copy()
            if away not in team_probabilities:
                team_probabilities[away] = initial_probabilities.copy()
            home_probabilities = team_probabilities[home]
            away_probabilities = team_probabilities[away]
            for result in OUTCOMES:
                probability_columns["x"][result].append(home_probabilities[result])
                probability_columns["y"][result].append(away_probabilities[result])
            combined_probability_columns["home_win"].append(
                (home_probabilities["win"] + away_probabilities["loss"]) / 2
            )
            combined_probability_columns["draw"].append(
                (home_probabilities["draw"] + away_probabilities["draw"]) / 2
            )
            combined_probability_columns["away_win"].append(
                (home_probabilities["loss"] + away_probabilities["win"]) / 2
            )

            if row.status not in COMPLETED_STATUSES:
                continue
            if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                raise ValueError("completed fixtures must include both teams' goals")

            if row.home_goals > row.away_goals:
                home_result, away_result = "win", "loss"
            elif row.home_goals < row.away_goals:
                home_result, away_result = "loss", "win"
            else:
                home_result = away_result = "draw"
            finished_results.append(
                (home, away, home_probabilities.copy(), away_probabilities.copy(),
                 home_result, away_result)
            )

        for home, away, home_probabilities, away_probabilities, home_result, away_result in finished_results:
            team_probabilities[home] = update_outcome_probabilities(
                home_result, home_probabilities, adjustment_rate
            )
            team_probabilities[away] = update_outcome_probabilities(
                away_result, away_probabilities, adjustment_rate
            )

    for side in ("x", "y"):
        for result in OUTCOMES:
            df[f"{side}_team_{result}_probability_before"] = probability_columns[side][result]
    for outcome, probabilities in combined_probability_columns.items():
        df[f"combined_{outcome}_probability_before"] = probabilities
    combined_probability_df = pd.DataFrame(combined_probability_columns, index=df.index)
    # This field intentionally reports the least probable event (the user's
    # chosen interpretation), not the most likely prediction.
    df["predicted_outcome"] = combined_probability_df.idxmin(axis=1)
    return df


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Write processed fixture probabilities to a DuckDB table."""
    with database_connection(db_path, connection) as con:
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        con.register("df_view", df)
        try:
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_view')
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