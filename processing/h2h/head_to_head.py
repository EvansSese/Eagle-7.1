"""Chronological, matchup-specific head-to-head features."""

from collections import defaultdict, deque
from functools import lru_cache
from itertools import groupby
import math
from numbers import Real

import pandas as pd

from processing.core import (
    COMPLETED_STATUSES, database_connection, get_bucket, goals_bucket,
    goals_per_match_bucket, points_bucket, points_per_match_bucket,
)
from processing.fair_odds.dixon_coles import (
    build_score_matrix, most_likely_score, outcome_probabilities,
    probability_to_decimal_odds,
)
from processing.goals.goal_metrics import GoalWindow, METRICS as GOAL_METRICS
from processing.outcome_probability.outcome_probability import _metrics as outcome_metrics
from processing.points.point_metrics import PointsWindow, METRICS as POINT_METRICS, match_points
from processing.power_ratings.power_ratings import (
    INITIAL_RATING, HOME_ADVANTAGE, _change, _replay,
)


WINDOWS = (3, 5, 10)
SPANS = ("all_time", "season")
SCOPES = ("overall", "home_only", "away_only", "last_3", "last_5", "last_10")


class TeamScope:
    """Incremental goals, points and power state for one team and scope."""

    def __init__(self, window=None):
        self.window = window
        self.goals = GoalWindow(window)
        self.points = PointsWindow(window)
        self.outcomes = deque(maxlen=window) if window else []
        self.power_history = deque(maxlen=window) if window else None
        self.rating = INITIAL_RATING
        self.cached_snapshot = None

    def snapshot(self):
        if self.cached_snapshot is not None:
            return self.cached_snapshot
        goals = dict(zip(GOAL_METRICS, self.goals.snapshot()))
        points = dict(zip(POINT_METRICS, self.points.snapshot()))
        outcomes = outcome_metrics(list(self.outcomes))
        rating = _replay(self.power_history) if self.window else self.rating
        self.cached_snapshot = goals, points, outcomes, rating
        return self.cached_snapshot

    def record(self, date, scored, conceded, opponent_rating, venue_advantage, prior_rating):
        self.cached_snapshot = None
        scored, conceded = int(scored), int(conceded)
        self.goals.add(scored, conceded)
        points, _ = match_points(scored, conceded)
        self.points.add(points)
        self.outcomes.append("win" if points == 3 else "draw" if points == 1 else "loss")
        if self.window:
            self.power_history.append((date, opponent_rating, scored - conceded, venue_advantage))
        else:
            self.rating += _change(prior_rating, opponent_rating, scored - conceded, venue_advantage)


def _new_teams(scopes):
    return {
        side: {
            scope: TeamScope(int(scope[5:]) if scope.startswith("last_") else None)
            for scope in scopes
        }
        for side in (0, 1)
    }


def _ratio(value, denominator):
    return value / denominator if denominator else 0.0


@lru_cache(maxsize=4096)
def _odds(x_xg, y_xg):
    matrix = build_score_matrix(x_xg, y_xg)
    outcomes = outcome_probabilities(matrix)
    home_score, away_score = most_likely_score(matrix)
    return {
        "home_win_probability": outcomes["home_win"],
        "draw_probability": outcomes["draw"],
        "away_win_probability": outcomes["away_win"],
        "home_correct_score": home_score,
        "away_correct_score": away_score,
        "home_win_odds": probability_to_decimal_odds(outcomes["home_win"]),
        "draw_odds": probability_to_decimal_odds(outcomes["draw"]),
        "away_win_odds": probability_to_decimal_odds(outcomes["away_win"]),
    }


def _model(x_goals, y_goals, home_avg, away_avg):
    """Use H2H scoring rates and the league's prior-round goal averages."""
    x_n, y_n = x_goals["matches_played"], y_goals["matches_played"]
    x_scored = _ratio(x_goals["goals_scored"], x_n)
    x_conceded = _ratio(x_goals["goals_conceded"], x_n)
    y_scored = _ratio(y_goals["goals_scored"], y_n)
    y_conceded = _ratio(y_goals["goals_conceded"], y_n)
    x_attack = _ratio(x_scored, home_avg)
    x_defense = _ratio(x_conceded, away_avg)
    y_attack = _ratio(y_scored, away_avg)
    y_defense = _ratio(y_conceded, home_avg)
    x_xg = round(x_attack * y_defense * home_avg, 2)
    y_xg = round(y_attack * x_defense * away_avg, 2)
    return {
        "x_attacking_strength": round(x_attack, 2),
        "x_defense_strength": round(x_defense, 2),
        "y_attacking_strength": round(y_attack, 2),
        "y_defense_strength": round(y_defense, 2),
        "x_expected_goals": x_xg,
        "y_expected_goals": y_xg,
        **_odds(x_xg, y_xg),
    }


def _league_round_order(df):
    first = df.groupby(["season", "round"], sort=False, dropna=False)["date"].min()
    ranks = {}
    for season in df["season"].unique():
        dates = first.loc[season].sort_values(kind="stable")
        ranks[season] = {label: rank for rank, label in enumerate(dates.index)}
    return ranks


def _legacy_columns(record, snapshots, models, home_avg, away_avg):
    """Keep the training dataset's existing H2H column names available."""
    record["h2h_previous_meetings"] = snapshots[("all_time", "overall", "x")][0]["matches_played"]
    record["home_league_avg_goals"], record["away_league_avg_goals"] = home_avg, away_avg
    for scope, suffix in (("overall", ""), *((f"last_{n}", f"_last{n}") for n in WINDOWS)):
        for side in ("x", "y"):
            goals, points, _, _ = snapshots[("all_time", scope, side)]
            record[f"{side}_team_matches_played{suffix or '_before'}"] = points["matches_played"]
            record[f"{side}_team_points{suffix or '_before'}"] = points["points"]
            record[f"{side}_team_points_per_match{suffix or '_before'}"] = points["points_per_match"]
            for metric in ("matches_played", "goals_scored", "goals_conceded",
                           "goals_scored_per_match", "goals_conceded_per_match"):
                record[f"{side}_team_{metric}{suffix}"] = goals[metric]
        xg = snapshots[("all_time", scope, "x")][0]
        yg = snapshots[("all_time", scope, "y")][0]
        xp = snapshots[("all_time", scope, "x")][1]
        yp = snapshots[("all_time", scope, "y")][1]
        for metric, buckets in (("points", points_bucket), ("points_per_match", points_per_match_bucket)):
            name = "points_per_match_diff_bucket" if metric == "points_per_match" else "points_diff_bucket"
            record[f"{name}{suffix}"] = get_bucket(buckets, xp[metric] - yp[metric])
        for metric in ("goals_scored", "goals_conceded", "goals_scored_per_match", "goals_conceded_per_match"):
            name = metric
            if not suffix and metric.endswith("_per_match"):
                name = metric.replace("_per_match", "_pm")
            bucket = goals_per_match_bucket if metric.endswith("_per_match") else goals_bucket
            record[f"{name}_diff_bucket{suffix}"] = get_bucket(bucket, xg[metric] - yg[metric])
        # Legacy names expose the all-time scope rather than league-wide metrics.
        model = models[("all_time", scope)]
        if suffix:
            for side in ("x", "y"):
                record[f"{side}_team_as{suffix}"] = model[f"{side}_attacking_strength"]
                record[f"{side}_team_ds{suffix}"] = model[f"{side}_defense_strength"]
                record[f"{side}_team_xg{suffix}"] = model[f"{side}_expected_goals"]
            record[f"x_league_avg_goals{suffix}"] = home_avg
            record[f"y_league_avg_goals{suffix}"] = away_avg
        else:
            for side, venue in (("x", "home"), ("y", "away")):
                record[f"{side}_team_{venue}_attacking_strength"] = model[f"{side}_attacking_strength"]
                record[f"{side}_team_{venue}_defense_strength"] = model[f"{side}_defense_strength"]
                record[f"{side}_team_{venue}_expected_goals"] = model[f"{side}_expected_goals"]
        for old, new in (
            ("x_team_win_probability", "home_win_probability"),
            ("draw_probability", "draw_probability"),
            ("y_team_win_probabilty", "away_win_probability"),
            ("x_team_cs", "home_correct_score"), ("y_team_cs", "away_correct_score"),
            ("home_win_odds", "home_win_odds"), ("draw_odds", "draw_odds"),
            ("away_win_odds", "away_win_odds"),
        ):
            record[f"{old}{suffix}"] = model[new]


def build_h2h_dataset(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Build all season and all-time matchup features in kickoff order."""
    required = {"fixture_id", "date", "round", "season", "league_id", "home_team", "away_team",
                "status", "home_goals", "away_goals"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"fixtures are missing required columns: {sorted(missing)}")
    df = fixtures.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df[["fixture_id", "date", "round", "season", "league_id"]].isna().any().any():
        raise ValueError("fixture IDs, dates, rounds, seasons and league IDs cannot be missing")
    if df["fixture_id"].duplicated().any():
        raise ValueError("fixture IDs must be unique")
    if df["league_id"].astype(str).nunique() > 1:
        raise ValueError("head-to-head data must belong to one league")
    if df.empty:
        raise ValueError("head-to-head fixtures cannot be empty")
    df["season"] = df["season"].astype(str)
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    ranks = _league_round_order(df)
    round_totals = defaultdict(lambda: [0, 0, 0])
    all_states = {}
    season_states = {}
    records = []

    for date, group in groupby(df.itertuples(index=False), key=lambda row: row.date):
        pending = []
        for row in group:
            home, away = row.home_team, row.away_team
            if not isinstance(home, str) or not home.strip() or not isinstance(away, str) or not away.strip() or home == away:
                raise ValueError("fixture teams must be non-empty and different")
            pair = tuple(sorted((home, away)))
            if pair not in all_states:
                all_states[pair] = _new_teams(SCOPES)
            season_key = (row.season, pair)
            if season_key not in season_states:
                season_states[season_key] = _new_teams(SCOPES[:3])
            all_state = all_states[pair]
            season_state = season_states[season_key]
            side_index = {"x": pair.index(home), "y": pair.index(away)}
            rank = ranks[row.season][row.round]
            home_total, away_total, league_count = round_totals[(row.season, rank - 1)]
            home_avg = _ratio(home_total, league_count)
            away_avg = _ratio(away_total, league_count)
            record = {"h2h_home_league_avg_goals_before": round(home_avg, 2),
                      "h2h_away_league_avg_goals_before": round(away_avg, 2)}
            snapshots = {}
            models = {}
            for span, state, scopes in (("all_time", all_state, SCOPES), ("season", season_state, SCOPES[:3])):
                for scope in scopes:
                    for side in ("x", "y"):
                        snapshot = state[side_index[side]][scope].snapshot()
                        snapshots[(span, scope, side)] = snapshot
                        goals, points, outcomes, rating = snapshot
                        prefix = f"h2h_{span}_{scope}_{side}_team_"
                        for metric, value in goals.items():
                            record[prefix + metric] = value
                        for metric, value in points.items():
                            if metric != "matches_played":
                                record[prefix + metric] = value
                        for metric, value in outcomes.items():
                            record[prefix + metric] = round(value, 2) if isinstance(value, float) else value
                        record[prefix + "power_rating"] = rating
                    x = snapshots[(span, scope, "x")]
                    y = snapshots[(span, scope, "y")]
                    record[f"h2h_{span}_{scope}_match_power_rating"] = x[3] - y[3]
                    record[f"h2h_{span}_{scope}_expected_goal_margin"] = x[3] - y[3] + HOME_ADVANTAGE
                    combined = {
                        "home_win": (x[2]["win_probability"] + y[2]["loss_probability"]) / 2,
                        "draw": (x[2]["draw_probability"] + y[2]["draw_probability"]) / 2,
                        "away_win": (x[2]["loss_probability"] + y[2]["win_probability"]) / 2,
                    }
                    for outcome, probability in combined.items():
                        record[f"h2h_{span}_{scope}_combined_{outcome}_probability"] = round(probability, 2)
                    record[f"h2h_{span}_{scope}_least_probable_outcome"] = min(combined, key=combined.get)
                    model = _model(x[0], y[0], home_avg, away_avg)
                    for metric, value in model.items():
                        record[f"h2h_{span}_{scope}_{metric}"] = value
                    models[(span, scope)] = model
            _legacy_columns(record, snapshots, models, home_avg, away_avg)
            records.append(record)

            if row.status not in COMPLETED_STATUSES:
                continue
            for goals in (row.home_goals, row.away_goals):
                if (isinstance(goals, bool) or not isinstance(goals, Real)
                        or not math.isfinite(goals) or goals < 0 or goals != int(goals)):
                    raise ValueError("completed fixtures must include finite, non-negative integer goals")
            pending.append((row, rank, all_state, season_state, side_index, snapshots))

        for row, rank, all_state, season_state, side_index, snapshots in pending:
            for span, state, scopes in (("all_time", all_state, SCOPES), ("season", season_state, SCOPES[:3])):
                for scope in scopes:
                    for side, other, scored, conceded, venue in (
                        ("x", "y", row.home_goals, row.away_goals, "home_only"),
                        ("y", "x", row.away_goals, row.home_goals, "away_only"),
                    ):
                        if scope in ("home_only", "away_only") and scope != venue:
                            continue
                        state[side_index[side]][scope].record(
                            date, scored, conceded, snapshots[(span, scope, other)][3],
                            HOME_ADVANTAGE if side == "x" else -HOME_ADVANTAGE,
                            snapshots[(span, scope, side)][3],
                        )
            total = round_totals[(row.season, rank)]
            total[0] += row.home_goals
            total[1] += row.away_goals
            total[2] += 1

    features = pd.DataFrame.from_records(records, index=df.index)
    text_columns = [
        column for column in features
        if column.endswith("_streak_outcome") or column.endswith("_least_probable_outcome")
    ]
    features[text_columns] = features[text_columns].astype("string")
    decimal_columns = [column for column in features if column.endswith("_consistency")]
    features[decimal_columns] = features[decimal_columns].astype(float)
    return pd.concat([df, features], axis=1)


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    """Replace a fixture table and enforce uniqueness."""
    with database_connection(db_path, connection) as con:
        con.register("h2h_output_view", df)
        try:
            con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM h2h_output_view')
            con.execute(f'CREATE UNIQUE INDEX "{table_name}-fixture_id-pk" ON "{table_name}" (fixture_id)')
        finally:
            con.unregister("h2h_output_view")


def process_league_h2h(combined_df: pd.DataFrame, league_id, db_path: str, windows=WINDOWS,
                       force_recompute: bool = False, connection=None) -> None:
    """Rebuild all H2H season tables and the complete league table."""
    if tuple(windows) != WINDOWS:
        raise ValueError("head-to-head windows must be (3, 5, 10)")
    league_id = str(league_id)
    fixtures = combined_df[combined_df["league_id"].astype(str) == league_id].copy()
    if fixtures.empty:
        raise ValueError(f"No fixtures found for league {league_id}")
    result = build_h2h_dataset(fixtures)
    with database_connection(db_path, connection) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            write_output(result, db_path, f"{league_id}-h2h", con)
            for season, rows in result.groupby("season", sort=True):
                write_output(rows, db_path, f"{league_id}-{season}-h2h", con)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
