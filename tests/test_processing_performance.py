import tempfile
import unittest
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from processing import process_seasons_data as workflow
from processing.core import COMPLETED_STATUSES
from processing.fair_odds import fair_odds
from processing.goals import goal_metrics
from processing.h2h import combine_fixtures, head_to_head
from processing.outcome_probability import outcome_probability
from processing.points import point_metrics
from processing.recent_fair_odds import last_n_fair_odds
from processing.recent_xg import last_n_xg
from processing.xG import expected_goals


def fixtures():
    rows = []
    for season in range(2022, 2025):
        for pair in range(2):
            for match in range(4):
                rows.append({
                    "fixture_id": season * 100 + pair * 10 + match,
                    "date": pd.Timestamp(season, 1 + match // 2, 1),
                    "round": f"Round {match // 2}",
                    "home_team": f"{'A' if match % 2 == 0 else 'B'}{pair}",
                    "away_team": f"{'B' if match % 2 == 0 else 'A'}{pair}",
                    "home_goals": (match + pair) % 4,
                    "away_goals": (season + match) % 3,
                    "status": ("FT", "AET", "PEN", "NS")[match],
                    "league_id": "39",
                    "season": str(season),
                    "source_table": f"39-{season}-fixtures",
                    "venue": "stadium",
                })
    return pd.DataFrame(rows)


def legacy_kickoffs(df):
    for date, kickoff in df.groupby("date", sort=True):
        yield date, kickoff.itertuples(index=False)


def legacy_league_averages(df, window=None):
    if window is not None and window <= 0:
        raise ValueError("window must be a positive integer")
    df = df.copy()
    df["round_number"] = df["round"].map(expected_goals.determine_round_order(df))
    df = df.sort_values(["round_number", "date", "fixture_id"]).reset_index(drop=True)
    df["x_league_avg_goals"] = 0.0
    df["y_league_avg_goals"] = 0.0
    history = deque(maxlen=window)
    for number in sorted(df["round_number"].unique()):
        home, away, count = (sum(row[i] for row in history) for i in range(3))
        mask = df["round_number"] == number
        df.loc[mask, "x_league_avg_goals"] = home / count if count else 0.0
        df.loc[mask, "y_league_avg_goals"] = away / count if count else 0.0
        finished = df[mask & df["status"].isin(COMPLETED_STATUSES)]
        history.append((finished["home_goals"].sum(), finished["away_goals"].sum(), len(finished)))
    return df


def legacy_score_matrix(lam, mu, rho=fair_odds.RHO, max_goals=fair_odds.MAX_GOALS):
    effective_rho = fair_odds.bounded_rho(lam, mu, rho)
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = (fair_odds.poisson_pmf(i, lam) * fair_odds.poisson_pmf(j, mu)
                 * fair_odds.dixon_coles_tau(i, j, lam, mu, effective_rho))
            matrix[i, j] = max(p, 0.0)
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


class ProcessingPerformanceTests(unittest.TestCase):
    def test_round_averages_match_previous_algorithm(self):
        data = fixtures()
        data["round"] = data["season"] + "::" + data["round"]
        data.loc[0, "round"] = None
        for window in (None, 1, 3, 10):
            with self.subTest(window=window):
                assert_frame_equal(
                    expected_goals.add_league_rolling_averages(data, window),
                    legacy_league_averages(data, window),
                    check_dtype=False,
                )

    def test_fixture_loader_can_filter_by_league(self):
        data = fixtures().drop(columns=["season", "source_table"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "country.ddb"
            with duckdb.connect(str(path)) as connection:
                connection.register("fixture_data", data)
                connection.execute('CREATE TABLE "39-2024-fixtures" AS SELECT * FROM fixture_data LIMIT 2')
                connection.execute('CREATE TABLE "40-2024-fixtures" AS SELECT * FROM fixture_data LIMIT 3')
                selected = combine_fixtures.load_all_fixtures(path, connection, league_id=39)
                all_rows = combine_fixtures.load_all_fixtures(path, connection)
                self.assertEqual(len(selected), 2)
                self.assertEqual(set(selected["league_id"]), {"39"})
                self.assertEqual(len(all_rows), 5)

    def test_score_matrices_match_previous_algorithm(self):
        for module in (fair_odds, last_n_fair_odds):
            for lam, mu, rho, maximum in (
                (0, 0, -0.05, 0), (0, 2, -0.05, 10), (1.4, 1.1, -0.05, 10),
                (20, 0.1, -0.5, 10), (3, 4, 0.9, 5),
            ):
                with self.subTest(module=module.__name__, lam=lam, mu=mu, rho=rho):
                    np.testing.assert_array_equal(
                        module.build_score_matrix(lam, mu, rho, maximum),
                        legacy_score_matrix(lam, mu, rho, maximum),
                    )

    def test_h2h_same_kickoff_and_season_scope(self):
        actual = head_to_head.build_h2h_dataset(fixtures()).set_index("fixture_id")
        first = actual.loc[202200]
        second = actual.loc[202201]
        self.assertEqual(first["x_team_points_before"], second["y_team_points_before"])
        self.assertEqual(first["y_team_points_before"], second["x_team_points_before"])
        self.assertEqual(actual.loc[202300, "h2h_season_overall_x_team_points"], 0)
        self.assertGreater(actual.loc[202300, "h2h_all_time_overall_x_team_matches_played"], 0)

    def test_h2h_rebuilds_all_season_and_combined_tables(self):
        data = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "country.ddb"
            with duckdb.connect(str(path)) as connection, patch.object(
                head_to_head, "build_h2h_dataset",
                wraps=head_to_head.build_h2h_dataset,
            ) as calculate:
                head_to_head.process_league_h2h(data, 39, path, connection=connection)
                self.assertEqual(calculate.call_count, 1)
                calculate.reset_mock()
                head_to_head.process_league_h2h(data, 39, path, connection=connection)
                calculate.assert_called_once()
                for season in ("2022", "2023", "2024"):
                    count = connection.execute(
                        f'SELECT COUNT(*) FROM "39-{season}-h2h"'
                    ).fetchone()[0]
                    self.assertEqual(count, 8)
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM "39-h2h"').fetchone()[0], 24)

    def test_workflow_reuses_fixtures_and_has_no_default_delays(self):
        data = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            league_db = Path(directory) / "leagues.ddb"
            country_db = Path(directory) / "country.ddb"
            with duckdb.connect(str(league_db)) as connection:
                connection.execute("CREATE TABLE leagues (league_id INTEGER, country_name VARCHAR, league_name VARCHAR, seasons INTEGER[])")
                connection.execute("INSERT INTO leagues VALUES (39, 'England', 'League', [2024])")
            with duckdb.connect(str(country_db)):
                pass
            with ExitStack() as stack:
                loader = stack.enter_context(patch.object(workflow.combine_fixtures, "load_all_fixtures", return_value=data))
                sleep = stack.enter_context(patch.object(workflow.time, "sleep"))
                for name in (
                    "process_goals_data", "process_points_data", "process_xg_data",
                    "process_rho_data", "process_fair_odds_data", "process_outcome_probability_data", "process_power_ratings_data",
                    "process_recent_goals_data", "process_recent_points_data",
                    "process_recent_xg_data", "process_recent_fair_odds_data",
                ):
                    stack.enter_context(patch.object(workflow, name))
                h2h = stack.enter_context(patch.object(workflow, "process_head_to_head_data"))
                workflow.process_seasons_data(39, [2024], league_db, country_db)
                loader.assert_called_once()
                self.assertIs(h2h.call_args.kwargs["combined"], data)
                sleep.assert_not_called()
                self.assertEqual(loader.call_args.kwargs["league_id"], 39)

    def test_workflow_writes_all_feature_tables(self):
        data = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            league_db = Path(directory) / "leagues.ddb"
            country_db = Path(directory) / "country.ddb"
            with duckdb.connect(str(league_db)) as connection:
                connection.execute("CREATE TABLE leagues (league_id INTEGER, country_name VARCHAR, league_name VARCHAR, seasons INTEGER[])")
                connection.execute("INSERT INTO leagues VALUES (39, 'England', 'League', [2022])")
            with duckdb.connect(str(country_db)) as connection:
                connection.register("fixture_data", data[data["season"] == "2022"].drop(
                    columns=["season", "source_table"]
                ))
                connection.execute('CREATE TABLE "39-2022-fixtures" AS SELECT * FROM fixture_data')
            progress = []
            workflow.process_seasons_data(
                39, [2022], league_db, country_db,
                progress_callback=lambda done, total, message: progress.append((done, total)),
            )
            with duckdb.connect(str(country_db), read_only=True) as connection:
                suffixes = ["goals", "points", "xg", "rho", "odds", "outcome-probability", "power-ratings", "h2h"]
                suffixes += [
                    f"{metric}-l{window}"
                    for metric in ("goals", "points", "xg", "odds")
                    for window in (3, 5, 10)
                ]
                for suffix in suffixes:
                    count = connection.execute(
                        f'SELECT COUNT(*) FROM "39-2022-{suffix}"'
                    ).fetchone()[0]
                    self.assertEqual(count, 8, suffix)
            self.assertEqual(progress[0], (0, 12))
            self.assertEqual(progress[-1], (12, 12))


if __name__ == "__main__":
    unittest.main()
