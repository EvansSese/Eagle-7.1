import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from processing import process_seasons_data
from processing.power_ratings import power_ratings as power


def fixture(number, home="A", away="B", home_goals=1, away_goals=0,
            status="FT", season="2024", league_id="39", date=None):
    return {
        "fixture_id": number,
        "date": pd.Timestamp(date or f"2024-01-{number:02d}"),
        "round": f"Round {number}",
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status,
        "season": season,
        "league_id": league_id,
    }


class PowerRatingsTests(unittest.TestCase):
    def test_published_examples(self):
        self.assertEqual(power._change(11, 9, 5, 0.5), 0.625)
        self.assertEqual(power._change(9, 11, -5, -0.5), -0.625)
        self.assertEqual(power._change(11, 9, 1, 0.5), -0.375)
        self.assertEqual(power._change(9, 11, -1, -0.5), 0.375)

    def test_season_overall_and_opponent_strength(self):
        data = pd.DataFrame([
            fixture(1, home_goals=5),
            fixture(2, home="A", away="C", home_goals=1),
            fixture(3, home="B", away="C", status="NS", home_goals=None, away_goals=None),
        ])
        result = power.add_power_ratings_before_match(data)
        self.assertEqual(result.iloc[0]["overall_match_power_rating_before"], 0)
        self.assertEqual(result.iloc[0]["overall_expected_goal_margin_before"], 0.5)
        self.assertEqual(result.iloc[1]["x_team_overall_power_rating_before"], 11.125)
        self.assertEqual(result.iloc[1]["y_team_overall_power_rating_before"], 10)
        self.assertEqual(result.iloc[2]["x_team_overall_power_rating_before"], 8.875)
        self.assertEqual(result.iloc[2]["y_team_overall_power_rating_before"], 10.15625)
        self.assertEqual(result.iloc[2]["overall_match_power_rating_before"], -1.28125)

    def test_venues_and_windows_replay_same_scope_opponent_ratings(self):
        rows = [
            fixture(1, home="A", away="B", home_goals=2),
            fixture(2, home="C", away="A", home_goals=0, away_goals=1),
            fixture(3, home="A", away="B", home_goals=0, away_goals=0),
            fixture(4, home="B", away="A", home_goals=0, away_goals=1),
            fixture(5, home="A", away="B", status="NS", home_goals=None, away_goals=None),
        ]
        result = power.add_power_ratings_before_match(pd.DataFrame(rows))
        first = result.iloc[1]
        self.assertEqual(first["y_team_home_only_power_rating_before"], 10.375)
        self.assertEqual(first["y_team_away_only_power_rating_before"], 10)
        current = result.iloc[-1]
        self.assertEqual(current["x_team_home_only_matches_before"], 2)
        self.assertEqual(current["x_team_away_only_matches_before"], 2)
        self.assertEqual(current["x_team_last_3_matches_before"], 3)
        self.assertEqual(current["x_team_last_5_matches_before"], 4)
        # A's last three: C away (opponent 10), B home (9.625),
        # B away (its last-3 pre-match rating 10.0078125).
        self.assertAlmostEqual(current["x_team_last_3_power_rating_before"], 10.423828125)
        self.assertNotEqual(current["x_team_last_3_power_rating_before"],
                            current["x_team_last_5_power_rating_before"])
        self.assertNotEqual(current["x_team_home_only_power_rating_before"],
                            current["x_team_away_only_power_rating_before"])
        self.assertEqual(
            current["venue_match_power_rating_before"],
            current["x_team_home_only_power_rating_before"]
            - current["y_team_away_only_power_rating_before"],
        )

    def test_all_window_sizes_are_bounded(self):
        rows = [fixture(i, home_goals=i % 3) for i in range(1, 13)]
        result = power.add_power_ratings_before_match(pd.DataFrame(rows)).iloc[-1]
        self.assertEqual(
            [result[f"x_team_last_{n}_matches_before"] for n in (3, 5, 10)],
            [3, 5, 10],
        )
        self.assertEqual(result["x_team_overall_matches_before"], 11)

    def test_unfinished_same_kickoff_and_season_reset(self):
        rows = [
            fixture(1, status="NS", home_goals=None, away_goals=None),
            fixture(2, home="A", away="C", date="2024-01-02"),
            fixture(3, home="A", away="D", date="2024-01-02"),
            fixture(4, home="A", away="E"),
            fixture(5, season="2025"),
            fixture(6, league_id="40"),
        ]
        result = power.add_power_ratings_before_match(pd.DataFrame(rows))
        self.assertEqual(result.iloc[1]["x_team_overall_power_rating_before"], 10)
        self.assertEqual(result.iloc[2]["x_team_overall_power_rating_before"], 10)
        self.assertEqual(result.iloc[3]["x_team_overall_power_rating_before"], 10.25)
        self.assertEqual(result.iloc[3]["x_team_overall_matches_before"], 2)
        self.assertEqual(result.iloc[4]["x_team_overall_power_rating_before"], 10)
        self.assertEqual(result.iloc[5]["x_team_overall_power_rating_before"], 10)

    def test_warmup_and_no_future_result_leakage(self):
        rows = [fixture(i) for i in range(1, 8)]
        result = power.add_power_ratings_before_match(pd.DataFrame(rows))
        self.assertFalse(result.iloc[4]["both_teams_at_least_5_matches_before"])
        self.assertTrue(result.iloc[5]["both_teams_at_least_5_matches_before"])
        rows[6]["home_goals"] = 20
        changed = power.add_power_ratings_before_match(pd.DataFrame(rows))
        pd.testing.assert_frame_equal(
            result.filter(regex="_before$"), changed.filter(regex="_before$")
        )

    def test_validation_and_empty(self):
        data = pd.DataFrame([fixture(1)])
        self.assertTrue(power.add_power_ratings_before_match(data.iloc[:0]).empty)
        for column, value in (("home_goals", None), ("away_goals", -1),
                              ("home_goals", 1.5), ("home_team", ""),
                              ("date", None), ("fixture_id", None), ("season", None)):
            invalid = data.copy()
            invalid[column] = invalid[column].astype(object)
            invalid.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                power.add_power_ratings_before_match(invalid)
        with self.assertRaises(ValueError):
            power.add_power_ratings_before_match(pd.concat([data, data]))

    def test_workflow_stores_current_season_with_unique_fixture_key(self):
        data = pd.DataFrame([fixture(1, season="2023"), fixture(2), fixture(3)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "country.ddb"
            with duckdb.connect(str(path)) as con:
                process_seasons_data.process_power_ratings_data(
                    39, 2024, "England", fixtures=data, database_path=path, connection=con
                )
                stored = con.execute('SELECT * FROM "39-2024-power-ratings" ORDER BY fixture_id').df()
                self.assertEqual(list(stored["fixture_id"]), [2, 3])
                self.assertEqual(stored.iloc[0]["x_team_overall_power_rating_before"], 10)
                self.assertEqual(stored.iloc[0]["round"], "Round 2")
                with self.assertRaises(duckdb.ConstraintException):
                    con.execute('INSERT INTO "39-2024-power-ratings" SELECT * FROM "39-2024-power-ratings" LIMIT 1')


if __name__ == "__main__":
    unittest.main()
