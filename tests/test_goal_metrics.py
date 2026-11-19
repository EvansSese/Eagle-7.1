import unittest

import pandas as pd

from processing.goals.goal_metrics import build_goal_features


def sample_fixtures() -> pd.DataFrame:
    rows = [
        (1, "2024-01-01", "A", "B", 3, 0),
        (2, "2024-01-08", "B", "A", 1, 1),
        (3, "2024-01-15", "A", "B", 5, 2),
        (4, "2024-01-22", "A", "B", None, None),
    ]
    df = pd.DataFrame(rows, columns=["fixture_id", "date", "home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    df["status"] = ["FT", "FT", "FT", "NS"]
    return df


class GoalMetricsTests(unittest.TestCase):
    def test_cumulative_metrics(self):
        out = build_goal_features(sample_fixtures()).set_index("fixture_id")
        self.assertEqual(out.loc[1, "x_team_goals_scored"], 0)
        self.assertEqual(out.loc[1, "x_team_goals_scored_per_match"], 0.0)
        last = out.loc[4]
        self.assertEqual(last["x_team_goals_scored"], 9)
        self.assertEqual(last["x_team_goals_conceded"], 3)
        self.assertEqual(last["x_team_goals_scored_per_match"], 3.0)
        self.assertEqual(last["x_team_home_matches_played"], 2)
        self.assertEqual(last["x_team_away_matches_played"], 1)
        self.assertEqual(last["x_team_away_goals_scored"], 1)
        self.assertEqual(last["x_team_scored_1_goal"], 1)
        self.assertEqual(last["x_team_scored_3_goals"], 1)
        self.assertEqual(last["x_team_scored_above_3_goals"], 1)
        self.assertEqual(last["x_team_clean_sheets"], 1)
        self.assertEqual(last["y_team_conceded_above_3_goals"], 1)
        self.assertEqual(last["y_team_goals_conceded_per_match"], 3.0)

    def test_window_metrics_use_only_recent_matches(self):
        out = build_goal_features(sample_fixtures(), window=2).set_index("fixture_id")
        last = out.loc[4]
        self.assertEqual(last["x_team_matches_played_last2"], 2)
        self.assertEqual(last["x_team_goals_scored_last2"], 6)
        self.assertEqual(last["x_team_home_goals_scored_last2"], 8)
        self.assertEqual(last["x_team_clean_sheets_last2"], 0)

    def test_history_seeds_totals(self):
        data = sample_fixtures()
        seeded = build_goal_features(data.iloc[[2, 3]], history=data.iloc[[0, 1]]).set_index("fixture_id")
        full = build_goal_features(data).set_index("fixture_id")
        self.assertEqual(seeded.loc[4, "x_team_goals_scored"], full.loc[4, "x_team_goals_scored"])


if __name__ == "__main__":
    unittest.main()
