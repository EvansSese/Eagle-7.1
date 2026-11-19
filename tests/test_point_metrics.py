import unittest

import pandas as pd

from processing.points.point_metrics import build_point_features


def sample_fixtures() -> pd.DataFrame:
    rows = [
        (1, "2024-01-01", "A", "B", 3, 0),
        (2, "2024-01-08", "B", "A", 1, 1),
        (3, "2024-01-15", "A", "B", 0, 2),
        (4, "2024-01-22", "A", "B", None, None),
    ]
    df = pd.DataFrame(rows, columns=["fixture_id", "date", "home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    df["status"] = ["FT", "FT", "FT", "NS"]
    return df


class PointMetricsTests(unittest.TestCase):
    def test_cumulative_metrics(self):
        out = build_point_features(sample_fixtures()).set_index("fixture_id")
        self.assertEqual(out.loc[1, "x_team_points_before"], 0)
        self.assertEqual(out.loc[1, "x_team_points_per_match_before"], 0.0)
        last = out.loc[4]
        # A: W (home), D (away), L (home) -> 4 points; B: L, D, W -> 4 points
        self.assertEqual(last["x_team_points_before"], 4)
        self.assertEqual(last["x_team_points_per_match_before"], 1.33)
        self.assertEqual(last["x_team_home_points_before"], 3)
        self.assertEqual(last["x_team_home_matches_played_before"], 2)
        self.assertEqual(last["x_team_away_points_before"], 1)
        self.assertEqual(last["x_team_freq_3_points_before"], 1)
        self.assertEqual(last["x_team_freq_1_point_before"], 1)
        self.assertEqual(last["x_team_freq_0_points_before"], 1)
        self.assertEqual(last["y_team_away_freq_3_points_before"], 1)
        self.assertEqual(last["y_team_home_freq_1_point_before"], 1)
        self.assertEqual(last["points_diff_before"], 0)

    def test_window_metrics_use_only_recent_matches(self):
        last = build_point_features(sample_fixtures(), window=2).set_index("fixture_id").loc[4]
        self.assertEqual(last["x_team_matches_played_last2"], 2)
        self.assertEqual(last["x_team_points_last2"], 1)
        self.assertEqual(last["x_team_home_points_last2"], 3)
        self.assertEqual(last["x_team_freq_3_points_last2"], 0)

    def test_history_seeds_totals(self):
        data = sample_fixtures()
        seeded = build_point_features(data.iloc[[2, 3]], history=data.iloc[[0, 1]]).set_index("fixture_id")
        full = build_point_features(data).set_index("fixture_id")
        self.assertEqual(seeded.loc[4, "x_team_points_before"], full.loc[4, "x_team_points_before"])


if __name__ == "__main__":
    unittest.main()
