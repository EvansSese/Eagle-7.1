import unittest

import pandas as pd

from processing.xG.expected_goals import build_expected_goals_table


def fixtures():
    rows = [
        (1, 1, "2025-08-01", "A", "B", 2, 0),
        (2, 1, "2025-08-01", "C", "D", 1, 1),
        (3, 2, "2025-08-08", "A", "D", 1, 1),
        (4, 2, "2025-08-08", "C", "B", 0, 3),
    ]
    return pd.DataFrame(
        [
            dict(league_id=1, fixture_id=f, round=f"Regular Season - {r}", date=d,
                 home_team=h, away_team=a, home_goals=hg, away_goals=ag, status="FT")
            for f, r, d, h, a, hg, ag in rows
        ]
    )


class ExpectedGoalsTests(unittest.TestCase):
    def setUp(self):
        out = build_expected_goals_table(fixtures())
        self.out = out.set_index("fixture_id")

    def test_round_one_is_zero(self):
        for fid in (1, 2):
            row = self.out.loc[fid]
            self.assertEqual(row["x_team_home_expected_goals"], 0)
            self.assertEqual(row["y_team_away_expected_goals"], 0)

    def test_round_two_formula(self):
        # Round 1: home goals 3 / 2 matches = 1.5, away goals 1 / 2 = 0.5
        row = self.out.loc[3]  # A (home, scored 2 at home) vs D (away, scored 1, conceded 1)
        self.assertEqual(row["home_league_avg_goals"], 1.5)
        self.assertEqual(row["away_league_avg_goals"], 0.5)
        att_x = 2 / 1.5
        def_y = 1 / 1.5
        att_y = 1 / 0.5
        def_x = 0 / 0.5
        self.assertAlmostEqual(row["x_team_home_attacking_strength"], round(att_x, 2))
        self.assertAlmostEqual(row["y_team_away_defense_strength"], round(def_y, 2))
        self.assertAlmostEqual(row["x_team_home_expected_goals"], round(att_x * def_y * 1.5, 2))
        self.assertAlmostEqual(row["y_team_away_expected_goals"], round(att_y * def_x * 0.5, 2))

    def test_league_average_uses_only_previous_round(self):
        df = fixtures()
        df["date"] = pd.to_datetime(df["date"])
        extra = df.iloc[[0]].copy()
        extra["fixture_id"] = 5
        extra["round"] = "Regular Season - 3"
        extra["date"] = pd.Timestamp("2025-08-15")
        extra["home_team"], extra["away_team"] = "B", "C"
        extra["home_goals"], extra["away_goals"] = 0, 0
        out = build_expected_goals_table(pd.concat([df, extra])).set_index("fixture_id")
        # Round 2: home goals 1 + 0, away goals 1 + 3 over 2 matches
        self.assertEqual(out.loc[5, "home_league_avg_goals"], 0.5)
        self.assertEqual(out.loc[5, "away_league_avg_goals"], 2.0)

    def test_unique_fixture_ids(self):
        self.assertTrue(self.out.index.is_unique)


if __name__ == "__main__":
    unittest.main()
