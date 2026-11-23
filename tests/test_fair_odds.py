import unittest

import numpy as np
import pandas as pd

from processing.fair_odds import fair_odds
from processing.fair_odds.dixon_coles import build_score_matrix, outcome_probabilities
from processing.fair_odds.team_form import TeamFormTracker


class DixonColesTests(unittest.TestCase):
    def test_matrix_sums_to_one_and_outcomes_too(self):
        matrix = build_score_matrix(1.6, 1.1)
        self.assertAlmostEqual(matrix.sum(), 1.0)
        self.assertAlmostEqual(sum(outcome_probabilities(matrix).values()), 1.0)

    def test_low_score_correction_with_negative_rho(self):
        base = build_score_matrix(1.4, 1.0, rho=0.0)
        adjusted = build_score_matrix(1.4, 1.0, rho=-0.1)
        self.assertGreater(adjusted[1, 1], base[1, 1])
        self.assertGreater(adjusted[0, 0], base[0, 0])
        self.assertLess(adjusted[1, 0], base[1, 0])
        self.assertLess(adjusted[0, 1], base[0, 1])
        self.assertTrue((adjusted >= 0).all())


class TeamFormTests(unittest.TestCase):
    def test_no_history_is_neutral(self):
        self.assertEqual(TeamFormTracker().form("A", pd.Timestamp("2025-01-01")), (1.0, 1.0))

    def test_recent_matches_weigh_more(self):
        tracker = TeamFormTracker(half_life_days=10)
        now = pd.Timestamp("2025-03-01")
        w = tracker.weights("A", now)
        self.assertEqual(w, [])
        tracker.record("A", now - pd.Timedelta(days=20), 0, 0, 1, 1)
        tracker.record("A", now - pd.Timedelta(days=10), 0, 0, 1, 1)
        tracker.record("A", now, 0, 0, 1, 1)
        np.testing.assert_allclose(tracker.weights("A", now), [0.25, 0.5, 1.0])

    def test_hot_attack_above_one_and_history_is_bounded(self):
        tracker = TeamFormTracker(max_matches=3)
        day = pd.Timestamp("2025-01-01")
        for i in range(5):
            tracker.record("A", day + pd.Timedelta(days=i), 3, 0, 1, 1)
        self.assertEqual(len(tracker.recent_matches("A")), 3)
        attack, defense = tracker.form("A", day + pd.Timedelta(days=6))
        self.assertGreater(attack, 1.0)
        self.assertLess(defense, 1.0)


class FairOddsTests(unittest.TestCase):
    def frame(self):
        rows = [
            (1, "2025-08-01", "A", "B", 3, 0, 1.5, 1.0),
            (2, "2025-08-01", "C", "D", 1, 1, 1.5, 1.0),
            (3, "2025-08-08", "A", "C", 1, 1, 1.5, 1.0),
        ]
        return pd.DataFrame([
            dict(league_id=1, fixture_id=f, round="r", date=pd.Timestamp(d), home_team=h, away_team=a,
                 home_goals=hg, away_goals=ag, status="FT",
                 x_team_home_expected_goals=x, y_team_away_expected_goals=y)
            for f, d, h, a, hg, ag, x, y in rows
        ])

    def test_form_uses_only_earlier_kickoffs(self):
        out = fair_odds.compute_match_probabilities(self.frame()).set_index("fixture_id")
        # Same-kickoff fixtures see no history; the later one does
        self.assertEqual(out.loc[1, "x_team_attack_form"], 1.0)
        self.assertEqual(out.loc[2, "x_team_attack_form"], 1.0)
        self.assertGreater(out.loc[3, "x_team_attack_form"], 1.0)  # A scored 3 vs 1.5 expected
        self.assertLess(out.loc[3, "x_team_adjusted_xg"] + 1e-9, 99)

    def test_probabilities_sum_to_one_and_raw_mode(self):
        out = fair_odds.compute_match_probabilities(self.frame())
        total = out[["x_team_win_probability", "draw_probability", "y_team_win_probabilty"]].sum(axis=1)
        np.testing.assert_allclose(total, 1.0)
        raw = fair_odds.compute_match_probabilities(self.frame(), use_form=False)
        self.assertNotIn("x_team_adjusted_xg", raw.columns)


class RhoEstimationTests(unittest.TestCase):
    def frame(self, results):
        rows = [
            dict(league_id=1, fixture_id=i, round=f"Regular Season - {r}", date=pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * r),
                 home_team="A", away_team="B", home_goals=hg, away_goals=ag, status="FT",
                 x_team_home_expected_goals=1.4, y_team_away_expected_goals=1.1)
            for i, (r, hg, ag) in enumerate(results, start=1)
        ]
        return pd.DataFrame(rows)

    def test_first_round_uses_default_and_no_lookahead(self):
        from processing.fair_odds import rho_estimation
        from processing.fair_odds.dixon_coles import RHO
        table = rho_estimation.build_rho_table(self.frame([(1, 0, 0)] * 10 + [(2, 3, 3)]))
        by_round = table.set_index("fixture_id")
        self.assertEqual(by_round.loc[1, "rho"], RHO)
        self.assertEqual(by_round.loc[1, "rho_matches_used"], 0)
        self.assertEqual(by_round.loc[11, "rho_matches_used"], 10)

    def test_many_one_nil_results_push_rho_positive_and_stored_per_fixture(self):
        from processing.fair_odds import rho_estimation
        rich = [(r, 1, 0) for r in range(1, 40) for _ in range(5)] + [(40, 1, 0)]
        table = rho_estimation.build_rho_table(self.frame(rich))
        self.assertTrue(table["fixture_id"].is_unique)
        self.assertGreater(table.iloc[-1]["rho"], 0)

    def test_odds_use_rho_table(self):
        rho_table = pd.DataFrame({"fixture_id": [1, 2, 3], "rho": [-0.2, -0.2, -0.2]})
        base = fair_odds.compute_match_probabilities(FairOddsTests().frame(), use_form=False)
        with_rho = fair_odds.compute_match_probabilities(FairOddsTests().frame(), use_form=False, rho_table=rho_table)
        self.assertIn("rho", with_rho.columns)
        self.assertFalse(np.allclose(base["draw_probability"], with_rho["draw_probability"]))

    def test_postponed_results_do_not_enter_rho_early(self):
        from processing.fair_odds import rho_estimation
        frame = self.frame([(1, 0, 0), (1, 1, 0), (2, 1, 1), (2, 1, 1)])
        frame.loc[frame["fixture_id"] == 2, "date"] = pd.Timestamp("2025-08-20")
        frame.loc[frame["fixture_id"] == 4, "date"] = pd.Timestamp("2025-08-20")
        out = rho_estimation.build_rho_table(frame).set_index("fixture_id")
        self.assertEqual(out.loc[3, "rho_matches_used"], 1)
        self.assertEqual(out.loc[4, "rho_matches_used"], 1)

if __name__ == "__main__":
    unittest.main()
