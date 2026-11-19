import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from processing import process_seasons_data
from processing.outcome_probability import outcome_probability as op
from processing.outcome_probability.outcome_probability_analysis import analyze_outcome_probabilities


def fixture(number, home="A", away="B", score=(1, 0), season="2024",
            round_label=None, status="FT", date=None):
    return {
        "fixture_id": number,
        "date": pd.Timestamp(date or f"2024-01-{number:02d}"),
        "round": round_label or f"Round {number}",
        "home_team": home,
        "away_team": away,
        "home_goals": score[0],
        "away_goals": score[1],
        "status": status,
        "season": season,
    }


class OutcomeProbabilityTests(unittest.TestCase):
    def test_scopes_and_consistency_use_completed_previous_results(self):
        rows = [
            fixture(1, score=(2, 0)),
            fixture(2, home="C", away="A", score=(0, 0)),
            fixture(3, score=(0, 1)),
            fixture(4, score=(3, 0), status="NS"),
            fixture(5),
        ]
        result = op.add_outcome_probabilities_before_match(pd.DataFrame(rows))
        current = result.iloc[-1]
        self.assertEqual(current["x_team_overall_matches_before"], 3)
        self.assertEqual(current["x_team_home_only_matches_before"], 2)
        self.assertEqual(current["x_team_away_only_matches_before"], 1)
        self.assertEqual(current["x_team_last_3_matches_before"], 3)
        self.assertEqual(current["x_team_overall_consistency_before"], 0)
        self.assertEqual(current["x_team_home_only_streak_outcome_before"], "loss")
        self.assertEqual(current["x_team_away_only_streak_outcome_before"], "draw")
        self.assertTrue(pd.isna(current["x_team_away_only_consistency_before"]))
        self.assertEqual(current["x_team_overall_win_probability_before"], 0.33)
        self.assertEqual(current["x_team_win_probability_before"], 0.33)

    def test_form_windows_and_strength(self):
        rows = [fixture(i, score=(1, 0)) for i in range(1, 13)]
        result = op.add_outcome_probabilities_before_match(pd.DataFrame(rows))
        last = result.iloc[-1]
        self.assertEqual([last[f"x_team_last_{n}_matches_before"] for n in (3, 5, 10)], [3, 5, 10])
        self.assertEqual(last["x_team_overall_streak_length_before"], 11)
        self.assertEqual(last["x_team_overall_consistency_before"], 1.0)
        self.assertGreater(last["x_team_win_probability_before"], last["x_team_loss_probability_before"])
        self.assertLess(last["y_team_win_probability_before"], last["y_team_loss_probability_before"])

    def test_streak_adjustment_requires_historical_evidence(self):
        no_evidence = op._metrics(["win"] * 6)
        self.assertEqual(no_evidence["streak_observations"], 0)
        results = ["win"] * 6 + ["loss"] + ["win"] * 6
        learned = op._metrics(results)
        self.assertEqual(learned["streak_observations"], 1)
        self.assertEqual(learned["streak_length"], 6)
        baseline_loss = (1 + 1) / (len(results) + op.PRIOR_MATCHES)
        self.assertGreater(learned["loss_probability"], baseline_loss)
        self.assertAlmostEqual(sum(learned[f"{o}_probability"] for o in op.OUTCOMES), 1)

    def test_windows_use_only_their_own_recent_results(self):
        rows = [fixture(i, score=(1, 0) if i <= 7 else (0, 1)) for i in range(1, 12)]
        result = op.add_outcome_probabilities_before_match(pd.DataFrame(rows)).iloc[-1]
        self.assertEqual(result["x_team_last_3_streak_length_before"], 3)
        self.assertEqual(result["x_team_last_5_streak_length_before"], 3)
        self.assertEqual(result["x_team_last_10_streak_length_before"], 3)
        self.assertEqual(result["x_team_last_3_win_probability_before"], 0.17)
        self.assertEqual(result["x_team_last_5_win_probability_before"], 0.38)
        self.assertEqual(result["x_team_last_10_win_probability_before"], 0.62)

    def test_first_round_season_reset_and_same_kickoff(self):
        rows = [
            fixture(1, round_label="R1"),
            fixture(2, round_label="R2"),
            fixture(3, home="A", away="C", round_label="R1"),
            fixture(4, home="A", away="D", round_label="R3", date="2024-01-04"),
            fixture(5, home="A", away="E", round_label="R3", date="2024-01-04"),
            fixture(6, season="2025", round_label="R1"),
        ]
        result = op.add_outcome_probabilities_before_match(pd.DataFrame(rows))
        for index in (0, 2, 5):
            for side in ("x", "y"):
                self.assertEqual(
                    [result.iloc[index][f"{side}_team_{o}_probability_before"] for o in op.OUTCOMES],
                    [0.33, 0.33, 0.33],
                )
                self.assertEqual(result.iloc[index][f"{side}_team_overall_matches_before"], 0)
        self.assertEqual(result.iloc[3]["x_team_overall_matches_before"], 3)
        self.assertEqual(result.iloc[4]["x_team_overall_matches_before"], 3)

    def test_input_validation_and_empty_input(self):
        data = pd.DataFrame([fixture(1)])
        self.assertTrue(op.add_outcome_probabilities_before_match(data.iloc[:0]).empty)
        for column, value in (("home_goals", None), ("away_goals", -1),
                              ("home_team", ""), ("date", None), ("season", None)):
            invalid = data.copy()
            invalid.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                op.add_outcome_probabilities_before_match(invalid)

    def test_database_workflow_and_analysis(self):
        data = pd.DataFrame([fixture(1), fixture(2, score=(0, 1)), fixture(3, score=(0, 0))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "country.ddb"
            with duckdb.connect(str(path)) as con:
                process_seasons_data.process_outcome_probability_data(
                    39, 2024, "England", fixtures=data, database_path=path, connection=con
                )
                stored = con.execute('SELECT * FROM "39-2024-outcome-probability" ORDER BY fixture_id').df()
                self.assertEqual(len(stored), 3)
                self.assertEqual(stored.iloc[1]["x_team_overall_matches_before"], 1)
                self.assertTrue(pd.isna(stored.iloc[0]["x_team_overall_consistency_before"]))
                self.assertEqual(analyze_outcome_probabilities(stored)["fixtures_analyzed"], 3)

    def test_null_only_streak_columns_are_stored_as_text(self):
        data = op.add_outcome_probabilities_before_match(pd.DataFrame([fixture(1)]))
        with duckdb.connect() as con:
            op.write_output(data, ":memory:", "outcomes", connection=con)
            schema = {row[0]: row[1] for row in con.execute("DESCRIBE outcomes").fetchall()}
            self.assertEqual(schema["x_team_overall_streak_outcome_before"], "VARCHAR")
            self.assertEqual(schema["x_team_overall_matches_before"], "BIGINT")


if __name__ == "__main__":
    unittest.main()
