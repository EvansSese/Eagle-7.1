import unittest

import pandas as pd

from training.market_targets import build_market_targets, derived_market_probabilities


class MarketTargetsTests(unittest.TestCase):
    def test_targets_and_thresholds(self):
        fixtures = pd.DataFrame(
            {"home_goals": [0, 1, 2, 1, 4], "away_goals": [0, 0, 0, 2, 0]},
            index=[10, 11, 12, 13, 14],
        )
        targets = build_market_targets(fixtures)
        self.assertEqual(targets["result"].tolist(), ["D", "H", "H", "A", "H"])
        self.assertEqual(targets["btts"].tolist(), [0, 0, 0, 1, 0])
        self.assertEqual(targets["over_0_5"].tolist(), [0, 1, 1, 1, 1])
        self.assertEqual(targets["over_1_5"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(targets["over_2_5"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(targets["over_3_5"].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(targets["correct_score"].tolist(), ["0-0", "1-0", "2-0", "1-2", "4-0"])
        self.assertEqual(targets.index.tolist(), fixtures.index.tolist())

    def test_invalid_scores_fail(self):
        for value in (None, -1, 0.5, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_market_targets(pd.DataFrame({"home_goals": [value], "away_goals": [0]}))

    def test_derived_probabilities_and_validation(self):
        result = pd.DataFrame({"H": [0.5], "D": [0.3], "A": [0.2]})
        over = pd.DataFrame({"over_0_5": [0.9], "over_1_5": [0.8],
                             "over_2_5": [0.6], "over_3_5": [0.4]})
        derived = derived_market_probabilities(result, over)
        self.assertAlmostEqual(derived.loc[0, "1X"], 0.8)
        self.assertAlmostEqual(derived.loc[0, "X2"], 0.5)
        self.assertAlmostEqual(derived.loc[0, "12"], 0.7)
        self.assertAlmostEqual(derived.loc[0, "under_2_5"], 0.4)
        result.loc[0, "H"] = 0.6
        with self.assertRaises(ValueError):
            derived_market_probabilities(result, over)


if __name__ == "__main__":
    unittest.main()
