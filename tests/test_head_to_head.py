import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from processing.h2h import head_to_head as h2h


def fixture(number, home, away, hg, ag, season="2024", status="FT", date=None, league="39"):
    return {
        "fixture_id": number, "date": pd.Timestamp(date or f"2024-01-{number:02d}"),
        "round": f"R{number}", "season": season, "league_id": league,
        "home_team": home, "away_team": away, "home_goals": hg,
        "away_goals": ag, "status": status,
    }


class HeadToHeadTests(unittest.TestCase):
    def test_only_meetings_contribute_and_season_resets(self):
        rows = [
            fixture(1, "A", "B", 2, 0, season="2023"),
            fixture(2, "B", "A", 0, 1, season="2023"),
            fixture(3, "A", "C", 10, 0),
            fixture(4, "A", "B", 1, 0),
            fixture(5, "B", "A", 0, 2),
            fixture(6, "A", "B", 0, 0),
            fixture(7, "B", "A", None, None, status="NS"),
        ]
        result = h2h.build_h2h_dataset(pd.DataFrame(rows)).set_index("fixture_id")
        current = result.loc[7]
        self.assertEqual(current["h2h_all_time_overall_y_team_points"], 13)
        self.assertEqual(current["h2h_all_time_overall_x_team_points"], 1)
        self.assertEqual(current["h2h_season_overall_y_team_points"], 7)
        self.assertEqual(current["h2h_all_time_overall_y_team_goals_scored"], 6)
        self.assertEqual(current["h2h_all_time_last_3_y_team_points"], 7)
        self.assertEqual(current["h2h_all_time_home_only_y_team_points"], 7)
        self.assertEqual(current["h2h_all_time_away_only_y_team_points"], 6)
        self.assertEqual(current["h2h_all_time_overall_y_team_streak_outcome"], "draw")
        self.assertGreater(current["h2h_all_time_overall_y_team_win_probability"],
                           current["h2h_all_time_overall_y_team_loss_probability"])
        self.assertGreater(current["h2h_all_time_overall_combined_away_win_probability"],
                           current["h2h_all_time_overall_combined_home_win_probability"])
        self.assertNotEqual(current["h2h_all_time_overall_y_team_power_rating"], 10)
        self.assertEqual(result.loc[4, "h2h_season_overall_x_team_matches_played"], 0)
        self.assertEqual(result.loc[4, "h2h_all_time_overall_x_team_matches_played"], 2)

    def test_tied_kickoff_and_scheduled_results_no_leakage(self):
        rows = [
            fixture(1, "A", "B", 1, 0),
            fixture(2, "A", "B", 2, 0, date="2024-01-02"),
            fixture(3, "B", "A", 0, 2, date="2024-01-02"),
            fixture(4, "A", "B", None, None, status="NS"),
            fixture(5, "A", "B", None, None, status="NS"),
        ]
        result = h2h.build_h2h_dataset(pd.DataFrame(rows)).set_index("fixture_id")
        self.assertEqual(result.loc[2, "h2h_previous_meetings"], 1)
        self.assertEqual(result.loc[3, "h2h_previous_meetings"], 1)
        self.assertEqual(result.loc[4, "h2h_previous_meetings"], 3)
        self.assertEqual(result.loc[5, "h2h_previous_meetings"], 3)
        self.assertEqual(result.loc[4, "h2h_all_time_overall_x_team_power_rating"], 11.0)

    def test_windows_and_league_baseline(self):
        rows = [fixture(i, "A", "B", 1, 0, season="2023") for i in range(1, 12)]
        rows += [fixture(12, "C", "D", 4, 2, season="2024"),
                 fixture(13, "A", "B", None, None, status="NS", season="2024")]
        result = h2h.build_h2h_dataset(pd.DataFrame(rows)).set_index("fixture_id")
        row = result.loc[13]
        self.assertEqual([row[f"h2h_all_time_last_{n}_x_team_matches_played"] for n in (3, 5, 10)], [3, 5, 10])
        self.assertEqual(row["h2h_home_league_avg_goals_before"], 4)
        self.assertEqual(row["h2h_away_league_avg_goals_before"], 2)
        self.assertEqual(row["h2h_season_overall_x_team_matches_played"], 0)
        self.assertEqual(row["h2h_all_time_overall_x_expected_goals"], 0.25)

    def test_league_baseline_excludes_future_and_simultaneous_results(self):
        rows = [
            fixture(1, "C", "D", 2, 1, date="2024-01-01"),
            fixture(2, "A", "B", None, None, status="NS", date="2024-01-02"),
            fixture(3, "E", "F", 10, 9, date="2024-01-02"),
            fixture(4, "G", "H", 20, 19, date="2024-01-03"),
        ]
        rows[2]["round"] = rows[0]["round"]
        rows[3]["round"] = rows[0]["round"]
        result = h2h.build_h2h_dataset(pd.DataFrame(rows)).set_index("fixture_id")
        self.assertEqual(result.loc[2, "h2h_home_league_avg_goals_before"], 2)
        self.assertEqual(result.loc[2, "h2h_away_league_avg_goals_before"], 1)

    def test_tables_rebuilt_and_validate(self):
        data = pd.DataFrame([fixture(1, "A", "B", 1, 0), fixture(2, "B", "A", 2, 0, season="2025")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "country.ddb"
            with duckdb.connect(str(path)) as con:
                h2h.process_league_h2h(data, 39, path, connection=con)
                h2h.process_league_h2h(data, 39, path, connection=con)
                self.assertEqual(con.execute('SELECT COUNT(*) FROM "39-h2h"').fetchone()[0], 2)
                self.assertEqual(con.execute('SELECT COUNT(*) FROM "39-2025-h2h"').fetchone()[0], 1)
                self.assertEqual(con.execute('SELECT h2h_previous_meetings FROM "39-2025-h2h"').fetchone()[0], 1)
                schema = {row[0]: row[1] for row in con.execute('DESCRIBE "39-2025-h2h"').fetchall()}
                self.assertEqual(schema["h2h_season_overall_x_team_consistency"], "DOUBLE")
                self.assertEqual(schema["h2h_season_overall_x_team_streak_outcome"], "VARCHAR")
                with self.assertRaises(duckdb.ConstraintException):
                    con.execute('INSERT INTO "39-2025-h2h" SELECT * FROM "39-2025-h2h"')
        invalid = data.copy()
        invalid.loc[0, "home_goals"] = -1
        with self.assertRaises(ValueError):
            h2h.build_h2h_dataset(invalid)


if __name__ == "__main__":
    unittest.main()
