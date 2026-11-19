import pandas as pd
import duckdb
from pathlib import Path
import logging
from typing import Callable
from core.database import connect_existing_database


COMPLETED_STATUSES = {"FT", "AET", "PEN"}
FEATURE_TABLE_SUFFIXES = (
    "points", "points-l3", "points-l5", "points-l10",
    "goals", "goals-l3", "goals-l5", "goals-l10",
    "xg", "xg-l3", "xg-l5", "xg-l10",
    "odds", "odds-l3", "odds-l5", "odds-l10", "h2h",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_league_details(
    league_id: int,
    leagues_db_path: str | Path = PROJECT_ROOT / "storage/database/leagues.ddb",
) -> tuple[list, str]:

    with connect_existing_database(leagues_db_path) as conn:
        result = conn.execute(
            "SELECT seasons, country_name FROM leagues WHERE league_id = ?",
            [league_id],
        ).fetchone()
    if result is None:
        return [], ""

    seasons, country_name = result
    seasons_sorted = sorted(seasons)
    return seasons_sorted, country_name

def join_tables(db_path: str, league_id: int, season: int)->pd.DataFrame:
    league_id = int(league_id)
    season = int(season)
    fixture_table = f"{league_id}-{season}-fixtures"
    expected_tables = [f"{league_id}-{season}-{suffix}" for suffix in FEATURE_TABLE_SUFFIXES]

    with connect_existing_database(db_path) as conn:
        fixture_count = conn.execute(
            f'SELECT count(*) FROM "{fixture_table}"'
        ).fetchone()[0]
        for table_name in expected_tables:
            row_count, unique_fixtures = conn.execute(
                f'''SELECT count(*), count(DISTINCT fixture_id)
                    FROM "{table_name}"'''
            ).fetchone()
            covered_fixtures = conn.execute(
                f'''SELECT count(*)
                    FROM "{fixture_table}" AS f
                    INNER JOIN "{table_name}" AS feature USING (fixture_id)'''
            ).fetchone()[0]
            if row_count != unique_fixtures:
                raise ValueError(f"{table_name!r} contains duplicate fixture IDs")
            if row_count != fixture_count:
                raise ValueError(
                    f"{table_name!r} contains {row_count} rows for "
                    f"{fixture_count} fixtures"
                )
            if covered_fixtures != fixture_count:
                raise ValueError(
                    f"{table_name!r} covers {covered_fixtures} of "
                    f"{fixture_count} fixtures"
                )

        joined_df = conn.execute(f"""
                            SELECT 
                            f.league_id,
                            f.fixture_id,
                            f.date,
                            f.venue,
                            f.round,
                            f.home_team,
                            f.away_team,
                            f.home_goals,
                            f.away_goals,
                            CASE
                                WHEN f.status IN ('FT', 'AET', 'PEN')
                                     AND f.home_goals IS NOT NULL
                                     AND f.away_goals IS NOT NULL
                                     AND f.home_goals > f.away_goals THEN 'H'
                                WHEN f.status IN ('FT', 'AET', 'PEN')
                                     AND f.home_goals IS NOT NULL
                                     AND f.away_goals IS NOT NULL
                                     AND f.away_goals > f.home_goals THEN 'A'
                                WHEN f.status IN ('FT', 'AET', 'PEN')
                                     AND f.home_goals IS NOT NULL
                                     AND f.away_goals IS NOT NULL
                                     AND f.home_goals = f.away_goals THEN 'D'
                                ELSE NULL
                            END AS result,
                            f.status,
                            
                            p.x_team_points_before,
                            p.y_team_points_before,
                            p.points_diff_bucket,
                            p.x_team_matches_played_before,
                            p.y_team_matches_played_before,
                            p.x_team_points_per_match_before,
                            p.y_team_points_per_match_before,
                            p.points_per_match_diff_bucket,
                            
                            pl3.x_team_points_last3,
                            pl3.y_team_points_last3,
                            pl3.points_diff_bucket_last3,
                            pl3.x_team_matches_played_last3,
                            pl3.y_team_matches_played_last3,
                            pl3.x_team_points_per_match_last3,
                            pl3.y_team_points_per_match_last3,
                            pl3.points_per_match_diff_bucket_last3,
                            
                            pl5.x_team_points_last5,
                            pl5.y_team_points_last5,
                            pl5.points_diff_bucket_last5,
                            pl5.x_team_matches_played_last5,
                            pl5.y_team_matches_played_last5,
                            pl5.x_team_points_per_match_last5,
                            pl5.y_team_points_per_match_last5,
                            pl5.points_per_match_diff_bucket_last5,
                            
                            pl10.x_team_points_last10,
                            pl10.y_team_points_last10,
                            pl10.points_diff_bucket_last10,
                            pl10.x_team_matches_played_last10,
                            pl10.y_team_matches_played_last10,
                            pl10.x_team_points_per_match_last10,
                            pl10.y_team_points_per_match_last10,
                            pl10.points_per_match_diff_bucket_last10,
                            
                            g.x_team_goals_scored,
                            g.y_team_goals_scored,
                            g.goals_scored_diff_bucket,
                            g.x_team_goals_conceded,
                            g.y_team_goals_conceded,
                            g.goals_conceded_diff_bucket,
                            g.x_team_matches_played,
                            g.y_team_matches_played,
                            g.x_team_goals_scored_per_match,
                            g.y_team_goals_scored_per_match,
                            g.goals_scored_pm_diff_bucket,
                            g.x_team_goals_conceded_per_match,
                            g.y_team_goals_conceded_per_match,
                            g.goals_conceded_pm_diff_bucket,
                            
                            gl3.x_team_goals_scored_last3,
                            gl3.y_team_goals_scored_last3,
                            gl3.goals_scored_diff_bucket_last3,
                            gl3.x_team_goals_conceded_last3,
                            gl3.y_team_goals_conceded_last3,
                            gl3.goals_conceded_diff_bucket_last3,
                            gl3.x_team_matches_played_last3,
                            gl3.y_team_matches_played_last3,
                            gl3.x_team_goals_scored_per_match_last3,
                            gl3.y_team_goals_scored_per_match_last3,
                            gl3.goals_scored_per_match_diff_bucket_last3,
                            gl3.x_team_goals_conceded_per_match_last3,
                            gl3.y_team_goals_conceded_per_match_last3,
                            gl3.goals_conceded_per_match_diff_bucket_last3,
                            
                            gl5.x_team_goals_scored_last5,
                            gl5.y_team_goals_scored_last5,
                            gl5.goals_scored_diff_bucket_last5,
                            gl5.x_team_goals_conceded_last5,
                            gl5.y_team_goals_conceded_last5,
                            gl5.goals_conceded_diff_bucket_last5,
                            gl5.x_team_matches_played_last5,
                            gl5.y_team_matches_played_last5,
                            gl5.x_team_goals_scored_per_match_last5,
                            gl5.y_team_goals_scored_per_match_last5,
                            gl5.x_team_goals_conceded_per_match_last5,
                            gl5.y_team_goals_conceded_per_match_last5,
                            gl5.goals_conceded_per_match_diff_bucket_last5,
                            
                            gl10.x_team_goals_scored_last10,
                            gl10.y_team_goals_scored_last10,
                            gl10.goals_scored_diff_bucket_last10,
                            gl10.x_team_goals_conceded_last10,
                            gl10.y_team_goals_conceded_last10,
                            gl10.goals_conceded_diff_bucket_last10,
                            gl10.x_team_matches_played_last10,
                            gl10.y_team_matches_played_last10,
                            gl10.x_team_goals_scored_per_match_last10,
                            gl10.y_team_goals_scored_per_match_last10,
                            gl10.x_team_goals_conceded_per_match_last10,
                            gl10.y_team_goals_conceded_per_match_last10,
                            gl10.goals_conceded_per_match_diff_bucket_last10,
                            
                            xg.x_team_home_attacking_strength AS x_team_as,
                            xg.y_team_away_attacking_strength AS y_team_as,
                            xg.x_team_home_defense_strength AS x_team_ds,
                            xg.y_team_away_defense_strength AS y_team_ds,
                            xg.x_team_home_expected_goals AS x_team_xg,
                            xg.y_team_away_expected_goals AS y_team_xg,
                            xg.home_league_avg_goals AS x_league_avg_goals,
                            xg.away_league_avg_goals AS y_league_avg_goals,
                            
                            xgl3.x_team_as AS x_team_as_last3,
                            xgl3.y_team_as AS y_team_as_last3,
                            xgl3.x_team_ds AS x_team_ds_last3,
                            xgl3.y_team_ds AS y_team_ds_last3,
                            xgl3.x_team_xg AS x_team_xg_last3,
                            xgl3.y_team_xg AS y_team_xg_last3,
                            xgl3.x_league_avg_goals AS x_league_avg_goals_last3,
                            xgl3.y_league_avg_goals AS y_league_avg_goals_last3,
                            
                            xgl5.x_team_as AS x_team_as_last5,
                            xgl5.y_team_as AS y_team_as_last5,
                            xgl5.x_team_ds AS x_team_ds_last5,
                            xgl5.y_team_ds AS y_team_ds_last5,
                            xgl5.x_team_xg AS x_team_xg_last5,
                            xgl5.y_team_xg AS y_team_xg_last5,
                            xgl5.x_league_avg_goals AS x_league_avg_goals_last5,
                            xgl5.y_league_avg_goals AS y_league_avg_goals_last5,
                            
                            xgl10.x_team_as AS x_team_as_last10,
                            xgl10.y_team_as AS y_team_as_last10,
                            xgl10.x_team_ds AS x_team_ds_last10,
                            xgl10.y_team_ds AS y_team_ds_last10,
                            xgl10.x_team_xg AS x_team_xg_last10,
                            xgl10.y_team_xg AS y_team_xg_last10,
                            xgl10.x_league_avg_goals AS x_league_avg_goals_last10,
                            xgl10.y_league_avg_goals AS y_league_avg_goals_last10,
                            
                            o.x_team_win_probability,
                            o.draw_probability,
                            o.y_team_win_probabilty,
                            o.x_team_cs,
                            o.y_team_cs,
                            o.home_win_odds,
                            o.draw_odds,
                            o.away_win_odds,
                            
                            ol3.x_team_win_probability AS x_team_win_probability_l3,
                            ol3.draw_probability AS draw_probability_l3,
                            ol3.y_team_win_probabilty AS y_team_win_probability_l3,
                            ol3.x_team_cs AS x_team_cs_l3,
                            ol3.y_team_cs AS y_team_cs_l3,
                            ol3.home_win_odds AS home_win_odds_l3,
                            ol3.draw_odds AS draw_odds_l3,
                            ol3.away_win_odds AS away_win_odds_l3,
                            
                            ol5.x_team_win_probability AS x_team_win_probability_l5,
                            ol5.draw_probability AS draw_probability_l5,
                            ol5.y_team_win_probabilty AS y_team_win_probability_l5,
                            ol5.x_team_cs AS x_team_cs_l5,
                            ol5.y_team_cs AS y_team_cs_l5,
                            ol5.home_win_odds AS home_win_odds_l5,
                            ol5.draw_odds AS draw_odds_l5,
                            ol5.away_win_odds AS away_win_odds_l5,
                            
                            ol10.x_team_win_probability AS x_team_win_probability_l10,
                            ol10.draw_probability AS draw_probability_l10,
                            ol10.y_team_win_probabilty AS y_team_win_probability_l10,
                            ol10.x_team_cs AS x_team_cs_l10,
                            ol10.y_team_cs AS y_team_cs_l10,
                            ol10.home_win_odds AS home_win_odds_l10,
                            ol10.draw_odds AS draw_odds_l10,
                            ol10.away_win_odds AS away_win_odds_l10,
                            
                            h2h.h2h_previous_meetings,
                            h2h.x_team_points_before AS x_team_points_before_h2h,
                            h2h.y_team_points_before AS y_team_points_before_h2h,
                            h2h.points_diff_bucket AS points_diff_bucket_h2h,
                            h2h.x_team_matches_played_before AS x_team_matches_played_before_h2h,
                            h2h.y_team_matches_played_before AS y_team_matches_played_before_h2h,
                            h2h.x_team_points_per_match_before AS x_team_points_per_match_before_h2h,
                            h2h.y_team_points_per_match_before AS y_team_points_per_match_before_h2h,
                            h2h.points_per_match_diff_bucket AS points_per_match_diff_bucket_h2h,
                            
                            h2h.x_team_points_last3 AS x_team_points_l3_h2h,
                            h2h.y_team_points_last3 AS y_team_points_l3_h2h,
                            h2h.points_diff_bucket_last3 AS points_diff_bucket_l3_h2h,
                            h2h.x_team_matches_played_last3 AS x_team_matches_played_l3_h2h,
                            h2h.y_team_matches_played_last3 AS y_team_matches_played_l3_h2h,
                            h2h.x_team_points_per_match_last3 AS x_team_points_per_match_l3_h2h,
                            h2h.y_team_points_per_match_last3 AS y_team_points_per_match_l3_h2h,
                            h2h.points_per_match_diff_bucket_last3 AS points_per_match_diff_bucket_l3_h2h,
                            
                            h2h.x_team_points_last5 AS x_team_points_l5_h2h,
                            h2h.y_team_points_last5 AS y_team_points_l5_h2h,
                            h2h.points_diff_bucket_last5 AS points_diff_bucket_l5_h2h,
                            h2h.x_team_matches_played_last5 AS x_team_matches_played_l5_h2h,
                            h2h.y_team_matches_played_last5 AS y_team_matches_played_l5_h2h,
                            h2h.x_team_points_per_match_last5 AS x_team_points_per_match_l5_h2h,
                            h2h.y_team_points_per_match_last5 AS y_team_points_per_match_l5_h2h,
                            h2h.points_per_match_diff_bucket_last5 AS points_per_match_diff_bucket_l5_h2h,
                            
                            h2h.x_team_points_last10 AS x_team_points_l10_h2h,
                            h2h.y_team_points_last10 AS y_team_points_l10_h2h,
                            h2h.points_diff_bucket_last10 AS points_diff_bucket_l10_h2h,
                            h2h.x_team_matches_played_last10 AS x_team_matches_played_l10_h2h,
                            h2h.y_team_matches_played_last10 AS y_team_matches_played_l10_h2h,
                            h2h.x_team_points_per_match_last10 AS x_team_points_per_match_l10_h2h,
                            h2h.y_team_points_per_match_last10 AS y_team_points_per_match_l10_h2h,
                            h2h.points_per_match_diff_bucket_last10 AS points_per_match_diff_bucket_l10_h2h,
                            
                            h2h.x_team_goals_scored AS x_team_goals_scored_h2h,
                            h2h.y_team_goals_scored AS y_team_goals_scored_h2h,
                            h2h.goals_scored_diff_bucket AS goals_scored_diff_bucket_h2h,
                            h2h.x_team_goals_conceded AS x_team_goals_conceded_h2h,
                            h2h.y_team_goals_conceded AS y_team_goals_conceded_h2h,
                            h2h.goals_conceded_diff_bucket AS goals_conceded_diff_bucket_h2h,
                            h2h.x_team_matches_played AS x_team_matches_played_h2h,
                            h2h.y_team_matches_played AS y_team_matches_played_h2h,
                            h2h.x_team_goals_scored_per_match AS x_team_goals_scored_per_match_h2h,
                            h2h.y_team_goals_scored_per_match AS y_team_goals_scored_per_match_h2h,
                            h2h.goals_scored_pm_diff_bucket AS goals_scored_pm_diff_bucket_h2h,
                            h2h.x_team_goals_conceded_per_match AS x_team_goals_conceded_per_match_h2h,
                            h2h.y_team_goals_conceded_per_match AS y_team_goals_conceded_per_match_h2h,
                            h2h.goals_conceded_pm_diff_bucket AS goals_conceded_pm_diff_bucket_h2h,
                            
                            h2h.x_team_goals_scored_last3 AS x_team_goals_scored_l3_h2h,
                            h2h.y_team_goals_scored_last3 AS y_team_goals_scored_l3_h2h,
                            h2h.goals_scored_diff_bucket_last3 AS goals_scored_diff_bucket_l3_h2h,
                            h2h.x_team_goals_conceded_last3 AS x_team_goals_conceded_l3_h2h,
                            h2h.y_team_goals_conceded_last3 AS y_team_goals_conceded_l3_h2h,
                            h2h.goals_conceded_diff_bucket_last3 AS goals_conceded_diff_bucket_l3_h2h,
                            h2h.x_team_goals_scored_per_match_last3 AS x_team_goals_scored_per_match_l3_h2h,
                            h2h.y_team_goals_scored_per_match_last3 AS y_team_goals_scored_per_match_l3_h2h,
                            h2h.goals_scored_per_match_diff_bucket_last3 AS goals_scored_per_match_diff_bucket_l3_h2h,
                            h2h.x_team_goals_conceded_per_match_last3 AS x_team_goals_conceded_per_match_l3_h2h,
                            h2h.y_team_goals_conceded_per_match_last3 AS y_team_goals_conceded_per_match_l3_h2h,
                            h2h.goals_conceded_per_match_diff_bucket_last3 AS goals_conceded_per_match_diff_bucket_l3_h2h,
                            
                            h2h.x_team_goals_scored_last5 AS x_team_goals_scored_l5_h2h,
                            h2h.y_team_goals_scored_last5 AS y_team_goals_scored_l5_h2h,
                            h2h.goals_scored_diff_bucket_last5 AS goals_scored_diff_bucket_l5_h2h,
                            h2h.x_team_goals_conceded_last5 AS x_team_goals_conceded_l5_h2h,
                            h2h.y_team_goals_conceded_last5 AS y_team_goals_conceded_l5_h2h,
                            h2h.goals_conceded_diff_bucket_last5 AS goals_conceded_diff_bucket_l5_h2h,
                            h2h.x_team_goals_scored_per_match_last5 AS x_team_goals_scored_per_match_l5_h2h,
                            h2h.y_team_goals_scored_per_match_last5 AS y_team_goals_scored_per_match_l5_h2h,
                            h2h.goals_scored_per_match_diff_bucket_last5 AS goals_scored_per_match_diff_bucket_l5_h2h,
                            h2h.x_team_goals_conceded_per_match_last5 AS x_team_goals_conceded_per_match_l5_h2h,
                            h2h.y_team_goals_conceded_per_match_last5 AS y_team_goals_conceded_per_match_l5_h2h,
                            h2h.goals_conceded_per_match_diff_bucket_last5 AS goals_conceded_per_match_diff_bucket_l5_h2h,
                            
                            h2h.x_team_goals_scored_last10 AS x_team_goals_scored_l10_h2h,
                            h2h.y_team_goals_scored_last10 AS y_team_goals_scored_l10_h2h,
                            h2h.goals_scored_diff_bucket_last10 AS goals_scored_diff_bucket_l10_h2h,
                            h2h.x_team_goals_conceded_last10 AS x_team_goals_conceded_l10_h2h,
                            h2h.y_team_goals_conceded_last10 AS y_team_goals_conceded_l10_h2h,
                            h2h.goals_conceded_diff_bucket_last10 AS goals_conceded_diff_bucket_l10_h2h,
                            h2h.x_team_goals_scored_per_match_last10 AS x_team_goals_scored_per_match_l10_h2h,
                            h2h.y_team_goals_scored_per_match_last10 AS y_team_goals_scored_per_match_l10_h2h,
                            h2h.goals_scored_per_match_diff_bucket_last10 AS goals_scored_per_match_diff_bucket_l10_h2h,
                            h2h.x_team_goals_conceded_per_match_last10 AS x_team_goals_conceded_per_match_l10_h2h,
                            h2h.y_team_goals_conceded_per_match_last10 AS y_team_goals_conceded_per_match_l10_h2h,
                            h2h.goals_conceded_per_match_diff_bucket_last10 AS goals_conceded_per_match_diff_bucket_l10_h2h,
                            
                            h2h.x_team_home_attacking_strength AS x_team_as_h2h,
                            h2h.y_team_away_attacking_strength AS y_team_as_h2h,
                            h2h.x_team_home_defense_strength AS x_team_ds_h2h,
                            h2h.y_team_away_defense_strength AS y_team_ds_h2h,
                            h2h.x_team_home_expected_goals AS x_team_xg_h2h,
                            h2h.y_team_away_expected_goals AS y_team_xg_h2h,
                            h2h.home_league_avg_goals AS x_league_avg_goals_h2h,
                            h2h.away_league_avg_goals AS y_league_avg_goals_h2h,
                            
                            h2h.x_team_as_last3 AS x_team_as_l3_h2h,
                            h2h.y_team_as_last3 AS y_team_as_l3_h2h,
                            h2h.x_team_ds_last3 AS x_team_ds_l3_h2h,
                            h2h.y_team_ds_last3 AS y_team_ds_l3_h2h,
                            h2h.x_team_xg_last3 AS x_team_xg_l3_h2h,
                            h2h.y_team_xg_last3 AS y_team_xg_l3_h2h,
                            h2h.x_league_avg_goals_last3 AS x_league_avg_goals_l3_h2h,
                            h2h.y_league_avg_goals_last3 AS y_league_avg_goals_l3_h2h,
                            
                            h2h.x_team_as_last5 AS x_team_as_l5_h2h,
                            h2h.y_team_as_last5 AS y_team_as_l5_h2h,
                            h2h.x_team_ds_last5 AS x_team_ds_l5_h2h,
                            h2h.y_team_ds_last5 AS y_team_ds_l5_h2h,
                            h2h.x_team_xg_last5 AS x_team_xg_l5_h2h,
                            h2h.y_team_xg_last5 AS y_team_xg_l5_h2h,
                            h2h.x_league_avg_goals_last5 AS x_league_avg_goals_l5_h2h,
                            h2h.y_league_avg_goals_last5 AS y_league_avg_goals_l5_h2h,
                            
                            h2h.x_team_as_last10 AS x_team_as_l10_h2h,
                            h2h.y_team_as_last10 AS y_team_as_l10_h2h,
                            h2h.x_team_ds_last10 AS x_team_ds_l10_h2h,
                            h2h.y_team_ds_last10 AS y_team_ds_l10_h2h,
                            h2h.x_team_xg_last10 AS x_team_xg_l10_h2h,
                            h2h.y_team_xg_last10 AS y_team_xg_l10_h2h,
                            h2h.x_league_avg_goals_last10 AS x_league_avg_goals_l10_h2h,
                            h2h.y_league_avg_goals_last10 AS y_league_avg_goals_l10_h2h,
                            
                            h2h.x_team_win_probability AS x_team_win_probability_h2h,
                            h2h.draw_probability AS draw_probability_h2h,
                            h2h.y_team_win_probabilty AS y_team_win_probability_h2h,
                            h2h.x_team_cs AS x_team_cs_h2h,
                            h2h.y_team_cs AS y_team_cs_h2h,
                            h2h.home_win_odds AS home_win_odds_h2h,
                            h2h.draw_odds AS draw_odds_h2h,
                            h2h.away_win_odds AS away_win_odds_h2h,
                            
                            h2h.x_team_win_probability_last3 AS x_team_win_probability_l3_h2h,
                            h2h.draw_probability_last3 AS draw_probability_l3_h2h,
                            h2h.y_team_win_probabilty_last3 AS y_team_win_probability_l3_h2h,
                            h2h.x_team_cs_last3 AS x_team_cs_l3_h2h,
                            h2h.y_team_cs_last3 AS y_team_cs_l3_h2h,
                            h2h.home_win_odds_last3 AS home_win_odds_l3_h2h,
                            h2h.draw_odds_last3 AS draw_odds_l3_h2h,
                            h2h.away_win_odds_last3 AS away_win_odds_l3_h2h,
                            
                            h2h.x_team_win_probability_last5 AS x_team_win_probability_l5_h2h,
                            h2h.draw_probability_last5 AS draw_probability_l5_h2h,
                            h2h.y_team_win_probabilty_last5 AS y_team_win_probability_l5_h2h,
                            h2h.x_team_cs_last5 AS x_team_cs_l5_h2h,
                            h2h.y_team_cs_last5 AS y_team_cs_l5_h2h,
                            h2h.home_win_odds_last5 AS home_win_odds_l5_h2h,
                            h2h.draw_odds_last5 AS draw_odds_l5_h2h,
                            h2h.away_win_odds_last5 AS away_win_odds_l5_h2h,
                            
                            h2h.x_team_win_probability_last10 AS x_team_win_probability_l10_h2h,
                            h2h.draw_probability_last10 AS draw_probability_l10_h2h,
                            h2h.y_team_win_probabilty_last10 AS y_team_win_probability_l10_h2h,
                            h2h.x_team_cs_last10 AS x_team_cs_l10_h2h,
                            h2h.y_team_cs_last10 AS y_team_cs_l10_h2h,
                            h2h.home_win_odds_last10 AS home_win_odds_l10_h2h,
                            h2h.draw_odds_last10 AS draw_odds_l10_h2h,
                            h2h.away_win_odds_last10 AS away_win_odds_l10_h2h
                            FROM "{fixture_table}" AS f
                            LEFT JOIN "{league_id}-{season}-points" AS p USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-points-l3" AS pl3 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-points-l5" AS pl5 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-points-l10" AS pl10 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-goals" AS g USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-goals-l3" AS gl3 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-goals-l5" AS gl5 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-goals-l10" AS gl10 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-xg" AS xg USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-xg-l3" AS xgl3 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-xg-l5" AS xgl5 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-xg-l10" AS xgl10 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-odds" AS o USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-odds-l3" AS ol3 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-odds-l5" AS ol5 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-odds-l10" AS ol10 USING (fixture_id)
                            LEFT JOIN "{league_id}-{season}-h2h" AS h2h USING (fixture_id)
                            """).df()
    if len(joined_df) != fixture_count:
        raise ValueError(
            f"Feature joins changed fixture count from {fixture_count} "
            f"to {len(joined_df)}"
        )
    if joined_df["fixture_id"].duplicated().any():
        raise ValueError("joined dataset contains duplicate fixture IDs")
    non_feature_columns = {
        "league_id", "fixture_id", "date", "venue", "round", "home_team",
        "away_team", "home_goals", "away_goals", "result", "status",
    }
    feature_columns = [
        column for column in joined_df.columns if column not in non_feature_columns
    ]
    missing_counts = joined_df[feature_columns].isna().sum()
    all_missing = missing_counts[missing_counts == len(joined_df)]
    if not all_missing.empty:
        raise ValueError(
            f"joined dataset contains entirely missing features: "
            f"{all_missing.index.tolist()}"
        )
    partial_missing = missing_counts[missing_counts > 0]
    for column, count in partial_missing.items():
        logger.warning(
            "Feature %s is missing for %d of %d fixtures",
            column,
            count,
            len(joined_df),
        )
    return joined_df

def save_to_db(
    df: pd.DataFrame,
    league_id: int,
    country: str,
    dataset_name: str,
    dataset_db_path: str | Path | None = None,
) -> None:
    dataset_db_path = Path(dataset_db_path) if dataset_db_path else (
        PROJECT_ROOT / "storage" / "database" / "datasets" / f"{country}.ddb"
    )
    dataset_table_name = f"{int(league_id)}-{dataset_name}"
    Path(dataset_db_path).parent.mkdir(parents=True, exist_ok=True)
    if "fixture_id" not in df or df["fixture_id"].duplicated().any():
        raise ValueError("dataset must have unique fixture_id values")
    with duckdb.connect(dataset_db_path) as conn:
        conn.register("df_view", df)
        try:
            conn.execute(f'DROP TABLE IF EXISTS "{dataset_table_name}"')
            conn.execute(
                f'CREATE TABLE "{dataset_table_name}" AS SELECT * FROM df_view'
            )
        finally:
            conn.unregister("df_view")


def prepare_all_seasons_dataset(
    league_id: int,
    leagues_db_path: str | Path = PROJECT_ROOT / "storage/database/leagues.ddb",
    country_db_path: str | Path | None = None,
    dataset_db_path: str | Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Prepare league-wide training data and a prediction table for each season."""
    seasons, country = get_league_details(league_id, leagues_db_path)
    if not country or not seasons:
        raise ValueError("No seasons are available for this league")
    seasons = sorted(set(seasons))
    country_db_path = Path(country_db_path) if country_db_path else (
        PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country}.ddb"
    )
    dataset_db_path = Path(dataset_db_path) if dataset_db_path else (
        PROJECT_ROOT / "storage" / "database" / "datasets" / f"{country}.ddb"
    )
    frames = []
    total = len(seasons) * 2 + 3
    completed = 0

    def report(message: str) -> None:
        nonlocal completed
        completed += 1
        if progress_callback:
            progress_callback(completed, total, message)

    if progress_callback:
        progress_callback(0, total, "Preparing all seasons.")
    for season in seasons:
        frame = join_tables(str(country_db_path), league_id, season)
        frames.append(frame)
        report(f"Joined season {season}.")
    combined = pd.concat(frames, ignore_index=True)
    if "fixture_id" not in combined or combined["fixture_id"].duplicated().any():
        raise ValueError("Fixtures must be unique across all seasons")
    dataset_db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(dataset_db_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for name, frame in [
                (f"{league_id}-training-dataset", combined),
                (f"{league_id}-prediction-dataset", combined),
                *[(f"{league_id}-{season}-prediction-dataset", frame)
                  for season, frame in zip(seasons, frames)],
            ]:
                connection.register("prepared_frame", frame)
                try:
                    connection.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM prepared_frame')
                finally:
                    connection.unregister("prepared_frame")
                report(f"Saved {name}.")
            connection.execute(
                f'CREATE OR REPLACE TABLE "{int(league_id)}-dataset-metadata" '
                "AS SELECT ?::INTEGER AS prediction_season, TRUE AS full_season, "
                "TRUE AS all_seasons, ?::INTEGER[] AS training_seasons",
                [seasons[-1], seasons],
            )
            report("Saved dataset metadata.")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return {
        "training_rows": len(combined),
        "prediction_rows": len(combined),
        "training_seasons": seasons,
        "message": f"Prepared {len(combined)} fixtures across {len(seasons)} seasons.",
    }


def prepare_league_dataset(
    league_id: int,
    prediction_season: int,
    leagues_db_path: str | Path = PROJECT_ROOT / "storage/database/leagues.ddb",
    country_db_path: str | Path | None = None,
    dataset_db_path: str | Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Build training data from prior seasons and prediction data for one season."""
    seasons, country = get_league_details(league_id, leagues_db_path)
    if not country or prediction_season not in seasons:
        raise ValueError("Season is not available for this league")
    if country_db_path is None:
        country_db_path = PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country}.ddb"
    country_db_path = Path(country_db_path)
    prior_seasons = [season for season in seasons if season < prediction_season]
    if not prior_seasons:
        raise ValueError("At least one earlier season is required to prepare training data")

    total_steps = len(prior_seasons) + 4
    completed_steps = 0

    def report_progress(message: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        if progress_callback is not None:
            progress_callback(
                completed_steps,
                total_steps,
                f"{message} ({completed_steps}/{total_steps}).",
            )

    if progress_callback is not None:
        progress_callback(
            0,
            total_steps,
            f"Preparing datasets: 0 of {total_steps} steps completed…",
        )

    training_frames = []
    for season in prior_seasons:
        training_frames.append(join_tables(str(country_db_path), league_id, season))
        report_progress(f"Joined training data for season {season}")
    training_df = pd.concat(training_frames, ignore_index=True)
    prediction_df = join_tables(str(country_db_path), league_id, prediction_season)
    report_progress(f"Prepared all fixtures for season {prediction_season}")

    dataset_db_path = Path(dataset_db_path) if dataset_db_path else (
        PROJECT_ROOT / "storage" / "database" / "datasets" / f"{country}.ddb"
    )
    dataset_db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(dataset_db_path)) as connection:
        connection.execute(f'DROP TABLE IF EXISTS "{int(league_id)}-dataset-metadata"')
    save_to_db(training_df, league_id, country, "training-dataset", dataset_db_path)
    report_progress("Saved training dataset")
    save_to_db(prediction_df, league_id, country, "prediction-dataset", dataset_db_path)
    report_progress("Saved prediction dataset")
    with duckdb.connect(str(dataset_db_path)) as connection:
        connection.execute(
            f'CREATE OR REPLACE TABLE "{int(league_id)}-dataset-metadata" '
            "AS SELECT ?::INTEGER AS prediction_season, TRUE AS full_season",
            [int(prediction_season)],
        )
    report_progress("Saved dataset metadata")
    return {
        "training_rows": len(training_df),
        "prediction_rows": len(prediction_df),
        "training_seasons": prior_seasons,
        "prediction_season": prediction_season,
    }


if __name__ == "__main__":
    league_ids = [39]
    for league_id in league_ids:
        seasons, country = get_league_details(league_id)
        result = prepare_league_dataset(league_id, seasons[-1])
        print(result)