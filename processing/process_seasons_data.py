import duckdb
import pandas as pd
import time
from pathlib import Path
from typing import Callable
from core.database import connect_existing_database

try:
    from .points import points
    from .goals import goals
    from .xG import expected_goals
    from .fair_odds import fair_odds, rho_estimation
    from .outcome_probability import outcome_probability
    from .power_ratings import power_ratings
    from .recent_goals import last_n_goals
    from .recent_points import last_n_points
    from .recent_xg import last_n_xg
    from .recent_fair_odds import last_n_fair_odds
    from .h2h import head_to_head, combine_fixtures
except ImportError:
    from points import points
    from goals import goals
    from xG import expected_goals
    from fair_odds import fair_odds, rho_estimation
    from outcome_probability import outcome_probability
    from power_ratings import power_ratings
    from recent_goals import last_n_goals
    from recent_points import last_n_points
    from recent_xg import last_n_xg
    from recent_fair_odds import last_n_fair_odds
    from h2h import head_to_head, combine_fixtures

FEATURE_PROCESSING_DELAY_SECONDS = 0
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def process_seasons_data(
    league_id: int,
    seasons_to_process: list,
    leagues_db_path: str | Path = PROJECT_ROOT / "storage/database/leagues.ddb",
    country_db_path: str | Path | None = None,
    delay_seconds: float = FEATURE_PROCESSING_DELAY_SECONDS,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    with connect_existing_database(leagues_db_path) as league_con:
        league_details = league_con.execute(
            "SELECT country_name, league_name, seasons FROM leagues WHERE league_id = ?",
            [league_id],
        ).fetchone()
    if league_details is None:
        raise ValueError(f"League {league_id} was not found")
    country_name, league_name, available_seasons = league_details
    print(f"Processing seasons data for {country_name} - {league_name} for seasons {available_seasons}")

    seasons = seasons_to_process if seasons_to_process else available_seasons
    total_steps = len(seasons) * 11 + 1
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

    def pause() -> None:
        if delay_seconds != 0:
            time.sleep(delay_seconds)

    if progress_callback is not None:
        progress_callback(
            0,
            total_steps,
            f"Starting feature processing for {len(seasons)} seasons…",
        )

    database_path = Path(country_db_path) if country_db_path else (
        PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    )
    if not database_path.is_file():
        raise FileNotFoundError(f"Fixture database does not exist: {database_path}")
    with duckdb.connect(str(database_path)) as connection:
        all_fixtures = combine_fixtures.load_all_fixtures(
            db_path=database_path, connection=connection, league_id=league_id
        )
        if all_fixtures.empty:
            raise ValueError(f"No fixtures found for league {league_id}")

        for season in seasons:
            print(f"Processing season: {season} - {league_name} - {country_name}")
            history = all_fixtures[
                (all_fixtures["league_id"].astype(str) == str(league_id))
                & (pd.to_numeric(all_fixtures["season"], errors="coerce") <= int(season))
            ].copy()
            history["round"] = (
                history["season"].astype(str) + "::" + history["round"].astype(str)
            )

            # Process goals
            process_goals_data(league_id=league_id, season=season, country_name=country_name, database_path=database_path, connection=connection)
            report_progress(f"Processed goals for season {season}")
            pause()

            # Process points
            process_points_data(league_id=league_id, season=season, country_name=country_name, database_path=database_path, connection=connection)
            report_progress(f"Processed points for season {season}")
            pause()

            # Process expected goals
            process_xg_data(
                league_id=league_id, season=season, country_name=country_name,
                fixtures=history, database_path=database_path, connection=connection,
            )
            report_progress(f"Processed expected goals for season {season}")
            pause()

            # Estimate the league/season Dixon-Coles rho
            process_rho_data(league_id=league_id, season=season, country_name=country_name, database_path=database_path, connection=connection)
            report_progress(f"Estimated Dixon-Coles rho for season {season}")
            pause()

            # Process fair odds data
            process_fair_odds_data(league_id=league_id, season=season, country_name=country_name, database_path=database_path, connection=connection)
            report_progress(f"Processed fair odds for season {season}")
            pause()

            # Process outcome probability data
            process_outcome_probability_data(
                league_id=league_id, season=season, country_name=country_name,
                fixtures=history, database_path=database_path, connection=connection,
            )
            report_progress(f"Processed outcome probabilities for season {season}")
            pause()

            process_power_ratings_data(
                league_id=league_id, season=season, country_name=country_name,
                fixtures=all_fixtures, database_path=database_path, connection=connection,
            )
            report_progress(f"Processed power ratings for season {season}")
            pause()

            # Process recent goals data
            process_recent_goals_data(
                league_id=league_id, season=season, country_name=country_name,
                fixtures=history, database_path=database_path, connection=connection,
            )
            report_progress(f"Processed recent goals for season {season}")
            pause()

            # Process recent points data
            process_recent_points_data(
                league_id=league_id, season=season, country_name=country_name,
                fixtures=history, database_path=database_path, connection=connection,
            )
            report_progress(f"Processed recent points for season {season}")
            pause()

            # Process recent xg data
            process_recent_xg_data(
                league_id=league_id, season=season, country_name=country_name,
                fixtures=history, database_path=database_path, connection=connection,
            )
            report_progress(f"Processed recent expected goals for season {season}")
            pause()

            # Process recent fair odds data
            process_recent_fair_odds_data(league_id=league_id, season=season, country_name=country_name, database_path=database_path, connection=connection)
            report_progress(f"Processed recent fair odds for season {season}")
            pause()

        # Process head-to-head data
        process_head_to_head_data(
            league_id=league_id,
            country_name=country_name,
            seasons_to_process=seasons,
            database_path=database_path,
            connection=connection,
            combined=all_fixtures,
        )
        report_progress("Processed head-to-head data")


def process_goals_data(league_id: int, country_name: str, season: int, database_path=None, connection=None) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    output_table = f"{league_id}-{season}-goals"

    fixtures = goals.load_fixtures(db_path=database_path, table_name=input_table, connection=connection)
    goals_df = goals.add_goals_before_match(fixtures)
    goals.write_output(goals_df, db_path=database_path, table_name=output_table, connection=connection)

def process_points_data(league_id: int, season: int, country_name:str, database_path=None, connection=None) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    output_table = f"{league_id}-{season}-points"

    fixtures = points.load_fixtures(db_path=database_path, table_name=input_table, connection=connection)
    points_df = points.add_points_before_match(fixtures)
    points.write_output(points_df, db_path=database_path, table_name=output_table, connection=connection)

def process_xg_data(
    league_id: int, season: int, country_name: str,
    fixtures: pd.DataFrame = None,
    database_path=None,
    connection=None,
) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    output_table = f"{league_id}-{season}-xg"

    fixtures = fixtures if fixtures is not None else expected_goals.load_fixtures(
        db_path=database_path, table_name=input_table, connection=connection
    )
    current_fixture_ids = _current_fixture_ids(fixtures, season)
    original_rounds = fixtures.set_index("fixture_id")["round"].to_dict()
    expected_goals_df = expected_goals.build_expected_goals_table(fixtures)
    expected_goals_df = expected_goals_df[
        expected_goals_df["fixture_id"].isin(current_fixture_ids)
    ].copy()
    expected_goals_df["round"] = expected_goals_df["fixture_id"].map(original_rounds)
    expected_goals.write_output(expected_goals_df, db_path=database_path, table_name=output_table, connection=connection)

def process_rho_data(league_id: int, season: int, country_name: str, database_path=None, connection=None) -> None:
    """Estimate the Dixon-Coles rho per round from the xG table and store it in "<league>-<season>-rho"."""
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    xg_df = fair_odds.load_expected_goals(db_path=database_path, table_name=f"{league_id}-{season}-xg", connection=connection)
    rho_df = rho_estimation.build_rho_table(xg_df)
    rho_estimation.write_output(rho_df, db_path=database_path, table_name=f"{league_id}-{season}-rho", connection=connection)

def process_fair_odds_data(league_id: int, season: int, country_name:str, database_path=None, connection=None) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-xg"
    output_table = f"{league_id}-{season}-odds"

    fixtures = fair_odds.load_expected_goals(db_path=database_path, table_name=input_table, connection=connection)
    rho_table = rho_estimation.load_rho_table(database_path, f"{league_id}-{season}-rho", connection=connection)
    fair_odds_df = fair_odds.compute_match_probabilities(fixtures, rho_table=rho_table)
    fair_odds.write_output(fair_odds_df, db_path=database_path, table_name=output_table, connection=connection)

def process_outcome_probability_data(
    league_id: int, season: int, country_name: str,
    fixtures: pd.DataFrame = None,
    database_path=None,
    connection=None,
) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    output_table = f"{league_id}-{season}-outcome-probability"

    fixtures = fixtures if fixtures is not None else outcome_probability.load_fixtures(
        db_path=database_path, table_name=input_table, connection=connection
    )
    current_fixture_ids = _current_fixture_ids(fixtures, season)
    outcome_probability_df = outcome_probability.add_outcome_probabilities_before_match(fixtures)
    outcome_probability_df = outcome_probability_df[
        outcome_probability_df["fixture_id"].isin(current_fixture_ids)
    ].copy()
    outcome_probability_df = outcome_probability_df.drop(
        columns=["season", "source_table"], errors="ignore"
    )
    outcome_probability.write_output(outcome_probability_df, db_path=database_path, table_name=output_table, connection=connection)

def process_power_ratings_data(
    league_id: int, season: int, country_name: str,
    fixtures: pd.DataFrame = None,
    database_path=None,
    connection=None,
) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    output_table = f"{league_id}-{season}-power-ratings"

    fixtures = fixtures if fixtures is not None else power_ratings.load_fixtures(
        db_path=database_path, table_name=input_table, connection=connection
    )
    if "league_id" in fixtures:
        fixtures = fixtures[fixtures["league_id"].astype(str) == str(league_id)].copy()
    if "season" in fixtures:
        fixtures = fixtures[pd.to_numeric(fixtures["season"], errors="coerce") == int(season)].copy()
    if fixtures.empty:
        raise ValueError(f"No fixtures found for league {league_id}, season {season}")
    ratings = power_ratings.add_power_ratings_before_match(fixtures)
    ratings = ratings.drop(columns=["season", "source_table"], errors="ignore")
    power_ratings.write_output(ratings, db_path=database_path, table_name=output_table, connection=connection)


def process_recent_goals_data(
    league_id: int, season: int, country_name: str,
    fixtures: pd.DataFrame = None,
    database_path=None,
    connection=None,
) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    form_windows = [3,5,10]
    fixtures = fixtures if fixtures is not None else last_n_goals.load_fixtures(
        db_path=database_path, table_name=input_table, connection=connection
    )
    current_fixture_ids = _current_fixture_ids(fixtures, season)

    for window in form_windows:
        output_table = f"{league_id}-{season}-goals-l{window}"
        recent_goals_df = last_n_goals.add_goals_last_n_matches(df=fixtures, window=window)
        recent_goals_df = _current_season_rows(
            recent_goals_df, current_fixture_ids
        )
        last_n_goals.write_output(recent_goals_df, db_path=database_path, table_name=output_table, connection=connection)

def process_recent_points_data(
    league_id: int, season: int, country_name: str,
    fixtures: pd.DataFrame = None,
    database_path=None,
    connection=None,
) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    form_windows = [3,5,10]
    fixtures = fixtures if fixtures is not None else last_n_points.load_fixtures(
        db_path=database_path, table_name=input_table, connection=connection
    )
    current_fixture_ids = _current_fixture_ids(fixtures, season)

    for window in form_windows:
        output_table = f"{league_id}-{season}-points-l{window}"
        recent_points_df = last_n_points.add_points_last_n_matches(df=fixtures, window=window)
        recent_points_df = _current_season_rows(
            recent_points_df, current_fixture_ids
        )
        last_n_points.write_output(recent_points_df, db_path=database_path, table_name=output_table, connection=connection)

def process_recent_xg_data(
    league_id: int, season: int, country_name: str,
    fixtures: pd.DataFrame = None,
    database_path=None,
    connection=None,
) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    input_table = f"{league_id}-{season}-fixtures"
    form_windows = [3,5,10]
    fixtures = fixtures if fixtures is not None else last_n_xg.load_fixtures(
        db_path=database_path, table_name=input_table, connection=connection
    )
    current_fixture_ids = _current_fixture_ids(fixtures, season)
    original_rounds = fixtures.set_index("fixture_id")["round"].to_dict()

    for window in form_windows:
        output_table = f"{league_id}-{season}-xg-l{window}"
        recent_xg_df = last_n_xg.build_expected_goals_table(df=fixtures, window=window)
        recent_xg_df = _current_season_rows(recent_xg_df, current_fixture_ids)
        recent_xg_df["round"] = recent_xg_df["fixture_id"].map(original_rounds)
        last_n_xg.write_output(recent_xg_df, db_path=database_path, table_name=output_table, connection=connection)


def _current_fixture_ids(fixtures: pd.DataFrame, season: int) -> set:
    if "season" not in fixtures:
        return set(fixtures["fixture_id"])
    current_season = pd.to_numeric(fixtures["season"], errors="coerce") == int(season)
    return set(fixtures.loc[current_season, "fixture_id"])


def _current_season_rows(df: pd.DataFrame, fixture_ids: set) -> pd.DataFrame:
    return df[df["fixture_id"].isin(fixture_ids)].drop(
        columns=["season", "source_table"], errors="ignore"
    ).copy()

def process_recent_fair_odds_data(league_id: int, season: int, country_name:str, database_path=None, connection=None) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    form_windows = [3,5,10]
    for window in form_windows:
        input_table = f"{league_id}-{season}-xg-l{window}"
        output_table = f"{league_id}-{season}-odds-l{window}"
        fixtures = last_n_fair_odds.load_expected_goals(db_path=database_path, table_name=input_table, connection=connection)
        recent_fair_odds_df = last_n_fair_odds.compute_match_probabilities(fixtures)
        last_n_fair_odds.write_output(recent_fair_odds_df, db_path=database_path, table_name=output_table, connection=connection)

def process_head_to_head_data(league_id: int, country_name:str, seasons_to_process:list, database_path=None, connection=None, combined=None) -> None:
    database_path = database_path or PROJECT_ROOT / "storage" / "database" / "leagues" / f"{country_name}.ddb"
    form_windows = [3,5,10]
    if combined is None:
        combined = combine_fixtures.load_all_fixtures(
            db_path=database_path, connection=connection, league_id=league_id
        )
    head_to_head.process_league_h2h(
        combined_df=combined,
        league_id=league_id,
        db_path=database_path,
        windows=form_windows,
        connection=connection,
    )


if __name__ == "__main__":
    league_ids = [39] # 119,203,207,208,61,62,89,140,141,135,136,78,357,110,39,113,94 -- 122
    for league in league_ids:
        process_seasons_data(league_id=league, seasons_to_process=[])