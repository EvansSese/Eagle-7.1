"""
head_to_head.py

Computes head-to-head (H2H) versions of every metric we've built —
points, goals, expected_goals, fair_odds, and their rolling
last-N-matches counterparts — one season at a time, writing each
season's result to its own table "{league_id}-{season}-h2h".

WHY THIS WAS SLOW BEFORE, AND WHAT CHANGED
-------------------------------------------
The previous version recomputed every metric from the ENTIRE
multi-season history every time — for every season, it re-grouped the
full combined dataset (all leagues, all seasons, INCLUDING seasons that
hadn't been reached yet in the loop) by matchup, 17 separate times (one
full pass per metric), and re-ran points/goals accumulation from
scratch across every single prior season. That's roughly O(seasons^2)
work for points/goals alone, and repeats the expensive matchup-key
grouping step far more often than necessary.

This version instead:

  1. SEEDS points/goals from the previous season's table. Both are
     plain running totals (no windowing), so a season's ENDING state
     can be derived exactly from that season's LAST row per matchup —
     take its "before this match" values and add that match's own
     result — and handed to points.py / goals.py as a starting point
     (see the new `initial_points` / `initial_goals_scored` etc.
     parameters added there). Each season then only needs to process
     ITS OWN fixtures for that matchup, not the entire history again.

  2. BOUNDS last-N metrics to a trailing window instead of full
     history. A rolling window only ever needs its own size worth of
     preceding matches — pulling exactly `max(windows)` trailing
     matches (plus the current season's own) is always exact, and its
     cost never grows with the number of seasons processed.

  3. BOUNDS expected_goals / fair_odds to "this matchup's history up to
     and including the current season" rather than the entire
     multi-season dataset (which previously included future seasons
     too, since it was never filtered). True incremental seeding isn't
     feasible for these (the league-average baseline is grouped by
     round, not a plain running sum), but a single matchup's total
     history is naturally small regardless of season count, so this
     bound keeps them cheap without changing what they compute.

  4. Groups matchup history once per season and passes each matchup only
     its own history, rather than rescanning the full history for every
     matchup.

  5. Reuses each expected-goals result when computing fair odds rather
     than running the same expected-goals calculation twice, and
     computes odds only for current-season fixtures.

  6. Reuses existing season output tables by default; missing tables are
     computed (and written) first, recursively if necessary. Set
     `force_recompute=True` to rebuild existing tables.

Requires: duckdb, pandas
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd
from processing.core import COMPLETED_STATUSES
from processing.core import database_connection

try:
    from .combine_fixtures import load_all_fixtures
except ImportError:
    from h2h.combine_fixtures import load_all_fixtures
from processing.points.points import add_points_before_match
from processing.goals.goals import add_goals_before_match
from processing.xG.expected_goals import build_expected_goals_table
from processing.fair_odds.fair_odds import compute_match_probabilities
from processing.recent_points.last_n_points import add_points_last_n_matches
from processing.recent_goals.last_n_goals import add_goals_last_n_matches
from processing.recent_xg.last_n_xg import build_expected_goals_table as build_expected_goals_table_last_n
from processing.recent_fair_odds.last_n_fair_odds import compute_match_probabilities as compute_match_probabilities_last_n
from processing.outcome_probability.outcome_probability import add_outcome_probabilities_before_match


_CORE_FIXTURE_COLUMNS = {
    "fixture_id", "date", "round", "home_team", "away_team", "home_goals", "away_goals",
    "status", "venue", "league_id", "season", "source_table", "matchup_key",
}
_OUTCOME_PROBABILITY_COLUMNS = {
    *(f"{side}_team_{outcome}_probability_before" for side in ("x", "y")
      for outcome in ("win", "draw", "loss")),
    "combined_home_win_probability_before",
    "combined_draw_probability_before",
    "combined_away_win_probability_before",
    "predicted_outcome",
}

# Pandas-heavy matchup work is typically faster without thread contention.
EXECUTOR_KIND = "thread"  # "thread" or "process"
DEFAULT_MAX_WORKERS = 1


def _make_executor(max_workers: int = None):
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS
    if EXECUTOR_KIND == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    return ThreadPoolExecutor(max_workers=max_workers)


def add_matchup_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'matchup_key' column: a canonical, order-independent id for
    the pair of teams in each fixture (sorted tuple of team names),
    using a vectorized operation to avoid per-row Python callbacks.
    """
    if df[["home_team", "away_team"]].isna().any().any():
        raise ValueError("fixture team names cannot be empty")
    df = df.copy()
    teams = df[["home_team", "away_team"]]
    df["matchup_key"] = list(zip(teams.min(axis=1), teams.max(axis=1)))
    return df


def _group_matchups(df: pd.DataFrame) -> dict:
    """Group fixture rows by canonical team pair with one vectorized pass."""
    if df.empty:
        return {}
    if df[["home_team", "away_team"]].isna().any().any():
        raise ValueError("fixture team names cannot be empty")

    home, away = df["home_team"], df["away_team"]
    home_first = home <= away
    keys = pd.DataFrame({
        "team_a": home.where(home_first, away),
        "team_b": away.where(home_first, home),
    }, index=df.index)
    return {
        key: df.iloc[positions]
        for key, positions in keys.groupby(["team_a", "team_b"], sort=False).indices.items()
    }


def add_h2h_meeting_count(df: pd.DataFrame, initial_count: int = 0) -> pd.DataFrame:
    """
    Adds 'h2h_previous_meetings': how many times, prior to this
    specific fixture, these two teams have played each other and the
    match finished (status == 'FT'). `initial_count` seeds the running
    count (see `extract_meeting_count_seed`) so a new season can resume
    from where the previous one left off instead of recounting from
    scratch.
    """
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    finished = df["status"].isin(COMPLETED_STATUSES).astype(int)
    df["h2h_previous_meetings"] = initial_count + finished.cumsum() - finished
    return df


def extract_meeting_count_seed(prev_matchup_h2h: pd.DataFrame) -> int:
    """Given a matchup's rows from the previous season's h2h table, returns
    the total meeting count AFTER that season's last match (its own
    'h2h_previous_meetings' value, plus 1 if that match finished)."""
    if prev_matchup_h2h.empty:
        return 0
    last = prev_matchup_h2h.sort_values(["date", "fixture_id"]).iloc[-1]
    count = int(last["h2h_previous_meetings"])
    if last["status"] in COMPLETED_STATUSES:
        count += 1
    return count


def _with_season_disambiguated_round(group_df: pd.DataFrame, build_xg_fn, window: int = None) -> pd.DataFrame:
    """
    Runs build_xg_fn (expected_goals.py / last_n_xg.py's
    build_expected_goals_table) safely on a head-to-head subset that may
    span multiple seasons: temporarily prefixes 'round' with 'season' so
    same-named rounds from different seasons ('Regular Season - 1' in
    2023 vs 2025) aren't collapsed into a single round by
    determine_round_order, then restores the original 'round' label in
    the output (keyed by fixture_id) so the result looks normal.
    """
    disambiguated = group_df.copy()
    original_round_by_fixture = dict(zip(disambiguated["fixture_id"], disambiguated["round"]))
    disambiguated["round"] = disambiguated["season"].astype(str) + " - " + disambiguated["round"].astype(str)

    result = build_xg_fn(disambiguated, window=window) if window is not None else build_xg_fn(disambiguated)
    result = result.copy()
    result["round"] = result["fixture_id"].map(original_round_by_fixture)
    return result


def suffix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Renames every non-core column by appending `suffix`. Needed because
    expected_goals / fair_odds and their last-N counterparts reuse
    IDENTICAL column names regardless of window — so merging multiple
    windows' results together would silently collide without this.
    """
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c not in _CORE_FIXTURE_COLUMNS}
    return df.rename(columns=rename_map)


def merge_h2h_results(base_df: pd.DataFrame, *other_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merges several H2H result DataFrames (all keyed by fixture_id) into
    one, keeping shared columns only once (from base_df) and adding
    only the genuinely new columns each subsequent DataFrame contributes.
    """
    frames = [base_df.copy().set_index("fixture_id")]
    existing_columns = set(base_df.columns)
    for df in other_dfs:
        new_columns = [
            column for column in df.columns
            if column not in existing_columns and column != "fixture_id"
        ]
        if new_columns:
            frames.append(df.loc[:, ["fixture_id", *new_columns]].copy().set_index("fixture_id"))
            existing_columns.update(new_columns)
    merged = pd.concat(frames, axis=1).copy()
    merged.index.name = "fixture_id"
    return merged.reset_index()


def process_matchup_for_season(team_a: str, team_b: str, current_season_matchup_rows: pd.DataFrame,
                                matchup_history: pd.DataFrame, prev_matchup_h2h: pd.DataFrame,
                                outcome_probability_rows: pd.DataFrame, windows) -> pd.DataFrame:
    """
    Computes every H2H metric for ONE matchup's fixtures in the CURRENT
    season only — this is the unit of work distributed across the
    executor. Must be a plain top-level function (not a closure) so it
    stays picklable for ProcessPoolExecutor.

    - meetings/points/goals: seeded from prev_season_h2h (if this
      matchup appeared there), then run over just this season's rows.
    - last-N metrics: bounded to this matchup's trailing max(windows)
      matches (pulled from history_upto_season) plus this season.
    - expected_goals/fair_odds (season and last-N): bounded to this
      matchup's full history up to and including the current season
      (naturally small, regardless of how many seasons exist).
    """
    current_rows = current_season_matchup_rows.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    # --- meetings, points, goals: seeded, run over just this season's rows ---
    meeting_seed = extract_meeting_count_seed(prev_matchup_h2h)
    h2h_meetings = add_h2h_meeting_count(current_rows, initial_count=meeting_seed)

    current_fixture_ids = set(current_rows["fixture_id"])
    prior_matchup_rows = matchup_history[~matchup_history["fixture_id"].isin(current_fixture_ids)]
    h2h_points = add_points_before_match(current_rows, history=prior_matchup_rows)
    h2h_goals = add_goals_before_match(current_rows, history=prior_matchup_rows)

    # --- expected_goals / fair_odds: bounded to full matchup history up to this season ---

    h2h_xg_full = _with_season_disambiguated_round(matchup_history, build_expected_goals_table)
    h2h_xg = h2h_xg_full[h2h_xg_full["fixture_id"].isin(current_fixture_ids)].reset_index(drop=True)

    h2h_odds = compute_match_probabilities(h2h_xg, use_form=False)

    # --- last-N metrics: bounded to trailing max(windows) matches + this season ---
    max_window = max(windows)
    trailing_history = matchup_history[~matchup_history["fixture_id"].isin(current_fixture_ids)]
    trailing_history = trailing_history.sort_values(["date", "fixture_id"]).tail(max_window)
    bounded_for_last_n = pd.concat([trailing_history, current_rows], ignore_index=True)
    bounded_for_last_n = bounded_for_last_n.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    last_n_pieces = []
    for window in windows:
        points_l_full = add_points_last_n_matches(bounded_for_last_n, window=window)
        points_l = points_l_full[points_l_full["fixture_id"].isin(current_fixture_ids)].reset_index(drop=True)

        goals_l_full = add_goals_last_n_matches(bounded_for_last_n, window=window)
        goals_l = goals_l_full[goals_l_full["fixture_id"].isin(current_fixture_ids)].reset_index(drop=True)

        xg_l_full = _with_season_disambiguated_round(bounded_for_last_n, build_expected_goals_table_last_n, window=window)
        xg_l_current = xg_l_full[xg_l_full["fixture_id"].isin(current_fixture_ids)].reset_index(drop=True)

        odds_l = compute_match_probabilities_last_n(xg_l_current)
        xg_l = suffix_columns(xg_l_current, f"_last{window}")
        odds_l = suffix_columns(odds_l, f"_last{window}")

        last_n_pieces.extend([points_l, goals_l, xg_l, odds_l])

    return merge_h2h_results(
        h2h_meetings, h2h_points, h2h_goals, h2h_xg, h2h_odds,
        outcome_probability_rows, *last_n_pieces,
    )


def _process_matchup_item(args: tuple) -> pd.DataFrame:
    """Top-level executor adapter so both thread and process pools can use it."""
    return process_matchup_for_season(*args)


def build_h2h_dataset_for_season(history_upto_season: pd.DataFrame, current_season_fixtures: pd.DataFrame,
                                  prev_season_h2h: pd.DataFrame, windows=(3, 5, 10),
                                  max_workers: int = None,
                                  outcome_probability_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes the head-to-head dataset for ONE season, processing each
    matchup present in current_season_fixtures concurrently (see
    EXECUTOR_KIND). `history_upto_season` should contain every fixture
    up to and including this season (used for bounding expected_goals/
    fair_odds and the last-N trailing window) — NOT future seasons.
    `prev_season_h2h` is the immediately preceding season's h2h table
    (or an empty/None DataFrame for the first season), used to seed
    points/goals/meeting-count.
    """
    current_matchups = _group_matchups(current_season_fixtures)
    history_matchups = _group_matchups(history_upto_season)
    previous_matchups = _group_matchups(prev_season_h2h) if prev_season_h2h is not None else {}
    if outcome_probability_df is None:
        outcome_probability_df = add_outcome_probabilities_before_match(history_upto_season)
    outcome_probability_columns = [
        "fixture_id",
        *[
            f"{side}_team_{outcome}_probability_before"
            for side in ("x", "y")
            for outcome in ("win", "draw", "loss")
        ],
        "combined_home_win_probability_before",
        "combined_draw_probability_before",
        "combined_away_win_probability_before",
        "predicted_outcome",
    ]
    missing_columns = set(outcome_probability_columns).difference(outcome_probability_df.columns)
    if missing_columns:
        raise ValueError(f"outcome probability data is missing columns: {sorted(missing_columns)}")
    current_fixture_ids = set(current_season_fixtures["fixture_id"])
    current_outcome_probabilities = outcome_probability_df[
        outcome_probability_df["fixture_id"].isin(current_fixture_ids)
    ][outcome_probability_columns]
    fixture_keys = current_season_fixtures[["fixture_id", "home_team", "away_team"]]
    outcome_rows = fixture_keys.merge(
        current_outcome_probabilities,
        on="fixture_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not outcome_rows["_merge"].eq("both").all():
        raise ValueError("outcome probability data does not cover every current-season fixture")
    outcome_matchups = _group_matchups(outcome_rows.drop(columns="_merge"))
    matchup_items = []
    empty_history = history_upto_season.iloc[0:0]
    for (team_a, team_b), rows in current_matchups.items():
        key = (team_a, team_b)
        matchup_items.append((
            team_a,
            team_b,
            rows.sort_values(["date", "fixture_id"]).reset_index(drop=True),
            history_matchups.get(key, empty_history),
            previous_matchups.get(key, pd.DataFrame()),
            outcome_matchups.get(key, pd.DataFrame()),
            windows,
        ))

    with _make_executor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_matchup_item, matchup_items))

    combined_results = pd.concat(results, ignore_index=True)
    return combined_results.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def write_output(df: pd.DataFrame, db_path: str, table_name: str, connection=None) -> None:
    with database_connection(db_path, connection) as con:
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        con.register("df_view", df)
        try:
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_view')
        finally:
            con.unregister("df_view")


def _read_table_if_exists(db_path: str, table_name: str, connection=None):
    if not Path(db_path).exists():
        return None
    with database_connection(db_path, connection, read_only=True) as con:
        existing_tables = {name for (name,) in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if table_name not in existing_tables:
            return None
        return con.execute(f'SELECT * FROM "{table_name}"').df()


def get_seasons_for_league(combined_df: pd.DataFrame, league_id) -> list:
    """Returns every season available for `league_id`, sorted."""
    league_id = str(league_id)
    league_rows = combined_df[combined_df["league_id"].astype(str) == league_id]
    return sorted(league_rows["season"].unique())


def ensure_season_h2h(combined_df: pd.DataFrame, league_id, season, db_path: str,
                       windows=(3, 5, 10), _cache: dict = None,
                       force_recompute: bool = False, _seasons: list = None,
                       _outcome_probability_df: pd.DataFrame = None,
                       connection=None, _outcome_probability_cache: dict = None) -> pd.DataFrame:
    """
    Returns the h2h table for `season`, computing (and writing) it —
    and, recursively, any missing earlier seasons it depends on — if
    it isn't already in the database. Set `force_recompute=True` to
    rebuild existing data. `_cache` is an in-memory
    memoization dict shared across the recursion within one run, so a
    season already computed earlier in this same run isn't re-read
    from disk or recomputed again.
    """
    league_id = str(league_id)
    _cache = {} if _cache is None else _cache
    cache_key = (league_id, season)
    if cache_key in _cache:
        return _cache[cache_key]

    table_name = f"{league_id}-{season}-h2h"
    existing = None if force_recompute else _read_table_if_exists(
        db_path, table_name, connection=connection
    )
    if existing is not None and _OUTCOME_PROBABILITY_COLUMNS.issubset(existing.columns):
        _cache[cache_key] = existing
        return existing

    if existing is not None:
        print(f"Rebuilding {table_name}: outcome probability columns are missing")

    if _outcome_probability_df is None:
        if _outcome_probability_cache is None:
            _outcome_probability_cache = {}
        if league_id not in _outcome_probability_cache:
            league_fixtures = combined_df[combined_df["league_id"].astype(str) == league_id]
            _outcome_probability_cache[league_id] = add_outcome_probabilities_before_match(league_fixtures)
        _outcome_probability_df = _outcome_probability_cache[league_id]

    print(f"Computing head-to-head for {table_name}")

    seasons = get_seasons_for_league(combined_df, league_id) if _seasons is None else _seasons
    season_index = seasons.index(season)

    if season_index == 0:
        prev_season_h2h = pd.DataFrame()
    else:
        prev_season = seasons[season_index - 1]
        prev_season_h2h = ensure_season_h2h(
            combined_df, league_id, prev_season, db_path, windows=windows,
            _cache=_cache, force_recompute=force_recompute, _seasons=seasons,
            _outcome_probability_df=_outcome_probability_df,
            _outcome_probability_cache=_outcome_probability_cache,
            connection=connection,
        )

    season_table = f"{league_id}-{season}-fixtures"
    current_season_fixtures = combined_df[combined_df["source_table"] == season_table]
    if current_season_fixtures.empty:
        raise ValueError(f"No rows found for source_table='{season_table}'.")

    history_upto_season = combined_df[
        (combined_df["league_id"] == league_id) & (combined_df["season"] <= season)
    ]

    h2h_dataset = build_h2h_dataset_for_season(
        history_upto_season, current_season_fixtures, prev_season_h2h,
        windows=windows, outcome_probability_df=_outcome_probability_df,
    )

    write_output(h2h_dataset, db_path, table_name, connection=connection)
    print(f"Wrote {len(h2h_dataset)} rows ({len(h2h_dataset.columns)} columns) to '{table_name}' in {db_path}")

    _cache[cache_key] = h2h_dataset
    return h2h_dataset


def process_league_h2h(combined_df: pd.DataFrame, league_id, db_path: str, windows=(3, 5, 10),
                       force_recompute: bool = False, connection=None) -> None:
    """
    Ensures a head-to-head dataset exists for every season of `league_id`,
    reusing existing tables unless `force_recompute=True`.
    """
    league_id = str(league_id)
    combined_df = combined_df[combined_df["league_id"].astype(str) == league_id].copy()
    seasons = get_seasons_for_league(combined_df, league_id)
    if not seasons:
        raise ValueError(f"No seasons found for league_id={league_id!r} in the combined fixtures.")

    cache = {}
    outcome_probability_cache = {}
    total_seasons = len(seasons)
    for season_number, season in enumerate(seasons, start=1):
        print(
            f"Processing season {season_number}/{total_seasons}: "
            f"{season} (league {league_id})"
        )
        result = ensure_season_h2h(
            combined_df, league_id, season, db_path, windows=windows,
            _cache=cache, force_recompute=force_recompute, _seasons=seasons,
            _outcome_probability_cache=outcome_probability_cache,
            connection=connection,
        )
        cache.clear()
        cache[(league_id, season)] = result


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/England.ddb"
    LEAGUE_ID = "39"
    FORM_WINDOWS = [3, 5, 10]

    combined = load_all_fixtures(DB_PATH)

    process_league_h2h(combined, LEAGUE_ID, DB_PATH, windows=FORM_WINDOWS)