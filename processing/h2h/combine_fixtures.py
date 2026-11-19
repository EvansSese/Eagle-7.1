"""
combine_fixtures.py

Discovers every fixtures table in the DuckDB database — tables named
following the "{league_id}-{season}-fixtures" convention — and
concatenates them into a single DataFrame spanning every league and
every season available.

This combined DataFrame is the shared foundation for head-to-head (H2H)
analysis: two teams' head-to-head history can span multiple seasons (or
even multiple competitions), so any H2H computation needs visibility
into ALL fixtures tables at once, not just the current season's.

Requires: duckdb, pandas
"""

import re
import duckdb
import pandas as pd
from processing.core import database_connection

FIXTURES_TABLE_PATTERN = re.compile(r"^(?P<league_id>\w+)-(?P<season>\w+)-fixtures$")


def list_fixture_tables(db_path: str, connection=None) -> list:
    """
    Returns a list of {"table_name", "league_id", "season"} dicts for
    every table in the database matching the
    "{league_id}-{season}-fixtures" naming convention. Tables that don't
    match (e.g. derived tables like "39-2025-points") are ignored.
    """
    with database_connection(db_path, connection, read_only=True) as con:
        all_tables = con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()

    matches = []
    for (table_name,) in all_tables:
        m = FIXTURES_TABLE_PATTERN.match(table_name)
        if m:
            matches.append({
                "table_name": table_name,
                "league_id": m.group("league_id"),
                "season": m.group("season"),
            })
    return matches


def load_all_fixtures(db_path: str, connection=None, league_id=None) -> pd.DataFrame:
    """
    Loads and concatenates every fixtures table found via
    list_fixture_tables into one combined DataFrame. Each row is tagged
    with 'season' and 'source_table' (and 'league_id' is normalized from
    the table name, overriding the raw column in case it's ever missing
    or inconsistent) so downstream code can tell which season/table a
    given fixture came from.

    If league_id is supplied, only that league's fixtures are loaded.
    Returns an empty DataFrame if no matching tables are found, rather
    than raising — callers decide whether that's an error for their use
    case.
    """
    fixture_tables = list_fixture_tables(db_path, connection=connection)
    if league_id is not None:
        fixture_tables = [info for info in fixture_tables if info["league_id"] == str(league_id)]
    if not fixture_tables:
        return pd.DataFrame()

    frames = []
    with database_connection(db_path, connection, read_only=True) as con:
        for info in fixture_tables:
            df = con.execute(f'SELECT * FROM "{info["table_name"]}"').df()
            df["season"] = info["season"]
            df["league_id"] = info["league_id"]
            df["source_table"] = info["table_name"]
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    DB_PATH = "../../storage/database/leagues/England.ddb"

    combined = load_all_fixtures(DB_PATH)

    if combined.empty:
        print(f"No '{{league_id}}-{{season}}-fixtures' tables found in {DB_PATH}")
    else:
        print(f"Loaded {len(combined)} fixtures across {combined['source_table'].nunique()} season table(s):")
        print(combined["source_table"].value_counts().to_string())