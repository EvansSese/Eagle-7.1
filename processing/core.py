import math
from contextlib import contextmanager
from itertools import groupby
from operator import attrgetter

import duckdb
from core.database import connect_existing_database


@contextmanager
def database_connection(db_path, connection=None, read_only: bool = False):
    """Reuse a caller-owned DuckDB connection, or open and own one for this call."""
    if connection is not None:
        yield connection
        return
    open_connection = connect_existing_database if read_only else duckdb.connect
    with open_connection(str(db_path)) as opened_connection:
        yield opened_connection

def write_fixture_table(df, db_path, table_name: str, connection=None) -> None:
    """Replace `table_name` with `df`, enforcing fixture_id as its unique key (used for joins)."""
    index_name = f"{table_name}-fixture_id-pk"
    with database_connection(db_path, connection) as con:
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        con.register("df_view", df)
        try:
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_view')
        finally:
            con.unregister("df_view")
        con.execute(f'CREATE UNIQUE INDEX "{index_name}" ON "{table_name}" (fixture_id)')

COMPLETED_STATUSES = frozenset({"FT", "AET", "PEN"})


def iter_fixture_kickoffs(df):
    """Iterate sorted fixtures in kickoff groups, without exposing same-time results."""
    columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "status"]
    return groupby(df[columns].itertuples(index=False), key=attrgetter("date"))


goals_bucket = {
    -3: (-math.inf, -20),  # -infinity < score <= -13
    -2: (-20, -10),  # -13 < score <= -4
    -1: (-10, -3),  # -4 < score <= -1
    0: (-3, 3),  # -1 < score <= 0   (captures exact 0)
    1: (3, 10),  # 0 < score <= 3
    2: (10, 20),  # 3 < score <= 6
    3: (20, math.inf)
}

goals_per_match_bucket = {
    -3: (-math.inf, -1),  # -infinity < score <= -13
    -2: (-1, -0.75),  # -13 < score <= -4
    -1: (-0.75, -0.25),  # -4 < score <= -1
    0: (-0.25, 0.25),  # -1 < score <= 0   (captures exact 0)
    1: (0.25, 0.75),  # 0 < score <= 3
    2: (0.75, 1),  # 3 < score <= 6
    3: (1, math.inf)
}

points_bucket = {
        -3: (-math.inf, -13),  # -infinity < score <= -13
        -2: (-13, -4),  # -13 < score <= -4
        -1: (-4, -1),  # -4 < score <= -1
        0: (-1, 0),  # -1 < score <= 0   (captures exact 0)
        1: (0, 3),  # 0 < score <= 3
        2: (3, 6),  # 3 < score <= 6
        3: (6, math.inf)  # 6 < score <= +infinity
    }

points_per_match_bucket = {
    -3: (-math.inf, -3 + (6 / 7)),  # -infinity < score <= -2.1429
    -2: (-3 + (6 / 7), -3 + 2 * (6 / 7)),  # -2.1429   < score <= -1.2857
    -1: (-3 + 2 * (6 / 7), -3 + 3 * (6 / 7)),  # -1.2857   < score <= -0.4286
    0: (-3 + 3 * (6 / 7), -3 + 4 * (6 / 7)),  # -0.4286   < score <=  0.4286
    1: (-3 + 4 * (6 / 7), -3 + 5 * (6 / 7)),  # 0.4286   < score <=  1.2857
    2: (-3 + 5 * (6 / 7), -3 + 6 * (6 / 7)),  # 1.2857   < score <=  2.1429
    3: (-3 + 6 * (6 / 7), math.inf)  # 2.1429   < score <= +infinity
}

def get_bucket(bucket_list:dict, value:float | int)-> int | None:
    score_key = None
    for key, (low, high) in bucket_list.items():
        # Use strict evaluation: 'low < score <= high'
        if low < value <= high:
            score_key = key
            break
    return score_key
