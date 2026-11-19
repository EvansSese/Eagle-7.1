from pathlib import Path

import duckdb


def connect_existing_database(path):
    """Open an existing database with the same configuration as writers.

    DuckDB rejects mixed read-only and read-write configurations for the
    same database within a process. This therefore requires write access
    even when the caller only performs reads.
    """
    database = Path(path)
    if not database.is_file():
        raise FileNotFoundError(f"Database does not exist: {database}")
    return duckdb.connect(str(database))
