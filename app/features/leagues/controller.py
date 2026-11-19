"""HTTP controller for the league catalogue."""

from pathlib import Path
from typing import Callable

import duckdb

from app.features.leagues import model


def serve_leagues(
    handler,
    database_path: Path,
    load_leagues: Callable[[Path], list[dict]] = model.get_leagues,
) -> None:
    """Send the catalogue or a logged database error using the dashboard handler.

    The loader can be supplied by an embedding application using the existing
    server API; otherwise the feature's catalogue model is used.
    """
    try:
        leagues = load_leagues(database_path)
    except (duckdb.Error, OSError) as error:
        handler.log_error("Unable to load leagues from %s: %s", database_path, error)
        handler._send_json({"error": "Unable to load leagues from the DuckDB database."}, status=500)
        return
    handler._send_json({"leagues": leagues})
