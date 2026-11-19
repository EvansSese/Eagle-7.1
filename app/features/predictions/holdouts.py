"""Read recorded holdout membership for the models used by the dashboard."""

import duckdb


def read_test_fixture_ids(
    connection: duckdb.DuckDBPyConnection,
    league_id: int,
    include_outcome_networks: bool,
) -> tuple[set[int], list[str]]:
    """Return the union of recorded test IDs and names of missing test tables.

    The outcome-network table is only relevant when those optional models
    participate in the ensemble. Missing tables make negative membership
    unknown, rather than proving that a fixture was not used for testing.
    """
    tables = [
        f"{league_id}-ensemble-classifier-test-predictions",
        f"{league_id}-goal-regression-test",
        f"{league_id}-deep-learning-test-predictions",
        f"{league_id}-sequence-model-test-predictions",
    ]
    if include_outcome_networks:
        tables.append(f"{league_id}-outcome-network-test-predictions")

    test_ids: set[int] = set()
    missing = []
    for table in tables:
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            missing.append(table)
            continue
        rows = connection.execute(
            f'SELECT DISTINCT fixture_id FROM "{table}" WHERE fixture_id IS NOT NULL'
        ).fetchall()
        test_ids.update(int(row[0]) for row in rows)
    return test_ids, missing
