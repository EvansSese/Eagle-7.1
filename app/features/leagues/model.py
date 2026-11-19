"""Read league catalogue data for the dashboard."""

from pathlib import Path

import duckdb


def get_leagues(database_path: Path) -> list[dict]:
    """Return leagues in country and league ID order for the league browser."""
    with duckdb.connect(str(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT
                league_id,
                league_name,
                country_name,
                country_flag,
                league_logo,
                seasons
            FROM leagues
            ORDER BY
                lower(coalesce(country_name, '')),
                league_id
            """
        ).fetchall()

    leagues = []
    for league_id, league_name, country_name, country_flag, league_logo, seasons in rows:
        available_seasons = seasons or []
        leagues.append(
            {
                "league_id": league_id,
                "league_name": league_name or "Unknown league",
                "country_name": country_name or "Unknown country",
                "country_flag": country_flag,
                "league_logo": league_logo,
                "seasons_available": bool(available_seasons),
                "season_count": len(available_seasons),
            }
        )
    return leagues
