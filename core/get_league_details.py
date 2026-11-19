from core.database import connect_existing_database

def get_league_details(leagues_db_path:str,league_id: int) -> dict:
    try:
        with connect_existing_database(leagues_db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM leagues WHERE league_id = ?",
                [league_id]
            )
            row = cursor.fetchone()
            columns = [column[0] for column in cursor.description]

        if row is None:
            return {
                "status": False,
                "error": "League not found"
            }

        # Convert row to key-value pairs
        league_details = dict(zip(columns, row))

        return {
            "status": True,
            "league_details": league_details
        }

    except Exception as e:
        return {
            "status": False,
            "error": str(e)
        }


if __name__ == "__main__":
    result = get_league_details(leagues_db_path="../storage/database/leagues.ddb", league_id=39)
    print(result)