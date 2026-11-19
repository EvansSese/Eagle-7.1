import os
from dotenv import load_dotenv
import http.client
import json
from pathlib import Path
import ssl
import time
from typing import Callable

# Load environment variables from env file to os_environment
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Read environment variables
BASE_URL = os.getenv("BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
SKIP_SSL_VERIFY = os.getenv("EAGLE_INSECURE_SKIP_SSL_VERIFY", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

input_dir = Path(__file__).resolve().parents[1] / "json"
input_file_path = input_dir / "leagues.json"


def fetch_fixtures(
    league_id: int,
    seasons: list,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    if not BASE_URL or not API_KEY:
        raise ValueError("BASE_URL and API_KEY must be configured")

    output_dir = input_dir / "fixtures" / str(league_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "x-apisports-key": API_KEY,
    }

    if SKIP_SSL_VERIFY:
        print(
            "WARNING: SSL certificate verification is disabled for fixture API requests."
        )
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    else:
        ssl_context = ssl.create_default_context()
    conn = http.client.HTTPSConnection(
        BASE_URL,
        timeout=30,
        context=ssl_context,
    )
    try:
        for index, season in enumerate(seasons):
            output_file_path = output_dir / f"{league_id}-{season}-fixtures.json"

            print(f"Fetching fixtures for {season} season")
            if progress_callback is not None:
                progress_callback(season, "fetching")

            conn.request("GET", f"/fixtures?season={season}&league={league_id}", headers=headers)
            response = conn.getresponse()
            response_body = response.read()
            if response.status != 200:
                raise RuntimeError(
                    f"Fixture API returned HTTP {response.status} "
                    f"for league {league_id}, season {season}: "
                    f"{response_body.decode('utf-8', errors='replace')}"
                )
            data = json.loads(response_body)
            if data.get("errors"):
                raise RuntimeError(
                    f"Fixture API reported errors for league {league_id}, "
                    f"season {season}: {data['errors']}"
                )
            if not isinstance(data.get("response"), list):
                raise ValueError("Fixture API response must contain a response list")
            with output_file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Successfully fetched fixtures for {season} season")
            if progress_callback is not None:
                progress_callback(season, "downloaded")

            if index < len(seasons) - 1:
                time.sleep(15)
    finally:
        conn.close()


def get_seasons(league_id: int)->list:
    seasons_list = []
    if os.path.exists(input_file_path):
        with input_file_path.open("r", encoding="utf-8") as f:
            leagues = json.load(f)

        for league in leagues['response']:
            l_id = league['league']['id']
            if l_id == league_id:
                print(f"{league['league']['name']} - {league['country']['name']}")
                for season in league['seasons']:
                    seasons_list.append(season['year'])
                    # print(season['year'])
    if not seasons_list:
        raise ValueError(f"No seasons found for league {league_id}")
    return seasons_list

if __name__ == "__main__":
    LEAGUE_IDS = [39] # 119,203,207,208,61,62,89,140,141,135,136,78,357,110,39,113,94 -- 122
    for LEAGUE_ID in LEAGUE_IDS:
        seasons = get_seasons(league_id=LEAGUE_ID)
        # print(seasons)
        fetch_fixtures(league_id=LEAGUE_ID, seasons=[seasons[-1]])
