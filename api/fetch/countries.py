import os
from dotenv import load_dotenv
import http.client
import json
from pathlib import Path

# Load environment variables from env file to os_environment
load_dotenv()

# Read environment variables
BASE_URL = os.getenv("BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")

output_dir = Path("../json")
file_path = output_dir / "countries.json"

# Automatically create the 'json' folder if it doesn't exist yet
output_dir.mkdir(parents=True, exist_ok=True)

# Function to fetch countries from the API
def fetch_countries():
    # Create the connection
    conn = http.client.HTTPSConnection(BASE_URL)

    # Set the headers
    headers = {
        "x-apisports-key": API_KEY,
    }

    # Send the request to the API
    try:
        conn.request("GET", "/countries", headers=headers)
        response = conn.getresponse()
        if response.status == 200:
            data = json.loads(response.read())
            with file_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=4, ensure_ascii=False))
        print("Successfully fetched countries")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    fetch_countries()