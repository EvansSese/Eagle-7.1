import os
import csv
import json
from pathlib import Path


def csv_to_json(file_path:str, league_id:int, season_year:str=None)-> None:
    #leagues = json.load(open(file_path))
    # Read the csv file
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        cols = [col for col in csv_reader.fieldnames]
        print(cols)

        json_list = []

        # Check if certain key is available in the csv header, if not return a specific value
        season_key = next((key for key in cols if key in ["Season", "season"]), None)
        date_key = next((key for key in cols if key in ["Date", "date"]), None)
        time_key = next((key for key in cols if key in ["Time", "time"]), None)
        home_team_key = next((key for key in cols if key in ["Home", "HomeTeam"]), None)
        away_team_key = next((key for key in cols if key in ["Away", "AwayTeam"]), None)
        home_score_key = next((key for key in cols if key in ["HG", "FTHG"]), None)
        away_score_key = next((key for key in cols if key in ["AG", "FTAG"]), None)
        result_key = next((key for key in cols if key in ["Res", "FTR"]), None)

        for row in csv_reader:
            json_row = {
                "league_id": league_id,
                "season": row.get(season_key) or season_year,
                "date": row.get(date_key),
                "time": row.get(time_key),
                "home_team": row.get(home_team_key),
                "away_team": row.get(away_team_key),
                "home_score": row.get(home_score_key),
                "away_score": row.get(away_score_key),
                "result": row.get(result_key),
            }
            json_list.append(json_row)

    # Write the json file
    json_file_path = f"./json/{season_year}/{league_id}-{season_year}-fixtures.json"
    dir_name = os.path.dirname(json_file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(json_file_path, 'w') as json_file:
        json.dump(json_list, json_file, indent=4)

if __name__ == "__main__":
    league_ids = json.load(open("csv_leagues.json"))
    dir_path = Path(f'./csv/')
    folders = [f.stem for f in dir_path.iterdir() if f.is_dir()]
    # print(folders)
    for folder in folders:
        season_year = folder
        dir_path_file = Path(f'./csv/{folder}')
        files = [f.stem for f in dir_path_file.iterdir() if f.is_file()]
        print(files)
        for file in files:
            print(f"{file}-{league_ids[file]}")
            csv_to_json(f"{dir_path_file}/{file}.csv", league_id=league_ids[file], season_year=season_year)