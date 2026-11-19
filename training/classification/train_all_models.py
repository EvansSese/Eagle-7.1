from pathlib import Path
from core.get_league_details import get_league_details
from classification.models import train_dt, train_gbc, train_knn, train_nb, train_rf, train_lr, train_svm

if __name__ == "__main__":
    league_ids = [62] #119,203,207,208,61, --62,89,(with errors) 140,141,135,136,78,357,110,39,113,94,122
    leagues_db_path = "../../storage/database/leagues.ddb"
    for league_id in league_ids:
        league_details = get_league_details(leagues_db_path=leagues_db_path,league_id=league_id)
        print(league_details)
        if league_details["status"]:
            db_path = f"../../storage/database/datasets/{league_details['league_details']['country_name']}.ddb"
            input_table = f"{league_id}-training-dataset"
            model_path = Path(f"../../storage/trained_classifier_models/v2/{league_id}/")
            train_dt.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
            train_gbc.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
            train_knn.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
            train_lr.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
            train_nb.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
            train_rf.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
            train_svm.main(db_path=db_path, input_table=input_table, league_id=league_id, model_path=model_path)
        else:
            print(f"{league_id} League not found")
