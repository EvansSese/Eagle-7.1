"""Predict upcoming fixtures using a previously trained classifier ensemble.

Run from the repository root, for example:
    python -m training.classification.predict_ensemble
    python -m training.classification.predict_ensemble --league-id 39
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.classification.train_ensemble import ROOT, predict_upcoming_fixtures

LOGGER = logging.getLogger("classifier_ensemble_prediction")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict upcoming fixtures with a saved classifier ensemble."
    )
    parser.add_argument("--league-id", type=int, default=39)
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "storage/database/datasets/England.ddb",
        help="DuckDB database containing the prediction dataset.",
    )
    parser.add_argument(
        "--input-table",
        help="Prediction dataset table (defaults to <league-id>-prediction-dataset).",
    )
    parser.add_argument(
        "--output-table",
        help="Output table (defaults to <league-id>-ensemble-classifier-upcoming-predictions).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Saved model directory (defaults to storage/trained_classifier_models/v3/<league-id>).",
    )
    args = parser.parse_args()

    input_table = args.input_table or f"{args.league_id}-prediction-dataset"
    model_dir = args.model_dir or (
        ROOT / "storage/trained_classifier_models/v3" / str(args.league_id)
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    predictions = predict_upcoming_fixtures(
        db_path=args.database,
        prediction_table=input_table,
        league_id=args.league_id,
        model_dir=model_dir,
        output_table=args.output_table,
    )
    if predictions.empty:
        print(f"No upcoming fixtures found in {input_table!r}.")
    else:
        print(predictions.to_string(index=False))
        LOGGER.info("Generated %d upcoming fixture predictions.", len(predictions))


if __name__ == "__main__":
    main()
