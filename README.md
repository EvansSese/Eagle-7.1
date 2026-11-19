# Eagle-7.1
Football outcome prediction using Machine Learning Algorithms.

## Local dashboard

Start the single-user dashboard from the repository root with:

```powershell
python -m app.server
```

Open `http://127.0.0.1:8000` to browse the three-pane match explorer: expand a
country and league in the left pane, choose a season to see round-grouped
fixtures in the center, then select a fixture to inspect its logos and processed
stats on the right. Browsing uses read-only DuckDB connections. To fetch
missing or updated fixtures, select one or more seasons in the left pane.
Fetching uses the existing API client and imports each completed season into
the country-specific database under `storage/database/leagues/`; it requires
`BASE_URL` and `API_KEY` in the repository `.env` file and observes the
existing inter-season API delay.

The league catalogue dashboard feature keeps its database model and HTTP
controller in `app/features/leagues/`; its browser view is part of the shared
dashboard template and JavaScript. `app.server.get_leagues` remains available
for existing callers. Other dashboard features still live in `app/server.py`.

Dashboard assets live in `app/static/`. The explorer, predictions, favorites,
and Model Lab have separate JavaScript and CSS files. `shared.js` contains
shared browser helpers and state; `dashboard.js` starts the page after the
feature scripts load. The stylesheets for shared components, responsive
layouts, typography, and themes are kept separately. Their order in
`app/templates/index.html` is significant for the CSS cascade and for the
deferred scripts' shared bindings.
Run the optional headless browser smoke test with
`$env:EAGLE_BROWSER_TEST="1"; python -m unittest tests.test_dashboard_assets`.

If a temporary local fetch is blocked by a certificate error, you can set
`EAGLE_INSECURE_SKIP_SSL_VERIFY=true` in the environment before starting the
dashboard. This disables TLS certificate and hostname checks for fixture API
requests; use it only as a temporary workaround on a trusted network. The
default is secure verification enabled. First check that the computer's date
and time are correct, since an expired certificate can indicate a server-side
issue.

The **Predictions** screen combines the trained classifiers, goal regressors,
MLP, CNN and RNN. Choose a league and season to
see fixtures grouped by round and ordered by kickoff, their top 3-way and double-chance
picks, and the most likely scoreline. Filter by played fixtures (FT, AET, PEN,
AWD, or WO), not-started fixtures (NS, including those past their scheduled
kickoff), or all statuses. Dates not yet known appear last. Select a fixture for ranked prediction
probabilities and processed feature stats. Create named favorite lists and
activate any number of them to save a selected fixture to every active list.
Lists and their fixtures are stored in the league catalog database and remain
available across dashboard sessions; lists start inactive each time the
dashboard loads. Prepare the league's all-season dataset in Model Lab after
fixture or feature updates and train all models before requesting predictions.
Fixtures without a known kickoff use empty sequence histories.
Predictions for played fixtures are retrospective:
models may have been exposed to related data, so these predictions are not an
reliable independent measure of model performance. Confidence is not a guarantee.
The **test** badge means a fixture appears in at least one participating model's
recorded test set, not necessarily every model's; missing test records are shown
as unknown. Round counters show correct top 3-way and double-chance picks over
all FT fixtures with valid scores and the respective prediction in that round,
independent of the status filter. Fixture analysis also evaluates the top double-chance pick. NS matches
are marked not yet played; other statuses are not evaluated.

Set `EAGLE_LEAGUES_DB` to use another league database, or set
`EAGLE_COUNTRY_DATABASES` to use another country-database directory. Set
`EAGLE_DATASET_DATABASES` to use another dataset-database directory. Set
`EAGLE_DASHBOARD_HOST` and `EAGLE_DASHBOARD_PORT` to change the local server
address.

The dashboard's **Model lab** presents a country/league browser beside the
processing and training workflow. After selecting a league, you can
fetch and import fixtures, process feature tables, and prepare training and
prediction datasets for all available seasons, then independently train/evaluate the
classification ensemble and goal-regression models. Operations run as
background jobs with explicit status and chronological holdout metrics.
The readiness table reports feature tables, datasets, and saved model artifacts
for the league.
Select workflow cards and use **Run selected**, or use **Run all** to run all
six steps in the displayed order. The queue stops on the first failure; later steps
are not run. Successful steps show a green check and can be re-run individually.
Progress shows the active step and its known progress. Completion history is
scoped to the league and retained while the server is running; it is
cleared on server restart.
The **Train neural networks** step trains the MLP on the prepared dataset and
the CNN, RNN, LSTM and autoencoder on raw team fixture histories, saving their
holdout metrics and upcoming predictions alongside their model artifacts.
After the classifier, regressor and neural artifacts are trained, choose
a season in the Predictions screen to generate 3-way, double-chance, and score
predictions for its fixtures.
Classifier artifacts are saved under
`storage/trained_classifier_models/v3/<league_id>` and regression artifacts
under `storage/trained_regressor_models/v3/<league_id>`. Fetching requires the API
credentials described above. Dataset tables are written to
`storage/database/datasets/<country>.ddb` by default.
Feature processing keeps one connection to the country database open across all
requested seasons and closes it when the run finishes.

The classification dataset labels only completed fixtures with valid scores
(`FT`, `AET`, or `PEN`); upcoming fixtures keep a null result and are excluded
from training. Feature histories use information available before each kickoff,
and classifier evaluation holds out the latest fixtures chronologically.

To compare regression models for home and away goals, set the configuration
values in the `if __name__ == "__main__"` block of
`training/regression/train_goal_models.py`, then run
`python -m training.regression.train_goal_models`. The script excludes
unstarted/unscored fixtures, makes a chronological train/test split, selects a
feature bundle for each model using a validation period inside the training
data, and reports score-regression metrics plus win/draw/loss accuracy from
rounded score predictions. It saves the prepared train/test datasets, model
comparison, selected features, and model-specific test predictions as tables
in the `storage/database/datasets/England.ddb` DuckDB database by default.
Fitted models are saved under `storage/trained_regressor_models/v3/39`. The
tables use the `39-goal-regression-*` naming prefix. The final test period is
not used to select features. Outcome accuracy is measured, not guaranteed to
reach any particular threshold. Upcoming `predicted_outcome` and
`predicted_double_chance` use the rounded-mean score forecast; the double-chance
pick always includes that forecast's outcome and selects its companion outcome
using validation-weighted model support. `predicted_outcome_weighted_vote`
preserves the separate validation-weighted vote result. The
`*_outcome_votes` columns remain unweighted model counts, while
`*_outcome_weighted_votes` columns store the weighted totals. Upcoming
predictions also include the alternative Poisson scoreline forecast in
`predicted_home_goals_poisson`, `predicted_away_goals_poisson`, and
`predicted_outcome_poisson`, using the validation-weighted mixture of goal-count
probabilities. The Poisson distribution also produces Over/Under picks and
Over probabilities for total-goal lines 0.5, 1.5, 2.5, and 3.5 in
`predicted_over_under_<line>` and `over_probability_<line>` columns (with the
decimal point represented by an underscore). The Poisson scoreline remains an
alternative for correct-score predictions, while the rounded-mean forecast
remains the primary 3-way and double-chance source. The chronological test
metrics include the accuracy of both forecast approaches and each Over/Under
line.
The existing `predicted_home_goals` and `predicted_away_goals` are the
rounded-mean forecast.

The Model lab can also run the classifier and regression trainers for the
selected league against the prepared training dataset. Both report their
chronological holdout metrics and save model artifacts under the selected
league's model directory.

To train the validation-selected classification ensemble, save its selected
models and parameters, evaluate the chronological holdout, and predict upcoming
fixtures, run `python -m training.classification.train_ensemble` from the
repository root. The script writes model artifacts under
`storage/trained_classifier_models/v3/<league_id>` and stores holdout metrics,
parameters, and predictions plus upcoming predictions in DuckDB. Its weighted
3-way vote includes per-outcome probabilities, model vote counts, and
probability-based confidence (0–1) for both the 3-way and double-chance picks,
along with the supporting raw and validation-weighted model vote totals.
To regenerate upcoming predictions from the saved models without retraining,
run `python -m training.classification.predict_ensemble`. Options include
`--league-id`, `--database`, `--input-table`, `--output-table`, and `--model-dir`.

To train neural networks for 3-way outcomes and goal scores, evaluate them on a
chronological holdout, then predict the prepared upcoming fixtures, run
`python -m training.deep_learning.train_match_networks --league-id 122 --db-path storage/database/datasets/Denmark.ddb`.
The trainer derives double
chance from the two most likely outcome classes and rounds neural goal
regression into a correct-score pick. It saves the models and metadata under
`storage/trained_deep_learning_models/v1/<league_id>` and writes holdout metrics,
test predictions, and upcoming predictions to the dataset database tables
`<league_id>-deep-learning-metrics`,
`<league_id>-deep-learning-test-predictions`, and
`<league_id>-deep-learning-upcoming-predictions`. The database must contain the
prepared `<league_id>-training-dataset` and `<league_id>-prediction-dataset`
tables.

To compare a CNN and RNN on raw team match histories, install TensorFlow with
`python -m pip install -r requirements.txt`, then run
`python -m training.deep_learning.train_sequence_models --league-id 122`. This
uses the league's raw season fixture tables and the previous 10 matches for
each team, with a chronological holdout. It saves per-model holdout metrics,
test and upcoming predictions in the dataset database, and trained Keras models
under `storage/trained_deep_learning_models/sequence/<league_id>`. Override
`--country-database` or `--dataset-database` when using non-default database
locations.

The LSTM and autoencoder use each team's previous 10 completed
matches (goals for and against, venue, points and time since match). The
autoencoder first learns to reconstruct training histories, then its encoder
is fine-tuned for 3-way classification and home/away goal prediction. Double
chance uses the two most likely outcomes; exact scores round the predicted
goals to the nearest non-negative integers. A grid of 16/32 hidden units and
0/0.25 dropout is selected by validation log loss on a later chronological
period; a separate later holdout reports outcome, double-chance and exact-score
accuracy, log loss and goal MAE. The grid also records validation goal MAE
and double-chance and exact-score accuracy; selection still prioritizes
3-way outcome log loss. Simultaneous
fixtures cannot see each other's results. Final models are refit on all
completed fixtures for upcoming predictions. To train these separately, run
`python -m training.deep_learning.train_outcome_networks --league-id 122 --country-database storage/database/leagues/Denmark.ddb --dataset-database storage/database/datasets/Denmark.ddb --model-dir storage/trained_deep_learning_models/sequence/122`.
The dataset database stores `<league_id>-outcome-network-grid`,
`<league_id>-outcome-network-metrics`, and the corresponding
`<league_id>-outcome-network-test-predictions` and
`<league_id>-outcome-network-upcoming-predictions` tables.
Retrain existing LSTM and autoencoder models to add score predictions; older
outcome-only models remain usable for outcome and double-chance ensembles.
TensorFlow uses a compatible GPU automatically when available; native Windows
TensorFlow 2.11 and later normally requires WSL2 for GPU acceleration.

To combine all saved classifiers, goal regressors, and neural models and
measure each model's contribution, run
`python -m training.ensemble_predict --league-id 122`. Each model contributes
within its family, and the classifier, goal-regression, and neural families
then receive equal weight in the final 3-way and double-chance probabilities.
Goal-regressor predictions are converted to outcomes by summing independent
Poisson scoreline probabilities, rather than assigning a fixed probability to
the outcome implied by rounded scores. Score forecasts still average the
available goal-model predictions. The script compares each model's
chronological holdout predictions on their shared fixtures and reports changes
in accuracy, log loss, and draw precision/recall when that model is removed.
Positive impact means the model improves the corresponding ensemble metric. It writes
upcoming predictions and the ablation report to
`<league_id>-all-model-upcoming-predictions` and
`<league_id>-all-model-importance` in the dataset DuckDB file. All classifier,
regressor, MLP, CNN, and RNN artifacts must already be trained for the league;
LSTM and autoencoder models are included when their metadata is available.

To predict upcoming fixtures after training, set the league, season, database,
input table, and optional model name in the `if __name__ == "__main__"` block
of `training/regression/predict_upcoming_fixtures.py`, then run
`python -m training.regression.predict_upcoming_fixtures`. The script selects
the highest-outcome-accuracy model from the stored model comparison table when
`MODEL_NAME` is `None`, loads its saved model and selected feature list, and replaces the
`39-2026-upcoming_fixtures_predictions` table in the datasets database.
Only future `NS` and `TBD` fixtures are included.
