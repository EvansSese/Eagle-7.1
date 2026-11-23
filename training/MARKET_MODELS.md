# Chronological market-model training

Use **Train market models** in Model Lab after processing all seasons and
preparing datasets. The card's market checkboxes select the targets trained
for every league in a batch; all targets are selected by default. Unselected
markets are not retrained or replaced. The full grid is computationally
expensive. Alternatively:

```powershell
.\.venv\Scripts\python.exe -m training.market_models --league-id 39 --dataset-database "storage\database\datasets\England.ddb" --country-database "storage\database\leagues\England.ddb"
```

The pipeline uses the prepared training, validation and test tables without
resplitting them. Each target is trained on the training split, chosen using
validation log loss for classification or MAE for regression, then scored
once on the test split. Selected models are not refitted on validation.
It compares rate/form features with all pre-match aggregate features, covering
goals, points, xG, odds, outcome probabilities, ratings and H2H. Season and
low-cardinality historical outcome fields are one-hot encoded. Fixture and
team identifiers, team names, venue, scores and result are not model inputs.
For matching numeric team metrics, the pipeline also includes x-team minus
y-team differences within the same scope and home-team home-scope minus
away-team away-scope differences where available. Missing operands remain
missing until training-only imputation.
Numeric missing and infinite values are imputed using training-set medians
with missingness indicators; categorical missing values get their own category.
Numerical scaling is fitted on training only for linear, SVM and feedforward
neural models. Tree models do not use scaling.

The traditional grid includes random forest, XGBoost, LightGBM, CatBoost,
logistic regression and probabilistic SVM for classifiers. Ridge regression and
SVR are their regression counterparts. The neural grid includes MLP, DNN,
LSTM, GRU, causal dilated TCN and Transformer models. Each algorithm has two
parameter choices. Sequential models use each team's ten earlier completed
fixtures, with goals for/against, venue, points, days since the match and a
presence indicator. Team names identify histories but are never input features.
Same-kickoff results do not enter each other's histories.

SVM probabilities use sigmoid calibration with training-only folds. Score
classes with a single training example remain in every training fold but have
no held-out examples for calibration, so their probabilities are not
reliably calibrated; their count is logged. Calibration fails explicitly if
there are not enough repeated examples to form at least two folds. Logistic
regression is allowed up to 2,000 iterations. Any candidate that still emits a
convergence warning is logged and recorded as `not_converged`, and cannot be
selected.

Targets are 1X2, BTTS, over 0.5/1.5/2.5/3.5, home goals, away goals, total
goals and exact score. The `training.market_targets.derived_market_probabilities`
helper derives double-chance probabilities from 1X2 and under probabilities
as complements of over probabilities. It does not force separately trained
over probabilities to be monotonic. Correct-score classes unseen during
training receive an effective probability of 1e-15 when calculating log loss;
the number of such fixtures is recorded in evaluation metrics.

Markets are trained in order: result, BTTS, over thresholds, correct score,
home goals, away goals and total goals. Each completed market is published
immediately, including predictions for every fixture in the current-season
prediction dataset, even fixtures already played. Model Lab displays these
predictions while the remaining models train. The Predictions screen displays
the current-season saved markets on each fixture and refreshes them while
open; it does not use legacy ensemble models for chronological datasets.
Double-chance and under
probabilities are included alongside the corresponding result and over
probabilities. The prediction dataset must have unique fixture IDs and valid
dates and team names matching the raw fixtures.

Each market appends its selected model and validation/test metrics to
`<league>-market-model-runs`, candidate validation scores to
`<league>-market-model-candidates`, and predictions to
`<league>-market-predictions` in the datasets database. Model Lab and
`GET /api/market-predictions?league_id=39` display only predictions from the
current dataset preparation. Re-preparing datasets invalidates the displayed
predictions until markets are retrained. A dataset change during training
causes publication to fail rather than associate predictions with new data.

Each current model is stored as
`storage\trained_market_models\v4\<league_id>\<target>.joblib`, replacing the
previous model for that market. The bundle includes its metadata and fitted
preprocessor, plus either the scikit-learn pipeline or serialized Keras model.
Use `training.market_models.load_market_artifact(path)` to load it, including
the custom recurrent layers. Historical run metadata is retained, but its
fixed artifact path points to the latest model, not an archived copy.
Sequence models use the six fixed history features documented above and
require a history tensor at inference. A failed database write restores the
previous artifact; an abrupt process termination between file replacement and
database commit can still leave them out of sync. Avoid concurrent training
from separate processes for the same league.

Existing legacy prediction actions do not consume these artifacts. Repeated
test evaluation across multiple model runs can bias comparisons; reserve a
newer holdout before making a final production assessment.
