# Outcome probability and consistency

The `<league>-<season>-outcome-probability` DuckDB table has one row per
fixture. The `x_team` and `y_team` prefixes refer to the home and away team,
respectively. Each has `overall`, `home_only`, `away_only`, `last_3`,
`last_5`, and `last_10` columns, ending in `_before`. Each scope reports
`matches`, `win_probability`, `draw_probability`, `loss_probability`,
`consistency`, `streak_outcome`, `streak_length`, and `streak_observations`.
Venue scopes use only matches at that venue; last-N scopes use the latest N
matches at either venue. All histories reset at the start of each season.

Only completed fixtures strictly before the kickoff contribute. Fixtures at the
same kickoff cannot see one another's results. The earliest round in each
season always uses the same empty-history prior, including postponed fixtures.
With no round column, a team's first match uses the prior. Unknown consistency
and streak outcome are NULL; consistency requires at least two results.

For each scope, the baseline is the observed win/draw/loss frequency smoothed
with one prior observation of each outcome. Consistency is the proportion of
adjacent results that are identical; it measures repeatability, not strength.
Streak length counts consecutive identical outcomes at the end of that scope.
The model finds earlier instances in the scope where the same outcome had
reached at least the current streak length, and counts the outcomes immediately
following those instances. It blends these counts with three baseline-equivalent
prior observations. `streak_observations` gives the number of such successors.
If there is no evidence about the current streak, the estimate stays at the
baseline. A winning streak does not automatically imply a loss or draw:
that assumption requires empirical validation and is not imposed by the model.
The smoothing strengths are heuristics, not fitted or calibrated parameters.

Probabilities and consistency are decimal fractions rounded to two places.
Rounding each independently means a probability triplet may sum to 0.99 or
1.01, including the equal 0.33/0.33/0.33 prior. The legacy
`x_team_*_probability_before` and `y_team_*_probability_before` fields alias
the overall estimates. Combined probabilities average the corresponding home
and away estimates; they are descriptive and not a calibrated match model.
For compatibility, `predicted_outcome` remains the *least* probable combined
outcome before rounding, with exact ties resolved home win, draw, then away win;
it is not the most likely prediction. Reprocessing replaces the outcome table; existing H2H
tables lacking these scoped metrics are rebuilt.
