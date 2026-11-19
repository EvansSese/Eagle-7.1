# Head-to-head features

Season processing rebuilds `<league>-<season>-h2h` for every available season
and `<league>-h2h` for the whole league. Both are keyed by `fixture_id`, retain
the fixture's original `season` and `round`, and contain pre-kickoff features.
Only completed (`FT`, `AET`, `PEN`) meetings between the two teams in the
selected league contribute to team metrics. Same-kickoff fixtures cannot see
one another; unfinished matches never update history. Cross-competition matches
are excluded.

Columns beginning `h2h_all_time_` use all earlier seasons and meetings;
`h2h_season_` resets at the start of each season. Both have `overall`,
`home_only`, and `away_only` scopes. All-time also has `last_3`, `last_5`, and
`last_10`, counting the most recent meetings at either venue. `x_team` and
`y_team` refer to the current fixture's home and away sides, respectively;
home/away-only describe the historical venue of the team, not the current
fixture. Counts, points, goal totals and frequencies are from those meetings.

Outcome probabilities, consistency, and streaks reuse the outcome module's
smoothed frequencies and observed-transition method. Combined outcome
probabilities average the relevant two team estimates; `least_probable_outcome`
uses the source module's least-probable-event convention and is not a prediction
of the most likely result. Power ratings begin at 10
for each matchup and span and use the power-ratings module's update; recent
windows replay only their selected meetings against the opponents' previously
recorded same-scope ratings. Neither metric incorporates meetings against
other opponents.

xG uses the existing attack/defence formula with the selected H2H scope's
per-team goals-per-match rates. The denominator is the league-wide home/away
goals average for the immediately previous round of the current season,
computed only from completed fixtures with an earlier kickoff. Thus the
league baseline is the deliberate exception to pair-only inputs. A missing
or zero baseline produces zero strength and xG, following the existing xG
module. Fair odds use the shared Dixon-Coles score grid and its default rho,
without team-form adjustment. When both xG values are zero the existing model
returns a deterministic 0-0 prediction and its zero-probability odds sentinel;
this is not evidence that a draw is certain.

The existing unprefixed H2H columns used by training remain available as
aliases of all-time overall and last-N metrics. Their xG and fair-odds values
now use the corresponding H2H scope and the league baseline above rather than
the previous pair-only round baseline; models trained on older tables should
be rebuilt. Probabilities are not calibrated and may be unreliable with few
meetings. Goal and point rates and outcome probabilities retain their source
modules' rounding; power ratings retain full precision.
