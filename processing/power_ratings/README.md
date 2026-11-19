# Football power ratings

The season-processing workflow writes `<league>-<season>-power-ratings` to the
country DuckDB database, with one uniquely keyed row per fixture. `x_team` and
`y_team` denote the home and away teams. Every `*_before` metric is calculated
before the fixture's kickoff; only completed fixtures (`FT`, `AET`, `PEN`)
strictly earlier in the same league and season contribute. Same-kickoff
fixtures cannot see one another's results. Unplayed fixtures do not update
ratings, and all teams start each season at 10.

The overall rating implements Paul Steele's method as described in
[Testing the Profitability of Football Power Ratings](https://football-data.co.uk/blog/power_ratings.php):

```
home_change = 0.25 * (home_goals - away_goals
                      - (home_rating - away_rating) - 0.5)
away_change = -home_change
```

The match power rating is `home_rating - away_rating`, following the linked
[ratings guide](https://betting.football-data.co.uk/ratings.php). The separate
`expected_goal_margin_before` adds 0.5 as in the update formula; both are
home-oriented and neither is a probability. The article's
profitability test excludes matches until **both** teams have played at least
five games; `both_teams_at_least_5_matches_before` exposes this criterion
without dropping fixtures. The article does not establish profitable betting.

The article does not define the other scopes. These are separate extensions:
for each team and scope, start at 10 and replay only the team's eligible prior
matches in chronological order. Each historical update uses the opposing
team's pre-match rating in the **same scope**, recorded at that historical
kickoff. Home/away advantage has the appropriate sign for the team's venue.
`home_only` and `away_only` select that team's home and away fixtures,
respectively; `last_3`, `last_5`, and `last_10` select its latest completed
fixtures at either venue. Rolling windows are replayed from 10 whenever a
fixture leaves the window, rather than subtracting old rating changes.
At the same timestamp, replayed changes use the same starting rating.
These scoped ratings do not necessarily conserve the total rating across
teams because each team's selected history differs. Each scoped match rating
compares both teams' ratings in that same scope. The additional `venue` match
rating compares the home team's `home_only` rating with the away team's
`away_only` rating; its expected goal margin also adds 0.5.

For each scope, columns include each team's `power_rating_before` and
`matches_before`, `match_power_rating_before`, and `expected_goal_margin_before`.
Ratings retain full
floating-point precision so subsequent calculations are not affected by
intermediate rounding. These values are not calibrated outcome probabilities;
fitting any probability model requires separate chronological validation.
