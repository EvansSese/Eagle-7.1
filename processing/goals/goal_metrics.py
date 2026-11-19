"""
goal_metrics.py

Shared, reusable building blocks for every goal-based feature (cumulative
season goals, last-N goals, and any future module that needs them).

For each fixture, every team's metrics are computed ONLY from matches that
finished before that fixture kicked off. Each metric is tracked for three
scopes:
  - overall : every match the team played
  - home    : only matches the team played at home
  - away    : only matches the team played away
and over one of two spans:
  - window=None : everything seen so far (cumulative)
  - window=N    : only the team's last N matches in that scope

Metrics per scope (column names are built by `metric_column`):
  matches_played, goals_scored, goals_conceded,
  goals_scored_per_match, goals_conceded_per_match,
  scored_1_goal, scored_2_goals, scored_3_goals, scored_above_3_goals,
  conceded_1_goal, conceded_2_goals, conceded_3_goals, conceded_above_3_goals,
  clean_sheets

Efficiency: every team/scope keeps running totals that are updated in O(1)
per match (a finished match is added, and the oldest one is subtracted once
a window is full). Cumulative mode stores no per-match history at all, and
windowed mode stores at most `window` (scored, conceded) pairs.
"""

from collections import deque

import pandas as pd

from processing.core import (
    COMPLETED_STATUSES,
    get_bucket,
    goals_bucket,
    goals_per_match_bucket,
    iter_fixture_kickoffs,
)

DECIMALS = 2
SCOPES = ("overall", "home", "away")
_SCOPE_PREFIX = {"overall": "", "home": "home_", "away": "away_"}
FREQUENCY_LABELS = ("1_goal", "2_goals", "3_goals", "above_3_goals")
METRICS = (
    "matches_played",
    "goals_scored",
    "goals_conceded",
    "goals_scored_per_match",
    "goals_conceded_per_match",
    *(f"scored_{label}" for label in FREQUENCY_LABELS),
    *(f"conceded_{label}" for label in FREQUENCY_LABELS),
    "clean_sheets",
)


def per_match(total: float, matches: int, decimals: int = DECIMALS) -> float:
    """Return total / matches rounded to `decimals`; 0.0 when no matches were played."""
    return round(total / matches, decimals) if matches > 0 else 0.0


def frequency_index(goals: int) -> int | None:
    """
    Map a single match's goals to a frequency slot: 1, 2, 3 goals -> 0, 1, 2
    and more than 3 goals -> 3. A blank (0-goal) match has no slot (None).
    """
    return None if goals <= 0 else min(goals, 4) - 1


def suffix_for(window: int | None) -> str:
    """Column suffix for a span: '' for cumulative, '_last{N}' for a window."""
    return "" if window is None else f"_last{window}"


def metric_column(side: str, scope: str, metric: str, window: int | None) -> str:
    """
    Build a column name, e.g. ('x', 'home', 'goals_scored', 3) ->
    'x_team_home_goals_scored_last3'. `side` is 'x' (home team of the
    fixture) or 'y' (away team); the 'overall' scope has no location tag.
    """
    return f"{side}_team_{_SCOPE_PREFIX[scope]}{metric}{suffix_for(window)}"


class GoalWindow:
    """
    Running goal totals for one team in one scope, over all matches
    (maxlen=None) or only the last `maxlen` matches.
    """

    __slots__ = ("maxlen", "games", "n", "scored", "conceded", "clean_sheets", "freq_scored", "freq_conceded")

    def __init__(self, maxlen: int | None = None):
        self.maxlen = maxlen
        self.games = deque() if maxlen is not None else None
        self.n = 0
        self.scored = 0
        self.conceded = 0
        self.clean_sheets = 0
        self.freq_scored = [0, 0, 0, 0]
        self.freq_conceded = [0, 0, 0, 0]

    def _apply(self, scored: int, conceded: int, sign: int) -> None:
        """Add (sign=1) or remove (sign=-1) one match from the running totals."""
        self.n += sign
        self.scored += sign * scored
        self.conceded += sign * conceded
        if conceded == 0:
            self.clean_sheets += sign
        slot = frequency_index(scored)
        if slot is not None:
            self.freq_scored[slot] += sign
        slot = frequency_index(conceded)
        if slot is not None:
            self.freq_conceded[slot] += sign

    def add(self, scored: int, conceded: int) -> None:
        """Record a finished match, evicting the oldest one if the window is full."""
        if self.games is not None:
            if len(self.games) == self.maxlen:
                self._apply(*self.games.popleft(), -1)
            self.games.append((scored, conceded))
        self._apply(scored, conceded, 1)

    def snapshot(self) -> tuple:
        """Current values, ordered exactly like METRICS."""
        return (
            self.n,
            self.scored,
            self.conceded,
            per_match(self.scored, self.n),
            per_match(self.conceded, self.n),
            *self.freq_scored,
            *self.freq_conceded,
            self.clean_sheets,
        )


class TeamGoals:
    """The three scoped GoalWindows (overall / home / away) of one team."""

    __slots__ = ("overall", "home", "away")

    def __init__(self, window: int | None):
        self.overall = GoalWindow(window)
        self.home = GoalWindow(window)
        self.away = GoalWindow(window)

    def snapshot(self) -> tuple:
        """Concatenated snapshots for SCOPES order: overall, home, away."""
        return self.overall.snapshot() + self.home.snapshot() + self.away.snapshot()


def feature_columns(window: int | None) -> list:
    """All output column names for a span, in the order `build_goal_features` emits them."""
    columns = []
    for side in ("x", "y"):
        columns += [metric_column(side, scope, m, window) for scope in SCOPES for m in METRICS]
    pm = "pm" if window is None else "per_match"
    sfx = suffix_for(window)
    columns += [
        f"goals_scored_diff{sfx}", f"goals_scored_diff_bucket{sfx}",
        f"goals_conceded_diff{sfx}", f"goals_conceded_diff_bucket{sfx}",
        f"goals_scored_{pm}_diff{sfx}", f"goals_scored_{pm}_diff_bucket{sfx}",
        f"goals_conceded_{pm}_diff{sfx}", f"goals_conceded_{pm}_diff_bucket{sfx}",
    ]
    return columns


def build_goal_features(df: pd.DataFrame, window: int | None = None,
                        history: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return `df` (sorted by date, fixture_id) with all goal features added
    for the given span. Fixtures sharing a kickoff time never see each
    other's results. The overall-scope scored/conceded totals and
    per-match rates also get home-minus-away difference columns and
    buckets.

    `history` is an optional frame of earlier fixtures (same columns) whose
    finished results seed the running totals before `df` is processed, so
    e.g. a new season can continue from a previous one. Rows of `history`
    are not part of the output.
    """
    if window is not None and window <= 0:
        raise ValueError("window must be a positive integer")
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")

    teams: dict = {}
    rows = []
    if history is not None and not history.empty:
        finished_history = history[history["status"].isin(COMPLETED_STATUSES)].sort_values(["date", "fixture_id"])
        for home, away, hg, ag in zip(finished_history["home_team"], finished_history["away_team"],
                                      finished_history["home_goals"], finished_history["away_goals"]):
            x = teams.setdefault(home, TeamGoals(window))
            y = teams.setdefault(away, TeamGoals(window))
            x.overall.add(int(hg), int(ag))
            x.home.add(int(hg), int(ag))
            y.overall.add(int(ag), int(hg))
            y.away.add(int(ag), int(hg))
    # Position of the overall goals_scored / goals_conceded / per-match values in a snapshot
    gs_i, gc_i, gspm_i, gcpm_i = (METRICS.index(m) for m in (
        "goals_scored", "goals_conceded", "goals_scored_per_match", "goals_conceded_per_match"))

    for _, kickoff in iter_fixture_kickoffs(df):
        finished = []
        for row in kickoff:
            home, away = row.home_team, row.away_team
            x = teams.get(home)
            if x is None:
                x = teams[home] = TeamGoals(window)
            y = teams.get(away)
            if y is None:
                y = teams[away] = TeamGoals(window)

            x_snap, y_snap = x.snapshot(), y.snapshot()
            gs_diff = x_snap[gs_i] - y_snap[gs_i]
            gc_diff = x_snap[gc_i] - y_snap[gc_i]
            gspm_diff = round(x_snap[gspm_i] - y_snap[gspm_i], DECIMALS)
            gcpm_diff = round(x_snap[gcpm_i] - y_snap[gcpm_i], DECIMALS)
            rows.append(x_snap + y_snap + (
                gs_diff, get_bucket(goals_bucket, gs_diff),
                gc_diff, get_bucket(goals_bucket, gc_diff),
                gspm_diff, get_bucket(goals_per_match_bucket, gspm_diff),
                gcpm_diff, get_bucket(goals_per_match_bucket, gcpm_diff),
            ))

            if row.status in COMPLETED_STATUSES:
                if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                    raise ValueError("completed fixtures must include both teams' goals")
                finished.append((x, y, int(row.home_goals), int(row.away_goals)))

        # Results only become visible once every fixture at this kickoff time is processed
        for x, y, home_goals, away_goals in finished:
            x.overall.add(home_goals, away_goals)
            x.home.add(home_goals, away_goals)
            y.overall.add(away_goals, home_goals)
            y.away.add(away_goals, home_goals)

    features = pd.DataFrame(rows, columns=feature_columns(window), index=df.index)
    return pd.concat([df, features], axis=1)
