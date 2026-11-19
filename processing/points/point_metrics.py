"""
point_metrics.py

Shared, reusable building blocks for every points-based feature (cumulative
season points, last-N points, and any future module that needs them).

For each fixture, a team's metrics use ONLY matches that finished before it
kicked off. A win is worth 3 points, a draw 1 and a loss 0. Metrics are
tracked for three scopes:
  - overall : every match the team played
  - home    : only matches the team played at home
  - away    : only matches the team played away
over one of two spans:
  - window=None : everything seen so far (suffix '_before')
  - window=N    : the team's last N matches in that scope (suffix '_last{N}')

Metrics per scope: matches_played, points, points_per_match (2 d.p.), and
points frequency - how many matches earned 3, 1 and 0 points
(freq_3_points, freq_1_point, freq_0_points).

Column names look like 'x_team_home_points_last5' (x = home team of the
fixture, y = away team; the overall scope carries no location tag).

Efficiency: running totals per team/scope are updated in O(1) per match
(adding the new match and, for full windows, removing the oldest). Cumulative
mode stores no match history; windowed mode stores at most N points values.
"""

from collections import deque

import pandas as pd

from processing.core import (
    COMPLETED_STATUSES,
    get_bucket,
    iter_fixture_kickoffs,
    points_bucket,
    points_per_match_bucket,
)
from processing.goals.goal_metrics import DECIMALS, per_match

SCOPES = ("overall", "home", "away")
_SCOPE_PREFIX = {"overall": "", "home": "home_", "away": "away_"}
METRICS = (
    "matches_played",
    "points",
    "points_per_match",
    "freq_3_points",
    "freq_1_point",
    "freq_0_points",
)
# Points value -> slot in the frequency counters (3, 1, 0 points)
_FREQ_SLOT = {3: 0, 1: 1, 0: 2}


def match_points(home_goals: int, away_goals: int) -> tuple:
    """Return (home_points, away_points): 3 for a win, 1 each for a draw, 0 for a loss."""
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def suffix_for(window: int | None) -> str:
    """Column suffix: '_before' for cumulative, '_last{N}' for a window."""
    return "_before" if window is None else f"_last{window}"


def metric_column(side: str, scope: str, metric: str, window: int | None) -> str:
    """Build a column name, e.g. ('x', 'home', 'points', 3) -> 'x_team_home_points_last3'."""
    return f"{side}_team_{_SCOPE_PREFIX[scope]}{metric}{suffix_for(window)}"


class PointsWindow:
    """
    Running points totals for one team in one scope, over all matches
    (maxlen=None) or only the last `maxlen` matches.
    """

    __slots__ = ("maxlen", "games", "n", "points", "freq")

    def __init__(self, maxlen: int | None = None):
        self.maxlen = maxlen
        self.games = deque() if maxlen is not None else None
        self.n = 0
        self.points = 0
        self.freq = [0, 0, 0]

    def _apply(self, pts: int, sign: int) -> None:
        """Add (sign=1) or remove (sign=-1) one match from the running totals."""
        self.n += sign
        self.points += sign * pts
        self.freq[_FREQ_SLOT[pts]] += sign

    def add(self, pts: int) -> None:
        """Record a finished match, evicting the oldest one if the window is full."""
        if self.games is not None:
            if len(self.games) == self.maxlen:
                self._apply(self.games.popleft(), -1)
            self.games.append(pts)
        self._apply(pts, 1)

    def snapshot(self) -> tuple:
        """Current values, ordered exactly like METRICS."""
        return (self.n, self.points, per_match(self.points, self.n), *self.freq)


class TeamPoints:
    """The three scoped PointsWindows (overall / home / away) of one team."""

    __slots__ = ("overall", "home", "away")

    def __init__(self, window: int | None):
        self.overall = PointsWindow(window)
        self.home = PointsWindow(window)
        self.away = PointsWindow(window)

    def snapshot(self) -> tuple:
        """Concatenated snapshots in SCOPES order: overall, home, away."""
        return self.overall.snapshot() + self.home.snapshot() + self.away.snapshot()


def _diff_columns(window: int | None) -> list:
    """Names of the home-minus-away difference/bucket columns (overall scope)."""
    if window is None:
        return ["points_diff_before", "points_diff_bucket",
                "ppm_diff_before", "points_per_match_diff_bucket"]
    sfx = f"_last{window}"
    return [f"points_diff{sfx}", f"points_diff_bucket{sfx}",
            f"points_per_match_diff{sfx}", f"points_per_match_diff_bucket{sfx}"]


def feature_columns(window: int | None) -> list:
    """All output column names for a span, in the order `build_point_features` emits them."""
    columns = []
    for side in ("x", "y"):
        columns += [metric_column(side, scope, m, window) for scope in SCOPES for m in METRICS]
    return columns + _diff_columns(window)


def build_point_features(df: pd.DataFrame, window: int | None = None,
                         history: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return `df` (sorted by date, fixture_id) with all points features added
    for the given span. Fixtures sharing a kickoff time never see each
    other's results.

    `history` is an optional frame of earlier fixtures (same columns) whose
    finished results seed the running totals before `df` is processed, so
    e.g. a new season can continue from a previous one. Its rows are not
    part of the output.
    """
    if window is not None and window <= 0:
        raise ValueError("window must be a positive integer")
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("fixture dates cannot be missing")

    teams: dict = {}
    rows = []
    if history is not None and not history.empty:
        done = history[history["status"].isin(COMPLETED_STATUSES)].sort_values(["date", "fixture_id"])
        for home, away, hg, ag in zip(done["home_team"], done["away_team"],
                                      done["home_goals"], done["away_goals"]):
            x = teams.setdefault(home, TeamPoints(window))
            y = teams.setdefault(away, TeamPoints(window))
            hp, ap = match_points(hg, ag)
            x.overall.add(hp)
            x.home.add(hp)
            y.overall.add(ap)
            y.away.add(ap)

    pts_i, ppm_i = METRICS.index("points"), METRICS.index("points_per_match")
    for _, kickoff in iter_fixture_kickoffs(df):
        finished = []
        for row in kickoff:
            x = teams.get(row.home_team)
            if x is None:
                x = teams[row.home_team] = TeamPoints(window)
            y = teams.get(row.away_team)
            if y is None:
                y = teams[row.away_team] = TeamPoints(window)

            x_snap, y_snap = x.snapshot(), y.snapshot()
            pts_diff = x_snap[pts_i] - y_snap[pts_i]
            ppm_diff = round(x_snap[ppm_i] - y_snap[ppm_i], DECIMALS)
            rows.append(x_snap + y_snap + (
                pts_diff, get_bucket(points_bucket, pts_diff),
                ppm_diff, get_bucket(points_per_match_bucket, ppm_diff),
            ))

            if row.status in COMPLETED_STATUSES:
                if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                    raise ValueError("completed fixtures must include both teams' goals")
                finished.append((x, y, *match_points(row.home_goals, row.away_goals)))

        # Results only become visible once every fixture at this kickoff time is processed
        for x, y, hp, ap in finished:
            x.overall.add(hp)
            x.home.add(hp)
            y.overall.add(ap)
            y.away.add(ap)

    features = pd.DataFrame(rows, columns=feature_columns(window), index=df.index)
    return pd.concat([df, features], axis=1)
