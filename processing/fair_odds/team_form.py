"""
team_form.py

Holds each team's recent match history and turns it into recency-weighted
"form" factors for the Dixon-Coles odds model.

For every team we keep only its last `max_matches` finished matches (a
bounded deque, so memory is O(teams * max_matches)). Each stored match has
the team's goals for/against and the pre-match expected goals (xG) for/against.
A match played `d` days before the fixture gets the exponential-decay weight
    w = 0.5 ** (d / half_life_days)
so newer games count more. Form compares what the team actually did with what
the xG model expected, with `prior_goals` of neutral pseudo-evidence so a
couple of matches cannot swing the factor wildly:

    attack_form  = (sum(w * goals_for)     + prior) / (sum(w * xg_for)     + prior)
    defense_form = (sum(w * goals_against) + prior) / (sum(w * xg_against) + prior)

attack_form > 1: scoring more than expected (hot attack);
defense_form > 1: conceding more than expected (leaky defence).
Both are clamped to [min_factor, max_factor] and equal 1.0 with no history.
Defaults: last 5 matches, 30-day half-life, clamp 0.3 to 2.25.
"""

from collections import deque

SECONDS_PER_DAY = 86400.0


class TeamFormTracker:
    """Recency-weighted attack/defence form per team. Feed matches via record(), read via form()."""

    def __init__(self, max_matches: int = 5, half_life_days: float = 30.0,
                 prior_goals: float = 1.0, min_factor: float = 0.3, max_factor: float = 2.25):
        if max_matches <= 0 or half_life_days <= 0 or prior_goals < 0 or min_factor > max_factor:
            raise ValueError("invalid form tracker settings")
        self.max_matches = max_matches
        self.half_life_days = half_life_days
        self.prior_goals = prior_goals
        self.min_factor = min_factor
        self.max_factor = max_factor
        self._history = {}  # team -> deque of (timestamp_seconds, gf, ga, xg_for, xg_against)

    def record(self, team, when, goals_for: float, goals_against: float,
               xg_for: float, xg_against: float) -> None:
        """Remember one finished match for `team`; the oldest match drops out beyond max_matches."""
        history = self._history.setdefault(team, deque(maxlen=self.max_matches))
        history.append((_to_seconds(when), goals_for, goals_against, xg_for, xg_against))

    def recent_matches(self, team) -> list:
        """The stored (timestamp_seconds, gf, ga, xg_for, xg_against) tuples, oldest first."""
        return list(self._history.get(team, ()))

    def weights(self, team, as_of) -> list:
        """Decay weight of each stored match relative to `as_of` (same order as recent_matches)."""
        now = _to_seconds(as_of)
        return [0.5 ** (max(now - m[0], 0.0) / SECONDS_PER_DAY / self.half_life_days)
                for m in self._history.get(team, ())]

    def form(self, team, as_of) -> tuple:
        """(attack_form, defense_form) for `team` using only recorded matches."""
        matches = self._history.get(team)
        if not matches:
            return 1.0, 1.0
        weights = self.weights(team, as_of)
        w_gf = w_ga = w_xf = w_xa = 0.0
        for w, (_, gf, ga, xf, xa) in zip(weights, matches):
            w_gf += w * gf
            w_ga += w * ga
            w_xf += w * xf
            w_xa += w * xa
        attack = self._clamp((w_gf + self.prior_goals) / (w_xf + self.prior_goals))
        defense = self._clamp((w_ga + self.prior_goals) / (w_xa + self.prior_goals))
        return attack, defense

    def _clamp(self, value: float) -> float:
        return max(self.min_factor, min(self.max_factor, value))


def _to_seconds(when) -> float:
    """Seconds since epoch for a pandas Timestamp / datetime."""
    return when.timestamp()
