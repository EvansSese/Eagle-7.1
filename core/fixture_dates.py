"""Normalize fixture kickoffs for chronological processing."""

import pandas as pd


def fixture_dates_utc(values: pd.Series, *, errors: str = "raise") -> pd.Series:
    """Return UTC-naive kickoffs; historical naive fixture timestamps represent UTC."""
    return pd.to_datetime(values, utc=True, errors=errors).dt.tz_localize(None)


def fixture_time_utc(value=None) -> pd.Timestamp:
    """Return a UTC-naive instant, interpreting a naive supplied value as UTC."""
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)
