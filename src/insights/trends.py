"""Weekly sentiment aggregation for the dashboard's time view."""

import pandas as pd

MIN_WEEK_REVIEWS = 10

_COLUMNS = [
    "week", "total", "positive", "neutral", "negative",
    "pct_negative", "net_sentiment", "low_sample",
]


def weekly_sentiment(
    df: pd.DataFrame,
    min_reviews: int = MIN_WEEK_REVIEWS,
    date_col: str = "review_date",
) -> pd.DataFrame:
    """Aggregate reviews to weeks (Monday start) with % negative and net sentiment.

    pct_negative and net_sentiment (pos% - neg%) are in percentage points.
    Weeks with fewer than `min_reviews` reviews are kept but flagged
    low_sample=True so callers can mute or drop them. Rows with unparseable
    dates are ignored. Returns an empty frame (with the same columns) if no
    dated rows remain.
    """
    # format="mixed": uploaded CSVs can mix "2026-01-05" and ISO timestamps; without it
    # pandas infers one format from the first row and silently drops the rest as NaT.
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    dated = pd.DataFrame({"date": dates, "sentiment": df["sentiment"]}).dropna(subset=["date"])
    if dated.empty:
        return pd.DataFrame(columns=_COLUMNS)

    dated["week"] = dated["date"].dt.to_period("W-SUN").dt.start_time
    counts = (
        dated.groupby(["week", "sentiment"]).size().unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts["pct_negative"] = 100 * counts["negative"] / counts["total"]
    counts["net_sentiment"] = 100 * (counts["positive"] - counts["negative"]) / counts["total"]
    counts["low_sample"] = counts["total"] < min_reviews

    return counts.reset_index()[_COLUMNS]
