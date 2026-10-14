import pandas as pd
import pytest

from src.insights.trends import MIN_WEEK_REVIEWS, weekly_sentiment


def _reviews():
    # Week of Mon 2026-01-05: 12 reviews (incl. Sunday 01-11 and an ISO-Z timestamp).
    week1 = (
        [("2026-01-05", "positive")] * 5
        + [("2026-01-07T10:00:00.000Z", "positive")]
        + [("2026-01-08", "neutral")] * 2
        + [("2026-01-09", "negative")] * 3
        + [("2026-01-11", "negative")]
    )
    # Week of Mon 2026-01-12: only 3 reviews -> low sample.
    week2 = [("2026-01-12", "negative"), ("2026-01-13", "positive"), ("2026-01-18", "positive")]
    junk = [("not a date", "negative"), (None, "positive")]
    return pd.DataFrame(week1 + week2 + junk, columns=["review_date", "sentiment"])


def test_weekly_aggregation_percentages_and_net_sentiment():
    weekly = weekly_sentiment(_reviews()).set_index("week")

    first = weekly.loc[pd.Timestamp("2026-01-05")]
    assert first["total"] == 12
    assert (first["positive"], first["neutral"], first["negative"]) == (6, 2, 4)
    assert first["pct_negative"] == pytest.approx(100 * 4 / 12)
    assert first["net_sentiment"] == pytest.approx(100 * (6 - 4) / 12)


def test_weeks_start_on_monday_and_unparseable_dates_are_ignored():
    weekly = weekly_sentiment(_reviews())

    assert list(weekly["week"]) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12")]
    assert weekly["total"].sum() == 15  # the 2 junk-date rows are dropped


def test_weeks_below_threshold_are_flagged_low_sample():
    weekly = weekly_sentiment(_reviews()).set_index("week")

    assert bool(weekly.loc[pd.Timestamp("2026-01-05"), "low_sample"]) is False
    assert bool(weekly.loc[pd.Timestamp("2026-01-12"), "low_sample"]) is True
    assert weekly.loc[pd.Timestamp("2026-01-12"), "total"] < MIN_WEEK_REVIEWS


def test_threshold_is_configurable():
    weekly = weekly_sentiment(_reviews(), min_reviews=3)

    assert not weekly["low_sample"].any()


def test_missing_sentiment_class_does_not_break_aggregation():
    df = pd.DataFrame({"review_date": ["2026-01-05"] * 2, "sentiment": ["positive", "positive"]})

    weekly = weekly_sentiment(df)

    assert weekly.loc[0, "negative"] == 0
    assert weekly.loc[0, "pct_negative"] == 0
    assert weekly.loc[0, "net_sentiment"] == 100


def test_no_dated_rows_returns_empty_frame():
    df = pd.DataFrame({"review_date": [None, None], "sentiment": ["positive", "negative"]})

    weekly = weekly_sentiment(df)

    assert weekly.empty
    assert "pct_negative" in weekly.columns
