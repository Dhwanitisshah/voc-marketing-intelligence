import pandas as pd

from src.insights import guards
from src.insights.guards import check_analysis

LONG = "The tablet is great for reading and the battery lasts all day long, delivery was quick too."


def _healthy(n=40):
    return pd.DataFrame(
        {
            "review_text": [LONG] * n,
            "sentiment": ["positive"] * (n - 10) + ["neutral"] * 5 + ["negative"] * 5,
            "aspects": [["price"]] * 10 + [[]] * (n - 10),
            "review_date": ["2026-01-05"] * n,
        }
    )


def test_healthy_analysis_has_no_warnings():
    assert check_analysis(_healthy()) == []


def test_mostly_neutral_triggers_warning_with_mapping_hint():
    df = _healthy().assign(sentiment="neutral")
    df.loc[0, "sentiment"] = "positive"  # 97.5% neutral

    warnings = check_analysis(df)

    assert len(warnings) == 1
    assert "neutral" in warnings[0] and "column mapping" in warnings[0]


def test_ninety_percent_is_the_boundary_for_neutral():
    n = 40
    df = _healthy(n).assign(sentiment=["neutral"] * 36 + ["positive"] * 4)  # exactly 90%

    assert not any("neutral" in w for w in check_analysis(df))


def test_low_aspect_coverage_triggers_warning():
    df = _healthy().assign(aspects=[[] for _ in range(40)])
    df.loc[0, "aspects"] = ["price"]  # 2.5% < 5%

    warnings = check_analysis(df)

    assert len(warnings) == 1 and "aspect keyword" in warnings[0] and "column mapping" in warnings[0]


def test_short_median_text_triggers_warning():
    df = _healthy().assign(review_text="user_0001")

    warnings = check_analysis(df)

    assert any("median review text" in w and "column mapping" in w for w in warnings)


def test_unparseable_dates_trigger_warning_over_20_percent():
    df = _healthy()
    df["review_date"] = ["not a date"] * 10 + ["2026-01-05"] * 30  # 25% fail

    warnings = check_analysis(df)

    assert len(warnings) == 1 and "dates" in warnings[0] and "date column" in warnings[0]


def test_dates_at_or_under_20_percent_are_fine():
    df = _healthy()
    df["review_date"] = ["not a date"] * 8 + ["2026-01-05"] * 32  # exactly 20%

    assert check_analysis(df) == []


def test_missing_date_column_values_skip_the_date_guard():
    df = _healthy().assign(review_date=None)

    assert check_analysis(df) == []


def test_mixed_date_formats_all_parse():
    df = _healthy()
    df["review_date"] = ["2026-01-05", "05/01/2026", "2026-01-05T10:00:00.000Z", "Jan 5, 2026"] * 10

    assert check_analysis(df) == []


def test_share_checks_are_skipped_on_tiny_datasets():
    df = pd.DataFrame(
        {
            "review_text": [LONG] * 5,
            "sentiment": ["neutral"] * 5,
            "aspects": [[] for _ in range(5)],
            "review_date": ["2026-01-05"] * 5,
        }
    )

    assert check_analysis(df) == []  # 5 < MIN_ROWS_FOR_SHARE_CHECKS; text length is still fine


def test_usernames_mapped_as_review_text_trigger_every_text_guard():
    # The real failure this exists for: a "Reviewer" column mapped as the review text.
    n = 200
    df = pd.DataFrame(
        {
            "review_text": [f"user_{i:04d}" for i in range(n)],
            "sentiment": ["neutral"] * n,
            "aspects": [[] for _ in range(n)],
            "review_date": ["2026-01-05"] * n,
        }
    )

    warnings = check_analysis(df)

    assert len(warnings) == 3
    assert all("column mapping" in w for w in warnings)


def test_empty_frame_has_no_warnings():
    assert check_analysis(_healthy().iloc[0:0]) == []


def test_thresholds_are_the_documented_ones():
    assert guards.NEUTRAL_SHARE_MAX == 0.90
    assert guards.ASPECT_COVERAGE_MIN == 0.05
    assert guards.MEDIAN_TEXT_LENGTH_MIN == 20
    assert guards.DATE_FAILURE_SHARE_MAX == 0.20
