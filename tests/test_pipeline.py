import pytest

from src import pipeline
from src.aspects.aspect_lexicon import ASPECT_KEYWORDS


@pytest.mark.slow
def test_analyze_adds_expected_columns_and_valid_sentiment_labels(sample_df):
    result = pipeline.analyze(sample_df)

    for col in ["clean_light", "clean_deep", "sentiment", "confidence", "aspects"]:
        assert col in result.columns
    for aspect in ASPECT_KEYWORDS:
        assert f"aspect_{aspect}" in result.columns

    assert set(result["sentiment"]).issubset({"positive", "neutral", "negative"})
    assert result["confidence"].between(0.0, 1.0).all()

    # aspect-level sentiment: one entry per tagged aspect, each with a valid label
    assert "aspect_details" in result.columns
    for aspects, details in zip(result["aspects"], result["aspect_details"]):
        assert set(details) == set(aspects)
        assert all(d["sentiment"] in {"positive", "neutral", "negative"} for d in details.values())


@pytest.mark.slow
def test_analyze_reports_monotonic_progress_through_every_stage(sample_df):
    events = []

    pipeline.analyze(sample_df, progress_callback=lambda frac, stage: events.append((frac, stage)))

    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions)
    assert 0.0 <= fractions[0] and fractions[-1] <= pipeline.STAGE_SPANS["Saving results"][0]
    stages = []
    for _, stage in events:
        if not stages or stages[-1] != stage:
            stages.append(stage)
    assert stages == [
        "Cleaning text", "Scoring sentiment", "Tagging aspects", "Scoring aspect sentences",
    ]


@pytest.mark.slow
def test_sentence_mode_separates_mixed_review_end_to_end():
    # Negative overall, but the UX sentence is praise: sentence mode must not inherit the review sentiment.
    import pandas as pd

    df = pd.DataFrame(
        {
            "review_text": [
                "The setup was really easy and the interface is lovely. "
                "But the speaker is awful and it broke after two days, a complete waste."
            ] * 3
            + ["Not easy to use at all. Total waste of money."],
            "rating": [1, 1, 1, 1],
            "date": ["2026-01-01"] * 4,
        }
    )

    result = pipeline.analyze(df)
    mixed, negated = result["aspect_details"].iloc[0], result["aspect_details"].iloc[3]

    assert result["sentiment"].iloc[0] == "negative"
    assert mixed["ux"]["sentiment"] == "positive"
    assert negated["ux"]["sentiment"] == "negative" and negated["ux"]["negated"] is True
