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
    assert stages == ["Cleaning text", "Scoring sentiment", "Tagging aspects"]
