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
