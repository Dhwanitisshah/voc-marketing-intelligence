import pytest

from src.sentiment.vader_model import score_vader, _label_from_compound


def test_label_from_compound_thresholds():
    assert _label_from_compound(0.05) == "positive"
    assert _label_from_compound(1.0) == "positive"
    assert _label_from_compound(-0.05) == "negative"
    assert _label_from_compound(-1.0) == "negative"
    assert _label_from_compound(0.0) == "neutral"
    assert _label_from_compound(0.049) == "neutral"
    assert _label_from_compound(-0.049) == "neutral"


def test_score_vader_strongly_positive_text():
    results = score_vader(["I absolutely love this product, it's fantastic and amazing!"])
    compound, label = results[0]
    assert compound >= 0.05
    assert label == "positive"


def test_score_vader_strongly_negative_text():
    results = score_vader(["This is terrible, awful, and completely broken. I hate it."])
    compound, label = results[0]
    assert compound <= -0.05
    assert label == "negative"


def test_score_vader_neutral_text():
    results = score_vader(["The package arrived on Tuesday."])
    compound, label = results[0]
    assert -0.05 < compound < 0.05
    assert label == "neutral"


def test_score_vader_handles_empty_text():
    results = score_vader(["", None])
    assert results == [(0.0, "neutral"), (0.0, "neutral")]


@pytest.mark.slow
def test_transformer_model_scores_text():
    from src.sentiment.transformer_model import score_transformer

    results = score_transformer(["I love this, best purchase ever!"])
    label, score = results[0]
    assert label in {"positive", "neutral", "negative"}
    assert 0.0 <= score <= 1.0
