import pandas as pd
import pytest

from src.preprocessing.clean_text import light_clean, deep_clean, preprocess


def test_light_clean_preserves_case_and_punctuation():
    text = "This is AMAZING!! Loved it."
    cleaned = light_clean(text)
    assert cleaned == text


def test_light_clean_strips_url():
    text = "Check it out at https://example.com/product for details"
    cleaned = light_clean(text)
    assert "https://example.com" not in cleaned
    assert "Check it out at" in cleaned


def test_light_clean_strips_html_tag():
    text = "<b>Great</b> product overall"
    cleaned = light_clean(text)
    assert "<b>" not in cleaned
    assert "</b>" not in cleaned
    assert "Great" in cleaned


@pytest.mark.slow
def test_deep_clean_lowercases_removes_stopwords_and_lemmatizes():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    cleaned = deep_clean("The support team never responded to my complaint", nlp)

    assert cleaned == cleaned.lower()
    assert "the" not in cleaned.split()
    assert "to" not in cleaned.split()
    assert "respond" in cleaned.split()


@pytest.mark.slow
def test_preprocess_adds_expected_columns():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    df = pd.DataFrame({"review_text": ["Delivery was fast and support was helpful."]})

    result = preprocess(df, nlp=nlp)

    assert "clean_light" in result.columns
    assert "clean_deep" in result.columns
    assert "word_count" in result.columns
    assert result["word_count"].iloc[0] == 7
