import pandas as pd

from src.aspects.aspect_tagger import tag_aspects, tag_dataframe
from src.aspects.aspect_lexicon import ASPECT_KEYWORDS


def test_tags_delivery_and_support_from_lemma_tokens():
    # "clean_deep"-style text: lowercased, lemmatized, stopwords removed.
    text = "delivery late refund never come"
    assert set(tag_aspects(text)) == {"delivery", "support"}


def test_no_keyword_matches_returns_empty_list():
    text = "beautiful morning sunshine happy day"
    assert tag_aspects(text) == []


def test_tag_dataframe_creates_correct_one_hot_columns():
    df = pd.DataFrame(
        {
            "clean_deep": [
                "delivery late arrive",
                "price expensive cost",
                "beautiful sunshine",
            ]
        }
    )

    result = tag_dataframe(df)

    for aspect in ASPECT_KEYWORDS:
        assert f"aspect_{aspect}" in result.columns

    assert result.loc[0, "aspect_delivery"] == True  # noqa: E712
    assert result.loc[0, "aspect_price"] == False  # noqa: E712
    assert result.loc[1, "aspect_price"] == True  # noqa: E712
    assert result.loc[2, "aspects"] == []
    assert all(not result.loc[2, f"aspect_{a}"] for a in ASPECT_KEYWORDS)
