import pandas as pd
import pytest

import config
from src.data.load_reviews import load_reviews


def test_loads_bundled_sample_and_returns_rows():
    df = load_reviews(config.SAMPLE_DATA_PATH)
    assert len(df) > 0
    assert config.REVIEW_TEXT_COL in df.columns


def test_drops_null_and_empty_review_text(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {"review_text": ["Great product", None, "", "  ", "Also great"]}
    ).to_csv(path, index=False)

    df = load_reviews(path)

    assert list(df["review_text"]) == ["Great product", "Also great"]


def test_missing_review_text_column_raises_key_error(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"not_the_text_column": ["hello"]}).to_csv(path, index=False)

    with pytest.raises(KeyError):
        load_reviews(path)


def test_falls_back_to_sample_when_raw_path_missing(tmp_path):
    bad_path = tmp_path / "does_not_exist.csv"

    df = load_reviews(bad_path)

    sample_df = pd.read_csv(config.SAMPLE_DATA_PATH)
    assert len(df) == len(sample_df)
