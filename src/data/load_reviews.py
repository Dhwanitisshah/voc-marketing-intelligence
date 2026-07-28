"""Phase 0 loader: read raw (or sample) customer review data into a clean DataFrame."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config


def load_reviews(path=None):
    """Load reviews from `path`, falling back to bundled sample data.

    Falls back to config.SAMPLE_DATA_PATH if config.RAW_DATA_PATH (or the
    given path) does not exist. Raises KeyError if the configured review
    text column is missing. Returns a cleaned DataFrame with null/empty
    review text dropped and whitespace stripped.
    """
    if path is None:
        path = config.RAW_DATA_PATH

    path = Path(path)
    if not path.exists():
        print(f"[info] {path} not found, falling back to sample data at {config.SAMPLE_DATA_PATH}")
        path = config.SAMPLE_DATA_PATH

    df = pd.read_csv(path)

    if config.REVIEW_TEXT_COL not in df.columns:
        raise KeyError(
            f"Expected review text column '{config.REVIEW_TEXT_COL}' not found. "
            f"Columns found: {list(df.columns)}"
        )

    df[config.REVIEW_TEXT_COL] = df[config.REVIEW_TEXT_COL].astype(str).str.strip()
    df = df[df[config.REVIEW_TEXT_COL].notna()]
    df = df[df[config.REVIEW_TEXT_COL].str.lower() != "nan"]
    df = df[df[config.REVIEW_TEXT_COL] != ""]
    df = df.reset_index(drop=True)

    return df


def summarize(df):
    """Print a quick summary of a reviews DataFrame."""
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if config.RATING_COL in df.columns:
        print("\nRating distribution:")
        print(df[config.RATING_COL].value_counts().sort_index())

    word_counts = df[config.REVIEW_TEXT_COL].str.split().str.len()
    print(f"\nAvg words per review: {word_counts.mean():.1f}")

    print("\nSample reviews:")
    for text in df[config.REVIEW_TEXT_COL].head(3):
        print(f"- {text}")


if __name__ == "__main__":
    reviews_df = load_reviews()
    summarize(reviews_df)
