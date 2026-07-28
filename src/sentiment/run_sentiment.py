"""Phase 2 orchestrator: run VADER + transformer sentiment models on reviews_clean.parquet.

Usage:
    python src/sentiment/run_sentiment.py [--sample N]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.sentiment.transformer_model import score_transformer
from src.sentiment.vader_model import score_vader


def _rating_to_sentiment(rating):
    if pd.isna(rating):
        return None
    rating = float(rating)
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add vader_label/vader_score and transformer_label/transformer_confidence columns."""
    df = df.copy()
    texts = df["clean_light"].tolist()

    vader_results = score_vader(texts)
    df["vader_score"] = [score for score, _ in vader_results]
    df["vader_label"] = [label for _, label in vader_results]

    transformer_results = score_transformer(texts)
    df["transformer_confidence"] = [conf for _, conf in transformer_results]
    df["transformer_label"] = [label for label, _ in transformer_results]

    if config.RATING_COL in df.columns:
        df["rating_sentiment"] = df[config.RATING_COL].apply(_rating_to_sentiment)

    return df


def main():
    parser = argparse.ArgumentParser(description="Run sentiment models on cleaned reviews.")
    parser.add_argument("--sample", type=int, default=None, help="Only process the first N rows.")
    args = parser.parse_args()

    in_path = config.PROCESSED_DIR / "reviews_clean.parquet"
    df = pd.read_parquet(in_path)

    if args.sample is not None:
        df = df.head(args.sample)

    print(f"[info] scoring {len(df)} reviews...")
    result_df = run_sentiment(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "reviews_sentiment.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"[info] saved {len(result_df)} rows to {out_path}")

    print("\nSample:")
    cols = ["clean_light", "vader_label", "vader_score", "transformer_label", "transformer_confidence"]
    if "rating_sentiment" in result_df.columns:
        cols.append("rating_sentiment")
    print(result_df[cols].head(5).to_string())


if __name__ == "__main__":
    main()
