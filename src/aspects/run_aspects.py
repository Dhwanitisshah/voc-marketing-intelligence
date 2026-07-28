"""Phase 3 orchestrator: keyword aspect tagging + BERTopic discovery, crossed with
sentiment to produce the aspect x sentiment summary that feeds Phase 5's LLM layer.

Usage:
    python src/aspects/run_aspects.py [--sample N]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.aspects.aspect_lexicon import ASPECT_KEYWORDS
from src.aspects.aspect_tagger import tag_dataframe
from src.aspects.topic_model import run_bertopic

SENTIMENT_LABELS = ["positive", "neutral", "negative"]


def build_aspect_sentiment_summary(df: pd.DataFrame) -> dict:
    """For each aspect, count positive/neutral/negative/total mentions using
    transformer_label, then rank aspects for top complaints and top praise."""
    per_aspect = {}
    for aspect in ASPECT_KEYWORDS:
        mask = df[f"aspect_{aspect}"]
        subset = df.loc[mask, "transformer_label"]
        total = int(mask.sum())
        counts = {label: int((subset == label).sum()) for label in SENTIMENT_LABELS}

        negative_share = counts["negative"] / total if total else 0.0
        positive_share = counts["positive"] / total if total else 0.0

        per_aspect[aspect] = {
            "total": total,
            "positive": counts["positive"],
            "neutral": counts["neutral"],
            "negative": counts["negative"],
            "negative_share": round(negative_share, 4),
            "positive_share": round(positive_share, 4),
        }

    mentioned = {a: s for a, s in per_aspect.items() if s["total"] > 0}
    top_complaints = sorted(
        mentioned.items(), key=lambda kv: (kv[1]["negative_share"], kv[1]["negative"]), reverse=True
    )
    top_praise = sorted(
        mentioned.items(), key=lambda kv: (kv[1]["positive_share"], kv[1]["positive"]), reverse=True
    )

    return {
        "aspects": per_aspect,
        "top_complaints": [aspect for aspect, _ in top_complaints],
        "top_praise": [aspect for aspect, _ in top_praise],
    }


def print_summary_table(summary: dict):
    header = f"{'Aspect':<12}{'Total':>8}{'Positive':>10}{'Neutral':>10}{'Negative':>10}{'Neg %':>8}"
    print(header)
    print("-" * len(header))
    for aspect, stats in summary["aspects"].items():
        neg_pct = f"{stats['negative_share']:.0%}" if stats["total"] else "-"
        print(
            f"{aspect:<12}{stats['total']:>8}{stats['positive']:>10}"
            f"{stats['neutral']:>10}{stats['negative']:>10}{neg_pct:>8}"
        )

    print(f"\nTop complaints (by negative share): {', '.join(summary['top_complaints']) or 'none'}")
    print(f"Top praise (by positive share):     {', '.join(summary['top_praise']) or 'none'}")


def main():
    parser = argparse.ArgumentParser(description="Run aspect mining on scored reviews.")
    parser.add_argument("--sample", type=int, default=None, help="Only process the first N rows.")
    args = parser.parse_args()

    in_path = config.PROCESSED_DIR / "reviews_sentiment.parquet"
    df = pd.read_parquet(in_path)

    if args.sample is not None:
        df = df.head(args.sample)

    print(f"[info] tagging aspects for {len(df)} reviews...")
    df = tag_dataframe(df)

    print("[info] running BERTopic discovery layer...")
    topics, topic_info = run_bertopic(df["clean_light"].tolist())
    if topics is not None:
        df["bertopic_topic"] = topics
        print("\nBERTopic topic info:")
        print(topic_info.to_string())

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "reviews_aspects.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[info] saved {len(df)} rows to {out_path}")

    summary = build_aspect_sentiment_summary(df)

    reports_dir = config.PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "aspect_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[info] saved aspect x sentiment summary to {summary_path}")

    print("\nAspect x Sentiment summary:")
    print_summary_table(summary)


if __name__ == "__main__":
    main()
