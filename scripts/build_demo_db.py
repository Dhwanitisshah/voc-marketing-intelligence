"""Build the small precomputed demo database that ships with the repo (data/demo/voc_demo.db).

A hosted deployment can't afford a multi-minute RoBERTa + BERTopic run on first load, so this
script does that work once, here: it samples reviews from an already-analysed dataset in your
local data/voc.db and stores, in a fresh database file,

  - the reviews with their review-level sentiment and aspect tags (reused from the source),
  - aspect-level sentiment under the current config (RoBERTa on the evidence sentences),
  - BERTopic topics (refit on the sampled reviews),
  - optionally a Gemini strategy (needs GEMINI_API_KEY), so the panel is populated without a key.

The app copies this file to data/voc.db on first start (db.ensure_demo_db) and opens the dataset
straight away. Usage:

    python scripts/build_demo_db.py [--source reviews_171e719d99] [--rows 1000] [--seed 42] [--no-strategy]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from src.db import database as db


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="reviews_171e719d99", help="dataset id in the local data/voc.db")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=config.DEMO_DB_PATH)
    parser.add_argument("--dataset-id", default=config.DEFAULT_DATASET_ID)
    parser.add_argument("--no-strategy", action="store_true", help="skip the Gemini call")
    args = parser.parse_args()

    source_db = db.DB_PATH
    if not source_db.exists():
        sys.exit(f"Source database {source_db} not found; analyse a dataset in the app first.")

    # 1. sample from the source dataset
    source = db.load_analysis(args.source)
    if source.empty:
        sys.exit(f"Dataset {args.source!r} not found in {source_db}.")
    sample = source.sample(n=min(args.rows, len(source)), random_state=args.seed).sort_values("id")
    print(f"Sampled {len(sample)} of {len(source)} reviews from {args.source}")

    frame = sample.rename(columns={"review_date": config.DATE_COL})[
        ["review_text", "clean_deep", "sentiment", "confidence", "aspects", "rating", config.DATE_COL]
    ].reset_index(drop=True)

    # 2. fresh database file (never touches the source DB)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    db.DB_PATH = args.out
    db._schema_ready.discard(str(args.out))
    db.init_db()
    db.save_analysis(frame, args.dataset_id, aspect_mode="review")
    print(f"Saved reviews to {args.out}")

    # 3. aspect-level sentiment under the current config
    from src.aspects.aspect_sentiment import mode_key, rebuild_dataset_aspects

    t0 = time.time()
    key = rebuild_dataset_aspects(args.dataset_id, config.ASPECT_SENTIMENT_MODE)
    print(f"Aspect sentiment ({key}) in {time.time() - t0:.0f}s; app expects {mode_key()}")

    # 4. topics
    from src.aspects.topic_model import discover_topics
    from src.preprocessing.clean_text import light_clean

    t0 = time.time()
    analysed = db.load_analysis(args.dataset_id)
    result = discover_topics([light_clean(t) for t in analysed["review_text"]])
    if result is None:
        print("Too few reviews for BERTopic; no topics stored.")
    else:
        db.save_topics(args.dataset_id, analysed["id"].tolist(), result)
        print(f"{len(result['topics'])} topics in {time.time() - t0:.0f}s")

    # 5. strategy (optional)
    if not args.no_strategy:
        from src.strategy.generate_strategy import generate_strategy

        try:
            strategy = generate_strategy(args.dataset_id)
            db.save_strategy(args.dataset_id, strategy)
            g = strategy["grounding"]
            print(f"Strategy by {strategy['model']}: {g['total'] - len(g['ungrounded'])}/{g['total']} figures grounded")
        except Exception as e:  # no key, outage, invalid response...
            print(f"Strategy skipped ({type(e).__name__}: {e}). Generate it in the app instead.")

    # 6. compact the file
    import sqlite3

    conn = sqlite3.connect(args.out)
    conn.execute("VACUUM")
    conn.close()
    print(f"Done: {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
