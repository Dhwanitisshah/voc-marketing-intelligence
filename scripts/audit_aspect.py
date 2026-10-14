"""Audit the keyword aspect tagger: sample reviews counted as *negative* mentions of one aspect.

For each sampled review it shows which lexicon keyword(s) triggered the tag and a
window of words around the first match, so you can judge by eye whether the
review is really complaining about that aspect, or just carries a negative
review-level sentiment that came from something else (the tagger scores the
whole review, not the aspect).

Usage:
    python scripts/audit_aspect.py [--dataset ID] [--aspect ux] [--sample 20] [--seed 42] [--window 15]
    python scripts/audit_aspect.py --export data/eval/aspect_audit.csv [--aspects ux price]

--export writes EVERY negative mention of the given aspects to a CSV for labelling
(draft_label / draft_reason / human_label are left empty). It refuses to overwrite
an existing file unless --force is given, so a hand-labelled file is never lost.
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.aspects.aspect_lexicon import ASPECT_KEYWORDS
from src.db import database as db

WORD_RE = re.compile(r"\S+")


def negative_mentions(dataset_id: str, aspect: str) -> list:
    """(review_id, review_text, clean_deep, rating, all_aspects) for negative reviews tagged with `aspect`."""
    conn = db._connect()
    try:
        return conn.execute(
            """
            SELECT ar.id, ar.review_text, ar.clean_deep, ar.rating, ar.aspects
            FROM review_aspects ra
            JOIN analyzed_reviews ar ON ar.id = ra.review_id
            WHERE ra.dataset_id = ? AND ra.aspect = ? AND ra.sentiment = 'negative'
            ORDER BY ar.id
            """,
            (dataset_id, aspect),
        ).fetchall()
    finally:
        conn.close()


def matched_keywords(clean_deep: str, keywords: set) -> list:
    """Lexicon keywords present as lemma tokens in clean_deep (exactly what the tagger matched)."""
    return sorted(set((clean_deep or "").split()) & keywords)


def keyword_window(text: str, keywords: set, nlp, window: int):
    """Return (matched surface word, windowed snippet with the match marked), or (None, head of text)."""
    words = list(WORD_RE.finditer(text))
    if not words:
        return None, text

    hit = None
    for token in nlp(text.lower()):
        if token.lemma_ in keywords:
            hit = token.idx
            break
    if hit is None:  # lemmatization differs from the cleaned text; fall back to a word-prefix match
        pattern = re.compile(r"\b(?:%s)" % "|".join(map(re.escape, keywords)), re.I)
        m = pattern.search(text)
        hit = m.start() if m else None
    if hit is None:
        return None, " ".join(w.group() for w in words[: 2 * window])

    index = next(i for i, w in enumerate(words) if w.start() <= hit < w.end())
    lo, hi = max(0, index - window), min(len(words), index + window + 1)
    snippet = [w.group() for w in words[lo:hi]]
    surface = snippet[index - lo]
    snippet[index - lo] = f"[[{surface}]]"
    return surface.strip(".,!?;:\"'()"), ("... " if lo else "") + " ".join(snippet) + (" ..." if hi < len(words) else "")


EXPORT_COLUMNS = [
    "review_id", "aspect", "keyword", "context_window", "full_review",
    "draft_label", "draft_reason", "human_label",
]


def export_mentions(dataset_id: str, aspects: list, path: Path, nlp, window: int):
    """Write all negative mentions of `aspects` to `path` (labels left blank)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for aspect in aspects:
            keywords = set(ASPECT_KEYWORDS[aspect])
            for review_id, text, clean_deep, _rating, _aspects in negative_mentions(dataset_id, aspect):
                _surface, snippet = keyword_window(text or "", keywords, nlp, window)
                writer.writerow(
                    {
                        "review_id": review_id,
                        "aspect": aspect,
                        "keyword": "|".join(matched_keywords(clean_deep, keywords)),
                        "context_window": snippet,
                        "full_review": text,
                        "draft_label": "",
                        "draft_reason": "",
                        "human_label": "",
                    }
                )
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="reviews_171e719d99")
    parser.add_argument("--aspect", default="ux", choices=sorted(ASPECT_KEYWORDS))
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=15, help="words kept on each side of the match")
    parser.add_argument("--export", type=Path, help="write all negative mentions to this CSV and exit")
    parser.add_argument("--aspects", nargs="+", default=["ux", "price"], choices=sorted(ASPECT_KEYWORDS))
    parser.add_argument("--force", action="store_true", help="allow --export to overwrite an existing file")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    if args.export:
        if args.export.exists() and not args.force:
            sys.exit(f"{args.export} already exists (it may hold human labels). Use --force to overwrite.")
        n = export_mentions(args.dataset, args.aspects, args.export, nlp, args.window)
        print(f"Wrote {n} negative mentions ({', '.join(args.aspects)}) to {args.export}")
        return

    keywords = set(ASPECT_KEYWORDS[args.aspect])

    rows = negative_mentions(args.dataset, args.aspect)
    if not rows:
        print(f"No negative '{args.aspect}' mentions found for dataset {args.dataset}.")
        return

    sample = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    print(
        f"Dataset {args.dataset}: {len(rows)} negative '{args.aspect}' mentions; "
        f"showing {len(sample)} (seed={args.seed}).\n"
    )

    for n, (review_id, text, clean_deep, rating, aspects) in enumerate(sample, 1):
        kws = matched_keywords(clean_deep, keywords)
        surface, snippet = keyword_window(text or "", keywords, nlp, args.window)
        others = [a for a in (aspects or "").split(",") if a and a != args.aspect]
        print(f"[{n:>2}] review #{review_id} | rating={rating} | matched={kws} | other aspects={others or '-'}")
        print(f"     {snippet}\n")

    # Which keywords produce the tag across ALL negative mentions (not just the sample)?
    counts = Counter(kw for _, _, clean_deep, _, _ in rows for kw in matched_keywords(clean_deep, keywords))
    print(f"Keyword hits across all {len(rows)} negative '{args.aspect}' mentions (a review can hit several):")
    for kw, c in counts.most_common():
        print(f"  {kw:<12}{c:>4}")
    single = sum(1 for _, _, cd, _, _ in rows if len(matched_keywords(cd, keywords)) == 1)
    print(f"\nReviews triggered by exactly one keyword: {single}/{len(rows)}")


if __name__ == "__main__":
    main()
