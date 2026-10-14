"""Before/after evaluation of aspect-level sentiment against the labelled audit set.

Compares, for the audited aspects (ux, price):
  review            every mention inherits the whole review's sentiment (the old behaviour)
  sentence          only the keyword's sentence is scored, negation-aware
  sentence+clause   as above, but also cut at "but" / "however" / ";" (ablation)

For each: negative mentions, precision of those mentions (share labelled `genuine`),
genuine complaints found, and the aspect's impact rank among all six aspects.

Labels come from data/eval/aspect_audit.csv: `human_label` where filled, otherwise
`draft_label`. Negative mentions with no label are reported, never silently dropped;
`--append-unlabelled` adds them to the CSV (found_in = the mode that flagged them) so they
can be labelled, and refreshes the `neg_<mode>` columns recording which modes count each row
as negative (scripts/eval_agreement.py reads those, so it needs no model).

Usage:
    python scripts/eval_aspects.py [--dataset ID] [--labels CSV] [--out docs/aspect_eval.md]
    python scripts/eval_aspects.py --append-unlabelled
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.aspects.aspect_lexicon import ASPECT_KEYWORDS
from src.aspects.aspect_sentiment import compute_aspect_details
from src.evaluation.agreement import flag_column
from src.db import database as db
from src.insights.ranking import describe_closeness, rank_complaints

AUDITED = ["ux", "price"]
KNOWN_FALSE_UX = [170, 229, 413, 593, 1345, 1399, 1585]  # the 7 false UX cases from the first audit
VARIANTS = [
    ("review", {"mode": "review"}),
    ("sentence", {"mode": "sentence", "split_clauses": False}),
    ("sentence+clause", {"mode": "sentence", "split_clauses": True}),
]
EXTRA_COLUMNS = ["found_in"]


def load_labels(path: Path):
    """{(aspect, review_id): (label, source)} with human_label taking precedence over draft_label."""
    labels, rows = {}, []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            human, draft = (r.get("human_label") or "").strip(), (r.get("draft_label") or "").strip()
            label, source = (human, "human") if human else (draft, "draft") if draft else ("", "none")
            labels[(r["aspect"], int(r["review_id"]))] = (label, source)
    return labels, rows


def negative_sets(df, details):
    """{aspect: {review_id: details}} for mentions counted as negative."""
    out = {a: {} for a in ASPECT_KEYWORDS}
    for review_id, per_aspect in zip(df["id"], details):
        for aspect, d in per_aspect.items():
            if d["sentiment"] == "negative":
                out[aspect][int(review_id)] = d
    return out


def aspect_summary(df, details) -> pd.DataFrame:
    rows = {a: {"aspect": a, "total": 0, "positive": 0, "neutral": 0, "negative": 0} for a in ASPECT_KEYWORDS}
    for per_aspect in details:
        for aspect, d in per_aspect.items():
            rows[aspect]["total"] += 1
            rows[aspect][d["sentiment"]] += 1
    return pd.DataFrame([r for r in rows.values() if r["total"]])


def evaluate(df, labels, split_runs):
    """Compute details for each variant; returns {variant: {...}} including wall-clock seconds."""
    results = {}
    for name, kwargs in VARIANTS:
        t0 = time.perf_counter()
        details = compute_aspect_details(df, **kwargs)
        seconds = time.perf_counter() - t0
        negatives = negative_sets(df, details)
        ranked = rank_complaints(aspect_summary(df, details))
        results[name] = {"details": details, "negatives": negatives, "ranked": ranked, "seconds": seconds}
    return results


def table(results, labels) -> str:
    lines = [
        "| Aspect | Mode | Negative mentions | Labelled | Genuine | Precision | Impact rank | Close to |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for aspect in AUDITED:
        for name, _ in VARIANTS:
            neg = results[name]["negatives"][aspect]
            labelled = [labels.get((aspect, rid), ("", "none"))[0] for rid in neg]
            known = [l for l in labelled if l]
            genuine = sum(1 for l in known if l == "genuine")
            precision = f"{genuine / len(known):.0%}" if known else "n/a"
            ranked = results[name]["ranked"].set_index("aspect")
            rank = int(ranked.loc[aspect, "rank"]) if aspect in ranked.index else "-"
            close = ", ".join(ranked.loc[aspect, "close_to"]) if aspect in ranked.index else ""
            unl = len(labelled) - len(known)
            note = f" ({unl} unlabelled)" if unl else ""
            lines.append(
                f"| {aspect} | {name} | {len(neg)} | {len(known)}{note} | {genuine} | {precision} | {rank} | {close or '-'} |"
            )
    return "\n".join(lines)


def known_cases_table(results, labels) -> str:
    lines = ["| Review | Draft label | " + " | ".join(n for n, _ in VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for rid in KNOWN_FALSE_UX:
        cells = []
        for name, _ in VARIANTS:
            counted = rid in results[name]["negatives"]["ux"]
            cells.append("still negative" if counted else "fixed")
        lines.append(f"| #{rid} | {labels.get(('ux', rid), ('?', ''))[0]} | " + " | ".join(cells) + " |")
    fixed = {
        name: sum(1 for rid in KNOWN_FALSE_UX if rid not in results[name]["negatives"]["ux"])
        for name, _ in VARIANTS
    }
    lines.append("| **Fixed** | | " + " | ".join(f"**{fixed[n]}/{len(KNOWN_FALSE_UX)}**" for n, _ in VARIANTS) + " |")
    return "\n".join(lines)


def unlabelled(results, labels, variant):
    """[(aspect, review_id, details)] negative in `variant` for audited aspects but absent from the labels."""
    out = []
    for aspect in AUDITED:
        for rid, d in results[variant]["negatives"][aspect].items():
            if not labels.get((aspect, rid), ("", ""))[0]:
                out.append((aspect, rid, d))
    return out


def append_unlabelled(path: Path, rows, new, df_by_id, results):
    fieldnames = list(rows[0].keys())
    flag_cols = [flag_column(name) for name, _ in VARIANTS]
    for col in EXTRA_COLUMNS + flag_cols:
        if col not in fieldnames:
            fieldnames.append(col)
    for r in rows:
        r.setdefault("found_in", "review")  # the original 75 were the review-level negatives
    seen = {(r["aspect"], int(r["review_id"])): r for r in rows}
    for aspect, rid, d, variant in new:
        key = (aspect, rid)
        if key in seen:
            if variant not in seen[key]["found_in"].split("+"):
                seen[key]["found_in"] += f"+{variant}"
            continue
        row = {c: "" for c in fieldnames}
        row.update(
            review_id=rid, aspect=aspect, keyword=d.get("keyword") or "",
            context_window=d.get("sentence") or "", full_review=df_by_id[rid], found_in=variant,
        )
        rows.append(row)
        seen[key] = row
    for r in rows:  # which modes count this mention as negative (a snapshot of this run)
        for name, _ in VARIANTS:
            r[flag_column(name)] = int(int(r["review_id"]) in results[name]["negatives"][r["aspect"]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="reviews_171e719d99")
    parser.add_argument("--labels", type=Path, default=Path("data/eval/aspect_audit.csv"))
    parser.add_argument("--out", type=Path, help="write the markdown tables here")
    parser.add_argument("--append-unlabelled", action="store_true", help="add unlabelled negatives to the CSV")
    parser.add_argument("--list-unlabelled", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from src.sentiment.transformer_model import score_probs

    score_probs(["warm up the model"])  # keep model load time out of the timings
    df = db.load_analysis(args.dataset).rename(columns={"review_text": "review_text"})
    labels, rows = load_labels(args.labels)
    results = evaluate(df, labels, None)

    if args.append_unlabelled or args.list_unlabelled:
        by_id = dict(zip(df["id"].astype(int), df["review_text"]))
        new = [(a, r, d, name) for name, _ in VARIANTS[1:] for a, r, d in unlabelled(results, labels, name)]
        counts = Counter((a, name) for a, _, _, name in new)
        print("Unlabelled negative mentions by (aspect, variant):", dict(counts))
        if args.append_unlabelled:
            append_unlabelled(args.labels, rows, new, by_id, results)
            print(f"Updated {args.labels}: appended unlabelled mentions and refreshed neg_<mode> flags")
        return

    sources = Counter(src for _, src in labels.values())
    md = [
        f"Dataset `{args.dataset}` ({len(df)} reviews). Labels: "
        f"{sources.get('human', 0)} human, {sources.get('draft', 0)} draft (draft used where `human_label` is empty).",
        "",
        table(results, labels),
        "",
        "### The 7 known false UX cases",
        "",
        known_cases_table(results, labels),
        "",
        "### Added runtime (3,000 reviews, model already loaded)",
        "",
        "| Mode | Aspect-sentiment step |",
        "|---|---:|",
        *[f"| {name} | {results[name]['seconds']:.1f} s |" for name, _ in VARIANTS],
        "",
        "### Complaint ranking per mode (all aspects, impact = negative mentions)",
        "",
    ]
    for name, _ in VARIANTS:
        ranked = results[name]["ranked"]
        md.append(f"**{name}**: " + ", ".join(f"{r.aspect} {r.impact}" for r in ranked.itertuples()))
        note = describe_closeness(ranked)
        md.append(f"  \n_{note or 'No statistically close neighbours.'}_\n")
    text = "\n".join(md)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
