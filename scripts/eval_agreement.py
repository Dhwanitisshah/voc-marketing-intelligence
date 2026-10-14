"""Recompute the aspect-audit results from HUMAN labels and write them into docs/aspect_eval.md.

Reads data/eval/aspect_audit.csv (human_label filled in with scripts/label_app.py) and produces:
  - precision per aspect and mode (review / sentence / sentence+clause) from human labels,
  - Cohen's kappa between draft_label and human_label (3-class and genuine-vs-not),
  - the list of rows where the two disagree.

Only the block between the HUMAN-LABELS markers in the document is replaced; the rest (draft-label
results, narrative) is left alone. Needs no model: the per-mode flags (neg_<mode>) are columns in the
CSV, refreshed by `python scripts/eval_aspects.py --append-unlabelled`.

Usage:
    python scripts/eval_agreement.py [--csv CSV] [--doc docs/aspect_eval.md] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.agreement import MODES, flag_column
from src.evaluation.labeling import load_rows
from src.evaluation.report import render_human_block, splice_block


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("data/eval/aspect_audit.csv"))
    parser.add_argument("--doc", type=Path, default=Path("docs/aspect_eval.md"))
    parser.add_argument("--dry-run", action="store_true", help="print the block instead of writing the doc")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fieldnames, rows = load_rows(args.csv)
    missing = [flag_column(m) for m in MODES if flag_column(m) not in fieldnames]
    if missing:
        sys.exit(f"{args.csv} has no {missing} column(s). Run: python scripts/eval_aspects.py --append-unlabelled")

    block = render_human_block(rows)
    if args.dry_run:
        print(block)
        return
    doc = args.doc.read_text(encoding="utf-8") if args.doc.exists() else ""
    args.doc.write_text(splice_block(doc, block), encoding="utf-8")
    human = sum(1 for r in rows if (r.get("human_label") or "").strip())
    print(f"Wrote human-label section to {args.doc} ({human}/{len(rows)} rows human-labelled).")


if __name__ == "__main__":
    main()
