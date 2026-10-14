"""Standalone Streamlit page for labelling rows of a CSV, one at a time.

    streamlit run scripts/label_app.py --server.fileWatcherType none -- --csv data/eval/aspect_audit.csv

Shows the aspect, the keyword highlighted in its context window, and the full review. Any draft
label is hidden in a collapsed expander so it doesn't anchor you. Each click saves `human_label`
to the CSV immediately; no other column (draft_label included) is ever modified.

Generic: point the --*-col options at another CSV, e.g. a random 300-review gold-standard set:

    streamlit run scripts/label_app.py -- --csv data/eval/gold.csv --id-col id \\
        --aspect-col "" --context-col review_text --labels positive,neutral,negative
"""

import argparse
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.labeling import (
    DEFAULT_GUIDE,
    DEFAULT_LABELS,
    highlight_html,
    labelled_count,
    load_rows,
    next_unlabelled,
    save_label,
    split_keywords,
)


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--csv", default="data/eval/aspect_audit.csv")
    p.add_argument("--id-col", default="review_id")
    p.add_argument("--aspect-col", default="aspect", help='"" if the CSV has no aspect column')
    p.add_argument("--keyword-col", default="keyword", help='"" for no keyword highlighting')
    p.add_argument("--context-col", default="context_window")
    p.add_argument("--text-col", default="full_review", help='"" to show only the context column')
    p.add_argument("--label-col", default="human_label")
    p.add_argument("--draft-col", default="draft_label", help='"" if there is no draft label')
    p.add_argument("--draft-reason-col", default="draft_reason")
    p.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma-separated label choices")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


args = parse_args()
LABELS = [x.strip() for x in args.labels.split(",") if x.strip()]
csv_path = Path(args.csv)
if not csv_path.is_absolute():
    csv_path = Path(__file__).resolve().parents[1] / csv_path

st.set_page_config(page_title="Label reviews", layout="centered")

if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}")
    st.stop()

fieldnames, rows = load_rows(csv_path)
if not rows:
    st.warning("The CSV has no rows.")
    st.stop()

key_cols = [c for c in (args.id_col, args.aspect_col) if c]
missing = [c for c in key_cols + [args.context_col] if c not in fieldnames]
if missing:
    st.error(f"Columns not in the CSV: {missing}. Available: {fieldnames}")
    st.stop()

st.session_state.setdefault("idx", 0)

# --- sidebar: filters + progress ---------------------------------------------------------
with st.sidebar:
    st.header("Progress")
    done = labelled_count(rows, args.label_col)
    st.progress(done / len(rows), text=f"{done} / {len(rows)} labelled")
    aspects = sorted({r[args.aspect_col] for r in rows}) if args.aspect_col in fieldnames and args.aspect_col else []
    chosen = st.multiselect("Aspects", aspects, default=aspects) if aspects else []
    st.caption(f"File: `{csv_path.name}`  \nSaved after every click.")
    with st.expander("Label guide"):
        for label in LABELS:
            if label in DEFAULT_GUIDE:
                st.markdown(f"**{label}**: {DEFAULT_GUIDE[label]}")
        if not any(label in DEFAULT_GUIDE for label in LABELS):
            st.caption("No guide for these labels.")

allowed = {i for i, r in enumerate(rows) if not aspects or r[args.aspect_col] in chosen}
if not allowed:
    st.info("No rows match the selected aspects.")
    st.stop()
order = sorted(allowed)

idx = st.session_state["idx"]
if idx not in allowed:  # filter changed: snap to the nearest allowed row
    idx = order[0]
    st.session_state["idx"] = idx
pos = order.index(idx)
row = rows[idx]


def go(i):
    st.session_state["idx"] = i


def set_label(value):
    save_label(csv_path, tuple(row[c] for c in key_cols), value, key_cols, args.label_col)
    # advance to the next unlabelled row (re-read, since the file just changed)
    _, fresh = load_rows(csv_path)
    nxt = next_unlabelled(fresh, idx, args.label_col, allowed)
    go(nxt if nxt is not None else idx)


# --- the row ---------------------------------------------------------------------------------
header = f"Row {pos + 1} of {len(order)}"
if args.aspect_col:
    header += f" · aspect: **{row[args.aspect_col]}**"
st.markdown(header)
keywords = split_keywords(row.get(args.keyword_col, "")) if args.keyword_col else []
if keywords:
    st.caption("Keyword(s): " + ", ".join(keywords))

with st.container(border=True):
    st.caption("Context")
    st.html(f"<div style='font-size:1.1em;line-height:1.5'>{highlight_html(row[args.context_col], keywords)}</div>")

if args.text_col and args.text_col in fieldnames and row[args.text_col] != row[args.context_col]:
    with st.container(border=True):
        st.caption("Full review")
        st.html(f"<div style='line-height:1.5'>{highlight_html(row[args.text_col], keywords, marker_pairs=False)}</div>")

current = (row.get(args.label_col) or "").strip()
if current:
    st.success(f"Your label: **{current}**")

with st.container(horizontal=True):
    for label in LABELS:
        st.button(
            label, key=f"set_{label}", on_click=set_label, args=(label,),
            type="primary" if label == current else "secondary",
        )
    st.button("skip", key="skip", on_click=lambda: go(next_unlabelled(rows, idx, args.label_col, allowed) or idx))
    if current:
        st.button("clear", key="clear", on_click=set_label, args=("",))

with st.container(horizontal=True):
    st.button("← prev", key="prev", on_click=go, args=(order[max(pos - 1, 0)],), disabled=pos == 0)
    st.button("next →", key="next", on_click=go, args=(order[min(pos + 1, len(order) - 1)],),
              disabled=pos == len(order) - 1)
    st.button("next unlabelled", key="next_unlabelled",
              on_click=lambda: go(next_unlabelled(rows, idx, args.label_col, allowed) or idx))

if args.draft_col and args.draft_col in fieldnames:
    with st.expander("Show draft label and reason (may anchor your judgement)", expanded=False):
        st.markdown(f"Draft label: **{row.get(args.draft_col) or '(none)'}**")
        if args.draft_reason_col in fieldnames:
            st.caption(row.get(args.draft_reason_col) or "")
