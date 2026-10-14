"""CSV helpers behind scripts/label_app.py: load, highlight, and save one label at a time.

Generic on purpose: any CSV with an id column and a text/context column can be labelled, so
the same tool serves the aspect audit now and a larger random gold-standard set later.
Saving only ever writes the label column; no other column (notably any draft label) is touched.
"""

import csv
import html
import os
import re
import tempfile
from pathlib import Path

DEFAULT_LABELS = ["genuine", "inherited", "false_trigger"]

DEFAULT_GUIDE = {
    "genuine": "The reviewer criticises the aspect itself.",
    "inherited": "The aspect is mentioned but not criticised (praise or neutral); the negativity is about something else.",
    "false_trigger": "The keyword matched something that isn't the aspect (an idiom, a competitor's price, build quality 'cheap', ...).",
}


def load_rows(path) -> tuple:
    """(fieldnames, rows) with every cell kept as the string stored in the file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_atomic(path, fieldnames, rows):
    """Write to a temp file in the same folder, then replace: a crash never leaves a half-written CSV."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def row_key(row: dict, key_cols) -> tuple:
    return tuple(row[c] for c in key_cols)


def save_label(path, key: tuple, label: str, key_cols, label_col: str = "human_label"):
    """Set `label_col` for the row identified by `key` (values of `key_cols`); "" clears it.

    Re-reads the file first, so labels saved from another session are not overwritten, and
    adds `label_col` if the file doesn't have it yet. Raises KeyError if no row matches.
    """
    fieldnames, rows = load_rows(path)
    if label_col not in fieldnames:
        fieldnames.append(label_col)
        for r in rows:
            r[label_col] = ""
    matches = [r for r in rows if row_key(r, key_cols) == tuple(key)]
    if not matches:
        raise KeyError(f"No row with {dict(zip(key_cols, key))} in {path}")
    for r in matches:
        r[label_col] = label
    _write_atomic(path, fieldnames, rows)


def labelled_count(rows, label_col: str = "human_label") -> int:
    return sum(1 for r in rows if (r.get(label_col) or "").strip())


def next_unlabelled(rows, after: int, label_col: str = "human_label", allowed=None):
    """Index of the next unlabelled row after `after` (wrapping round), or None if all are labelled."""
    n = len(rows)
    for step in range(1, n + 1):
        i = (after + step) % n
        if allowed is not None and i not in allowed:
            continue
        if not (rows[i].get(label_col) or "").strip():
            return i
    return None


_MARKER_RE = re.compile(r"\[\[(.+?)\]\]")


def highlight_html(text: str, keywords=(), marker_pairs: bool = True) -> str:
    """HTML-escape `text` and wrap keyword hits in <mark>.

    `[[word]]` markers (as written by the audit export) are converted; otherwise any word
    starting with one of `keywords` (case-insensitive) is marked. Safe to pass to st.html.
    """
    escaped = html.escape(text or "")
    if marker_pairs and _MARKER_RE.search(escaped):
        return _MARKER_RE.sub(lambda m: f"<mark>{m.group(1)}</mark>", escaped)
    words = [html.escape(k) for k in keywords if k]
    if not words:
        return escaped
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")(\w*)", re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def split_keywords(value: str) -> list:
    return [k.strip() for k in re.split(r"[|,;]", value or "") if k.strip()]
