"""Phase 3: keyword-based aspect tagger (the reliable backbone).

Matches ASPECT_KEYWORDS against the "clean_deep" column, which is already
lowercased and lemmatized, so keywords are matched as whole lemma tokens
rather than substrings (avoids false positives like "ship" inside "worship").
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aspects.aspect_lexicon import ASPECT_KEYWORDS


def tag_aspects(text: str) -> list:
    """Return the list of aspects whose keywords appear as lemma tokens in `text`."""
    if not isinstance(text, str) or not text:
        return []

    tokens = set(text.split())
    matched = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if tokens & set(keywords):
            matched.append(aspect)
    return matched


def tag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add an "aspects" list column plus one-hot "aspect_<name>" boolean columns."""
    df = df.copy()
    df["aspects"] = df["clean_deep"].apply(tag_aspects)

    for aspect in ASPECT_KEYWORDS:
        df[f"aspect_{aspect}"] = df["aspects"].apply(lambda aspects: aspect in aspects)

    return df
