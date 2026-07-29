"""Phase 4: composed analysis pipeline for the dashboard.

Chains the existing Phase 1-3 modules into a single `analyze()` call:
preprocess (clean_light/clean_deep) -> transformer sentiment on clean_light
-> keyword aspect tagging on clean_deep. VADER and BERTopic are intentionally
skipped here; the dashboard only needs the transformer + keyword tagger.
"""

import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing.clean_text import preprocess
from src.sentiment.transformer_model import score_transformer
from src.aspects.aspect_tagger import tag_dataframe

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full clean -> sentiment -> aspect chain over a raw reviews df.

    `df` must contain config.REVIEW_TEXT_COL. Returns a new df with
    clean_light, clean_deep, sentiment, confidence, aspects, and
    aspect_<name> boolean columns added.
    """
    if config.REVIEW_TEXT_COL not in df.columns:
        raise KeyError(
            f"Expected review text column '{config.REVIEW_TEXT_COL}' not found. "
            f"Columns found: {list(df.columns)}"
        )

    df = preprocess(df, nlp=_get_nlp())

    results = score_transformer(df["clean_light"].tolist())
    df["sentiment"] = [label for label, _ in results]
    df["confidence"] = [score for _, score in results]

    df = tag_dataframe(df)

    return df
