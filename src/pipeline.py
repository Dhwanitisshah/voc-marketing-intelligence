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


# Share of the overall progress bar each stage owns: (start, end) fractions.
STAGE_SPANS = {
    "Cleaning text": (0.0, 0.10),
    "Scoring sentiment": (0.10, 0.85),
    "Tagging aspects": (0.85, 0.90),
    "Saving results": (0.90, 1.0),
}


def analyze(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """Run the full clean -> sentiment -> aspect chain over a raw reviews df.

    `df` must contain config.REVIEW_TEXT_COL. Returns a new df with
    clean_light, clean_deep, sentiment, confidence, aspects, and
    aspect_<name> boolean columns added.

    `progress_callback(fraction, stage)`, if given, is called with the overall
    progress in [0, 1] and the current stage name from STAGE_SPANS (the caller
    owns the final "Saving results" stage, which happens after analyze()).
    """
    def report(stage, done=0.0):
        if progress_callback:
            start, end = STAGE_SPANS[stage]
            progress_callback(start + (end - start) * done, stage)

    if config.REVIEW_TEXT_COL not in df.columns:
        raise KeyError(
            f"Expected review text column '{config.REVIEW_TEXT_COL}' not found. "
            f"Columns found: {list(df.columns)}"
        )

    report("Cleaning text")
    df = preprocess(df, nlp=_get_nlp())

    report("Scoring sentiment")
    results = score_transformer(
        df["clean_light"].tolist(),
        progress_callback=lambda done, total: report("Scoring sentiment", done / total if total else 1.0),
    )
    df["sentiment"] = [label for label, _ in results]
    df["confidence"] = [score for _, score in results]

    report("Tagging aspects")
    df = tag_dataframe(df)
    report("Tagging aspects", 1.0)

    return df
