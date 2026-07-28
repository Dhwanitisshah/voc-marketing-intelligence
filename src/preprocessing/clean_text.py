"""Phase 1: two-track text cleaning.

Different downstream tasks need different cleaning, so every review gets two
cleaned columns:
  - clean_light : for sentiment models (VADER/transformers), which use case,
    punctuation, and emojis as signal. Only strips URLs/HTML and normalizes
    whitespace/encoding.
  - clean_deep  : for TF-IDF, topic/aspect mining, and word clouds. Lowercased,
    punctuation/digits stripped, lemmatized, stopwords removed.
"""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import spacy

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.data.load_reviews import load_reviews

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def light_clean(text: str) -> str:
    """Strip URLs/HTML and normalize whitespace/encoding; keep case, punctuation, emojis."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


def deep_clean(text: str, nlp) -> str:
    """Lowercase, strip punctuation/digits, then lemmatize and drop stopwords via spaCy."""
    return _deep_clean_batch([text], nlp)[0]


def _deep_clean_batch(texts, nlp) -> list:
    """Batch version of deep_clean using nlp.pipe() for efficiency."""
    prepped = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        text = text.lower()
        text = NON_ALPHA_RE.sub(" ", text)
        text = WHITESPACE_RE.sub(" ", text).strip()
        prepped.append(text)

    cleaned = []
    for doc in nlp.pipe(prepped, batch_size=64, disable=["parser", "ner"]):
        lemmas = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_space and token.lemma_.strip()
        ]
        cleaned.append(" ".join(lemmas))

    return cleaned


def preprocess(df: pd.DataFrame, nlp=None) -> pd.DataFrame:
    """Add clean_light, clean_deep, and word_count columns to `df`."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    df = df.copy()
    text_col = config.REVIEW_TEXT_COL

    df["clean_light"] = df[text_col].apply(light_clean)
    df["clean_deep"] = _deep_clean_batch(df[text_col].tolist(), nlp)
    df["word_count"] = df[text_col].str.split().str.len()

    return df


if __name__ == "__main__":
    reviews_df = load_reviews()
    processed_df = preprocess(reviews_df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "reviews_clean.parquet"
    processed_df.to_parquet(out_path, index=False)
    print(f"[info] saved {len(processed_df)} rows to {out_path}\n")

    print("Before / after sample:\n")
    sample = processed_df.head(5)
    for _, row in sample.iterrows():
        print(f"ORIGINAL:    {row[config.REVIEW_TEXT_COL]}")
        print(f"CLEAN_LIGHT: {row['clean_light']}")
        print(f"CLEAN_DEEP:  {row['clean_deep']}")
        print(f"WORD_COUNT:  {row['word_count']}")
        print("-" * 80)
