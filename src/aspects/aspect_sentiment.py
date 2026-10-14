"""Aspect-level sentiment: lightweight aspect-based sentiment analysis (ABSA).

The keyword tagger says WHICH aspects a review mentions. This module decides how the
reviewer feels ABOUT each one. Two modes (config.ASPECT_SENTIMENT_MODE):

  "review"   every mention inherits the whole review's sentiment. A 1-star review that
             says "easy to set up, but the speaker is terrible" counts as a negative UX
             mention.
  "sentence" only the sentence holding the aspect keyword is scored with the same
             RoBERTa model (class probabilities are averaged if several sentences mention
             the aspect). Positive-polarity keywords ("easy", "worth", "value", ...) that
             are negated ("not easy", "no value") are counted as negative mentions.

The scorer is injectable so the logic can be tested without loading a model.
"""

import re
import sys
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.aspects.aspect_lexicon import ASPECT_KEYWORDS, POSITIVE_POLARITY_KEYWORDS
from src.preprocessing.clean_text import light_clean

MODES = ("sentence", "review")
LABELS = ("negative", "neutral", "positive")

# Lemmas that negate a following positive keyword. spaCy lemmatizes "n't" to "not".
NEGATION_LEMMAS = {
    "not", "no", "never", "nothing", "nobody", "none", "neither", "nor",
    "hardly", "barely", "scarcely", "without", "lack", "lacking",
}
NEGATION_WINDOW = 3  # content tokens looked at before the keyword
CONTRAST_WORDS = {"but", "however", "although", "though"}

_sentence_nlp = None


def get_sentence_nlp():
    """spaCy pipeline with a fast sentence segmenter, tagger and lemmatizer (no parser/NER)."""
    global _sentence_nlp
    if _sentence_nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
        nlp.enable_pipe("senter")
        _sentence_nlp = nlp
    return _sentence_nlp


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"ASPECT_SENTIMENT_MODE must be one of {MODES}, got {mode!r}")
    return mode


def mode_key(mode: str | None = None, split_clauses: bool | None = None) -> str:
    """Identifier stored with a dataset: "review", "sentence" or "sentence+clause".

    Datasets whose stored key differs from the current config's are rebuilt on load, so
    flipping ASPECT_SPLIT_CLAUSES invalidates caches just as changing the mode does.
    """
    mode = validate_mode(mode or config.ASPECT_SENTIMENT_MODE)
    if mode == "review":
        return "review"
    split = config.ASPECT_SPLIT_CLAUSES if split_clauses is None else split_clauses
    return "sentence+clause" if split else "sentence"


@dataclass
class Mention:
    aspect: str
    keyword: str
    sentence: str
    negated: bool


def is_negated(tokens, index: int, window: int = NEGATION_WINDOW) -> bool:
    """Is `tokens[index]` preceded by a negation cue within `window` content tokens?

    Scanning stops at punctuation, so "not good, easy to use" does not negate "easy".
    """
    seen = 0
    for j in range(index - 1, -1, -1):
        tok = tokens[j]
        if tok.is_punct:
            return False
        if tok.lemma_.lower() in NEGATION_LEMMAS or tok.lower_ in NEGATION_LEMMAS:
            return True
        seen += 1
        if seen >= window:
            return False
    return False


def _spans(sent, split_clauses: bool):
    """Yield token lists: the whole sentence, or its clauses if `split_clauses`."""
    tokens = list(sent)
    if not split_clauses:
        yield tokens
        return
    current = []
    for tok in tokens:
        if tok.lower_ in CONTRAST_WORDS or tok.text == ";":
            if current:
                yield current
            current = [] if tok.text == ";" else [tok]
        else:
            current.append(tok)
    if current:
        yield current


def find_mentions(doc, aspects, split_clauses: bool = False) -> dict:
    """{aspect: [Mention, ...]} - one mention per sentence (or clause) containing a keyword."""
    found = {}
    for aspect in aspects:
        keywords = set(ASPECT_KEYWORDS.get(aspect, ()))
        for sent in doc.sents:
            for span in _spans(sent, split_clauses):
                hits = [
                    i for i, tok in enumerate(span)
                    if tok.lemma_.lower() in keywords or tok.lower_ in keywords
                ]
                if not hits:
                    continue
                negated_hits = [
                    i for i in hits
                    if (span[i].lemma_.lower() in POSITIVE_POLARITY_KEYWORDS
                        or span[i].lower_ in POSITIVE_POLARITY_KEYWORDS)
                    and is_negated(span, i)
                ]
                negated = bool(negated_hits)
                first = span[negated_hits[0] if negated_hits else hits[0]]  # report the negated keyword if any
                text = re.sub(r"\s+", " ", "".join(t.text_with_ws for t in span)).strip()
                found.setdefault(aspect, []).append(
                    Mention(aspect, first.lemma_.lower(), text, negated)
                )
    return found


def _label_from_probs(probs: dict) -> str:
    return max(LABELS, key=lambda label: probs.get(label, 0.0))


def _resolve(mentions, probs_by_sentence: dict, review_sentiment: str) -> dict:
    """Combine one aspect's mentions into {"sentiment", "keyword", "sentence", "negated"}."""
    if not mentions:  # keyword not found in the raw text (lemma mismatch): fall back to review level
        return {"sentiment": review_sentiment, "keyword": None, "sentence": None, "negated": False}

    scored = []
    for m in mentions:
        probs = dict(probs_by_sentence[m.sentence])
        if m.negated and _label_from_probs(probs) != "positive":
            probs = {"negative": 1.0, "neutral": 0.0, "positive": 0.0}  # negated positive keyword
        scored.append((m, probs))

    mean = {label: sum(p.get(label, 0.0) for _, p in scored) / len(scored) for label in LABELS}
    label = _label_from_probs(mean)
    best_mention, _ = max(scored, key=lambda mp: mp[1].get(label, 0.0))
    return {
        "sentiment": label,
        "keyword": best_mention.keyword,
        "sentence": best_mention.sentence,
        "negated": best_mention.negated,
    }


def compute_aspect_details(
    df,
    mode: str | None = None,
    nlp=None,
    scorer=None,
    split_clauses: bool | None = None,
    progress_callback=None,
):
    """Aspect-level sentiment for every review: a list (one dict per row) of {aspect: details}.

    `df` needs `aspects` (list per row), `sentiment` (review level) and either
    `clean_light` or the raw review text column. `details` has keys sentiment,
    keyword, sentence, negated. `progress_callback(fraction)` reports 0..1.
    """
    mode = validate_mode(mode or config.ASPECT_SENTIMENT_MODE)
    aspect_lists = list(df["aspects"])
    sentiments = list(df["sentiment"])

    if mode == "review":
        details = [
            {a: {"sentiment": s, "keyword": None, "sentence": None, "negated": False} for a in aspects}
            for aspects, s in zip(aspect_lists, sentiments)
        ]
        if progress_callback:
            progress_callback(1.0)
        return details

    if scorer is None:
        from src.sentiment.transformer_model import score_probs as scorer
    if split_clauses is None:
        split_clauses = config.ASPECT_SPLIT_CLAUSES
    nlp = nlp or get_sentence_nlp()

    texts = list(df["clean_light"]) if "clean_light" in df.columns else [
        light_clean(t) for t in df[config.REVIEW_TEXT_COL]
    ]
    todo = [i for i, aspects in enumerate(aspect_lists) if aspects]

    mentions_by_row = {}
    docs = nlp.pipe([texts[i] if isinstance(texts[i], str) else "" for i in todo], batch_size=64)
    for n, (i, doc) in enumerate(zip(todo, docs), 1):
        mentions_by_row[i] = find_mentions(doc, aspect_lists[i], split_clauses)
        if progress_callback and n % 200 == 0:
            progress_callback(0.3 * n / len(todo))

    def whole_review(i):
        return re.sub(r"\s+", " ", texts[i] if isinstance(texts[i], str) else "").strip()

    # A mention whose sentence IS the whole review was already scored (by the same model, on
    # the same text) when the review-level sentiment was computed: reuse it instead of re-scoring.
    unique_sentences = sorted(
        {
            m.sentence
            for i, per_aspect in mentions_by_row.items()
            for ms in per_aspect.values()
            for m in ms
            if m.sentence != whole_review(i)
        }
    )
    probs = {}
    if unique_sentences:
        scored = scorer(
            unique_sentences,
            progress_callback=(lambda done, total: progress_callback(0.3 + 0.7 * done / total))
            if progress_callback else None,
        )
        probs = dict(zip(unique_sentences, scored))

    details = []
    for i, aspects in enumerate(aspect_lists):
        per_aspect = mentions_by_row.get(i, {})
        row_probs = ChainMap(
            {whole_review(i): {label: float(label == sentiments[i]) for label in LABELS}}, probs
        )
        details.append(
            {a: _resolve(per_aspect.get(a, []), row_probs, sentiments[i]) for a in aspects}
        )
    if progress_callback:
        progress_callback(1.0)
    return details


def rebuild_dataset_aspects(
    dataset_id: str,
    mode: str | None = None,
    progress_callback=None,
    scorer=None,
    split_clauses: bool | None = None,
):
    """Recompute and store aspect-level sentiment for an already-analysed dataset.

    Uses the stored review text, review-level sentiment and aspect tags, so the full
    pipeline (cleaning, review sentiment, topics) is not re-run. Returns the stored key
    (see mode_key).
    """
    from src.db import database as db

    mode = validate_mode(mode or config.ASPECT_SENTIMENT_MODE)
    df = db.load_analysis(dataset_id)
    df = df.rename(columns={"review_text": config.REVIEW_TEXT_COL})
    details = compute_aspect_details(
        df, mode, scorer=scorer, split_clauses=split_clauses, progress_callback=progress_callback
    )

    rows = []
    for review_id, per_aspect in zip(df["id"], details):
        for aspect, d in per_aspect.items():
            rows.append((int(review_id), aspect, d["sentiment"], d["keyword"], d["sentence"], int(d["negated"])))
    key = mode_key(mode, split_clauses)
    db.replace_review_aspects(dataset_id, rows, key)
    return key
