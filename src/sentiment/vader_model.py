"""Phase 2: VADER lexicon-based sentiment baseline.

Runs on clean_light text (case, punctuation, and emojis are signal for VADER).
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

_analyzer = SentimentIntensityAnalyzer()


def _label_from_compound(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def score_vader(texts):
    """Score a list of texts with VADER.

    Returns a list of (compound_score, label) tuples.
    """
    results = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            results.append((0.0, "neutral"))
            continue
        compound = _analyzer.polarity_scores(text)["compound"]
        results.append((compound, _label_from_compound(compound)))
    return results
