"""Phase 2: transformer-based sentiment model.

Runs on clean_light text. Uses cardiffnlp/twitter-roberta-base-sentiment-latest,
a 3-class (negative/neutral/positive) RoBERTa model fine-tuned on tweets, which
outputs labels directly so no threshold mapping is needed.

If torch/model download is a problem (slow connection, disk space), a lighter
binary fallback is "distilbert-base-uncased-finetuned-sst-2-english" (POSITIVE/
NEGATIVE only, no neutral class) — swap MODEL_NAME below if needed.
"""

from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    # cardiffnlp historically also emits LABEL_0/1/2 depending on config version
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

_pipe = None


def _get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            framework="pt",
            truncation=True,
            max_length=256,
        )
    return _pipe


def score_transformer(texts, batch_size=32):
    """Score a list of texts with the transformer model.

    Returns a list of (label, confidence) tuples. Loads the model once and
    scores in batches via the HuggingFace pipeline for efficiency.
    """
    texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
    pipe = _get_pipeline()

    results = []
    outputs = pipe(texts, batch_size=batch_size, truncation=True, max_length=256)
    for out in outputs:
        label = _LABEL_MAP.get(out["label"].lower(), out["label"].lower())
        results.append((label, float(out["score"])))
    return results
