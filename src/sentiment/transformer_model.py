"""Phase 2: transformer-based sentiment model.

Runs on clean_light text. Uses cardiffnlp/twitter-roberta-base-sentiment-latest,
a 3-class (negative/neutral/positive) RoBERTa model fine-tuned on tweets, which
outputs labels directly so no threshold mapping is needed.

If torch/model download is a problem (slow connection, disk space), a lighter
binary fallback is "distilbert-base-uncased-finetuned-sst-2-english" (POSITIVE/
NEGATIVE only, no neutral class) — swap MODEL_NAME below if needed.
"""

from tqdm import tqdm
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


def score_transformer(texts, batch_size=32, progress_callback=None):
    """Score a list of texts with the transformer model.

    Returns a list of (label, confidence) tuples. Loads the model once and
    scores in batches via the HuggingFace pipeline for efficiency. If given,
    `progress_callback(done, total)` is called after each batch.
    """
    texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
    pipe = _get_pipeline()

    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring reviews"):
        batch = texts[i:i + batch_size]
        outputs = pipe(batch, truncation=True, max_length=256)
        for out in outputs:
            label = _LABEL_MAP.get(out["label"].lower(), out["label"].lower())
            results.append((label, float(out["score"])))
        if progress_callback:
            progress_callback(len(results), len(texts))
    return results


def score_probs(texts, batch_size=32, progress_callback=None):
    """Class probabilities for each text: a list of {"negative": p, "neutral": p, "positive": p}.

    Same model and truncation as score_transformer, but keeps the whole distribution
    so several short texts (e.g. the sentences mentioning one aspect) can be averaged.
    Texts are batched in order of length (so batches pad to similar lengths, which is
    several times faster on CPU) and returned in the original order.
    `progress_callback(done, total)` is called after each batch.
    """
    texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
    pipe = _get_pipeline()

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results = [None] * len(texts)
    for lo in range(0, len(order), batch_size):
        idx = order[lo:lo + batch_size]
        outputs = pipe([texts[i] for i in idx], top_k=None, truncation=True, max_length=256)
        for i, scores in zip(idx, outputs):
            results[i] = {
                _LABEL_MAP.get(sc["label"].lower(), sc["label"].lower()): float(sc["score"])
                for sc in scores
            }
        if progress_callback:
            progress_callback(min(lo + batch_size, len(order)), len(order))
    return results
