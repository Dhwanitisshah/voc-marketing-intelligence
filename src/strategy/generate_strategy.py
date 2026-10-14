"""Phase 5: LLM strategy layer.

Turns the aspect x sentiment analysis already computed in Phase 3/4 into an
executive summary and prioritized marketing recommendations. The LLM reasons
only over aggregates and example snippets we hand it here — it never sees raw
data it could invent numbers from, and the prompt tells it not to fabricate
statistics.

Uses the free-tier Gemini API via the new unified `google-genai` SDK
(package: google-genai, import: `from google import genai`) — NOT the
deprecated `google-generativeai` package.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.db import database as db
from src.insights.ranking import MIN_MENTIONS, negative_heavy_topics, rank_complaints
from src.strategy.validation import (
    StrategyValidationError,
    check_grounding,
    parse_strategy,
)

load_dotenv()

# "gemini-flash-latest" is the free-tier alias that always points at the
# current default Flash model; confirm the exact current free model ID in
# Google AI Studio's model list (https://aistudio.google.com/) as offerings
# change — pinned versions (e.g. gemini-2.5-flash) can be retired for new API
# keys. Override with GEMINI_MODEL / GEMINI_FALLBACK_MODEL in .env.
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_FALLBACK_MODEL = "gemini-flash-lite-latest"
RETRYABLE_STATUS_CODES = (429, 503)  # rate limited / model overloaded


def primary_model() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def fallback_model() -> str:
    return os.environ.get("GEMINI_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL


SYSTEM_INSTRUCTION = """You are a marketing research analyst producing an executive brief from
customer review analytics. You will be given aggregate sentiment counts, per-aspect sentiment
breakdowns, discovered topic aggregates, and a handful of representative negative review snippets.

Rules:
- Every claim you make MUST be grounded strictly in the provided data. Cite the specific
  numbers/percentages given to you when making a claim (e.g. "42% of reviews mentioning
  shipping were negative").
- Do NOT fabricate, estimate, or round statistics that were not given to you.
- If the data is sparse or an aspect has few mentions, say so rather than overstating confidence.
- Aspects are ranked by impact (the number of negative mentions), not by negative share, and the
  top_complaint_aspects list is already in that order. Treat it as the priority order.
- An aspect with "low_confidence": true has fewer mentions than the minimum sample size. Never give
  it "high" priority; state explicitly that the sample is too small to be conclusive.
- "negative_heavy_topics" are themes discovered from the reviews that no keyword lexicon defined.
  They carry aggregates only. Use them to point at complaint themes the aspect list misses, but do not
  invent review content for them beyond their keywords and counts.
- Order "recommendations" to follow top_complaint_aspects (highest impact first); recommendations
  about negative_heavy_topics come after the aspect-based ones.
- Only use numbers that appear in the data (or simple percentages computed from its counts).
- Respond with clean JSON only, matching the requested schema exactly."""

JSON_SCHEMA_HINT = """Respond with JSON matching exactly this schema:
{
  "executive_summary": "3-4 sentence overview",
  "recommendations": [
    {
      "aspect": "string",
      "finding": "string, grounded in the given numbers",
      "recommendation": "string, concrete marketing action",
      "priority": "high" | "medium" | "low"
    }
  ]
}"""


def _status_code(exc: Exception):
    """HTTP status carried by a google-genai APIError (`code`), if any."""
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _generate(model: str, system_instruction: str, user_prompt: str, api_key: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    return response.text


def call_llm(system_instruction: str, user_prompt: str) -> tuple:
    """Isolated LLM call. Returns (response_text, model_used).

    Tries the primary model; on a 429/503 retries once on the fallback model.
    Any other error, or a failure of the fallback, propagates to the caller.
    Swap providers (e.g. a local Ollama model) by editing only this function.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and add it to a .env file in the project root as GEMINI_API_KEY=your_key_here."
        )

    model = primary_model()
    try:
        return _generate(model, system_instruction, user_prompt, api_key), model
    except Exception as exc:
        fallback = fallback_model()
        if _status_code(exc) not in RETRYABLE_STATUS_CODES or fallback == model:
            raise
        return _generate(fallback, system_instruction, user_prompt, api_key), fallback


def build_context(dataset_id: str, max_aspects: int = 3, examples_per_aspect: int = 3) -> dict:
    """Pull grounding data from SQLite: overall counts, per-aspect breakdown, and a few
    representative negative snippets for the top complaint aspects."""
    counts = db.sentiment_counts(dataset_id)
    total = sum(counts.values())

    aspect_df = db.aspect_sentiment_summary(dataset_id)

    aspects = []
    top_complaint_aspects = []
    if not aspect_df.empty:
        ranked = rank_complaints(aspect_df)
        for _, row in ranked.iterrows():
            aspects.append(
                {
                    "aspect": row["aspect"],
                    "total_mentions": int(row["total"]),
                    "positive": int(row["positive"]),
                    "neutral": int(row["neutral"]),
                    "negative": int(row["negative"]),
                    "negative_share": round(float(row["negative_share"]), 3),
                    "positive_share": round(float(row["positive_share"]), 3),
                    "impact": int(row["impact"]),
                    "low_confidence": bool(row["low_confidence"]),
                }
            )
        top_complaint_aspects = ranked.head(max_aspects)["aspect"].tolist()

    examples = {}
    for aspect in top_complaint_aspects:
        snippets = db.negative_examples_for_aspect(dataset_id, aspect, limit=examples_per_aspect)
        # Bound total example text so the prompt stays compact.
        examples[aspect] = [s[:300] for s in snippets]

    return {
        "dataset_id": dataset_id,
        "total_reviews": total,
        "sentiment_counts": counts,
        "ranking_method": (
            "aspects ranked by impact = negative mention count; aspects with fewer than "
            f"{MIN_MENTIONS} mentions are flagged low_confidence and ranked after the rest"
        ),
        "aspect_breakdown": aspects,
        "top_complaint_aspects": top_complaint_aspects,
        "negative_heavy_topics": _negative_topic_aggregates(dataset_id, counts, total),
        "example_negative_reviews": examples,
    }


def _negative_topic_aggregates(dataset_id: str, counts: dict, total: int, limit: int = 5) -> list:
    """Aggregates (no review text) for the topics most over-indexed on negative sentiment.

    Empty when topic discovery hasn't been run for the dataset.
    """
    if not total or not db.topics_exist(dataset_id):
        return []

    baseline = counts.get("negative", 0) / total
    heavy = negative_heavy_topics(db.topic_sentiment_summary(dataset_id), baseline, limit=limit)
    return [
        {
            "topic": row["label"],
            "keywords": list(row["keywords"]),
            "size": int(row["total"]),
            "negative": int(row["negative"]),
            "negative_share": round(float(row["negative_share"]), 3),
        }
        for _, row in heavy.iterrows()
    ]


def hash_context(context: dict) -> str:
    """Stable fingerprint of the exact context handed to the model."""
    canonical = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_strategy_stale(dataset_id: str, record: dict | None = None) -> bool | None:
    """Has the data behind a cached strategy changed since it was generated?

    None if no strategy exists. True if the context hash differs, or if the
    strategy predates hash tracking (so freshness can't be confirmed).
    """
    record = record or db.load_strategy_record(dataset_id)
    if record is None:
        return None
    if not record["context_hash"]:
        return True
    return record["context_hash"] != hash_context(build_context(dataset_id))


def generate_strategy(dataset_id: str) -> dict:
    """Generate a validated strategy.

    Returns the Strategy fields plus "model" (which model answered),
    "context_hash" (see is_strategy_stale) and "grounding" (see
    validation.check_grounding). Raises StrategyValidationError if the model
    returns an invalid response twice.
    """
    context = build_context(dataset_id)

    user_prompt = (
        "Here is the computed review analytics data for this dataset:\n\n"
        f"{json.dumps(context, indent=2)}\n\n"
        f"{JSON_SCHEMA_HINT}"
    )

    raw_text, model = call_llm(SYSTEM_INSTRUCTION, user_prompt)
    try:
        parsed = parse_strategy(raw_text)
    except StrategyValidationError as first_error:
        retry_prompt = (
            f"{user_prompt}\n\nYour previous response was rejected: {first_error} "
            "Respond again with ONLY valid JSON matching the schema exactly."
        )
        raw_text, model = call_llm(SYSTEM_INSTRUCTION, retry_prompt)
        parsed = parse_strategy(raw_text)  # a second failure surfaces as the error

    result = parsed.model_dump()
    result["model"] = model
    result["context_hash"] = hash_context(context)
    result["grounding"] = check_grounding(result, context)
    return result


if __name__ == "__main__":
    datasets = db.list_datasets()
    if not datasets:
        print("No datasets found in data/voc.db. Run the dashboard or a pipeline script first.")
        sys.exit(0)

    for ds in datasets:
        print(f"\n{'=' * 80}\nDataset: {ds}\n{'=' * 80}")
        strategy = generate_strategy(ds)
        print(json.dumps(strategy, indent=2))
