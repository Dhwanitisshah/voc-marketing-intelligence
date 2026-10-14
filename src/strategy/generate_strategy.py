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

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.db import database as db
from src.insights.ranking import MIN_MENTIONS, negative_heavy_topics, rank_complaints

load_dotenv()

# "gemini-flash-latest" is the free-tier alias that always points at the
# current default Flash model; confirm the exact current free model ID in
# Google AI Studio's model list (https://aistudio.google.com/) as offerings
# change — pinned versions (e.g. gemini-2.5-flash) can be retired for new API
# keys. "gemini-flash-lite-latest" is an even lighter/cheaper free option if
# flash is rate-limited or too slow.
MODEL_NAME = "gemini-flash-latest"

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


def call_llm(system_instruction: str, user_prompt: str) -> str:
    """Isolated LLM call. Swap providers (e.g. a local Ollama model) by editing only this function."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and add it to a .env file in the project root as GEMINI_API_KEY=your_key_here."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    return response.text


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


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def generate_strategy(dataset_id: str) -> dict:
    context = build_context(dataset_id)

    user_prompt = (
        "Here is the computed review analytics data for this dataset:\n\n"
        f"{json.dumps(context, indent=2)}\n\n"
        f"{JSON_SCHEMA_HINT}"
    )

    raw_text = call_llm(SYSTEM_INSTRUCTION, user_prompt)

    cleaned = _strip_json_fences(raw_text)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": raw_text}

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
