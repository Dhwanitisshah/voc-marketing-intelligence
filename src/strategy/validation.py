"""Validation for the LLM strategy output: a pydantic schema and a numeric grounding check.

The schema guarantees the response has the shape the dashboard renders. The
grounding check goes after the prompt's "don't fabricate statistics" rule with
code: every figure that appears in the strategy text must appear in, or be
derivable from, the context we sent to the model.
"""

import re
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

__all__ = [
    "Recommendation",
    "Strategy",
    "StrategyValidationError",
    "check_grounding",
    "parse_strategy",
]


class StrategyValidationError(RuntimeError):
    """The model's response could not be parsed into the Strategy schema."""


class Recommendation(BaseModel):
    aspect: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    priority: Literal["high", "medium", "low"]

    @field_validator("priority", mode="before")
    @classmethod
    def _normalize_priority(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


class Strategy(BaseModel):
    executive_summary: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(min_length=1)


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


def parse_strategy(raw_text: str) -> Strategy:
    """Parse + validate the model response, raising StrategyValidationError with a readable reason."""
    try:
        return Strategy.model_validate_json(_strip_json_fences(raw_text))
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'response'}: {err['msg']}" for err in e.errors()
        )
        raise StrategyValidationError(
            f"The model's response did not match the expected strategy format ({problems})."
        ) from e


# --- Grounding check ---------------------------------------------------------

# A "figure": digits (optionally with thousands commas / decimals / a trailing %),
# not glued to letters, underscores or other digits -- so ids like "reviews_171e719d99"
# and tokens like "4K" or "2nd" are not mistaken for statistics.
_FIGURE_RE = re.compile(
    r"(?<![A-Za-z_\d.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)(?![A-Za-z_\d])"
)


def _figures(text: str):
    """Yield (display, value, decimals) for each figure in `text`."""
    for m in _FIGURE_RE.finditer(text):
        whole, frac, pct = m.group(1).replace(",", ""), m.group(2) or "", m.group(3)
        yield m.group(0), float(whole + frac), len(frac) - 1 if frac else 0


def _walk_numbers(node, out: set):
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.add(float(node))
        if isinstance(node, float) and 0 < node <= 1:
            out.add(node * 100)  # a share quoted as a percentage
    elif isinstance(node, str):
        out.update(value for _, value, _ in _figures(node))  # e.g. figures inside review snippets
    elif isinstance(node, dict):
        for v in node.values():
            _walk_numbers(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_numbers(v, out)


def _percent_variants(part: float, whole: float) -> list:
    return [100 * part / whole] if whole else []


def _allowed_values(context: dict) -> set:
    """Every number the model was given, plus simple arithmetic derivations of them."""
    allowed: set = set()
    _walk_numbers({k: v for k, v in context.items() if k != "dataset_id"}, allowed)

    total = context.get("total_reviews") or 0
    sentiments = context.get("sentiment_counts", {})
    allowed.update(sentiments.values())
    for count in sentiments.values():
        allowed.update(_percent_variants(count, total))

    def add_breakdown(counts: list, whole: float):
        # counts like [positive, neutral, negative]: each, each as % of whole, and sums of any subset.
        for r in range(1, len(counts) + 1):
            for subset in combinations(counts, r):
                s = sum(subset)
                allowed.add(float(s))
                allowed.update(_percent_variants(s, whole))

    add_breakdown(list(sentiments.values()), total)

    aspects = context.get("aspect_breakdown", [])
    for a in aspects:
        mentions = a.get("total_mentions", 0)
        add_breakdown([a.get("positive", 0), a.get("neutral", 0), a.get("negative", 0)], mentions)
        allowed.update(_percent_variants(mentions, total))
    if aspects:
        allowed.add(float(sum(a.get("negative", 0) for a in aspects)))
        top = set(context.get("top_complaint_aspects", []))
        allowed.add(float(sum(a.get("negative", 0) for a in aspects if a.get("aspect") in top)))

    for t in context.get("negative_heavy_topics", []):
        allowed.update(_percent_variants(t.get("negative", 0), t.get("size", 0)))
        allowed.update(_percent_variants(t.get("size", 0), total))

    for key in ("aspect_breakdown", "top_complaint_aspects", "negative_heavy_topics"):
        allowed.add(float(len(context.get(key, []))))

    return allowed


def _is_grounded(value: float, decimals: int, allowed: set) -> bool:
    tolerance = 0.5 * 10 ** -decimals + 1e-9  # the figure is a correctly rounded version of an allowed value
    return any(abs(a - value) <= tolerance for a in allowed)


def _strategy_texts(strategy: dict):
    yield "executive summary", strategy.get("executive_summary", "")
    for rec in strategy.get("recommendations", []):
        label = rec.get("aspect", "?")
        yield f"{label} finding", rec.get("finding", "")
        yield f"{label} recommendation", rec.get("recommendation", "")


def check_grounding(strategy: dict, context: dict) -> dict:
    """Verify every figure in the strategy text against the context sent to the model.

    Returns {"total": n_figures, "ungrounded": ["47% (support finding)", ...]}.
    """
    allowed = _allowed_values(context)
    total = 0
    ungrounded = []
    for where, text in _strategy_texts(strategy):
        for display, value, decimals in _figures(text):
            total += 1
            if not _is_grounded(value, decimals, allowed):
                ungrounded.append(f"{display} ({where})")
    return {"total": total, "ungrounded": ungrounded}
