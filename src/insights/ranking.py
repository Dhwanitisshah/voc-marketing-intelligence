"""Impact-based complaint ranking, shared by the dashboard and the strategy layer.

Ranking aspects by negative *share* alone over-weights tiny samples (an aspect
with 2 mentions, 1 negative, is "50% negative"). Impact is the number of
negative mentions (mentions x negative_share), so a high-volume aspect with a
modest share outranks a rarely-mentioned one. Aspects with fewer than
MIN_MENTIONS mentions are flagged low-confidence and always sorted after the
well-supported ones.
"""

import math

import pandas as pd

MIN_MENTIONS = 30
# Two adjacent counts are "statistically close" when their rough +/- CLOSE_SIGMA*sqrt(n)
# intervals (Poisson counting noise) overlap, i.e. |a - b| <= CLOSE_SIGMA * (sqrt(a) + sqrt(b)).
CLOSE_SIGMA = 1.0


def rank_complaints(aspect_df: pd.DataFrame, min_mentions: int = MIN_MENTIONS) -> pd.DataFrame:
    """Rank aspects by impact (negative mention count), well-supported aspects first.

    `aspect_df` needs the columns produced by db.aspect_sentiment_summary:
    aspect, total, positive, neutral, negative. Returns a copy with added
    negative_share, positive_share, impact, and low_confidence columns, sorted
    so every aspect with >= min_mentions precedes every low-confidence one.
    """
    if aspect_df.empty:
        return aspect_df.assign(
            negative_share=[], positive_share=[], impact=[], low_confidence=[], rank=[], close_to=[]
        )

    ranked = aspect_df.assign(
        negative_share=aspect_df["negative"] / aspect_df["total"],
        positive_share=aspect_df["positive"] / aspect_df["total"],
        impact=aspect_df["negative"],
        low_confidence=aspect_df["total"] < min_mentions,
    )
    ranked = ranked.sort_values(
        ["low_confidence", "impact", "negative_share"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["close_to"] = _close_neighbours(ranked)
    return ranked


def counts_are_close(a: float, b: float, sigma: float = CLOSE_SIGMA) -> bool:
    """Do two counts' rough +/- sigma*sqrt(n) intervals overlap (so their order isn't reliable)?"""
    if max(a, b) <= 0:
        return False
    return abs(a - b) <= sigma * (math.sqrt(a) + math.sqrt(b))


def _close_neighbours(ranked: pd.DataFrame) -> list:
    """For each row, the aspects directly above/below it (same confidence tier) that are statistically close."""
    names = ranked["aspect"].tolist()
    counts = ranked["impact"].tolist()
    tiers = ranked["low_confidence"].tolist()
    close = [[] for _ in names]
    for i in range(len(names) - 1):
        if tiers[i] == tiers[i + 1] and counts_are_close(counts[i], counts[i + 1]):
            close[i].append(names[i + 1])
            close[i + 1].append(names[i])
    return close


def describe_closeness(ranked: pd.DataFrame) -> str | None:
    """One sentence naming the adjacent aspects whose order isn't statistically reliable, or None."""
    if ranked.empty or "close_to" not in ranked.columns:
        return None
    pairs = []
    rows = ranked.to_dict("records")
    for upper, lower in zip(rows, rows[1:]):
        if lower["aspect"] in upper["close_to"]:
            pairs.append(f"{upper['aspect']} and {lower['aspect']} ({upper['impact']} vs {lower['impact']})")
    if not pairs:
        return None
    note = "Statistically close negative-mention counts, so their order is not reliable: " + "; ".join(pairs) + "."
    if rows and rows[0]["close_to"]:
        note += f" There is no clear #1: {rows[0]['aspect']} is not distinguishable from {rows[0]['close_to'][0]}."
    return note


def negative_heavy_topics(
    topic_df: pd.DataFrame, baseline_negative_share: float, limit: int = 5
) -> pd.DataFrame:
    """Topics whose negative share exceeds the dataset-wide baseline, by negative count.

    `topic_df` is db.topic_sentiment_summary. Over-indexing on the baseline keeps
    a big topic that is merely as negative as the dataset average out of the
    list; ranking by negative count (not share) mirrors the aspect impact ranking.
    """
    if topic_df.empty:
        return topic_df.assign(negative_share=[])

    scored = topic_df.assign(negative_share=topic_df["negative"] / topic_df["total"])
    heavy = scored[(scored["negative_share"] > baseline_negative_share) & (scored["negative"] > 0)]
    return heavy.sort_values(
        ["negative", "negative_share"], ascending=False, kind="mergesort"
    ).head(limit).reset_index(drop=True)
