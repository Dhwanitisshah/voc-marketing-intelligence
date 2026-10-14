"""Impact-based complaint ranking, shared by the dashboard and the strategy layer.

Ranking aspects by negative *share* alone over-weights tiny samples (an aspect
with 2 mentions, 1 negative, is "50% negative"). Impact is the number of
negative mentions (mentions x negative_share), so a high-volume aspect with a
modest share outranks a rarely-mentioned one. Aspects with fewer than
MIN_MENTIONS mentions are flagged low-confidence and always sorted after the
well-supported ones.
"""

import pandas as pd

MIN_MENTIONS = 30


def rank_complaints(aspect_df: pd.DataFrame, min_mentions: int = MIN_MENTIONS) -> pd.DataFrame:
    """Rank aspects by impact (negative mention count), well-supported aspects first.

    `aspect_df` needs the columns produced by db.aspect_sentiment_summary:
    aspect, total, positive, neutral, negative. Returns a copy with added
    negative_share, positive_share, impact, and low_confidence columns, sorted
    so every aspect with >= min_mentions precedes every low-confidence one.
    """
    if aspect_df.empty:
        return aspect_df.assign(
            negative_share=[], positive_share=[], impact=[], low_confidence=[]
        )

    ranked = aspect_df.assign(
        negative_share=aspect_df["negative"] / aspect_df["total"],
        positive_share=aspect_df["positive"] / aspect_df["total"],
        impact=aspect_df["negative"],
        low_confidence=aspect_df["total"] < min_mentions,
    )
    return ranked.sort_values(
        ["low_confidence", "impact", "negative_share"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


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
