import pandas as pd

from src.insights.ranking import MIN_MENTIONS, negative_heavy_topics, rank_complaints
from src.strategy import generate_strategy as strategy_mod


def _aspect_df(rows):
    return pd.DataFrame(
        [
            {"aspect": a, "total": t, "positive": t - n, "neutral": 0, "negative": n}
            for a, t, n in rows
        ],
        columns=["aspect", "total", "positive", "neutral", "negative"],
    )


def test_high_volume_price_outranks_low_volume_delivery():
    # delivery has the higher negative *share* (60% vs 8%) but fewer negative mentions.
    df = _aspect_df([("delivery", 50, 30), ("price", 500, 40)])

    ranked = rank_complaints(df)

    assert list(ranked["aspect"]) == ["price", "delivery"]
    assert list(ranked["impact"]) == [40, 30]
    # negative share is still reported alongside impact
    assert ranked.set_index("aspect").loc["delivery", "negative_share"] == 0.6


def test_old_share_ranking_would_have_put_delivery_first():
    df = _aspect_df([("delivery", 50, 30), ("price", 500, 40)])

    by_share = (df["negative"] / df["total"]).idxmax()

    assert df.loc[by_share, "aspect"] == "delivery"  # guards the premise of the test above


def test_low_sample_aspects_are_flagged():
    df = _aspect_df([("ux", MIN_MENTIONS, 3), ("packaging", MIN_MENTIONS - 1, 3)])

    flags = rank_complaints(df).set_index("aspect")["low_confidence"]

    assert bool(flags["ux"]) is False
    assert bool(flags["packaging"]) is True


def test_low_sample_never_ranks_above_well_supported():
    # packaging has more negative mentions (20) than price (5) but only 25 mentions.
    df = _aspect_df([("packaging", 25, 20), ("price", 200, 5)])

    ranked = rank_complaints(df)

    assert list(ranked["aspect"]) == ["price", "packaging"]


def test_min_mentions_is_configurable():
    df = _aspect_df([("packaging", 25, 20), ("price", 200, 5)])

    ranked = rank_complaints(df, min_mentions=10)

    assert list(ranked["aspect"]) == ["packaging", "price"]
    assert not ranked["low_confidence"].any()


def test_rank_complaints_handles_empty_frame():
    ranked = rank_complaints(_aspect_df([]))

    assert ranked.empty
    assert {"impact", "low_confidence", "negative_share"} <= set(ranked.columns)


def test_negative_heavy_topics_over_index_and_rank_by_count():
    topics = pd.DataFrame(
        [
            {"topic_id": 0, "label": "big", "total": 100, "negative": 5},    # 5% - at baseline
            {"topic_id": 1, "label": "mid", "total": 40, "negative": 12},    # 30%
            {"topic_id": 2, "label": "small", "total": 10, "negative": 6},   # 60%
            {"topic_id": 3, "label": "clean", "total": 30, "negative": 0},
        ]
    )

    heavy = negative_heavy_topics(topics, baseline_negative_share=0.05, limit=5)

    assert list(heavy["topic_id"]) == [1, 2]


def test_build_context_ranks_by_impact_and_flags_low_confidence(tmp_db, seed_reviews):
    seed_reviews(
        "ds_rank",
        [
            (["price"], "negative", 40),
            (["price"], "positive", 460),
            (["delivery"], "negative", 6),
            (["delivery"], "positive", 4),
        ],
    )

    context = strategy_mod.build_context("ds_rank")

    assert context["top_complaint_aspects"] == ["price", "delivery"]
    by_aspect = {a["aspect"]: a for a in context["aspect_breakdown"]}
    assert by_aspect["price"]["impact"] == 40
    assert by_aspect["price"]["low_confidence"] is False
    assert by_aspect["delivery"]["low_confidence"] is True
    assert by_aspect["delivery"]["negative_share"] == 0.6  # share still reported
    assert "low_confidence" in strategy_mod.SYSTEM_INSTRUCTION
    assert str(MIN_MENTIONS) in context["ranking_method"]
