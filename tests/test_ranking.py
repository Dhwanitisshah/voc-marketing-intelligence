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


# --- statistical closeness ---------------------------------------------------------------


def test_counts_within_counting_noise_are_close():
    from src.insights.ranking import counts_are_close

    assert counts_are_close(40, 35) is True      # 5 <= sqrt(40) + sqrt(35)
    assert counts_are_close(35, 20) is False     # 15 > 5.9 + 4.5
    assert counts_are_close(100, 60) is False
    assert counts_are_close(0, 0) is False       # nothing to compare


def test_adjacent_close_aspects_are_flagged_both_ways():
    df = _aspect_df([("ux", 755, 40), ("price", 547, 35), ("support", 150, 20)])

    ranked = rank_complaints(df).set_index("aspect")

    assert list(ranked["rank"]) == [1, 2, 3]
    assert ranked.loc["ux", "close_to"] == ["price"]
    assert ranked.loc["price", "close_to"] == ["ux"]        # 35 vs 20 is not close
    assert ranked.loc["support", "close_to"] == []


def test_low_confidence_tier_is_not_compared_with_well_supported_aspects():
    # 25 mentions -> low confidence: ranked after "price" even though 20 vs 18 negatives are close.
    df = _aspect_df([("price", 200, 18), ("packaging", 25, 20)])

    ranked = rank_complaints(df).set_index("aspect")

    assert ranked.loc["price", "close_to"] == [] and ranked.loc["packaging", "close_to"] == []


def test_describe_closeness_names_the_pair_and_denies_a_clear_first():
    from src.insights.ranking import describe_closeness

    df = _aspect_df([("ux", 755, 40), ("price", 547, 35), ("support", 150, 5)])

    note = describe_closeness(rank_complaints(df))

    assert "ux and price (40 vs 35)" in note
    assert "no clear #1" in note.lower()


def test_describe_closeness_is_none_when_the_order_is_clear():
    from src.insights.ranking import describe_closeness

    df = _aspect_df([("ux", 755, 100), ("price", 547, 35), ("support", 150, 5)])

    assert describe_closeness(rank_complaints(df)) is None


def test_clear_second_place_gap_still_notes_closeness_without_denying_first():
    from src.insights.ranking import describe_closeness

    df = _aspect_df([("ux", 755, 100), ("price", 547, 36), ("support", 150, 35)])

    note = describe_closeness(rank_complaints(df))

    assert "price and support" in note and "no clear #1" not in note.lower()


def test_strategy_context_exposes_closeness_to_the_model(tmp_db, seed_reviews):
    seed_reviews(
        "ds_close",
        [
            (["ux"], "negative", 40), (["ux"], "positive", 360),
            (["price"], "negative", 35), (["price"], "positive", 365),
            (["support"], "negative", 5), (["support"], "positive", 95),
        ],
    )

    context = strategy_mod.build_context("ds_close")

    by_aspect = {a["aspect"]: a for a in context["aspect_breakdown"]}
    assert by_aspect["ux"]["statistically_close_to"] == ["price"]
    assert by_aspect["ux"]["rank"] == 1
    assert by_aspect["support"]["statistically_close_to"] == []
    assert "no clear #1" in context["ranking_note"].lower()
    assert "statistically_close_to" in strategy_mod.SYSTEM_INSTRUCTION


def test_strategy_context_has_no_note_when_ranking_is_clear(tmp_db, seed_reviews):
    seed_reviews("ds_clear", [(["ux"], "negative", 60), (["price"], "negative", 10), (["price"], "positive", 90)])

    assert strategy_mod.build_context("ds_clear")["ranking_note"] is None
