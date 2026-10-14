import pytest

from src.aspects import topic_model
from src.db import database as db
from src.strategy import generate_strategy as strategy_mod

RESULT = {
    "assignments": [0, 0, 0, 1, 1, -1],
    "topics": [
        {"topic_id": 0, "label": "battery, life, charge", "keywords": ["battery", "life", "charge"], "size": 3},
        {"topic_id": 1, "label": "screen, glare", "keywords": ["screen", "glare"], "size": 2},
    ],
}
SPEC = [
    (["ux"], "positive", 2),     # ids 0-1 -> topic 0
    (["ux"], "negative", 1),     # id 2    -> topic 0
    (["price"], "negative", 2),  # ids 3-4 -> topic 1
    ([], "positive", 1),         # id 5    -> outlier
]


def test_topics_not_cached_until_saved(tmp_db, seed_reviews):
    seed_reviews("ds1", SPEC)

    assert db.topics_exist("ds1") is False
    assert db.load_topics("ds1").empty
    assert db.coverage_summary("ds1")["topic_matched"] is None


def test_save_and_load_topics_round_trip(tmp_db, seed_reviews):
    ids = seed_reviews("ds1", SPEC)

    db.save_topics("ds1", ids, RESULT)

    assert db.topics_exist("ds1") is True
    loaded = db.load_topics("ds1")
    assert list(loaded["topic_id"]) == [0, 1]
    assert list(loaded["size"]) == [3, 2]
    assert loaded.loc[0, "keywords"] == ["battery", "life", "charge"]

    summary = db.topic_sentiment_summary("ds1").set_index("topic_id")
    assert summary.loc[0, ["total", "positive", "negative"]].tolist() == [3, 2, 1]
    assert summary.loc[1, ["total", "positive", "negative"]].tolist() == [2, 0, 2]
    assert -1 not in summary.index  # outliers never appear as a topic

    examples = db.topic_examples("ds1", per_topic=2)
    assert set(examples) == {0, 1}
    assert len(examples[0]) == 2 and len(examples[1]) == 2


def test_coverage_summary_compares_lexicon_and_topics(tmp_db, seed_reviews):
    ids = seed_reviews("ds1", SPEC)
    db.save_topics("ds1", ids, RESULT)

    coverage = db.coverage_summary("ds1")

    assert coverage == {"total": 6, "lexicon_matched": 5, "topic_matched": 5}


def test_save_topics_replaces_previous_run_and_is_per_dataset(tmp_db, seed_reviews):
    ids1 = seed_reviews("ds1", SPEC)
    ids2 = seed_reviews("ds2", SPEC)
    db.save_topics("ds1", ids1, RESULT)
    db.save_topics("ds2", ids2, RESULT)

    rerun = {
        "assignments": [0] * 6,
        "topics": [{"topic_id": 0, "label": "all", "keywords": ["all"], "size": 6}],
    }
    db.save_topics("ds1", ids1, rerun)

    assert list(db.load_topics("ds1")["label"]) == ["all"]
    assert list(db.load_topics("ds2")["topic_id"]) == [0, 1]  # untouched


def test_strategy_context_includes_negative_topics_as_aggregates_only(tmp_db, seed_reviews):
    ids = seed_reviews("ds1", SPEC)
    db.save_topics("ds1", ids, RESULT)

    context = strategy_mod.build_context("ds1")

    topics = context["negative_heavy_topics"]
    # overall negative share is 3/6; topic 1 is 100% negative, topic 0 is 33% -> only topic 1.
    assert [t["topic"] for t in topics] == ["screen, glare"]
    assert set(topics[0]) == {"topic", "keywords", "size", "negative", "negative_share"}
    assert topics[0]["negative"] == 2 and topics[0]["negative_share"] == 1.0


def test_strategy_context_has_no_topics_without_cache(tmp_db, seed_reviews):
    seed_reviews("ds1", SPEC)

    assert strategy_mod.build_context("ds1")["negative_heavy_topics"] == []


def test_discover_topics_skips_tiny_datasets():
    assert topic_model.discover_topics(["a short review"] * (topic_model.MIN_DOCS - 1)) is None


@pytest.mark.slow
def test_discover_topics_returns_storable_result():
    themes = {
        "battery": "battery life drains charge charging power hours",
        "screen": "screen display glare bright resolution pixels",
        "setup": "setup install pairing wifi router instructions",
    }
    texts = [
        f"The {theme} is {i % 5} out of five, {words} {words.split()[i % 6]} again"
        for theme, words in themes.items()
        for i in range(30)
    ]

    result = topic_model.discover_topics(texts)

    assert result is not None
    assert len(result["assignments"]) == len(texts)
    assert result["topics"], "expected at least one non-outlier topic"
    assigned = sum(1 for t in result["assignments"] if t != topic_model.OUTLIER_TOPIC)
    assert sum(t["size"] for t in result["topics"]) == assigned
    sizes = [t["size"] for t in result["topics"]]
    assert sizes == sorted(sizes, reverse=True)
    assert all(t["keywords"] and t["label"] for t in result["topics"])
