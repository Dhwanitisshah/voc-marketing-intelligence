from src.db import database as db


def test_init_db_is_idempotent(tmp_db):
    db.init_db()
    db.init_db()  # should not raise or duplicate tables


def test_save_and_load_analysis_round_trips_rows_and_aspects(tmp_db, enriched_df):
    db.save_analysis(enriched_df, "ds1")

    loaded = db.load_analysis("ds1")

    assert len(loaded) == len(enriched_df)
    assert list(loaded["sentiment"]) == list(enriched_df["sentiment"])
    assert list(loaded["aspects"]) == list(enriched_df["aspects"])


def test_dataset_exists_true_and_false(tmp_db, enriched_df):
    assert db.dataset_exists("ds1") is False

    db.save_analysis(enriched_df, "ds1")

    assert db.dataset_exists("ds1") is True


def test_sentiment_counts_matches_inserted_data(tmp_db, enriched_df):
    db.save_analysis(enriched_df, "ds1")

    counts = db.sentiment_counts("ds1")

    expected = enriched_df["sentiment"].value_counts().to_dict()
    assert counts == expected


def test_aspect_sentiment_summary_group_by_counts(tmp_db, enriched_df):
    db.save_analysis(enriched_df, "ds1")

    summary = db.aspect_sentiment_summary("ds1").set_index("aspect")

    # Rebuild expected per-aspect counts directly from enriched_df for comparison.
    expected = {}
    for _, row in enriched_df.iterrows():
        for aspect in row["aspects"]:
            bucket = expected.setdefault(aspect, {"total": 0, "positive": 0, "neutral": 0, "negative": 0})
            bucket["total"] += 1
            bucket[row["sentiment"]] += 1

    for aspect, counts in expected.items():
        assert summary.loc[aspect, "total"] == counts["total"]
        assert summary.loc[aspect, "positive"] == counts["positive"]
        assert summary.loc[aspect, "neutral"] == counts["neutral"]
        assert summary.loc[aspect, "negative"] == counts["negative"]


def test_save_and_load_strategy_round_trips(tmp_db):
    strategy = {"executive_summary": "All good.", "recommendations": []}

    db.save_strategy("ds1", strategy)
    loaded = db.load_strategy("ds1")

    assert loaded == strategy


def test_load_strategy_returns_none_when_absent(tmp_db):
    assert db.load_strategy("nonexistent") is None
