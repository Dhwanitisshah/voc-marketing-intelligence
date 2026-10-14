import pytest

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


def _synthetic_df(n):
    import pandas as pd

    sentiments = ["positive", "negative", "neutral"]
    aspects = [["price"], ["delivery", "ux"], []]
    return pd.DataFrame(
        {
            "review_text": [f"review {i}" for i in range(n)],
            "clean_deep": [f"review {i}" for i in range(n)],
            "sentiment": [sentiments[i % 3] for i in range(n)],
            "confidence": [0.5 + (i % 5) / 10 for i in range(n)],
            "aspects": [aspects[i % 3] for i in range(n)],
            "rating": [float(i % 5 + 1) if i % 4 else float("nan") for i in range(n)],
            "date": ["2026-01-05" if i % 4 else None for i in range(n)],
        }
    )


def test_save_analysis_batches_keep_aspect_rows_aligned(tmp_db, monkeypatch):
    monkeypatch.setattr(db, "SAVE_BATCH_SIZE", 4)  # force several batches over 10 rows
    df = _synthetic_df(10)
    calls = []

    db.save_analysis(df, "ds1", progress_callback=lambda done, total: calls.append((done, total)))

    assert calls == [(4, 10), (8, 10), (10, 10)]
    loaded = db.load_analysis("ds1")
    assert list(loaded["review_text"]) == list(df["review_text"])
    assert list(loaded["aspects"]) == list(df["aspects"])
    # every review_aspects row points at a review that really has that aspect and sentiment
    conn = db._connect()
    try:
        rows = conn.execute(
            "SELECT ra.aspect, ra.sentiment, ar.aspects, ar.sentiment "
            "FROM review_aspects ra JOIN analyzed_reviews ar ON ar.id = ra.review_id"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == sum(len(a) for a in df["aspects"])
    assert all(aspect in aspects.split(",") and s == s2 for aspect, s, aspects, s2 in rows)


def test_save_analysis_missing_rating_and_date_become_null(tmp_db):
    db.save_analysis(_synthetic_df(8), "ds1")

    loaded = db.load_analysis("ds1")

    assert loaded["rating"].isna().sum() == 2 and loaded["review_date"].isna().sum() == 2
    assert loaded["review_date"].dropna().eq("2026-01-05").all()


def test_save_analysis_appends_without_id_collisions(tmp_db):
    db.save_analysis(_synthetic_df(5), "ds1")
    db.save_analysis(_synthetic_df(5), "ds2")

    ids = db.load_analysis("ds1")["id"].tolist() + db.load_analysis("ds2")["id"].tolist()

    assert len(set(ids)) == 10


def test_save_analysis_failure_rolls_back(tmp_db, monkeypatch):
    monkeypatch.setattr(db, "SAVE_BATCH_SIZE", 2)

    def boom(done, total):
        if done >= 4:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        db.save_analysis(_synthetic_df(6), "ds1", progress_callback=boom)

    assert db.dataset_exists("ds1") is False  # earlier batches were rolled back too
