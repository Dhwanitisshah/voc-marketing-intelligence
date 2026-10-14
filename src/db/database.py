"""Phase 4: SQLite persistence for analyzed review datasets.

Every function opens its own short-lived connection (check_same_thread=False)
so this module is safe to call from Streamlit's rerun model without holding a
connection across reruns.

Aspects are stored two ways: a comma-joined string on `analyzed_reviews` (for
display / reconstructing the original list), and one row per (review, aspect)
in `review_aspects` (for GROUP BY aggregation without exploding strings in
Python on every query).
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config

DB_PATH = config.PROJECT_ROOT / "data" / "voc.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    """Create tables and indexes if they don't already exist."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyzed_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                review_text TEXT,
                clean_deep TEXT,
                sentiment TEXT,
                confidence REAL,
                aspects TEXT,
                rating REAL,
                review_date TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyzed_reviews_dataset "
            "ON analyzed_reviews (dataset_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_aspects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                review_id INTEGER NOT NULL,
                aspect TEXT NOT NULL,
                sentiment TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_aspects_dataset "
            "ON review_aspects (dataset_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy (
                dataset_id TEXT PRIMARY KEY,
                strategy_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                dataset_id TEXT NOT NULL,
                topic_id INTEGER NOT NULL,
                label TEXT,
                keywords TEXT,
                size INTEGER,
                created_at TEXT,
                PRIMARY KEY (dataset_id, topic_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_topics (
                dataset_id TEXT NOT NULL,
                review_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                PRIMARY KEY (dataset_id, review_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_topics_dataset_topic "
            "ON review_topics (dataset_id, topic_id)"
        )
        conn.commit()
    finally:
        conn.close()


def dataset_exists(dataset_id: str) -> bool:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM analyzed_reviews WHERE dataset_id = ? LIMIT 1",
            (dataset_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def save_analysis(df: pd.DataFrame, dataset_id: str):
    """Write an enriched (post-pipeline.analyze) dataframe under `dataset_id`."""
    rating_col = config.RATING_COL if config.RATING_COL in df.columns else None
    date_col = config.DATE_COL if config.DATE_COL in df.columns else None

    conn = _connect()
    try:
        cur = conn.cursor()
        for _, row in df.iterrows():
            aspects = row["aspects"] if isinstance(row["aspects"], list) else []
            cur.execute(
                """
                INSERT INTO analyzed_reviews
                    (dataset_id, review_text, clean_deep, sentiment, confidence, aspects, rating, review_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    row[config.REVIEW_TEXT_COL],
                    row.get("clean_deep", ""),
                    row["sentiment"],
                    float(row["confidence"]),
                    ",".join(aspects),
                    row[rating_col] if rating_col else None,
                    str(row[date_col]) if date_col else None,
                ),
            )
            review_id = cur.lastrowid
            for aspect in aspects:
                cur.execute(
                    """
                    INSERT INTO review_aspects (dataset_id, review_id, aspect, sentiment)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dataset_id, review_id, aspect, row["sentiment"]),
                )
        conn.commit()
    finally:
        conn.close()


def load_analysis(dataset_id: str) -> pd.DataFrame:
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM analyzed_reviews WHERE dataset_id = ? ORDER BY id",
            conn,
            params=(dataset_id,),
        )
        df["aspects"] = df["aspects"].apply(
            lambda s: s.split(",") if isinstance(s, str) and s else []
        )
        return df
    finally:
        conn.close()


def list_datasets() -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT dataset_id FROM analyzed_reviews ORDER BY dataset_id"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def aspect_sentiment_summary(dataset_id: str) -> pd.DataFrame:
    """Per-aspect total/positive/neutral/negative counts for `dataset_id`."""
    conn = _connect()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                aspect,
                COUNT(*) AS total,
                SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS negative
            FROM review_aspects
            WHERE dataset_id = ?
            GROUP BY aspect
            ORDER BY total DESC
            """,
            conn,
            params=(dataset_id,),
        )
        return df
    finally:
        conn.close()


def sentiment_counts(dataset_id: str) -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT sentiment, COUNT(*) FROM analyzed_reviews "
            "WHERE dataset_id = ? GROUP BY sentiment",
            (dataset_id,),
        ).fetchall()
        return {sentiment: count for sentiment, count in rows}
    finally:
        conn.close()


def negative_examples_for_aspect(dataset_id: str, aspect: str, limit: int = 3) -> list:
    """A few representative negative review snippets tagged with `aspect`."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT ar.review_text
            FROM review_aspects ra
            JOIN analyzed_reviews ar ON ar.id = ra.review_id
            WHERE ra.dataset_id = ? AND ra.aspect = ? AND ra.sentiment = 'negative'
            LIMIT ?
            """,
            (dataset_id, aspect, limit),
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


def save_strategy(dataset_id: str, strategy_dict: dict):
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO strategy (dataset_id, strategy_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                strategy_json = excluded.strategy_json,
                created_at = excluded.created_at
            """,
            (dataset_id, json.dumps(strategy_dict), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_strategy(dataset_id: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT strategy_json FROM strategy WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


# --- Discovered topics (BERTopic cache) -------------------------------------

OUTLIER_TOPIC = -1


def save_topics(dataset_id: str, review_ids: list, result: dict):
    """Replace the cached topics for `dataset_id`.

    `review_ids` are analyzed_reviews.id values, aligned with
    result["assignments"] (as returned by topic_model.discover_topics).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        conn.execute("DELETE FROM topics WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM review_topics WHERE dataset_id = ?", (dataset_id,))
        conn.executemany(
            "INSERT INTO topics (dataset_id, topic_id, label, keywords, size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (dataset_id, t["topic_id"], t["label"], json.dumps(t["keywords"]), t["size"], now)
                for t in result["topics"]
            ],
        )
        conn.executemany(
            "INSERT INTO review_topics (dataset_id, review_id, topic_id) VALUES (?, ?, ?)",
            [
                (dataset_id, int(review_id), int(topic_id))
                for review_id, topic_id in zip(review_ids, result["assignments"])
            ],
        )
        conn.commit()
    finally:
        conn.close()


def topics_exist(dataset_id: str) -> bool:
    """True once topic discovery has been run for this dataset (even if it found no topics)."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM review_topics WHERE dataset_id = ? LIMIT 1", (dataset_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def load_topics(dataset_id: str) -> pd.DataFrame:
    """Cached topics (topic_id, label, keywords list, size), largest first."""
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT topic_id, label, keywords, size FROM topics "
            "WHERE dataset_id = ? ORDER BY size DESC, topic_id",
            conn,
            params=(dataset_id,),
        )
        df["keywords"] = df["keywords"].apply(lambda s: json.loads(s) if s else [])
        return df
    finally:
        conn.close()


def topic_sentiment_summary(dataset_id: str) -> pd.DataFrame:
    """Per-topic positive/neutral/negative counts, joined with label/keywords/size."""
    conn = _connect()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                t.topic_id, t.label, t.keywords, t.size,
                COUNT(*) AS total,
                SUM(CASE WHEN ar.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN ar.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                SUM(CASE WHEN ar.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative
            FROM review_topics rt
            JOIN topics t ON t.dataset_id = rt.dataset_id AND t.topic_id = rt.topic_id
            JOIN analyzed_reviews ar ON ar.id = rt.review_id
            WHERE rt.dataset_id = ?
            GROUP BY t.topic_id, t.label, t.keywords, t.size
            ORDER BY t.size DESC, t.topic_id
            """,
            conn,
            params=(dataset_id,),
        )
        df["keywords"] = df["keywords"].apply(lambda s: json.loads(s) if s else [])
        return df
    finally:
        conn.close()


def topic_examples(dataset_id: str, per_topic: int = 3) -> dict:
    """{topic_id: [up to `per_topic` review texts]}, highest sentiment-confidence first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT topic_id, review_text FROM (
                SELECT rt.topic_id, ar.review_text,
                       ROW_NUMBER() OVER (
                           PARTITION BY rt.topic_id ORDER BY ar.confidence DESC, ar.id
                       ) AS rn
                FROM review_topics rt
                JOIN analyzed_reviews ar ON ar.id = rt.review_id
                WHERE rt.dataset_id = ? AND rt.topic_id != ? AND ar.review_text IS NOT NULL
            )
            WHERE rn <= ?
            ORDER BY topic_id, rn
            """,
            (dataset_id, OUTLIER_TOPIC, per_topic),
        ).fetchall()
        examples = {}
        for topic_id, text in rows:
            examples.setdefault(topic_id, []).append(text)
        return examples
    finally:
        conn.close()


def coverage_summary(dataset_id: str) -> dict:
    """How many reviews the 6-aspect lexicon matched vs how many got a non-outlier topic.

    topic_matched is None when topic discovery hasn't been run for the dataset.
    """
    conn = _connect()
    try:
        total, lexicon_matched = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN aspects IS NOT NULL AND aspects != '' THEN 1 ELSE 0 END) "
            "FROM analyzed_reviews WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        topic_matched = None
        if topics_exist(dataset_id):
            topic_matched = conn.execute(
                "SELECT COUNT(*) FROM review_topics WHERE dataset_id = ? AND topic_id != ?",
                (dataset_id, OUTLIER_TOPIC),
            ).fetchone()[0]
        return {
            "total": total or 0,
            "lexicon_matched": lexicon_matched or 0,
            "topic_matched": topic_matched,
        }
    finally:
        conn.close()
