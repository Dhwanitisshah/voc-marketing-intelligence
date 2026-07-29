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
