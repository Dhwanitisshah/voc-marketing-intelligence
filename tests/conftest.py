"""Shared fixtures for the test suite.

Isolation guarantees:
  - tmp_db points src.db.database at a throwaway sqlite file so no test ever
    touches the real data/voc.db.
  - mock_llm replaces src.strategy.generate_strategy.call_llm with a canned
    response so no test ever calls the live Gemini API.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.db import database as db
from src.preprocessing.clean_text import preprocess
from src.aspects.aspect_tagger import tag_dataframe

CANNED_STRATEGY_JSON = json.dumps(
    {
        "executive_summary": "Overall sentiment is mixed across 4 reviews, split 2 positive "
        "and 2 negative. Delivery is the top complaint at 50% negative share.",
        "recommendations": [
            {
                "aspect": "delivery",
                "finding": "Delivery had a 50% negative share across 2 mentions.",
                "recommendation": "Improve shipping time estimates and tracking updates.",
                "priority": "high",
            },
            {
                "aspect": "price",
                "finding": "Price had a 100% positive share across 1 mention.",
                "recommendation": "Promote price-to-value messaging in marketing copy.",
                "priority": "low",
            },
        ],
    }
)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point src.db.database at a throwaway sqlite file for the duration of the test."""
    db_path = tmp_path / "test_voc.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    yield db_path


@pytest.fixture
def sample_df():
    """A small hand-built dataframe touching every aspect keyword bucket."""
    return pd.DataFrame(
        {
            "review_text": [
                "Delivery was super fast, arrived a day early and packaging was solid.",
                "Shipping took over three weeks and support was unresponsive when I asked for updates.",
                "Price is a bit high compared to competitors but the quality justifies it.",
                "App UI is confusing, took me forever to find the checkout button.",
                "Absolutely disappointed with the quality, item broke after two days of use.",
                "Customer support was incredibly helpful and resolved my issue within minutes.",
            ],
            "rating": [5, 1, 4, 2, 1, 5],
            "date": [
                "2026-01-03",
                "2026-01-05",
                "2026-01-07",
                "2026-01-09",
                "2026-01-10",
                "2026-01-12",
            ],
        }
    )


@pytest.fixture(scope="session")
def _session_nlp():
    """Load spaCy once per test session so slow tests don't pay the load cost repeatedly."""
    import spacy

    return spacy.load("en_core_web_sm")


@pytest.fixture
def enriched_df(sample_df, _session_nlp):
    """sample_df run through the real preprocess + aspect tagger, with a hand-set fake
    sentiment/confidence column so DB tests never load the transformer model."""
    df = preprocess(sample_df, nlp=_session_nlp)
    df["sentiment"] = ["positive", "negative", "positive", "negative", "negative", "positive"]
    df["confidence"] = [0.9, 0.85, 0.7, 0.6, 0.95, 0.8]
    df = tag_dataframe(df)
    return df


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the live Gemini call with a canned valid JSON response."""
    import src.strategy.generate_strategy as strategy_mod

    def _fake_call_llm(system_instruction, user_prompt):
        return CANNED_STRATEGY_JSON

    monkeypatch.setattr(strategy_mod, "call_llm", _fake_call_llm)
    return CANNED_STRATEGY_JSON
