"""Schema validation, grounding check, staleness detection and model fallback.

No test here makes a real Gemini call: `call_llm` is replaced, or the google-genai
client is swapped for a fake.
"""

import json

import pytest

from src.db import database as db
from src.strategy import generate_strategy as strategy_mod
from src.strategy.validation import (
    StrategyValidationError,
    check_grounding,
    parse_strategy,
)

VALID = {
    "executive_summary": "Price drives most negative mentions.",
    "recommendations": [
        {"aspect": "price", "finding": "f", "recommendation": "r", "priority": "high"}
    ],
}


# --- Schema ------------------------------------------------------------------


def test_parse_strategy_accepts_valid_json_and_fenced_json():
    assert parse_strategy(json.dumps(VALID)).recommendations[0].aspect == "price"
    assert parse_strategy("```json\n" + json.dumps(VALID) + "\n```").executive_summary


def test_parse_strategy_normalizes_priority_case():
    data = {**VALID, "recommendations": [{**VALID["recommendations"][0], "priority": "High"}]}

    assert parse_strategy(json.dumps(data)).recommendations[0].priority == "high"


@pytest.mark.parametrize(
    "payload, expected_fragment",
    [
        ("not json", "response"),
        (json.dumps({"executive_summary": "x"}), "recommendations"),
        (json.dumps({**VALID, "recommendations": []}), "recommendations"),
        (
            json.dumps({**VALID, "recommendations": [{**VALID["recommendations"][0], "priority": "urgent"}]}),
            "priority",
        ),
        (json.dumps({**VALID, "recommendations": [{"aspect": "price"}]}), "finding"),
    ],
)
def test_parse_strategy_rejects_schema_violations(payload, expected_fragment):
    with pytest.raises(StrategyValidationError) as exc:
        parse_strategy(payload)

    assert expected_fragment in str(exc.value)


def test_invalid_response_is_retried_once_then_succeeds(tmp_db, enriched_df, monkeypatch):
    db.save_analysis(enriched_df, "ds1")
    prompts = []
    responses = iter(["{}", json.dumps(VALID)])

    def fake_llm(system_instruction, user_prompt):
        prompts.append(user_prompt)
        return next(responses), "m"

    monkeypatch.setattr(strategy_mod, "call_llm", fake_llm)

    result = strategy_mod.generate_strategy("ds1")

    assert len(prompts) == 2
    assert "previous response was rejected" in prompts[1]
    assert result["recommendations"][0]["aspect"] == "price"


def test_two_invalid_responses_raise_and_do_not_retry_again(tmp_db, enriched_df, monkeypatch):
    db.save_analysis(enriched_df, "ds1")
    calls = []

    def fake_llm(system_instruction, user_prompt):
        calls.append(1)
        return json.dumps({"executive_summary": "x", "recommendations": "nope"}), "m"

    monkeypatch.setattr(strategy_mod, "call_llm", fake_llm)

    with pytest.raises(StrategyValidationError):
        strategy_mod.generate_strategy("ds1")

    assert len(calls) == 2


# --- Grounding check -----------------------------------------------------------

CONTEXT = {
    "dataset_id": "reviews_171e719d99",
    "total_reviews": 3000,
    "sentiment_counts": {"positive": 2675, "neutral": 176, "negative": 149},
    "aspect_breakdown": [
        {"aspect": "price", "total_mentions": 547, "positive": 488, "neutral": 24, "negative": 35,
         "negative_share": 0.064, "positive_share": 0.892, "impact": 35, "low_confidence": False},
        {"aspect": "delivery", "total_mentions": 45, "positive": 39, "neutral": 3, "negative": 3,
         "negative_share": 0.067, "positive_share": 0.867, "impact": 3, "low_confidence": False},
    ],
    "top_complaint_aspects": ["price", "delivery"],
    "negative_heavy_topics": [],
    "example_negative_reviews": {"price": ["Paid 20 dollars more than the listed price."]},
}


def _strategy(summary="", finding="", recommendation=""):
    return {
        "executive_summary": summary,
        "recommendations": [
            {"aspect": "price", "finding": finding, "recommendation": recommendation, "priority": "high"}
        ],
    }


def test_ungrounded_number_is_detected():
    strategy = _strategy(finding="Price had a 47% negative share across 547 mentions.")

    result = check_grounding(strategy, CONTEXT)

    assert result["total"] == 2
    assert result["ungrounded"] == ["47% (price finding)"]


def test_numbers_in_context_and_derivations_are_grounded():
    strategy = _strategy(
        summary="Of 3,000 reviews, 2,675 were positive and 149 negative (5.0% negative).",
        finding="35 of 547 price mentions were negative: 6.4%, or 0.064 as a share; 3 of 45 delivery (7%).",
        recommendation="Address the 2 top complaints; 38 negative mentions in total, 20 dollars in one snippet.",
    )

    result = check_grounding(strategy, CONTEXT)

    assert result["ungrounded"] == []
    assert result["total"] > 8


def test_ids_and_glued_tokens_are_not_treated_as_figures():
    strategy = _strategy(summary="Dataset reviews_171e719d99 shows 4K screens and 2nd-gen devices.")

    assert check_grounding(strategy, CONTEXT) == {"total": 0, "ungrounded": []}


def test_rounding_must_match_the_stated_precision():
    # 35/547 = 6.398% -> 6.4% is right; 6.9% is not.
    good = check_grounding(_strategy(finding="6.4% negative"), CONTEXT)
    bad = check_grounding(_strategy(finding="6.9% negative"), CONTEXT)

    assert good["ungrounded"] == []
    assert bad["ungrounded"] == ["6.9% (price finding)"]


def test_generate_strategy_attaches_grounding_model_and_hash(tmp_db, enriched_df, mock_llm):
    db.save_analysis(enriched_df, "ds1")

    result = strategy_mod.generate_strategy("ds1")

    assert result["model"] == "mock-model"
    assert result["context_hash"] == strategy_mod.hash_context(strategy_mod.build_context("ds1"))
    assert set(result["grounding"]) == {"total", "ungrounded"}


# --- Staleness ----------------------------------------------------------------


def test_strategy_is_fresh_until_ranking_input_changes(tmp_db, seed_reviews, mock_llm):
    seed_reviews("ds1", [(["price"], "negative", 40), (["price"], "positive", 460)])
    db.save_strategy("ds1", strategy_mod.generate_strategy("ds1"))

    assert strategy_mod.is_strategy_stale("ds1") is False

    # more negative delivery reviews -> a new aspect enters the ranking
    seed_reviews("ds1", [(["delivery"], "negative", 35)])

    assert strategy_mod.is_strategy_stale("ds1") is True


def test_strategy_becomes_stale_when_topics_change(tmp_db, seed_reviews, mock_llm):
    ids = seed_reviews("ds1", [(["price"], "negative", 10), (["price"], "positive", 40)])
    db.save_strategy("ds1", strategy_mod.generate_strategy("ds1"))
    assert strategy_mod.is_strategy_stale("ds1") is False

    topics = {
        "assignments": [0] * 10 + [-1] * 40,
        "topics": [{"topic_id": 0, "label": "refund", "keywords": ["refund"], "size": 10}],
    }
    db.save_topics("ds1", ids, topics)

    assert strategy_mod.is_strategy_stale("ds1") is True


def test_no_strategy_is_neither_fresh_nor_stale(tmp_db):
    assert strategy_mod.is_strategy_stale("missing") is None


def test_strategy_without_hash_counts_as_stale(tmp_db, seed_reviews):
    seed_reviews("ds1", [(["price"], "negative", 5)])
    db.save_strategy("ds1", VALID)  # legacy-style: no context_hash

    assert strategy_mod.is_strategy_stale("ds1") is True


def test_init_db_migrates_old_strategy_table(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE strategy (dataset_id TEXT PRIMARY KEY, strategy_json TEXT, created_at TEXT)")
    conn.execute("INSERT INTO strategy VALUES ('ds1', '{\"executive_summary\": \"old\"}', 'then')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)

    db.init_db()

    record = db.load_strategy_record("ds1")
    assert record["strategy"] == {"executive_summary": "old"}
    assert record["context_hash"] is None and record["model"] is None


# --- Model fallback (mocked google-genai client) ----------------------------------


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def _install_fake_client(monkeypatch, behaviours):
    """behaviours: {model_name: exception | text}. Returns the list of models called."""
    from google import genai

    called = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            called.append(model)
            outcome = behaviours[model]
            if isinstance(outcome, Exception):
                raise outcome
            return _FakeResponse(outcome)

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    return called


def _api_error(code):
    from google.genai import errors

    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"message": f"http {code}", "status": "TEST"}})


@pytest.mark.parametrize("code", [503, 429])
def test_retryable_error_falls_back_to_lite_model(monkeypatch, code):
    called = _install_fake_client(
        monkeypatch,
        {strategy_mod.DEFAULT_MODEL: _api_error(code), strategy_mod.DEFAULT_FALLBACK_MODEL: "from lite"},
    )

    text, model = strategy_mod.call_llm("sys", "prompt")

    assert (text, model) == ("from lite", strategy_mod.DEFAULT_FALLBACK_MODEL)
    assert called == [strategy_mod.DEFAULT_MODEL, strategy_mod.DEFAULT_FALLBACK_MODEL]


def test_successful_primary_does_not_touch_fallback(monkeypatch):
    called = _install_fake_client(monkeypatch, {strategy_mod.DEFAULT_MODEL: "from primary"})

    assert strategy_mod.call_llm("sys", "prompt") == ("from primary", strategy_mod.DEFAULT_MODEL)
    assert called == [strategy_mod.DEFAULT_MODEL]


def test_non_retryable_error_does_not_fall_back(monkeypatch):
    called = _install_fake_client(
        monkeypatch, {strategy_mod.DEFAULT_MODEL: _api_error(400), strategy_mod.DEFAULT_FALLBACK_MODEL: "x"}
    )

    with pytest.raises(Exception, match="400"):
        strategy_mod.call_llm("sys", "prompt")

    assert called == [strategy_mod.DEFAULT_MODEL]


def test_fallback_failure_propagates_after_exactly_one_retry(monkeypatch):
    called = _install_fake_client(
        monkeypatch,
        {strategy_mod.DEFAULT_MODEL: _api_error(503), strategy_mod.DEFAULT_FALLBACK_MODEL: _api_error(503)},
    )

    with pytest.raises(Exception, match="503"):
        strategy_mod.call_llm("sys", "prompt")

    assert called == [strategy_mod.DEFAULT_MODEL, strategy_mod.DEFAULT_FALLBACK_MODEL]


def test_primary_model_is_read_from_env(monkeypatch):
    called = _install_fake_client(monkeypatch, {"custom-model": "ok"})
    monkeypatch.setenv("GEMINI_MODEL", "custom-model")

    assert strategy_mod.call_llm("sys", "prompt") == ("ok", "custom-model")
    assert called == ["custom-model"]


def test_no_second_attempt_when_fallback_equals_primary(monkeypatch):
    called = _install_fake_client(monkeypatch, {"same": _api_error(503)})
    monkeypatch.setenv("GEMINI_MODEL", "same")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "same")

    with pytest.raises(Exception, match="503"):
        strategy_mod.call_llm("sys", "prompt")

    assert called == ["same"]
