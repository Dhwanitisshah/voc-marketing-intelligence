import json

import pytest

from src.db import database as db
from src.strategy import generate_strategy as strategy_mod


def _seed(tmp_db, enriched_df, dataset_id="ds1"):
    db.save_analysis(enriched_df, dataset_id)
    return dataset_id


def test_build_context_shape(tmp_db, enriched_df):
    dataset_id = _seed(tmp_db, enriched_df)

    context = strategy_mod.build_context(dataset_id)

    assert context["dataset_id"] == dataset_id
    assert context["total_reviews"] == len(enriched_df)
    assert context["sentiment_counts"] == db.sentiment_counts(dataset_id)
    assert isinstance(context["aspect_breakdown"], list)
    assert isinstance(context["top_complaint_aspects"], list)
    assert isinstance(context["example_negative_reviews"], dict)

    for aspect_entry in context["aspect_breakdown"]:
        assert {"aspect", "total_mentions", "positive", "neutral", "negative",
                "negative_share", "positive_share"} <= aspect_entry.keys()

    for aspect, snippets in context["example_negative_reviews"].items():
        assert aspect in context["top_complaint_aspects"]
        for snippet in snippets:
            assert len(snippet) <= 300


def test_generate_strategy_parses_mocked_json(tmp_db, enriched_df, mock_llm):
    dataset_id = _seed(tmp_db, enriched_df)

    result = strategy_mod.generate_strategy(dataset_id)

    assert "executive_summary" in result
    assert "recommendations" in result
    assert result["recommendations"][0]["aspect"] == "delivery"
    assert result["recommendations"][0]["priority"] == "high"


def test_strip_json_fences_handles_fenced_response():
    fenced = "```json\n" + json.dumps({"a": 1}) + "\n```"
    assert strategy_mod._strip_json_fences(fenced) == json.dumps({"a": 1})


def test_generate_strategy_falls_back_to_raw_on_invalid_json(tmp_db, enriched_df, monkeypatch):
    dataset_id = _seed(tmp_db, enriched_df)
    monkeypatch.setattr(strategy_mod, "call_llm", lambda system_instruction, user_prompt: "not json at all")

    result = strategy_mod.generate_strategy(dataset_id)

    assert result == {"raw": "not json at all"}


def test_call_llm_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        strategy_mod.call_llm("system", "prompt")
