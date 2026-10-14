"""Deployment prep: settings precedence, demo database, lazy heavy imports, pinned requirements."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import config
from src import settings
from src.db import database as db
from src.strategy import generate_strategy as strategy_mod

ROOT = Path(__file__).resolve().parents[1]


# --- settings: st.secrets first, then environment -------------------------------------------------------


def test_secret_takes_priority_over_environment(monkeypatch):
    monkeypatch.setattr(settings, "_streamlit_secret", lambda name: "from-secrets")
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    assert settings.get_setting("GEMINI_API_KEY") == "from-secrets"


def test_environment_is_the_fallback_then_the_default(monkeypatch):
    monkeypatch.setattr(settings, "_streamlit_secret", lambda name: None)
    monkeypatch.setenv("SOME_SETTING", "from-env")
    monkeypatch.delenv("MISSING_SETTING", raising=False)

    assert settings.get_setting("SOME_SETTING") == "from-env"
    assert settings.get_setting("MISSING_SETTING", "fallback") == "fallback"
    assert settings.get_setting("MISSING_SETTING") is None


def test_empty_secret_does_not_shadow_the_environment(monkeypatch):
    monkeypatch.setattr(settings, "_streamlit_secret", lambda name: "")
    monkeypatch.setenv("SOME_SETTING", "from-env")

    assert settings.get_setting("SOME_SETTING") == "from-env"


def test_streamlit_secret_lookup_never_raises_without_a_secrets_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .streamlit/secrets.toml here

    assert settings._streamlit_secret("DEFINITELY_NOT_A_SECRET") is None


def test_int_setting_parses_and_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "_streamlit_secret", lambda name: None)
    monkeypatch.setenv("CAP", "2000")
    monkeypatch.setenv("BAD", "lots")

    assert settings.get_int_setting("CAP") == 2000
    assert settings.get_int_setting("BAD", 5) == 5
    assert settings.get_int_setting("UNSET_INT", 7) == 7


def test_gemini_key_is_read_from_secrets_when_the_environment_has_none(monkeypatch):
    from google import genai

    seen = {}

    class FakeClient:
        def __init__(self, api_key):
            seen["key"] = api_key
            self.models = self

        def generate_content(self, model, contents, config):
            return type("R", (), {"text": "ok"})()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "_streamlit_secret", lambda name: "secret-key" if name == "GEMINI_API_KEY" else None)

    assert strategy_mod.call_llm("sys", "prompt")[0] == "ok"
    assert seen["key"] == "secret-key"


def test_missing_key_everywhere_still_gives_the_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "_streamlit_secret", lambda name: None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        strategy_mod.call_llm("sys", "prompt")


# --- demo database seeding ------------------------------------------------------------------------------------


def test_demo_db_is_copied_when_there_is_no_database_yet(tmp_path, monkeypatch):
    demo = tmp_path / "demo" / "voc_demo.db"
    demo.parent.mkdir()
    demo.write_bytes(b"demo-bytes")
    target = tmp_path / "data" / "voc.db"
    monkeypatch.setattr(config, "DEMO_DB_PATH", demo)
    monkeypatch.setattr(db, "DB_PATH", target)

    assert db.ensure_demo_db() is True
    assert target.read_bytes() == b"demo-bytes"


def test_existing_database_is_never_overwritten_by_the_demo(tmp_path, monkeypatch):
    demo = tmp_path / "voc_demo.db"
    demo.write_bytes(b"demo-bytes")
    target = tmp_path / "voc.db"
    target.write_bytes(b"my-real-data")
    monkeypatch.setattr(config, "DEMO_DB_PATH", demo)
    monkeypatch.setattr(db, "DB_PATH", target)

    assert db.ensure_demo_db() is False
    assert target.read_bytes() == b"my-real-data"


def test_nothing_happens_when_no_demo_ships(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_DB_PATH", tmp_path / "absent.db")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "voc.db")

    assert db.ensure_demo_db() is False and not (tmp_path / "voc.db").exists()


@pytest.mark.skipif(not config.DEMO_DB_PATH.exists(), reason="demo database not built (scripts/build_demo_db.py)")
def test_committed_demo_database_is_complete_and_current(tmp_path, monkeypatch):
    """Guards against shipping a demo that would trigger a 100 s rebuild or a stale strategy on first load."""
    from src.aspects.aspect_sentiment import mode_key

    copy = tmp_path / "voc.db"
    shutil.copyfile(config.DEMO_DB_PATH, copy)
    monkeypatch.setattr(db, "DB_PATH", copy)
    ds = config.DEFAULT_DATASET_ID

    assert db.dataset_exists(ds)
    assert len(db.load_analysis(ds)) >= 500
    assert db.get_aspect_mode(ds) == mode_key(), "demo was built under a different aspect setting: rerun build_demo_db.py"
    assert db.topics_exist(ds) and len(db.load_topics(ds)) >= 5
    assert not db.aspect_sentiment_summary(ds).empty
    if db.load_strategy_record(ds):
        assert strategy_mod.is_strategy_stale(ds) is False, "cached demo strategy is out of date: rerun build_demo_db.py"
    assert copy.stat().st_size < 5 * 1024 * 1024


# --- hosting weight ------------------------------------------------------------------------------------------------


def test_viewing_the_app_does_not_import_torch_or_transformers():
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import src.pipeline, src.sentiment.transformer_model, src.aspects.aspect_sentiment, src.db.database;"
        "print('torch' in sys.modules or 'transformers' in sys.modules)"
    )

    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# --- packaging -------------------------------------------------------------------------------------------------------


def _requirement_lines(name):
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def test_requirements_are_pinned_and_cpu_only():
    lines = _requirement_lines("requirements.txt")
    packages = [l for l in lines if not l.startswith("--")]

    assert any(l == "--index-url https://download.pytorch.org/whl/cpu" for l in lines)
    assert any(l.startswith("--extra-index-url https://pypi.org") for l in lines)
    unpinned = [l for l in packages if not (re.match(r"^[A-Za-z0-9_.\-]+==[\w.+]+$", l) or l.startswith("https://"))]
    assert not unpinned, f"unpinned requirements: {unpinned}"
    assert any(l.startswith("torch==") for l in packages)


def test_dev_requirements_extend_the_runtime_ones():
    assert "-r requirements.txt" in _requirement_lines("requirements-dev.txt")
    assert any(l.startswith("pytest==") for l in _requirement_lines("requirements-dev.txt"))


def test_ci_runs_the_fast_suite_only():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'pytest -m "not slow"' in ci and "requirements-dev.txt" in ci


def test_dockerfile_is_cpu_non_root_and_honours_the_port():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "requirements.txt" in docker and "USER user" in docker and "${PORT}" in docker
    assert "fileWatcherType=none" in docker


def test_secrets_file_is_ignored_but_the_example_is_shipped():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".streamlit/secrets.toml" in ignore
    assert (ROOT / ".streamlit" / "secrets.toml.example").exists()
