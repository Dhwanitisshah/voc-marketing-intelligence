"""Sentence-level aspect sentiment and negation. A fake scorer stands in for RoBERTa."""

import pandas as pd
import pytest

import config
from src.aspects import aspect_sentiment as A
from src.db import database as db

NEG_WORDS = ("terrible", "bad", "hate", "crash", "awful", "too much")
POS_WORDS = ("great", "love", "easy", "nice")


def fake_scorer_factory(calls=None, overrides=None):
    """Keyword-driven stand-in for score_probs; records every batch it is asked to score."""

    def scorer(sentences, progress_callback=None):
        if calls is not None:
            calls.append(list(sentences))
        out = []
        for s in sentences:
            low = s.lower()
            if overrides and low in overrides:
                out.append(overrides[low])
            elif any(w in low for w in NEG_WORDS):
                out.append({"negative": 0.8, "neutral": 0.1, "positive": 0.1})
            elif any(w in low for w in POS_WORDS):
                out.append({"negative": 0.1, "neutral": 0.1, "positive": 0.8})
            else:
                out.append({"negative": 0.1, "neutral": 0.8, "positive": 0.1})
        if progress_callback:
            progress_callback(len(sentences), len(sentences))
        return out

    return scorer


@pytest.fixture(scope="module")
def sentence_nlp():
    return A.get_sentence_nlp()


def _df(text, aspects, sentiment="negative"):
    return pd.DataFrame({"review_text": [text], "aspects": [aspects], "sentiment": [sentiment]})


def _ux(text, sentence_nlp, sentiment="negative", scorer=None, **kwargs):
    df = _df(text, ["ux"], sentiment)
    return A.compute_aspect_details(
        df, "sentence", nlp=sentence_nlp, scorer=scorer or fake_scorer_factory(), **kwargs
    )[0]["ux"]


# --- modes -------------------------------------------------------------------


def test_review_mode_inherits_review_sentiment():
    df = _df("Easy to set up. The speaker is terrible.", ["ux"], "negative")

    details = A.compute_aspect_details(df, "review")

    assert details == [{"ux": {"sentiment": "negative", "keyword": None, "sentence": None, "negated": False}}]


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="ASPECT_SENTIMENT_MODE"):
        A.compute_aspect_details(_df("x", []), "paragraph")


def test_default_mode_comes_from_config(monkeypatch, sentence_nlp):
    monkeypatch.setattr(config, "ASPECT_SENTIMENT_MODE", "review")

    d = A.compute_aspect_details(_df("Easy to set up. Speaker is terrible.", ["ux"]))[0]["ux"]

    assert d["sentiment"] == "negative" and d["sentence"] is None


# --- sentence-level scoring -----------------------------------------------------


def test_only_the_keyword_sentence_is_scored(sentence_nlp):
    calls = []
    d = _ux(
        "Echo was easy to set up. The speaker is terrible. Shipping was slow.",
        sentence_nlp, scorer=fake_scorer_factory(calls),
    )

    assert calls == [["Echo was easy to set up."]]  # not the speaker or shipping sentences
    assert d["sentiment"] == "positive"  # the review is negative, the UX mention is not
    assert d["keyword"] == "easy" and d["sentence"] == "Echo was easy to set up."


def test_complaint_sentence_is_negative_even_in_a_positive_review(sentence_nlp):
    d = _ux("Love this tablet. The apps crash all the time.", sentence_nlp, sentiment="positive")

    assert d["sentiment"] == "negative"
    assert d["sentence"] == "The apps crash all the time."


def test_multiple_sentences_are_averaged_into_one_mention(sentence_nlp):
    scorer = fake_scorer_factory(
        overrides={
            "apps are great.": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
            "apps keep crashing.": {"negative": 0.6, "neutral": 0.2, "positive": 0.2},
        }
    )

    d = _ux("Apps are great. Apps keep crashing.", sentence_nlp, scorer=scorer)

    assert d["sentiment"] == "positive"  # mean positive 0.55 beats mean negative 0.325
    assert d["sentence"] == "Apps are great."  # the evidence is the sentence behind the winning label


def test_single_sentence_review_reuses_review_score_without_rescoring(sentence_nlp):
    calls = []

    d = _ux("The app does not work well.", sentence_nlp, sentiment="negative", scorer=fake_scorer_factory(calls))

    assert calls == []  # the sentence is the whole review: already scored at review level
    assert d["sentiment"] == "negative"


def test_keyword_missing_from_raw_text_falls_back_to_review_sentiment(sentence_nlp):
    df = _df("A perfectly ordinary sentence.", ["delivery"], "neutral")

    d = A.compute_aspect_details(df, "sentence", nlp=sentence_nlp, scorer=fake_scorer_factory())[0]["delivery"]

    assert d == {"sentiment": "neutral", "keyword": None, "sentence": None, "negated": False}


def test_reviews_without_aspects_are_never_scored(sentence_nlp):
    calls = []

    details = A.compute_aspect_details(
        _df("Whatever.", []), "sentence", nlp=sentence_nlp, scorer=fake_scorer_factory(calls)
    )

    assert details == [{}] and calls == []


def test_clause_splitting_scores_only_the_keyword_clause(sentence_nlp):
    text = "The sound is terrible but it was easy to set up."
    whole = _ux(text, sentence_nlp, scorer=fake_scorer_factory(), split_clauses=False)  # "terrible" drags the sentence negative
    calls = []
    clause = _ux(text, sentence_nlp, scorer=fake_scorer_factory(calls), split_clauses=True)

    assert whole["sentiment"] == "negative" or whole["sentence"] == text
    assert calls == [["but it was easy to set up."]]
    assert clause["sentiment"] == "positive"


# --- negation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, aspect, keyword",
    [
        ("The tablet is not easy to use.", "ux", "easy"),
        ("The tablet is not incredibly easy to use or load.", "ux", "easy"),
        ("There is no value in this.", "price", "value"),
        ("It isn't worth the money.", "price", "worth"),
        ("The support staff were never helpful.", "support", "helpful"),
    ],
)
def test_negated_positive_keyword_is_a_negative_mention_even_if_scored_neutral(
    sentence_nlp, text, aspect, keyword
):
    # Prefix keeps the review from being a single sentence, so the scorer really runs.
    df = _df("Got it last week. " + text, [aspect], "negative")
    neutral = {"negative": 0.1, "neutral": 0.8, "positive": 0.1}
    scorer = fake_scorer_factory(overrides={text.lower(): neutral})

    d = A.compute_aspect_details(df, "sentence", nlp=sentence_nlp, scorer=scorer)[0][aspect]

    assert d["negated"] is True and d["sentiment"] == "negative" and d["keyword"] == keyword


def test_negation_does_not_override_a_positive_sentence(sentence_nlp):
    # "can't believe how easy" contains a negation cue but is praise; the model says positive.
    d = _ux("Got it last week. I can't believe how easy this is.", sentence_nlp, sentiment="positive")

    assert d["negated"] is True and d["sentiment"] == "positive"


def test_negation_stops_at_punctuation(sentence_nlp):
    d = _ux("Got it last week. Not good, easy to use though.", sentence_nlp, sentiment="positive")

    assert d["negated"] is False


def test_negating_a_neutral_keyword_does_not_force_negative(sentence_nlp):
    # "apps" is not a positive-polarity keyword: "not many apps" is left to the model.
    neutral = {"negative": 0.1, "neutral": 0.8, "positive": 0.1}
    scorer = fake_scorer_factory(overrides={"there are not many apps.": neutral})

    d = _ux("Got it last week. There are not many apps.", sentence_nlp, scorer=scorer)

    assert d["negated"] is False and d["sentiment"] == "neutral"


def test_is_negated_window_limits(sentence_nlp):
    doc = sentence_nlp("It is not at all a very easy thing")
    easy = next(t.i for t in doc if t.text == "easy")

    assert A.is_negated(doc, easy, window=3) is False  # "not" is 5 tokens away
    assert A.is_negated(doc, easy, window=6) is True


# --- persistence + mode invalidation -------------------------------------------------------


def _seed_review_mode(tmp_db):
    df = pd.DataFrame(
        {
            "review_text": [
                "Easy to set up. The speaker is terrible.",
                "Love it. The apps crash all the time.",
                "Nothing to report.",
            ],
            "clean_deep": ["easy set speaker terrible", "love app crash time", "nothing report"],
            "sentiment": ["negative", "positive", "neutral"],
            "confidence": [0.9, 0.9, 0.9],
            "aspects": [["ux"], ["ux"], []],
            "rating": [2, 4, 3],
            "date": ["2026-01-01"] * 3,
        }
    )
    db.save_analysis(df, "ds1")  # no aspect_details: review-level
    return df


def test_legacy_dataset_reports_review_mode_and_unknown_dataset_none(tmp_db):
    _seed_review_mode(tmp_db)

    assert db.get_aspect_mode("ds1") == "review"
    assert db.get_aspect_mode("nope") is None


def test_rebuild_switches_mode_and_changes_aspect_counts(tmp_db, sentence_nlp):
    _seed_review_mode(tmp_db)
    before = db.aspect_sentiment_summary("ds1").iloc[0]
    assert (before["positive"], before["negative"]) == (1, 1)  # review-level: one of each

    mode = A.rebuild_dataset_aspects("ds1", "sentence", scorer=fake_scorer_factory(), split_clauses=False)

    assert mode == "sentence" and db.get_aspect_mode("ds1") == "sentence"
    after = db.aspect_sentiment_summary("ds1").iloc[0]
    # review 1: "Easy to set up." -> positive; review 2: "The apps crash..." -> negative
    assert (after["positive"], after["negative"]) == (1, 1)
    conn = db._connect()
    try:
        rows = conn.execute(
            "SELECT review_id, sentiment, keyword, sentence FROM review_aspects ORDER BY review_id"
        ).fetchall()
    finally:
        conn.close()
    assert [(r[1], r[2]) for r in rows] == [("positive", "easy"), ("negative", "app")]
    assert rows[1][3] == "The apps crash all the time."


def test_rebuild_back_to_review_mode_restores_inherited_sentiment(tmp_db, sentence_nlp):
    _seed_review_mode(tmp_db)
    A.rebuild_dataset_aspects("ds1", "sentence", scorer=fake_scorer_factory())

    A.rebuild_dataset_aspects("ds1", "review")

    assert db.get_aspect_mode("ds1") == "review"
    summary = db.aspect_sentiment_summary("ds1").iloc[0]
    assert (summary["positive"], summary["negative"]) == (1, 1)


def test_negative_examples_prefer_the_evidence_sentence(tmp_db, sentence_nlp):
    _seed_review_mode(tmp_db)
    # review-level: the whole negative review, including its positive UX sentence
    assert db.negative_examples_for_aspect("ds1", "ux") == ["Easy to set up. The speaker is terrible."]

    A.rebuild_dataset_aspects("ds1", "sentence", scorer=fake_scorer_factory())

    assert db.negative_examples_for_aspect("ds1", "ux") == ["The apps crash all the time."]


def test_save_analysis_stores_aspect_details_and_mode(tmp_db):
    df = _seed_review_mode(tmp_db).assign(
        aspect_details=[
            {"ux": {"sentiment": "positive", "keyword": "easy", "sentence": "Easy to set up.", "negated": False}},
            {"ux": {"sentiment": "negative", "keyword": "app", "sentence": "Apps crash.", "negated": True}},
            {},
        ]
    )

    db.save_analysis(df, "ds2", aspect_mode="sentence")

    assert db.get_aspect_mode("ds2") == "sentence"
    conn = db._connect()
    try:
        rows = conn.execute(
            "SELECT sentiment, keyword, sentence, negated FROM review_aspects WHERE dataset_id='ds2' ORDER BY review_id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("positive", "easy", "Easy to set up.", 0), ("negative", "app", "Apps crash.", 1)]


def test_init_db_adds_aspect_columns_to_an_old_database(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE review_aspects (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, "
        "review_id INTEGER NOT NULL, aspect TEXT NOT NULL, sentiment TEXT)"
    )
    conn.execute("INSERT INTO review_aspects (dataset_id, review_id, aspect, sentiment) VALUES ('ds1', 1, 'ux', 'negative')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)

    db.init_db()

    conn = db._connect()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(review_aspects)")}
        old_row = conn.execute("SELECT sentiment, sentence FROM review_aspects").fetchone()
    finally:
        conn.close()
    assert {"keyword", "sentence", "negated"} <= cols
    assert old_row == ("negative", None)


@pytest.mark.slow
def test_score_probs_returns_distributions_in_input_order():
    from src.sentiment.transformer_model import score_probs

    texts = [
        "This is the most wonderful, well designed and delightful device I have ever owned, I love it so much",
        "Terrible.",
        "It arrived on Tuesday.",
    ]

    results = score_probs(texts, batch_size=2)

    assert len(results) == 3
    assert all(abs(sum(r.values()) - 1.0) < 1e-3 and set(r) == {"negative", "neutral", "positive"} for r in results)
    assert max(results[0], key=results[0].get) == "positive"
    assert max(results[1], key=results[1].get) == "negative"


def test_first_connection_migrates_an_old_database_without_init_db(tmp_path, monkeypatch):
    # Scripts / the strategy CLI never call init_db(); an old DB must still just work.
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE analyzed_reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, "
        "review_text TEXT, clean_deep TEXT, sentiment TEXT, confidence REAL, aspects TEXT, rating REAL, review_date TEXT)"
    )
    conn.execute(
        "CREATE TABLE review_aspects (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, "
        "review_id INTEGER NOT NULL, aspect TEXT NOT NULL, sentiment TEXT)"
    )
    conn.execute("INSERT INTO analyzed_reviews (dataset_id, review_text, sentiment, aspects) VALUES ('ds1', 'Slow app.', 'negative', 'ux')")
    conn.execute("INSERT INTO review_aspects (dataset_id, review_id, aspect, sentiment) VALUES ('ds1', 1, 'ux', 'negative')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(db, "_schema_ready", set())

    assert db.get_aspect_mode("ds1") == "review"                      # dataset_meta did not exist
    assert db.negative_examples_for_aspect("ds1", "ux") == ["Slow app."]  # ra.sentence did not exist


def test_clause_splitting_is_on_by_default():
    assert config.ASPECT_SPLIT_CLAUSES is True
    assert A.mode_key() == "sentence+clause"


@pytest.mark.parametrize(
    "mode, split, expected",
    [("review", True, "review"), ("review", False, "review"), ("sentence", False, "sentence"), ("sentence", True, "sentence+clause")],
)
def test_mode_key_combinations(mode, split, expected):
    assert A.mode_key(mode, split) == expected


def test_default_config_gives_clause_split_without_an_explicit_flag(sentence_nlp):
    calls = []

    d = _ux("The sound is terrible but it was easy to set up.", sentence_nlp, scorer=fake_scorer_factory(calls))

    assert calls == [["but it was easy to set up."]] and d["sentiment"] == "positive"


def test_changing_only_the_clause_flag_makes_a_cached_dataset_stale(tmp_db, sentence_nlp, monkeypatch):
    _seed_review_mode(tmp_db)
    A.rebuild_dataset_aspects("ds1", "sentence", scorer=fake_scorer_factory(), split_clauses=False)
    assert db.get_aspect_mode("ds1") == "sentence"

    # the app compares the stored key with mode_key(): with clauses now on, they differ -> rebuild
    assert db.get_aspect_mode("ds1") != A.mode_key()

    A.rebuild_dataset_aspects("ds1", "sentence", scorer=fake_scorer_factory())  # config default: clauses on
    assert db.get_aspect_mode("ds1") == A.mode_key() == "sentence+clause"
