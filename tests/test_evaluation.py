"""Labeling helpers, label_app.py, agreement statistics and the human-label report."""

import csv
import sys
from pathlib import Path

import pytest

from src.evaluation import agreement as ag
from src.evaluation import labeling as lb
from src.evaluation import report as rp

FIELDS = [
    "review_id", "aspect", "keyword", "context_window", "full_review",
    "draft_label", "draft_reason", "human_label",
    "neg_review", "neg_sentence", "neg_sentence_clause",
]


def make_row(i, aspect="ux", draft="genuine", human="", flags=(1, 1, 1), text=None):
    return {
        "review_id": str(i), "aspect": aspect, "keyword": "app|ui",
        "context_window": f"... the [[app]] is slow, \"really\", {i} ...",
        "full_review": text or f"Review {i}: the app is slow,\nsecond line with a comma, and \"quotes\".",
        "draft_label": draft, "draft_reason": f"reason {i}", "human_label": human,
        "neg_review": flags[0], "neg_sentence": flags[1], "neg_sentence_clause": flags[2],
    }


def write_csv(path, rows, fields=FIELDS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "audit.csv"
    write_csv(path, [make_row(1), make_row(2, draft="inherited"), make_row(3, aspect="price", draft="false_trigger")])
    return path


# --- saving labels ---------------------------------------------------------------------------


def test_save_label_changes_only_the_human_label_of_one_row(csv_path):
    _, before = lb.load_rows(csv_path)

    lb.save_label(csv_path, ("2", "ux"), "false_trigger", ["review_id", "aspect"])

    _, after = lb.load_rows(csv_path)
    assert after[1]["human_label"] == "false_trigger"
    for i, (b, a) in enumerate(zip(before, after)):
        changed = {k for k in b if b[k] != a[k]}
        assert changed == ({"human_label"} if i == 1 else set())  # draft_label et al. untouched


def test_save_label_key_includes_aspect_so_same_review_id_is_not_confused(tmp_path):
    path = tmp_path / "a.csv"
    write_csv(path, [make_row(7, aspect="ux"), make_row(7, aspect="price")])

    lb.save_label(path, ("7", "price"), "genuine", ["review_id", "aspect"])

    _, rows = lb.load_rows(path)
    assert [r["human_label"] for r in rows] == ["", "genuine"]


def test_save_label_can_clear_and_adds_a_missing_label_column(tmp_path):
    path = tmp_path / "a.csv"
    fields = [f for f in FIELDS if f != "human_label"]
    write_csv(path, [{k: v for k, v in make_row(1).items() if k in fields}], fields)

    lb.save_label(path, ("1", "ux"), "genuine", ["review_id", "aspect"])
    assert lb.load_rows(path)[1][0]["human_label"] == "genuine"
    lb.save_label(path, ("1", "ux"), "", ["review_id", "aspect"])
    assert lb.load_rows(path)[1][0]["human_label"] == ""


def test_save_label_unknown_row_raises_and_leaves_file_alone(csv_path):
    before = csv_path.read_bytes()

    with pytest.raises(KeyError):
        lb.save_label(csv_path, ("999", "ux"), "genuine", ["review_id", "aspect"])

    assert csv_path.read_bytes() == before


def test_multiline_quoted_cells_survive_a_save_round_trip(csv_path):
    lb.save_label(csv_path, ("1", "ux"), "genuine", ["review_id", "aspect"])

    _, rows = lb.load_rows(csv_path)

    assert rows[0]["full_review"] == make_row(1)["full_review"]
    assert rows[0]["context_window"] == make_row(1)["context_window"]


def test_no_temp_files_are_left_behind(csv_path):
    lb.save_label(csv_path, ("1", "ux"), "genuine", ["review_id", "aspect"])

    assert [p.name for p in csv_path.parent.iterdir()] == ["audit.csv"]


def test_progress_and_next_unlabelled():
    rows = [make_row(1, human="genuine"), make_row(2), make_row(3, human="inherited"), make_row(4)]

    assert lb.labelled_count(rows) == 2
    assert lb.next_unlabelled(rows, after=0) == 1
    assert lb.next_unlabelled(rows, after=1) == 3
    assert lb.next_unlabelled(rows, after=3) == 1                      # wraps
    assert lb.next_unlabelled(rows, after=0, allowed={0, 2, 3}) == 3   # honours the aspect filter
    assert lb.next_unlabelled([make_row(1, human="genuine")], after=0) is None


# --- highlighting ---------------------------------------------------------------------------------


def test_highlight_escapes_html_before_marking():
    out = lb.highlight_html("<script>alert(1)</script> the [[app]] & more")

    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<mark>app</mark>" in out and "&amp;" in out


def test_highlight_marks_keyword_prefixes_case_insensitively_without_markers():
    out = lb.highlight_html("Apps are ok. The UI is fine. Pineapple.", ["app", "ui"], marker_pairs=False)

    assert "<mark>Apps</mark>" in out and "<mark>UI</mark>" in out
    assert "Pineapple" in out and "<mark>Pineapple" not in out  # only word starts


def test_highlight_without_keywords_or_markers_is_just_escaped_text():
    assert lb.highlight_html("a < b") == "a &lt; b"


def test_split_keywords():
    assert lb.split_keywords("app|ui") == ["app", "ui"] and lb.split_keywords("") == []


# --- label_app.py (driven through Streamlit's AppTest) ---------------------------------------------


def _app(csv_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["label_app.py", "--csv", str(csv_path)])
    script = Path(__file__).resolve().parents[1] / "scripts" / "label_app.py"
    at = AppTest.from_file(str(script), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _button(at, label):
    return [b for b in at.button if b.label == label][0]


def test_label_app_saves_after_each_click_and_advances(csv_path, monkeypatch):
    at = _app(csv_path, monkeypatch)

    _button(at, "genuine").click().run()
    assert lb.load_rows(csv_path)[1][0]["human_label"] == "genuine"
    assert "Row 2 of 3" in at.markdown[0].value  # moved on to the next unlabelled row

    _button(at, "false_trigger").click().run()
    _button(at, "skip").click().run()
    _, rows = lb.load_rows(csv_path)
    assert [r["human_label"] for r in rows] == ["genuine", "false_trigger", ""]
    assert [r["draft_label"] for r in rows] == ["genuine", "inherited", "false_trigger"]  # never touched


def test_label_app_hides_the_draft_label_in_a_collapsed_expander(csv_path, monkeypatch):
    at = _app(csv_path, monkeypatch)

    expanders = [e for e in at.expander if "draft" in e.label.lower()]
    assert len(expanders) == 1 and expanders[0].proto.expanded is False

    inside = [m.value for m in expanders[0].markdown] + [c.value for c in expanders[0].caption]
    assert any("reason 1" in t for t in inside) and any("genuine" in t for t in inside)  # it IS in the expander...
    outside = [
        e.value for e in list(at.markdown) + list(at.caption) + list(at.sidebar.markdown) + list(at.sidebar.caption)
        if e.value not in inside
    ]
    assert not any("reason 1" in t for t in outside)  # ...and nowhere else on the page


def test_label_app_navigation_and_progress(csv_path, monkeypatch):
    at = _app(csv_path, monkeypatch)

    assert _button(at, "← prev").disabled is True
    _button(at, "next →").click().run()
    assert "Row 2 of 3" in at.markdown[0].value
    _button(at, "← prev").click().run()
    assert "Row 1 of 3" in at.markdown[0].value
    assert at.sidebar.get("progress")[0].proto.text == "0 / 3 labelled"


def test_label_app_progress_counter_updates(csv_path, monkeypatch):
    at = _app(csv_path, monkeypatch)

    _button(at, "genuine").click().run()

    assert at.sidebar.get("progress")[0].proto.text == "1 / 3 labelled"


def test_label_app_clear_removes_a_label(csv_path, monkeypatch):
    at = _app(csv_path, monkeypatch)
    _button(at, "genuine").click().run()
    _button(at, "← prev").click().run()

    _button(at, "clear").click().run()

    assert lb.load_rows(csv_path)[1][0]["human_label"] == ""


def test_label_app_is_generic_over_columns_and_labels(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "gold.csv"
    write_csv(path, [{"id": "a1", "text": "Great phone!", "label": ""}, {"id": "a2", "text": "Broke fast.", "label": ""}],
              ["id", "text", "label"])
    monkeypatch.setattr(
        sys, "argv",
        ["label_app.py", "--csv", str(path), "--id-col", "id", "--aspect-col", "", "--keyword-col", "",
         "--context-col", "text", "--text-col", "", "--label-col", "label", "--draft-col", "",
         "--labels", "positive,neutral,negative"],
    )
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "scripts" / "label_app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    _button(at, "negative").click().run()

    assert lb.load_rows(path)[1][0]["label"] == "negative"


# --- agreement maths ----------------------------------------------------------------------------------


def test_cohens_kappa_known_values():
    assert ag.cohens_kappa(["g", "g", "i", "i"], ["g", "g", "i", "i"]) == pytest.approx(1.0)
    assert ag.cohens_kappa(["g", "g", "i", "i"], ["g", "i", "i", "i"]) == pytest.approx(0.5)
    assert ag.cohens_kappa(["g", "i"], ["i", "g"]) == pytest.approx(-1.0)


def test_cohens_kappa_undefined_cases():
    assert ag.cohens_kappa([], []) is None
    assert ag.cohens_kappa(["g", "g"], ["g", "g"]) is None  # no variation
    with pytest.raises(ValueError):
        ag.cohens_kappa(["g"], ["g", "i"])


def test_kappa_interpretation_bands():
    assert ag.kappa_interpretation(0.9) == "almost perfect"
    assert ag.kappa_interpretation(0.5) == "moderate"
    assert ag.kappa_interpretation(-0.1) == "no better than chance"
    assert ag.kappa_interpretation(None) == "undefined"


def test_precision_rows_use_the_mode_flags_and_exclude_unlabelled():
    rows = [
        make_row(1, human="genuine", flags=(1, 1, 1)),
        make_row(2, human="inherited", flags=(1, 0, 0)),     # only the review mode counts it
        make_row(3, human="", flags=(1, 1, 0)),               # unlabelled
        make_row(4, aspect="price", human="genuine", flags=(0, 1, 1)),
    ]

    table = {(r["aspect"], r["mode"]): r for r in ag.precision_rows(rows, "human_label", ["ux", "price"])}

    assert table[("ux", "review")]["negative"] == 3 and table[("ux", "review")]["labelled"] == 2
    assert table[("ux", "review")]["precision"] == pytest.approx(0.5)
    assert table[("ux", "sentence")]["precision"] == pytest.approx(1.0)       # row 2 not counted in this mode
    assert table[("ux", "sentence")]["unlabelled"] == 1
    assert table[("ux", "sentence+clause")]["negative"] == 1
    assert table[("price", "review")]["precision"] is None                    # nothing flagged


def test_agreement_summary_lists_disagreements():
    rows = [
        make_row(1, draft="genuine", human="genuine"),
        make_row(2, draft="inherited", human="genuine"),
        make_row(3, draft="genuine", human=""),                                # not paired
    ]

    s = ag.agreement_summary(rows)

    assert (s["n"], s["agree"]) == (2, 1)
    assert [r["review_id"] for r in s["disagreements"]] == ["2"]


# --- report -------------------------------------------------------------------------------------------------


def test_report_says_so_when_there_are_no_human_labels():
    block = rp.render_human_block([make_row(1), make_row(2)])

    assert "No human labels yet" in block and "| Aspect |" not in block
    assert block.startswith(rp.BEGIN) and block.endswith(rp.END)


def test_report_marks_its_numbers_as_human_and_lists_disagreements():
    rows = [
        make_row(1, draft="genuine", human="genuine"),
        make_row(2, draft="genuine", human="inherited"),
        make_row(3, draft="inherited", human="inherited", flags=(1, 0, 0)),
        make_row(4, draft="false_trigger", human="false_trigger", flags=(1, 0, 0)),
    ]

    block = rp.render_human_block(rows, aspects=("ux",))

    assert "Source: human labels" in block and "4 of 4 rows labelled" in block
    assert "Precision (human)" in block and "Cohen's κ" in block
    assert "#2 | ux | genuine | **inherited**" in block
    assert "| ux | review | 4 | 4 | 1 | **25%** | 50% |" in block


def test_splice_replaces_only_the_generated_block_and_is_idempotent():
    block1 = rp.render_human_block([make_row(1)])
    doc = f"# Title\n\nhand-written text\n\n{block1}\n\n## Tail\nkeep me\n"
    block2 = rp.render_human_block([make_row(1, human="genuine")], aspects=("ux",))

    once = rp.splice_block(doc, block2)
    twice = rp.splice_block(once, block2)

    assert "hand-written text" in once and "keep me" in once and "No human labels yet" not in once
    assert once == twice
    assert once.count(rp.BEGIN) == 1


def test_splice_appends_when_markers_are_missing():
    out = rp.splice_block("# Doc\n", rp.render_human_block([make_row(1)]))

    assert out.startswith("# Doc") and rp.BEGIN in out
