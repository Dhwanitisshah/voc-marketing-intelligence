"""Agreement and precision maths for the aspect audit (used by scripts/eval_agreement.py)."""

from collections import Counter

LABELS = ["genuine", "inherited", "false_trigger"]
MODES = ["review", "sentence", "sentence+clause"]


def flag_column(mode: str) -> str:
    """CSV column recording whether a mode counts the mention as negative, e.g. neg_sentence_clause."""
    return "neg_" + mode.replace("+", "_")


def cohens_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa between two equal-length label lists; None if undefined (no rows, or no variation)."""
    if len(a) != len(b):
        raise ValueError("label lists must have equal length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return None  # both raters used one single label: kappa is undefined
    return (observed - expected) / (1 - expected)


def kappa_interpretation(k: float | None) -> str:
    if k is None:
        return "undefined"
    for bound, name in [(0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"), (0.2, "fair"), (0.0, "slight")]:
        if k > bound:
            return name
    return "no better than chance"


def paired_labels(rows, draft_col="draft_label", human_col="human_label"):
    """Rows where both a draft and a human label exist."""
    return [r for r in rows if (r.get(draft_col) or "").strip() and (r.get(human_col) or "").strip()]


def agreement_summary(rows, draft_col="draft_label", human_col="human_label") -> dict:
    pairs = paired_labels(rows, draft_col, human_col)
    draft = [r[draft_col].strip() for r in pairs]
    human = [r[human_col].strip() for r in pairs]
    binary = lambda labels: ["genuine" if x == "genuine" else "not genuine" for x in labels]  # noqa: E731
    return {
        "n": len(pairs),
        "agree": sum(d == h for d, h in zip(draft, human)),
        "kappa": cohens_kappa(draft, human),
        "kappa_binary": cohens_kappa(binary(draft), binary(human)),
        "disagreements": [r for r in pairs if r[draft_col].strip() != r[human_col].strip()],
    }


def precision_rows(rows, label_col: str, aspects, modes=MODES) -> list:
    """Per (aspect, mode): negative mentions, how many carry a label, genuine count and precision.

    A mention belongs to a mode if its `neg_<mode>` flag is "1"/"true". Unlabelled mentions are
    counted separately and excluded from precision.
    """
    out = []
    for aspect in aspects:
        for mode in modes:
            col = flag_column(mode)
            flagged = [
                r for r in rows
                if r["aspect"] == aspect and str(r.get(col, "")).strip().lower() in {"1", "true"}
            ]
            labelled = [r for r in flagged if (r.get(label_col) or "").strip()]
            genuine = sum(1 for r in labelled if r[label_col].strip() == "genuine")
            out.append(
                {
                    "aspect": aspect, "mode": mode, "negative": len(flagged), "labelled": len(labelled),
                    "unlabelled": len(flagged) - len(labelled), "genuine": genuine,
                    "precision": genuine / len(labelled) if labelled else None,
                }
            )
    return out
