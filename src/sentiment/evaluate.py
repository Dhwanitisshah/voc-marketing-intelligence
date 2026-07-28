"""Phase 2 evaluation: compare VADER and transformer sentiment against rating-derived
ground truth (if available), and report agreement rate between the two models.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config

LABELS = ["negative", "neutral", "positive"]


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def evaluate(df: pd.DataFrame) -> str:
    """Build the full report text and return it (also printed by caller)."""
    lines = []

    agreement = (df["vader_label"] == df["transformer_label"]).mean()
    lines.append("=" * 72)
    lines.append("SENTIMENT MODEL COMPARISON")
    lines.append("=" * 72)
    lines.append(f"\nModel agreement rate (VADER vs transformer): {agreement:.1%}\n")

    if "rating_sentiment" not in df.columns:
        lines.append("[info] No rating column available — skipping accuracy evaluation.")
        return "\n".join(lines)

    eval_df = df.dropna(subset=["rating_sentiment"])
    y_true = eval_df["rating_sentiment"]

    vader_metrics = _metrics(y_true, eval_df["vader_label"])
    transformer_metrics = _metrics(y_true, eval_df["transformer_label"])

    lines.append(f"Evaluated against rating_sentiment ground truth (n={len(eval_df)})\n")
    header = f"{'Metric':<18}{'VADER':>12}{'Transformer':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_macro", "F1 (macro)"),
    ]:
        lines.append(f"{label:<18}{vader_metrics[key]:>12.3f}{transformer_metrics[key]:>14.3f}")

    lines.append("\n" + "-" * 72)
    lines.append("VADER classification report")
    lines.append("-" * 72)
    lines.append(classification_report(y_true, eval_df["vader_label"], labels=LABELS, zero_division=0))

    lines.append("VADER confusion matrix (rows=true, cols=pred; order=negative,neutral,positive)")
    lines.append(str(confusion_matrix(y_true, eval_df["vader_label"], labels=LABELS)))

    lines.append("\n" + "-" * 72)
    lines.append("Transformer classification report")
    lines.append("-" * 72)
    lines.append(classification_report(y_true, eval_df["transformer_label"], labels=LABELS, zero_division=0))

    lines.append("Transformer confusion matrix (rows=true, cols=pred; order=negative,neutral,positive)")
    lines.append(str(confusion_matrix(y_true, eval_df["transformer_label"], labels=LABELS)))

    return "\n".join(lines)


def main():
    in_path = config.PROCESSED_DIR / "reviews_sentiment.parquet"
    df = pd.read_parquet(in_path)

    report = evaluate(df)
    print(report)

    reports_dir = config.PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "sentiment_comparison.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[info] saved report to {out_path}")


if __name__ == "__main__":
    main()
