"""Sanity checks on a finished analysis, to catch silent failures.

The classic one: the wrong CSV column is mapped as the review text (usernames, ids,
product names). The pipeline happily runs, and every review comes back neutral with
no aspects. These checks turn that into a visible warning.
"""

import pandas as pd

NEUTRAL_SHARE_MAX = 0.90
ASPECT_COVERAGE_MIN = 0.05
MEDIAN_TEXT_LENGTH_MIN = 20
DATE_FAILURE_SHARE_MAX = 0.20
# Share-based checks are meaningless on a handful of rows.
MIN_ROWS_FOR_SHARE_CHECKS = 20

MAPPING_HINT = "Check your column mapping: the review text column may point at the wrong column."
DATE_HINT = "Check your column mapping: the date column may be wrong, or its format isn't recognised."


def check_analysis(df: pd.DataFrame) -> list:
    """Return warning messages for a loaded analysis (db.load_analysis); [] if all is well.

    Expects review_text, sentiment, aspects (list per row) and, optionally, review_date.
    """
    warnings = []
    n = len(df)
    if n == 0:
        return warnings

    if n >= MIN_ROWS_FOR_SHARE_CHECKS:
        neutral_share = (df["sentiment"] == "neutral").mean()
        if neutral_share > NEUTRAL_SHARE_MAX:
            warnings.append(
                f"{neutral_share:.0%} of reviews were scored neutral (expected well under "
                f"{NEUTRAL_SHARE_MAX:.0%}). {MAPPING_HINT}"
            )

        with_aspect = df["aspects"].apply(lambda a: bool(a)).mean()
        if with_aspect < ASPECT_COVERAGE_MIN:
            warnings.append(
                f"Only {with_aspect:.1%} of reviews matched any aspect keyword "
                f"(threshold {ASPECT_COVERAGE_MIN:.0%}). {MAPPING_HINT}"
            )

    median_length = df["review_text"].fillna("").astype(str).str.len().median()
    if median_length < MEDIAN_TEXT_LENGTH_MIN:
        warnings.append(
            f"The median review text is only {median_length:.0f} characters "
            f"(minimum expected {MEDIAN_TEXT_LENGTH_MIN}). {MAPPING_HINT}"
        )

    if "review_date" in df.columns and df["review_date"].notna().any():
        parsed = pd.to_datetime(df["review_date"], errors="coerce", utc=True, format="mixed")
        failed_share = parsed.isna().mean()
        if failed_share > DATE_FAILURE_SHARE_MAX:
            warnings.append(
                f"{failed_share:.0%} of dates could not be parsed (limit {DATE_FAILURE_SHARE_MAX:.0%}). "
                f"{DATE_HINT}"
            )

    return warnings
