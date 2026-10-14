"""Best-guess mapping of an uploaded CSV's columns to review text / rating / date.

The dashboard pre-selects these guesses; the user can still override them.
"""

import pandas as pd

TEXT_KEYWORDS = ["review", "text", "comment", "content", "body", "feedback", "description"]
RATING_KEYWORDS = ["rating", "score", "stars", "star"]
DATE_KEYWORDS = ["date", "time"]

# A column whose name matches a text keyword is only trusted if its values are at
# least this fraction as long (on average) as the longest text column. Otherwise
# "Reviewer" / "review_id" style columns beat the real review body.
NAMED_TEXT_MIN_LENGTH_RATIO = 0.5


def guess_column(columns, keywords, prefer_all=None):
    """Case-insensitive substring match against column names.

    Returns the first column containing any keyword, preferring one that
    contains every keyword in `prefer_all` if given.
    """
    lower = {c: c.lower() for c in columns}
    if prefer_all:
        for col, low in lower.items():
            if all(k in low for k in prefer_all):
                return col
    for col, low in lower.items():
        if any(k in low for k in keywords):
            return col
    return None


def _is_text_column(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _avg_length(series: pd.Series) -> float:
    avg = series.dropna().astype(str).str.len().mean()
    return float(avg) if pd.notna(avg) else 0.0


def guess_review_text_column(df: pd.DataFrame) -> str | None:
    """Pick the column most likely to hold the review body.

    Column names are a hint, not proof: "Reviewer" contains "review" but holds
    usernames. So candidates are compared by average text length too.
    """
    text_cols = [c for c in df.columns if _is_text_column(df[c])]
    if not text_cols:
        return None

    lengths = {c: _avg_length(df[c]) for c in text_cols}
    longest = max(text_cols, key=lengths.get)

    named = [c for c in text_cols if any(k in c.lower() for k in TEXT_KEYWORDS)]
    trusted = [c for c in named if lengths[c] >= NAMED_TEXT_MIN_LENGTH_RATIO * lengths[longest]]
    if not trusted:
        return longest

    both = [c for c in trusted if "review" in c.lower() and "text" in c.lower()]
    return max(both or trusted, key=lengths.get)
