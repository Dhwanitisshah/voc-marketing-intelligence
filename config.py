"""Central configuration: paths and editable column mappings.

Every downstream script should import paths and column names from here
rather than hardcoding them, so the dataset schema only needs to change
in one place.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_reviews.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Column mappings -----------------------------------------------------
# Edit these if your source CSV uses different column names.
REVIEW_TEXT_COL = "review_text"
RATING_COL = "rating"
DATE_COL = "date"

# --- Aspect sentiment ----------------------------------------------------
# "sentence": score only the sentence containing the aspect keyword (lightweight
#             aspect-based sentiment), with negation handling for positive keywords.
# "review":   each aspect mention inherits the whole review's sentiment.
# Datasets analysed under a different mode are rebuilt automatically when loaded.
ASPECT_SENTIMENT_MODE = "sentence"
# Also cut sentences at "but" / "however" / "although" / ";" and score only the
# clause holding the keyword ("easy to set up, but the speaker is bad").
ASPECT_SPLIT_CLAUSES = False
