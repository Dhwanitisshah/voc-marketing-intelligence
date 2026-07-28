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
