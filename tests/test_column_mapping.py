import pandas as pd

from src.data.column_mapping import (
    DATE_KEYWORDS,
    RATING_KEYWORDS,
    guess_column,
    guess_review_text_column,
)

LONG = "I bought this tablet for my kids and the battery lasts all day, the screen is bright and sharp."


def _df(**cols):
    return pd.DataFrame(cols)


def test_reviewer_name_column_does_not_beat_the_real_review_text():
    # "Reviewer" contains "review" and comes first, but holds usernames.
    df = _df(Reviewer=["user_1", "user_2"], **{"Customer Comment": [LONG, LONG]})

    assert guess_review_text_column(df) == "Customer Comment"


def test_review_id_column_does_not_beat_review_body():
    df = _df(review_id=["r-1001", "r-1002"], review_body=[LONG, LONG])

    assert guess_review_text_column(df) == "review_body"


def test_explicit_review_text_name_wins_among_long_columns():
    df = _df(title_text=[LONG + " extra words here"] * 2, review_text=[LONG] * 2)

    assert guess_review_text_column(df) == "review_text"


def test_unnamed_text_column_is_found_by_length():
    df = _df(user=["bob", "amy"], notes_xyz=[LONG, LONG], count=[1, 2])

    assert guess_review_text_column(df) == "notes_xyz"


def test_short_named_column_loses_to_much_longer_unnamed_column():
    df = _df(Reviewer=["bob", "amy"], Opinion=[LONG, LONG])

    assert guess_review_text_column(df) == "Opinion"


def test_no_text_columns_returns_none():
    assert guess_review_text_column(_df(a=[1, 2], b=[3.5, 4.5])) is None


def test_rating_and_date_guesses():
    columns = ["Reviewer", "Posted Time", "Star Rating", "Customer Comment", "Helpful Votes"]

    assert guess_column(columns, RATING_KEYWORDS) == "Star Rating"
    assert guess_column(columns, DATE_KEYWORDS) == "Posted Time"
    assert guess_column(["a", "b"], RATING_KEYWORDS) is None


def test_same_file_with_a_different_mapping_gets_a_different_dataset_id():
    from src.data.column_mapping import derive_dataset_id

    df = _df(Reviewer=["user_1", "user_2"], **{"Customer Comment": [LONG, LONG]})
    wrong = df.rename(columns={"Reviewer": "review_text"})
    right = df.rename(columns={"Customer Comment": "review_text"})

    id_wrong = derive_dataset_id("reviews.csv", wrong, {"Reviewer": "review_text"})
    id_right = derive_dataset_id("reviews.csv", right, {"Customer Comment": "review_text"})

    assert id_wrong != id_right
    assert id_right == derive_dataset_id("reviews.csv", right, {"Customer Comment": "review_text"})  # stable
    assert id_right.startswith("reviews_")
