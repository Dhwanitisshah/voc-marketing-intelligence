"""Phase 4: Streamlit dashboard for VOC sentiment + aspect analysis.

Uploads a CSV, runs it through src.pipeline.analyze() once, persists the
result to SQLite (src.db.database), and visualizes sentiment/aspect
breakdowns. Re-selecting an already-analyzed dataset loads straight from
SQLite instead of reprocessing.
"""

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src import pipeline
from src.aspects.aspect_lexicon import ASPECT_KEYWORDS
from src.db import database as db
from src.sentiment.transformer_model import _get_pipeline as _load_transformer_pipeline
from src.strategy.generate_strategy import generate_strategy

st.set_page_config(page_title="VOC Marketing Intelligence", layout="wide")

SENTIMENT_COLORS = {"positive": "#2E7D32", "neutral": "#9E9E9E", "negative": "#C62828"}
PRIORITY_COLORS = {"high": "#C62828", "medium": "#F9A825", "low": "#2E7D32"}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def render_strategy(strategy: dict):
    if "raw" in strategy:
        st.warning("Model response wasn't valid JSON — showing raw output.")
        st.text(strategy["raw"])
        return

    st.markdown(f"**{strategy.get('executive_summary', '')}**")

    recs = sorted(
        strategy.get("recommendations", []),
        key=lambda r: PRIORITY_ORDER.get(r.get("priority", "low"), 3),
    )
    for rec in recs:
        priority = rec.get("priority", "low")
        color = PRIORITY_COLORS.get(priority, "#9E9E9E")
        with st.container(border=True):
            st.markdown(
                f"**{rec.get('aspect', '')}** "
                f"<span style='background-color:{color};color:white;padding:2px 8px;"
                f"border-radius:8px;font-size:0.75em;'>{priority.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"*Finding:* {rec.get('finding', '')}")
            st.markdown(f"*Recommendation:* {rec.get('recommendation', '')}")


@st.cache_resource(show_spinner="Loading sentiment model (first run only)...")
def load_models():
    """Warm the transformer pipeline once per server process."""
    return _load_transformer_pipeline()


def derive_dataset_id(filename: str, df: pd.DataFrame) -> str:
    row_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()[:10]
    stem = Path(filename).stem
    return f"{stem}_{row_hash}"


db.init_db()
load_models()

st.title("VOC marketing intelligence")
st.caption("Sentiment and aspect-level analysis of customer reviews.")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload a reviews CSV", type="csv")
    st.caption(
        f"Needs a `{config.REVIEW_TEXT_COL}` column. First-time analysis of a "
        "large file takes a while since the transformer model scores every row."
    )

    existing_datasets = db.list_datasets()
    selected_dataset = st.selectbox(
        "Or reload a previously analyzed dataset",
        ["-- select --"] + existing_datasets,
    )

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    dataset_id = derive_dataset_id(uploaded_file.name, raw_df)

    if dataset_id != st.session_state.get("dataset_id"):
        if db.dataset_exists(dataset_id):
            st.session_state["dataset_id"] = dataset_id
        else:
            try:
                with st.spinner("Running sentiment + aspect analysis..."):
                    enriched_df = pipeline.analyze(raw_df)
                    db.save_analysis(enriched_df, dataset_id)
            except KeyError as e:
                st.error(str(e))
                st.caption(
                    "Rename your CSV's text column to match, or update "
                    "REVIEW_TEXT_COL in config.py."
                )
                st.stop()
            st.session_state["dataset_id"] = dataset_id
elif selected_dataset != "-- select --":
    st.session_state["dataset_id"] = selected_dataset

dataset_id = st.session_state.get("dataset_id")

if not dataset_id:
    st.info("Upload a CSV or select a previously analyzed dataset from the sidebar to begin.")
    st.stop()

df = db.load_analysis(dataset_id)
st.caption(f"Dataset: `{dataset_id}` · {len(df)} reviews")

# --- Headline metrics -----------------------------------------------------
counts = db.sentiment_counts(dataset_id)
total = sum(counts.values())
pct_positive = 100 * counts.get("positive", 0) / total if total else 0
pct_negative = 100 * counts.get("negative", 0) / total if total else 0

with st.container(horizontal=True):
    st.metric("Total reviews", f"{total:,}", border=True)
    st.metric("Positive", f"{pct_positive:.1f}%", border=True)
    st.metric("Negative", f"{pct_negative:.1f}%", border=True)

# --- Sentiment donut + aspect x sentiment stacked bar ----------------------
col1, col2 = st.columns([1, 2])

with col1:
    with st.container(border=True):
        st.subheader("Sentiment breakdown")
        donut_df = pd.DataFrame({"sentiment": list(counts.keys()), "count": list(counts.values())})
        fig = px.pie(
            donut_df, names="sentiment", values="count", hole=0.5,
            color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

aspect_summary = db.aspect_sentiment_summary(dataset_id)

with col2:
    with st.container(border=True):
        st.subheader("Aspect x sentiment")
        if aspect_summary.empty:
            st.caption("No aspects were detected in this dataset.")
        else:
            long_df = aspect_summary.melt(
                id_vars=["aspect", "total"],
                value_vars=["positive", "neutral", "negative"],
                var_name="sentiment",
                value_name="count",
            )
            fig = px.bar(
                long_df.sort_values("total", ascending=True),
                x="count", y="aspect", color="sentiment", orientation="h",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

# --- Top complaints / top praise -------------------------------------------
col3, col4 = st.columns(2)

if not aspect_summary.empty:
    aspect_summary = aspect_summary.assign(
        negative_share=aspect_summary["negative"] / aspect_summary["total"],
        positive_share=aspect_summary["positive"] / aspect_summary["total"],
    )

    with col3:
        with st.container(border=True):
            st.subheader("Top complaints")
            top_complaints = aspect_summary.sort_values("negative_share", ascending=False).head(5)
            fig = px.bar(
                top_complaints, x="negative_share", y="aspect", orientation="h",
                color_discrete_sequence=[SENTIMENT_COLORS["negative"]],
            )
            fig.update_layout(xaxis_tickformat=".0%", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        with st.container(border=True):
            st.subheader("Top praise")
            top_praise = aspect_summary.sort_values("positive_share", ascending=False).head(5)
            fig = px.bar(
                top_praise, x="positive_share", y="aspect", orientation="h",
                color_discrete_sequence=[SENTIMENT_COLORS["positive"]],
            )
            fig.update_layout(xaxis_tickformat=".0%", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

# --- Word cloud of negative reviews ----------------------------------------
with st.container(border=True):
    st.subheader("Common words in negative reviews")
    negative_text = " ".join(df.loc[df["sentiment"] == "negative", "clean_deep"].dropna())
    if negative_text.strip():
        wc = WordCloud(width=1200, height=400, background_color="white").generate(negative_text)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.caption("No negative reviews to build a word cloud from.")

# --- Sentiment over time (if a date column exists) --------------------------
if df["review_date"].notna().any():
    with st.container(border=True):
        st.subheader("Sentiment over time")
        time_df = df.copy()
        time_df["review_date"] = pd.to_datetime(time_df["review_date"], errors="coerce")
        time_df = time_df.dropna(subset=["review_date"])
        if not time_df.empty:
            daily = (
                time_df.groupby([time_df["review_date"].dt.date, "sentiment"])
                .size()
                .reset_index(name="count")
                .rename(columns={"review_date": "date"})
            )
            fig = px.line(
                daily, x="date", y="count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS, markers=True,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No parseable dates found.")

# --- Filterable review table -------------------------------------------------
with st.container(border=True):
    st.subheader("Reviews")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        sentiment_filter = st.multiselect(
            "Filter by sentiment", options=sorted(counts.keys()), default=[]
        )
    with filter_col2:
        aspect_filter = st.multiselect(
            "Filter by aspect", options=sorted(ASPECT_KEYWORDS.keys()), default=[]
        )

    filtered = df
    if sentiment_filter:
        filtered = filtered[filtered["sentiment"].isin(sentiment_filter)]
    if aspect_filter:
        filtered = filtered[
            filtered["aspects"].apply(lambda a: any(x in a for x in aspect_filter))
        ]

    st.dataframe(
        filtered[["review_text", "sentiment", "confidence", "aspects", "rating", "review_date"]],
        hide_index=True,
        width="stretch",
    )

# --- Marketing strategy (Phase 5 LLM layer) --------------------------------
with st.container(border=True):
    st.subheader("Marketing strategy")

    cached_strategy = db.load_strategy(dataset_id)
    button_label = "Regenerate marketing strategy" if cached_strategy else "Generate marketing strategy"

    if st.button(button_label):
        with st.spinner("Calling Gemini..."):
            try:
                strategy = generate_strategy(dataset_id)
            except RuntimeError as e:
                st.error(str(e))
                strategy = None
        if strategy is not None:
            db.save_strategy(dataset_id, strategy)
            cached_strategy = strategy

    if cached_strategy:
        render_strategy(cached_strategy)
    else:
        st.caption("No strategy generated yet for this dataset.")
