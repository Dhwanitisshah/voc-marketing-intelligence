# VOC Marketing Intelligence

A Voice of Customer (VOC) analysis system that turns raw customer reviews into
sentiment scores, aspect-level complaints and praise (delivery, price, quality,
support, packaging, UI, etc.), and LLM-generated marketing strategy
recommendations — built up in phases from a simple data pipeline to a full
dashboard and report.

## Structure

```
voc-marketing-intelligence/
├── config.py               # central paths + editable column mappings
├── requirements.txt         # dependencies, grouped by phase
├── data/
│   ├── raw/                 # your real reviews.csv goes here (gitignored)
│   ├── processed/           # pipeline outputs (gitignored)
│   └── sample/               # bundled sample_reviews.csv for development
├── src/
│   ├── data/
│   │   └── load_reviews.py  # Phase 0 loader
│   ├── preprocessing/
│   │   └── clean_text.py    # Phase 1 two-track text cleaning
│   ├── sentiment/
│   │   ├── vader_model.py       # Phase 2 lexicon baseline (VADER)
│   │   ├── transformer_model.py # Phase 2 transformer model (RoBERTa)
│   │   ├── run_sentiment.py     # Phase 2 orchestrator
│   │   └── evaluate.py          # Phase 2 model comparison / evaluation
│   └── aspects/
│       ├── aspect_lexicon.py    # Phase 3 editable seed keyword lexicon
│       ├── aspect_tagger.py     # Phase 3 keyword aspect tagger (backbone)
│       ├── topic_model.py       # Phase 3 BERTopic discovery layer
│       └── run_aspects.py       # Phase 3 orchestrator
│   ├── pipeline.py            # Phase 4 composed analyze() (clean -> sentiment -> aspects)
│   ├── db/
│   │   └── database.py        # Phase 4 SQLite persistence (data/voc.db) + Phase 5 strategy cache
│   └── strategy/
│       └── generate_strategy.py  # Phase 5 Gemini-backed marketing strategy layer
├── notebooks/
│   └── 01_eda.ipynb          # Phase 1 exploratory analysis
├── reports/                  # generated reports
└── dashboard/
    └── app.py                 # Phase 4 Streamlit app
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install pandas
```

As later phases are implemented, install the rest of `requirements.txt`:

```bash
pip install -r requirements.txt
```

To use your own data, drop a CSV at `data/raw/reviews.csv` with at least a
`review_text` column (see `config.py` for column-name mappings). Without it,
the loader falls back to the bundled sample dataset.

Run the Phase 0 loader:

```bash
python src/data/load_reviews.py
```

Run the Phase 1 cleaning pipeline (writes `data/processed/reviews_clean.parquet`):

```bash
python src/preprocessing/clean_text.py
```

Then explore the output in `notebooks/01_eda.ipynb`.

Run the Phase 2 sentiment pipeline (writes `data/processed/reviews_sentiment.parquet`;
use `--sample N` to dev on a subset):

```bash
python src/sentiment/run_sentiment.py --sample 200
```

Then evaluate both models against rating-derived ground truth (writes
`reports/sentiment_comparison.txt`):

```bash
python src/sentiment/evaluate.py
```

Run the Phase 3 aspect mining pipeline (writes `data/processed/reviews_aspects.parquet`
and `reports/aspect_summary.json`; use `--sample N` to dev on a subset). Below 50
reviews BERTopic is skipped automatically and only keyword tagging runs:

```bash
python src/aspects/run_aspects.py
```

Edit `src/aspects/aspect_lexicon.py` to extend the seed keywords for your domain.

Run the Phase 4 dashboard (composes Phases 1-3 via `src/pipeline.py`, persists
analyzed datasets to `data/voc.db` via `src/db/database.py`):

```bash
streamlit run dashboard/app.py
```

Upload a CSV from the sidebar (or reload a dataset you've already analyzed) to
explore sentiment/aspect breakdowns, top complaints/praise, a negative-review
word cloud, and a filterable review table. The first analysis of a dataset
runs the transformer sentiment model over every row, so large files take a
while the first time; reselecting the same dataset afterwards loads straight
from SQLite instead of reprocessing.

### Phase 5: marketing strategy layer

Turns the aspect x sentiment analysis into an executive summary and
prioritized marketing recommendations, using the free Google Gemini API. The
LLM only ever reasons over aggregates and example snippets pulled from
`data/voc.db` (via `src/db/database.py`) — it's instructed never to invent
numbers.

Setup:

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY=your_key_here`. `.env`
   is gitignored so your key never gets committed.
3. Install `google-genai` (already in `requirements.txt`).

Run it standalone against every dataset already in `data/voc.db`:

```bash
python src/strategy/generate_strategy.py
```

Or click "Generate marketing strategy" in the dashboard's "Marketing
strategy" section — it only calls the API on click, caches the result in
SQLite (`strategy` table), and reloads the cached result on future visits
until you click "Regenerate".

## Roadmap

| Phase | Focus |
|-------|-------|
| 0 | ✅ Setup — repo structure, environment, dataset loader, sample data |
| 1 | ✅ Data pipeline — cleaning, normalization, exploratory analysis |
| 2 | ✅ Sentiment analysis — VADER lexicon baseline vs. RoBERTa transformer, evaluated against rating-derived ground truth |
| 3 | ✅ Aspect mining — keyword tagger backbone + BERTopic discovery layer, crossed with sentiment |
| 4 | ✅ Dashboard — interactive Streamlit app for exploring results |
| 5 | ✅ LLM strategy layer — Gemini-generated marketing recommendations, cached in SQLite, surfaced in the dashboard |
| 6 | Report / paper — write-up of findings and methodology |
