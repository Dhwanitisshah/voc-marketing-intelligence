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
│   └── sentiment/
│       ├── vader_model.py       # Phase 2 lexicon baseline (VADER)
│       ├── transformer_model.py # Phase 2 transformer model (RoBERTa)
│       ├── run_sentiment.py     # Phase 2 orchestrator
│       └── evaluate.py          # Phase 2 model comparison / evaluation
├── notebooks/
│   └── 01_eda.ipynb          # Phase 1 exploratory analysis
├── reports/                  # generated reports
└── dashboard/                 # Phase 4 Streamlit app
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

## Roadmap

| Phase | Focus |
|-------|-------|
| 0 | ✅ Setup — repo structure, environment, dataset loader, sample data |
| 1 | ✅ Data pipeline — cleaning, normalization, exploratory analysis |
| 2 | ✅ Sentiment analysis — VADER lexicon baseline vs. RoBERTa transformer, evaluated against rating-derived ground truth |
| 3 | Aspect mining — topic modeling to surface complaint/praise themes |
| 4 | Dashboard — interactive Streamlit app for exploring results |
| 5 | LLM strategy layer — Claude-generated marketing recommendations |
| 6 | Report / paper — write-up of findings and methodology |
