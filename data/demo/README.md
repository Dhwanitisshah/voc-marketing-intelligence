# Demo database

`voc_demo.db` is a small, precomputed SQLite database so a hosted copy of the app opens instantly
instead of spending minutes running RoBERTa and BERTopic. On a fresh install (no `data/voc.db`) the
app copies it into place and opens the dataset `demo_amazon_electronics_1000`.

**Contents:** a random sample of 1,000 reviews (seed 42) from the first 3,000 cleaned reviews of the
Datafiniti *Consumer Reviews of Amazon Products* dataset, with review-level sentiment, keyword aspects,
sentence+clause aspect sentiment, 9 BERTopic topics and a Gemini-generated strategy.

**Data source and licence:** [Datafiniti Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
on Kaggle. Check the licence terms on that page (it is shared under a non-commercial licence, as far as
I know) before redistributing or deploying publicly; the review text belongs to the original reviewers.
If the terms don't fit your use, delete this folder and build a demo from your own data.

**Rebuild** (needed whenever `ASPECT_SENTIMENT_MODE` / `ASPECT_SPLIT_CLAUSES`, the aspect lexicon or the
strategy context changes; `tests/test_deploy.py` fails if the demo is out of date):

```bash
python scripts/build_demo_db.py --source reviews_171e719d99 --rows 1000
```

Uses your local `data/voc.db` as the source and a Gemini key (optional: `--no-strategy` skips it).
