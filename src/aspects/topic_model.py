"""Phase 3: BERTopic discovery layer.

Fits on "clean_light" (real text, not lemmas) because embeddings need natural
language, not a bag of lemmas. Uses a CountVectorizer(stop_words="english")
as the vectorizer_model so the topic keyword representations stay readable
(no stopwords cluttering topic labels).

BERTopic needs enough documents to form clusters. Below MIN_DOCS we skip it
entirely rather than let it fail or produce meaningless topics — the keyword
tagger in aspect_tagger.py still runs and remains the reliable signal.
"""

MIN_DOCS = 50


def run_bertopic(texts: list):
    """Fit BERTopic on `texts`. Returns (topics, topic_info) or (None, None) if skipped."""
    if len(texts) < MIN_DOCS:
        print(
            f"[info] only {len(texts)} documents (< {MIN_DOCS}) — skipping BERTopic, "
            "dataset is too small for topic modeling. Keyword tagging will run alone."
        )
        return None, None

    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model)

    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()

    return topics, topic_info
