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
OUTLIER_TOPIC = -1


def _fit(texts: list):
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # BERTopic's default UMAP settings, but seeded so a dataset's topics are
    # reproducible between runs.
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    topic_model = BERTopic(
        vectorizer_model=CountVectorizer(stop_words="english"), umap_model=umap_model
    )
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


def run_bertopic(texts: list):
    """Fit BERTopic on `texts`. Returns (topics, topic_info) or (None, None) if skipped."""
    if len(texts) < MIN_DOCS:
        print(
            f"[info] only {len(texts)} documents (< {MIN_DOCS}) — skipping BERTopic, "
            "dataset is too small for topic modeling. Keyword tagging will run alone."
        )
        return None, None

    topic_model, topics = _fit(texts)
    return topics, topic_model.get_topic_info()


def discover_topics(texts: list, top_n_words: int = 8):
    """Fit BERTopic and return it in a storage-friendly shape, or None if too few docs.

    Returns {"assignments": [topic_id per text, -1 = outlier],
             "topics": [{"topic_id", "label", "keywords", "size"}, ...]}
    where topics exclude the outlier bucket and are sorted by size, largest first.
    """
    if len(texts) < MIN_DOCS:
        return None

    topic_model, assignments = _fit(texts)

    sizes = {}
    for topic_id in assignments:
        sizes[topic_id] = sizes.get(topic_id, 0) + 1

    topics = []
    for topic_id, size in sizes.items():
        if topic_id == OUTLIER_TOPIC:
            continue
        keywords = [word for word, _ in topic_model.get_topic(topic_id)[:top_n_words]]
        topics.append(
            {
                "topic_id": int(topic_id),
                "label": ", ".join(keywords[:3]),
                "keywords": keywords,
                "size": int(size),
            }
        )
    topics.sort(key=lambda t: t["size"], reverse=True)

    return {"assignments": [int(t) for t in assignments], "topics": topics}
