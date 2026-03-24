"""Batch topic discovery (fit_transform on the given batch)."""

from __future__ import annotations

from typing import Any, Dict, List

import hdbscan
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from inference.topic.loaders import _load_embedder_all
from inference.topic.payload import build_topic_payload, empty_topic_payload


def _single_doc_topic_payload(text: str, top_k_words: int = 10) -> Dict[str, Any]:
    """Fallback for one valid review: build a single pseudo-topic from term frequency."""
    doc = str(text or "").strip()
    if not doc:
        return empty_topic_payload(1)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        x = vectorizer.fit_transform([doc])
        idx = x.toarray()[0].argsort()[::-1]
        vocab = vectorizer.get_feature_names_out()
        keywords = [str(vocab[i]) for i in idx[: max(1, int(top_k_words or 10))]]
    except Exception:
        keywords = []

    return {
        "topics": [0],
        "keywords_by_topic": {0: keywords},
        "counts": {0: 1},
        "outlier_count": 0,
        "probs": [[1.0]],
        "coherence": {
            "c_v_overall": 1.0 if keywords else 0.0,
            "c_v_by_topic": {0: 1.0 if keywords else 0.0},
            "proxy_overall": 1.0 if keywords else 0.0,
            "proxy_by_topic": {0: 1.0 if keywords else 0.0},
            "available": False,
            "estimated": True,
            "source": "proxy_fallback",
            "error": "",
        },
        "raw_topic_rows": [
            {
                "topic_id": 0,
                "count": 1,
                "share": 1.0,
                "words": ", ".join(keywords),
                "mean_confidence": 1.0,
                "median_confidence": 1.0,
                "max_confidence": 1.0,
                "coherence_c_v": None,
                "coherence_proxy": 1.0 if keywords else 0.0,
            }
        ],
        "raw_review_rows": [{"review_idx": 1, "topic_id": 0, "confidence": 1.0, "text": doc[:180]}],
    }


def discover_topics_batch(
    texts: List[str],
    top_k_words: int = 10,
    min_topic_size: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Discover topics from the provided batch itself (fit_transform)."""
    if not texts:
        return empty_topic_payload(0)

    cleaned = [str(t or "").strip() for t in texts]
    valid_indices = [i for i, text in enumerate(cleaned) if text]
    valid_texts = [cleaned[i] for i in valid_indices]

    if not valid_texts:
        return empty_topic_payload(len(texts))
    if len(valid_texts) == 1:
        base = _single_doc_topic_payload(valid_texts[0], top_k_words=top_k_words)
        topics_full = [-1] * len(texts)
        probs_full: List[Any] = [None] * len(texts)
        topics_full[valid_indices[0]] = int(base["topics"][0])
        probs_full[valid_indices[0]] = base["probs"][0]
        return build_topic_payload(
            topics=topics_full,
            probs=probs_full,
            texts=cleaned,
            keywords_by_topic=base.get("keywords_by_topic") or {},
        )

    embedder = _load_embedder_all()

    n_neighbors = max(2, min(15, len(valid_texts) - 1))
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=min(5, max(2, len(valid_texts) - 1)),
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min(int(min_topic_size or 5), len(valid_texts))),
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=1,
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_k_words,
        calculate_probabilities=True,
        verbose=False,
    )

    try:
        batch_topics, batch_probs = topic_model.fit_transform(valid_texts)
        batch_topics = [int(t) for t in batch_topics]
    except Exception:
        topics_full = [-1] * len(texts)
        for idx in valid_indices:
            topics_full[idx] = -1
        return build_topic_payload(
            topics=topics_full,
            probs=[None] * len(texts),
            texts=cleaned,
            keywords_by_topic={},
        )

    topics_full = [-1] * len(texts)
    probs_full: List[Any] = [None] * len(texts)
    for j, src_idx in enumerate(valid_indices):
        topics_full[src_idx] = int(batch_topics[j])
        if batch_probs is not None:
            probs_full[src_idx] = batch_probs[j]

    keywords_by_topic: Dict[int, List[str]] = {}
    for topic_id in sorted(set(topics_full)):
        if int(topic_id) == -1:
            continue
        topic_words = topic_model.get_topic(int(topic_id)) or []
        keywords_by_topic[int(topic_id)] = [word for (word, _score) in topic_words[:top_k_words]]

    return build_topic_payload(
        topics=topics_full,
        probs=probs_full,
        texts=cleaned,
        keywords_by_topic=keywords_by_topic,
    )
