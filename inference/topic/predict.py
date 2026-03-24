"""Prediction utilities for topic models."""

from __future__ import annotations

from typing import Any, Dict, List

from bertopic import BERTopic

from inference.topic.loaders import load_topic_model, load_topic_model_all
from inference.topic.payload import build_topic_payload


def predict_topic_single(
    text: str,
    cluster_label: str,
    top_k_words: int = 10,
) -> Dict[str, Any]:
    """Predict topic for one review using a cluster-specific trained model."""
    model = load_topic_model(cluster_label)

    topics, probs = model.transform([text])
    topic_id = int(topics[0])

    keywords: List[str] = []
    if topic_id != -1:
        topic_words = model.get_topic(topic_id) or []
        keywords = [word for word, _score in topic_words[:top_k_words]]

    confidence = None
    try:
        if probs is not None:
            row = probs[0]
            confidence = float(max(row)) if hasattr(row, "__len__") else float(row)
    except Exception:
        confidence = None

    return {
        "topic_id": topic_id,
        "keywords": keywords,
        "is_outlier": topic_id == -1,
        "confidence": confidence,
    }


def predict_topic_batch(
    texts: List[str],
    cluster_label: str,
    top_k_words: int = 10,
) -> Dict[str, Any]:
    model = load_topic_model(cluster_label)
    return _predict_topic_batch_with_model(model=model, texts=texts, top_k_words=top_k_words)


def predict_topic_batch_all(
    texts: List[str],
    top_k_words: int = 10,
) -> Dict[str, Any]:
    model = load_topic_model_all()
    return _predict_topic_batch_with_model(model=model, texts=texts, top_k_words=top_k_words)


def _predict_topic_batch_with_model(
    model: BERTopic,
    texts: List[str],
    top_k_words: int = 10,
) -> Dict[str, Any]:
    topics, probs = model.transform(texts)
    topics = [int(t) for t in topics]

    counts: Dict[int, int] = {}
    for t in topics:
        counts[t] = counts.get(t, 0) + 1

    keywords_by_topic: Dict[int, List[str]] = {}
    for t in sorted(counts.keys()):
        if t == -1:
            continue
        topic_words = model.get_topic(t) or []
        keywords_by_topic[t] = [w for (w, _s) in topic_words[:top_k_words]]

    return build_topic_payload(
        topics=topics,
        probs=probs,
        texts=texts,
        keywords_by_topic=keywords_by_topic,
    )
