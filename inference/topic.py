from __future__ import annotations

import os
from typing import Dict, Any, List

import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# =========================================================
# Paths
# =========================================================
TOPIC_MODELS_DIR = "artifacts/bertopic_by_cluster"
ALL_TOPIC_MODEL_PATH = "artifacts/bertopic_all_n30.model"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_CLUSTER_TO_MODEL = {
    "Cluster 1 (AI-Charged Product/Service Providers)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_1.model"
    ),
    "Cluster 2 (AI Development Facilitators)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_2.model"
    ),
    "Cluster 3 (Data Analytics Providers)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_3.model"
    ),
    "Cluster 4 (Deep Tech Researchers)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_4.model"
    ),
}


# =========================================================
# Load embedding model (once)
# =========================================================
@st.cache_resource
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# =========================================================
# Load BERTopic model (per cluster, cached)
# =========================================================
@st.cache_resource
def load_topic_model(cluster_label: str) -> BERTopic:
    model_path = _CLUSTER_TO_MODEL.get(cluster_label)
    if not model_path:
        raise ValueError(f"Unknown cluster label: {cluster_label}")

    embedder = _load_embedder()
    model = BERTopic.load(model_path, embedding_model=embedder)
    return model


@st.cache_resource
def load_topic_model_all() -> BERTopic:
    embedder = _load_embedder()
    return BERTopic.load(ALL_TOPIC_MODEL_PATH, embedding_model=embedder)


# =========================================================
# Single-review topic prediction
# =========================================================
def predict_topic_single(
    text: str,
    cluster_label: str,
    top_k_words: int = 10,
) -> Dict[str, Any]:
    """
    Returns:
      {
        topic_id: int,
        keywords: List[str],
        is_outlier: bool,
        confidence: Optional[float]
      }
    """
    model = load_topic_model(cluster_label)

    topics, probs = model.transform([text])
    topic_id = int(topics[0])

    # Extract keywords
    keywords: List[str] = []
    if topic_id != -1:
        topic_words = model.get_topic(topic_id) or []
        keywords = [word for word, _score in topic_words[:top_k_words]]

    # Extract confidence (if available)
    confidence = None
    try:
        if probs is not None:
            p = probs[0]
            confidence = float(max(p)) if hasattr(p, "__len__") else float(p)
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
    """
    Returns:
      {
        topics: List[int],                       # topic_id per review
        keywords_by_topic: Dict[int, List[str]], # topic_id -> keywords
        counts: Dict[int, int],                  # topic_id -> count in this batch
        outlier_count: int,
      }
    """
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
    """Shared batch prediction logic for any loaded BERTopic model."""

    topics, probs = model.transform(texts)  # topics: list[int]
    topics = [int(t) for t in topics]

    # Counts
    counts: Dict[int, int] = {}
    for t in topics:
        counts[t] = counts.get(t, 0) + 1

    outlier_count = counts.get(-1, 0)

    # Keywords for topics that appeared (excluding -1)
    keywords_by_topic: Dict[int, List[str]] = {}
    for t in sorted(counts.keys()):
        if t == -1:
            continue
        topic_words = model.get_topic(t) or []
        keywords_by_topic[t] = [w for (w, _s) in topic_words[:top_k_words]]

    return {
        "topics": topics,
        "keywords_by_topic": keywords_by_topic,
        "counts": counts,
        "outlier_count": outlier_count,
        "probs": probs,  # optional; you can ignore if you want
    }
