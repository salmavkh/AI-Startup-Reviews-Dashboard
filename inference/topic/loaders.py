"""Model loading utilities for topic modeling."""

import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from inference.topic.constants import (
    ALL_TOPIC_MODEL_PATH,
    CLUSTER_TO_MODEL,
    EMBEDDING_MODEL_NAME,
)


@st.cache_resource
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def _load_embedder_all() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_topic_model(cluster_label: str) -> BERTopic:
    model_path = CLUSTER_TO_MODEL.get(cluster_label)
    if not model_path:
        raise ValueError(f"Unknown cluster label: {cluster_label}")

    embedder = _load_embedder()
    return BERTopic.load(model_path, embedding_model=embedder)


@st.cache_resource
def load_topic_model_all() -> BERTopic:
    embedder = _load_embedder_all()
    return BERTopic.load(ALL_TOPIC_MODEL_PATH, embedding_model=embedder)
