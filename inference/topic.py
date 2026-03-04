from __future__ import annotations

import os
import re
import statistics
from typing import Dict, Any, List

import hdbscan
import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


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

    counts: Dict[int, int] = {}
    for t in topics:
        counts[t] = counts.get(t, 0) + 1
    outlier_count = counts.get(-1, 0)

    keywords_by_topic: Dict[int, List[str]] = {}
    for t in sorted(counts.keys()):
        if t == -1:
            continue
        topic_words = model.get_topic(t) or []
        keywords_by_topic[t] = [w for (w, _s) in topic_words[:top_k_words]]

    return _build_topic_payload(
        topics=topics,
        probs=probs,
        texts=texts,
        keywords_by_topic=keywords_by_topic,
    )


def _confidence_from_prob_row(row: Any) -> float | None:
    if row is None:
        return None
    try:
        if hasattr(row, "__len__") and not isinstance(row, (str, bytes)):
            vals = []
            for x in row:
                try:
                    vals.append(float(x))
                except Exception:
                    continue
            if vals:
                return max(vals)
            return None
        return float(row)
    except Exception:
        return None


def _tokenize_for_coherence(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z]{2,}", str(text or "").lower())]


def _topic_words_for_cv(words: List[str], max_terms: int = 10) -> List[str]:
    """
    Build unigram candidates for c_v from BERTopic top words.
    Handles phrases like "free trial" by splitting to ["free", "trial"].
    """
    out: list[str] = []
    seen = set()
    for w in words[: max(1, int(max_terms or 10))]:
        token = str(w or "").strip().lower()
        if not token:
            continue
        parts = re.findall(r"[a-zA-Z]{2,}", token)
        for p in parts:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _compute_coherence_metrics(
    texts: List[str],
    topics: List[int],
    keywords_by_topic: Dict[int, List[str]],
) -> Dict[str, Any]:
    """
    Returns:
      {
        c_v_overall: Optional[float],      # requires gensim
        c_v_by_topic: Dict[int, float],    # requires gensim
        proxy_overall: float,              # always available
        proxy_by_topic: Dict[int, float],  # always available
        available: bool,                   # whether c_v was computed
        error: str                         # c_v error if unavailable
      }
    """
    # Lightweight proxy coherence based on keyword coverage in assigned docs.
    proxy_by_topic: Dict[int, float] = {}
    weighted_sum = 0.0
    weighted_n = 0
    for tid, words in keywords_by_topic.items():
        docs = [texts[i] for i, t in enumerate(topics) if int(t) == int(tid) and i < len(texts)]
        if not docs:
            proxy_by_topic[int(tid)] = 0.0
            continue
        toks = [w.lower() for w in words[:10] if isinstance(w, str) and w.strip()]
        if not toks:
            proxy_by_topic[int(tid)] = 0.0
            continue
        coverages = []
        for d in docs:
            dlow = str(d or "").lower()
            hit = sum(1 for w in toks if w in dlow)
            coverages.append(hit / max(1, len(toks)))
        score = float(sum(coverages) / max(1, len(coverages)))
        proxy_by_topic[int(tid)] = score
        weighted_sum += score * len(docs)
        weighted_n += len(docs)
    proxy_overall = float(weighted_sum / max(1, weighted_n))

    # Try exact c_v coherence via gensim if present.
    try:
        from gensim.corpora import Dictionary  # type: ignore
        from gensim.models.coherencemodel import CoherenceModel  # type: ignore
    except Exception as exc:
        return {
            "c_v_overall": proxy_overall,
            "c_v_by_topic": dict(proxy_by_topic),
            "proxy_overall": proxy_overall,
            "proxy_by_topic": proxy_by_topic,
            "available": False,
            "estimated": True,
            "source": "proxy_fallback",
            "error": str(exc),
        }

    tokenized_docs = [_tokenize_for_coherence(t) for t in texts]
    dictionary = Dictionary(tokenized_docs)

    valid_topic_ids = []
    topic_word_lists = []
    for tid in sorted(keywords_by_topic.keys()):
        words = _topic_words_for_cv(keywords_by_topic.get(tid) or [], max_terms=10)
        if words:
            valid_topic_ids.append(int(tid))
            topic_word_lists.append(words)

    if not topic_word_lists:
        return {
            "c_v_overall": proxy_overall,
            "c_v_by_topic": dict(proxy_by_topic),
            "proxy_overall": proxy_overall,
            "proxy_by_topic": proxy_by_topic,
            "available": False,
            "estimated": True,
            "source": "proxy_fallback",
            "error": "",
        }

    cm = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    cv_overall = float(cm.get_coherence())
    cv_values = cm.get_coherence_per_topic()
    cv_by_topic = {
        int(valid_topic_ids[i]): float(cv_values[i]) for i in range(min(len(valid_topic_ids), len(cv_values)))
    }
    # Backfill missing topics with proxy so UI never gets None for coherence_c_v.
    for tid in keywords_by_topic.keys():
        if int(tid) not in cv_by_topic:
            cv_by_topic[int(tid)] = float(proxy_by_topic.get(int(tid), cv_overall))

    return {
        "c_v_overall": cv_overall,
        "c_v_by_topic": cv_by_topic,
        "proxy_overall": proxy_overall,
        "proxy_by_topic": proxy_by_topic,
        "available": True,
        "estimated": False,
        "source": "gensim_c_v",
        "error": "",
    }


def _build_raw_topic_rows(
    topics: List[int],
    probs: Any,
    texts: List[str],
    keywords_by_topic: Dict[int, List[str]],
    counts: Dict[int, int],
    coherence: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = max(1, len(topics))
    confidence_by_topic: Dict[int, List[float]] = {}
    per_review_rows = []
    for i, tid in enumerate(topics):
        conf = _confidence_from_prob_row(probs[i]) if probs is not None and i < len(probs) else None
        if conf is not None:
            confidence_by_topic.setdefault(int(tid), []).append(float(conf))
        per_review_rows.append(
            {
                "review_idx": i + 1,
                "topic_id": int(tid),
                "confidence": conf,
                "text": str(texts[i] or "")[:180],
            }
        )

    rows = []
    for tid, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        words = keywords_by_topic.get(int(tid), [])
        confs = confidence_by_topic.get(int(tid), [])
        coherence_c_v = coherence.get("c_v_by_topic", {}).get(int(tid))
        if coherence_c_v is None:
            coherence_c_v = coherence.get("c_v_overall")
        if coherence_c_v is None:
            coherence_c_v = coherence.get("proxy_by_topic", {}).get(int(tid), coherence.get("proxy_overall", 0.0))
        rows.append(
            {
                "topic_id": int(tid),
                "count": int(count),
                "share": float(count / total),
                "words": ", ".join(words),
                "mean_confidence": float(sum(confs) / len(confs)) if confs else None,
                "median_confidence": float(statistics.median(confs)) if confs else None,
                "max_confidence": float(max(confs)) if confs else None,
                "coherence_c_v": coherence_c_v,
                "coherence_proxy": coherence.get("proxy_by_topic", {}).get(int(tid), 0.0),
            }
        )
    return rows, per_review_rows


def _build_topic_payload(
    topics: List[int],
    probs: Any,
    texts: List[str],
    keywords_by_topic: Dict[int, List[str]],
) -> Dict[str, Any]:
    counts: Dict[int, int] = {}
    for t in topics:
        counts[int(t)] = counts.get(int(t), 0) + 1
    outlier_count = int(counts.get(-1, 0))

    coherence = _compute_coherence_metrics(
        texts=texts,
        topics=topics,
        keywords_by_topic=keywords_by_topic,
    )
    raw_topic_rows, raw_review_rows = _build_raw_topic_rows(
        topics=topics,
        probs=probs,
        texts=texts,
        keywords_by_topic=keywords_by_topic,
        counts=counts,
        coherence=coherence,
    )

    return {
        "topics": topics,
        "keywords_by_topic": keywords_by_topic,
        "counts": counts,
        "outlier_count": outlier_count,
        "probs": probs,
        "coherence": coherence,
        "raw_topic_rows": raw_topic_rows,
        "raw_review_rows": raw_review_rows,
    }


def _empty_topic_payload(n_texts: int) -> Dict[str, Any]:
    topics = [-1 for _ in range(max(0, int(n_texts or 0)))]
    return {
        "topics": topics,
        "keywords_by_topic": {},
        "counts": {-1: len(topics)} if topics else {},
        "outlier_count": len(topics),
        "probs": [None for _ in topics],
        "coherence": {
            "c_v_overall": 0.0,
            "c_v_by_topic": {},
            "proxy_overall": 0.0,
            "proxy_by_topic": {},
            "available": False,
            "estimated": True,
            "source": "proxy_fallback",
            "error": "",
        },
        "raw_topic_rows": [],
        "raw_review_rows": [],
    }


def _single_doc_topic_payload(text: str, top_k_words: int = 10) -> Dict[str, Any]:
    """Fallback for one valid review: build a single pseudo-topic from term frequency."""
    doc = str(text or "").strip()
    if not doc:
        return _empty_topic_payload(1)

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
    """
    Discover topics from the provided batch itself (fit_transform).
    This does not use or overwrite the saved production model.
    """
    if not texts:
        return _empty_topic_payload(0)

    cleaned = [str(t or "").strip() for t in texts]
    valid_indices = [i for i, t in enumerate(cleaned) if t]
    valid_texts = [cleaned[i] for i in valid_indices]

    if not valid_texts:
        return _empty_topic_payload(len(texts))
    if len(valid_texts) == 1:
        base = _single_doc_topic_payload(valid_texts[0], top_k_words=top_k_words)
        topics_full = [-1] * len(texts)
        probs_full: List[Any] = [None] * len(texts)
        topics_full[valid_indices[0]] = int(base["topics"][0])
        probs_full[valid_indices[0]] = base["probs"][0]
        return _build_topic_payload(
            topics=topics_full,
            probs=probs_full,
            texts=cleaned,
            keywords_by_topic=base.get("keywords_by_topic") or {},
        )

    embedder = _load_embedder()

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
        # Fallback for unstable tiny/noisy batches.
        topics_full = [-1] * len(texts)
        for idx in valid_indices:
            topics_full[idx] = -1
        return _build_topic_payload(
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
    for t in sorted(set(topics_full)):
        if int(t) == -1:
            continue
        topic_words = topic_model.get_topic(int(t)) or []
        keywords_by_topic[int(t)] = [w for (w, _score) in topic_words[:top_k_words]]

    return _build_topic_payload(
        topics=topics_full,
        probs=probs_full,
        texts=cleaned,
        keywords_by_topic=keywords_by_topic,
    )
