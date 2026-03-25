"""Shared review-analysis pipeline for Streamlit pages."""

import hashlib
import json

import streamlit as st

from helpers.topic_summary import topic_summary_or_empty
from inference.emotion import predict_proba_single
from inference.topic.keywords import extract_keywords_batch
from inference.topic.llm_label import llm_label_topics_from_keywords
from inference.sentiment import predict_single as predict_sentiment_single
from inference.topic import discover_topics_batch
from inference.emotion.va import predict_va_single, summarize_va

GENERAL_CLUSTER_LABEL = "General online review feedback"


def _average_discrete_emotion_probs(
    texts: list[str],
    per_review_probs: list[dict[str, float]],
) -> dict:
    valid_indices = [
        i for i, t in enumerate(texts or [])
        if str(t or "").strip() and i < len(per_review_probs)
    ]
    if not valid_indices:
        return {"method": "prob", "total": 0, "percentages": {}, "counts": None}

    summed: dict[str, float] = {}
    for idx in valid_indices:
        dist = per_review_probs[idx] or {}
        for label, prob in dist.items():
            try:
                p = float(prob)
            except Exception:
                continue
            summed[str(label)] = summed.get(str(label), 0.0) + p

    total = len(valid_indices)
    avg = {label: score / total for label, score in summed.items()}
    percentages = dict(sorted(avg.items(), key=lambda kv: -kv[1]))
    return {"method": "prob", "total": total, "percentages": percentages, "counts": None}


def reviews_signature(rows: list[dict] | None) -> str:
    payload = []
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        payload.append(
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "content": r.get("content"),
                "date": r.get("date"),
                "platform": r.get("platform"),
                "reviewer": r.get("reviewer"),
            }
        )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def run_review_analysis(
    rows: list[dict],
    *,
    keywords_per_review: int = 5,
    keywords_overall: int = 20,
    topic_examples_per_theme: int = 2,
    cluster_label: str = GENERAL_CLUSTER_LABEL,
    include_signature: bool = False,
) -> dict:
    texts = [(r.get("content") or "").strip() for r in rows]

    topic_res = None
    topic_summary_text = ""
    topic_labels = {}

    if texts:
        with st.spinner("Running topic model..."):
            try:
                topic_res = discover_topics_batch(
                    texts,
                    top_k_words=10,
                    min_topic_size=5,
                )
            except Exception as exc:
                st.warning(f"Topic model failed: {exc}")
                topic_res = None

    if topic_res:
        with st.spinner("Generating high-level topic names..."):
            try:
                topic_labels = llm_label_topics_from_keywords(
                    cluster_label=cluster_label,
                    keywords_by_topic=topic_res.get("keywords_by_topic") or {},
                )
            except Exception as exc:
                st.warning(f"Topic naming failed: {exc}")
                topic_labels = {}
            topic_res["topic_labels"] = topic_labels

        with st.spinner("Generating LLM topic summary..."):
            try:
                topic_summary_text = topic_summary_or_empty(
                    topic_res=topic_res,
                    total_reviews=len(texts),
                    cluster_label=cluster_label,
                    review_texts=texts,
                    examples_per_topic=topic_examples_per_theme,
                )
            except Exception as exc:
                st.warning(f"LLM summary failed: {exc}")
                topic_summary_text = ""

    with st.spinner("Running sentiment..."):
        sentiments = [predict_sentiment_single(t or "") for t in texts]

    with st.spinner("Extracting keywords..."):
        if keywords_per_review <= 0 and keywords_overall <= 0:
            keyword_res = {"per_review": [[] for _ in texts], "overall": []}
        else:
            try:
                keyword_res = extract_keywords_batch(
                    texts=texts,
                    per_review_top_n=keywords_per_review,
                    overall_top_n=keywords_overall,
                )
            except Exception as exc:
                st.warning(f"Keyword extraction failed: {exc}")
                keyword_res = {"per_review": [[] for _ in texts], "overall": []}

    with st.spinner("Running emotion analysis..."):
        va_by_review = [predict_va_single(t or "") for t in texts]
        va_overall = summarize_va(va_by_review)
        discrete_by_review = [predict_proba_single(t or "") for t in texts]
        discrete_overall = _average_discrete_emotion_probs(texts, discrete_by_review)

    result = {
        "cluster_label": cluster_label,
        "topic": topic_res,
        "topic_summary": topic_summary_text,
        "keywords": keyword_res,
        "review_count": len(rows),
        "sentiment": sentiments,
        "emotion": {
            "va": va_overall,
            "discrete": discrete_overall,
        },
        "emotion_by_review": {
            "va": va_by_review,
            "discrete": discrete_by_review,
        },
        "reviews": rows,
    }
    if include_signature:
        result["reviews_signature"] = reviews_signature(rows)
    return result
