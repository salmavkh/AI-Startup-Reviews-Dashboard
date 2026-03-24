from __future__ import annotations

import re
from typing import Any, Dict, List

import streamlit as st
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _normalize_keyword(term: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(term or "").strip().lower())
    cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)
    return cleaned


@st.cache_resource
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def _load_keybert_model():
    from keybert import KeyBERT

    return KeyBERT(model=_load_embedder())


def extract_keywords_single(
    text: str,
    top_n: int = 5,
    keyphrase_ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> List[Dict[str, Any]]:
    content = str(text or "").strip()
    if not content:
        return []

    model = _load_keybert_model()
    rows = model.extract_keywords(
        content,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        top_n=max(1, int(top_n)),
        use_mmr=True,
        diversity=0.4,
    )

    out: List[Dict[str, Any]] = []
    for keyword, score in rows or []:
        term = str(keyword or "").strip()
        if not term:
            continue
        out.append({"keyword": term, "score": float(score)})
    return out


def extract_keywords_batch(
    texts: List[str],
    per_review_top_n: int = 5,
    overall_top_n: int = 15,
    keyphrase_ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> Dict[str, Any]:
    per_review: List[List[Dict[str, Any]]] = []
    aggregated: Dict[str, Dict[str, Any]] = {}

    for idx, text in enumerate(texts or []):
        entries = extract_keywords_single(
            text=str(text or ""),
            top_n=per_review_top_n,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
        )
        per_review.append(entries)

        seen_in_review = set()
        for item in entries:
            raw_term = str(item.get("keyword") or "").strip()
            score = float(item.get("score") or 0.0)
            normalized = _normalize_keyword(raw_term)
            if not normalized:
                continue

            bucket = aggregated.setdefault(
                normalized,
                {
                    "keyword": raw_term,
                    "review_count": 0,
                    "mentions": 0,
                    "score_sum": 0.0,
                },
            )

            bucket["mentions"] += 1
            bucket["score_sum"] += score

            if idx not in seen_in_review:
                bucket["review_count"] += 1
                seen_in_review.add(idx)

            if len(raw_term) > len(str(bucket.get("keyword") or "")):
                bucket["keyword"] = raw_term

    overall = []
    for row in aggregated.values():
        mentions = int(row.get("mentions", 0))
        score_sum = float(row.get("score_sum", 0.0))
        avg_score = score_sum / mentions if mentions > 0 else 0.0
        overall.append(
            {
                "keyword": str(row.get("keyword") or ""),
                "review_count": int(row.get("review_count", 0)),
                "mentions": mentions,
                "avg_score": avg_score,
            }
        )

    overall.sort(
        key=lambda x: (
            -int(x.get("review_count", 0)),
            -int(x.get("mentions", 0)),
            -float(x.get("avg_score", 0.0)),
            str(x.get("keyword", "")),
        )
    )

    return {
        "per_review": per_review,
        "overall": overall[: max(1, int(overall_top_n))],
    }
