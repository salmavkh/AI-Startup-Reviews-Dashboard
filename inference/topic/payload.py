"""Payload builders for topic outputs."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

from inference.topic.coherence import compute_coherence_metrics


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


def build_topic_payload(
    topics: List[int],
    probs: Any,
    texts: List[str],
    keywords_by_topic: Dict[int, List[str]],
) -> Dict[str, Any]:
    counts: Dict[int, int] = {}
    for t in topics:
        counts[int(t)] = counts.get(int(t), 0) + 1
    outlier_count = int(counts.get(-1, 0))

    coherence = compute_coherence_metrics(
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


def empty_topic_payload(n_texts: int) -> Dict[str, Any]:
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
