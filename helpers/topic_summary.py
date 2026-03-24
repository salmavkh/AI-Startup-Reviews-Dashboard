"""Shared helpers for building LLM topic-summary payloads."""

from inference.topic.llm_summary import llm_topic_summary


def collect_topic_examples_for_payload(
    topic_res: dict,
    review_texts: list[str],
    per_topic_limit: int = 2,
) -> dict:
    """Pick representative snippets for each discovered topic."""
    examples_by_topic = {}
    topics = topic_res.get("topics") or []
    keywords_by_topic = topic_res.get("keywords_by_topic") or {}
    if not isinstance(topics, list) or len(topics) != len(review_texts):
        return examples_by_topic

    per_topic_limit = max(1, min(5, int(per_topic_limit or 2)))
    scored = {}

    for idx, topic_id in enumerate(topics):
        try:
            tid = int(topic_id)
        except Exception:
            continue
        if tid == -1:
            continue

        text = str(review_texts[idx] or "").strip()
        if not text:
            continue

        text_lower = text.lower()
        topic_keywords = [
            str(k).strip().lower()
            for k in keywords_by_topic.get(tid, [])[:10]
            if str(k).strip()
        ]
        overlap = sum(1 for kw in topic_keywords if kw and kw in text_lower)
        if overlap <= 0:
            continue

        snippet = text[:180] + ("..." if len(text) > 180 else "")
        scored.setdefault(tid, []).append((overlap, snippet))

    for tid, rows in scored.items():
        rows.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        picked = []
        seen = set()
        for _overlap, snippet in rows:
            if snippet in seen:
                continue
            seen.add(snippet)
            picked.append(snippet)
            if len(picked) >= per_topic_limit:
                break
        if picked:
            examples_by_topic[tid] = picked

    return examples_by_topic


def build_top_topics_payload(
    topic_res: dict,
    total_reviews: int,
    review_texts: list[str] | None = None,
    examples_per_topic: int = 2,
) -> list:
    if not topic_res or total_reviews <= 0:
        return []

    counts = topic_res.get("counts") or {}
    keywords_by_topic = topic_res.get("keywords_by_topic") or {}
    examples_by_topic = {}
    if review_texts:
        examples_by_topic = collect_topic_examples_for_payload(
            topic_res=topic_res,
            review_texts=review_texts,
            per_topic_limit=examples_per_topic,
        )

    items = [(tid, c) for tid, c in counts.items() if tid != -1]
    items.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            "topic_id": tid,
            "count": c,
            "pct": c / total_reviews,
            "keywords": keywords_by_topic.get(tid, []),
            "examples": examples_by_topic.get(tid, []),
        }
        for tid, c in items[:5]
    ]


def topic_summary_or_empty(
    topic_res: dict,
    total_reviews: int,
    cluster_label: str,
    review_texts: list[str] | None = None,
    examples_per_topic: int = 2,
) -> str:
    top_topics_payload = build_top_topics_payload(
        topic_res=topic_res,
        total_reviews=total_reviews,
        review_texts=review_texts,
        examples_per_topic=examples_per_topic,
    )
    if not top_topics_payload:
        return ""
    return llm_topic_summary(cluster_label, top_topics_payload)
