"""Topic coherence and proxy metrics."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _tokenize_for_coherence(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z]{2,}", str(text or "").lower())]


def _topic_words_for_cv(words: List[str], max_terms: int = 10) -> List[str]:
    """Build unigram candidates for c_v from BERTopic top words."""
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


def compute_coherence_metrics(
    texts: List[str],
    topics: List[int],
    keywords_by_topic: Dict[int, List[str]],
) -> Dict[str, Any]:
    """Compute proxy coherence and optionally exact c_v via gensim."""
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
