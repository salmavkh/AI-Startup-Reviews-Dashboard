from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import json
import re
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _read_groq_api_key() -> str:
    key = str(os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st

        key = str(st.secrets.get("GROQ_API_KEY") or "").strip()
    except Exception:
        key = ""
    return key


@lru_cache(maxsize=1)
def _get_client() -> Optional[Groq]:
    key = _read_groq_api_key()
    if not key:
        return None
    return Groq(api_key=key)


def _parse_json_label_payload(raw_text: str) -> Optional[Dict[str, str]]:
    """Best-effort parser for JSON and JSON-like LLM outputs."""
    text = str(raw_text or "").strip()
    if not text:
        return None

    candidates: List[str] = [text]

    # Handle fenced JSON blocks: ```json ... ```
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend([c.strip() for c in fenced if c and c.strip()])

    # Handle embedded JSON object inside other text.
    if "{" in text and "}" in text:
        first = text.find("{")
        last = text.rfind("}")
        if 0 <= first < last:
            candidates.append(text[first : last + 1].strip())

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label") or "").strip()
        explanation = str(obj.get("explanation") or "").strip()
        if label or explanation:
            return {
                "label": label or "Topic summary unavailable",
                "explanation": explanation,
            }

    # Last fallback: regex extract JSON fields if payload is malformed/truncated.
    def _extract_json_string_field(field_name: str) -> str:
        pattern = rf'"{field_name}"\s*:\s*"((?:\\.|[^"\\])*)"'
        m = re.search(pattern, text, flags=re.DOTALL)
        if not m:
            return ""
        try:
            return json.loads(f"\"{m.group(1)}\"").strip()
        except Exception:
            return m.group(1).strip()

    label = _extract_json_string_field("label")
    explanation = _extract_json_string_field("explanation")
    if label or explanation:
        return {
            "label": label or "Topic summary unavailable",
            "explanation": explanation,
        }

    return None

def llm_label_topic(
    cluster_label: str,
    topic_id: int,
    keywords: List[str],
    review_text: str,
) -> Dict[str, str]:
    # If outlier, keep it simple and avoid wasting tokens
    if topic_id == -1:
        return {
            "label": "Unassigned / Misc",
            "explanation": "This review doesn’t clearly match any learned topic for this cluster.",
        }

    prompt = f"""
You are helping label BERTopic topics for an academic dashboard.

Cluster: {cluster_label}
Topic ID: {topic_id}
Topic keywords: {", ".join(keywords)}
Example review: {review_text}

Task:
1) Produce a short human-readable topic label (max 6 words).
2) Produce a 1–2 sentence explanation grounded ONLY in the keywords and the example review.
Do NOT invent product features or facts.

Return JSON with keys: label, explanation.
"""
    client = _get_client()
    if client is None:
        return {
            "label": _fallback_label_from_keywords(keywords),
            "explanation": "GROQ_API_KEY is not configured, so this label is keyword-based.",
        }

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()
    if not text:
        return {"label": "Topic summary unavailable", "explanation": ""}

    parsed_payload = _parse_json_label_payload(text)
    if parsed_payload is not None:
        return parsed_payload

    # Fallback: parse simple line-based output.
    label = ""
    explanation = ""
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        lower = line.lower()
        if lower.startswith("label:"):
            label = line.split(":", 1)[1].strip()
        elif lower.startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()

    if not label:
        label = "Topic summary unavailable"

    return {"label": label, "explanation": explanation}


def _fallback_label_from_keywords(keywords: List[str], max_terms: int = 4) -> str:
    terms = []
    for w in (keywords or [])[: max(1, int(max_terms or 4))]:
        token = str(w or "").strip()
        if not token:
            continue
        terms.append(token.title())
    if not terms:
        return "General Topic"
    return " / ".join(terms)


def llm_label_topic_from_keywords(
    cluster_label: str,
    topic_id: int,
    keywords: List[str],
) -> str:
    """Return a short high-level label from BERTopic top words."""
    if int(topic_id) == -1:
        return "Unassigned / Misc"

    clean_keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not clean_keywords:
        return "General Topic"

    prompt = f"""
You are labeling a BERTopic topic for an analytics dashboard.

Cluster: {cluster_label}
Topic ID: {topic_id}
Top words: {", ".join(clean_keywords[:12])}

Task:
- Return one concise high-level label (max 6 words).
- Use only the top words.
- No explanation text.

Return JSON with key: label
"""
    try:
        client = _get_client()
        if client is None:
            return _fallback_label_from_keywords(clean_keywords)

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = str(resp.choices[0].message.content or "").strip()
        if not text:
            return _fallback_label_from_keywords(clean_keywords)

        parsed_payload = _parse_json_label_payload(text)
        if isinstance(parsed_payload, dict):
            label = str(parsed_payload.get("label") or "").strip()
            if label:
                return label

        for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
            if line.lower().startswith("label:"):
                val = line.split(":", 1)[1].strip()
                if val:
                    return val
        return _fallback_label_from_keywords(clean_keywords)
    except Exception:
        return _fallback_label_from_keywords(clean_keywords)


def llm_label_topics_from_keywords(
    cluster_label: str,
    keywords_by_topic: Dict[int, List[str]],
) -> Dict[int, str]:
    """Batch label helper for topic_id -> high-level label."""
    out: Dict[int, str] = {}
    for tid in sorted(keywords_by_topic.keys()):
        if int(tid) == -1:
            continue
        out[int(tid)] = llm_label_topic_from_keywords(
            cluster_label=cluster_label,
            topic_id=int(tid),
            keywords=keywords_by_topic.get(int(tid)) or [],
        )
    return out
