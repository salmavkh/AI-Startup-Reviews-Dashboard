from __future__ import annotations
from typing import Dict, Any, List
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
def _get_client() -> Groq | None:
    key = _read_groq_api_key()
    if not key:
        return None
    return Groq(api_key=key)

def llm_topic_summary(cluster_label: str, top_topics: List[Dict[str, Any]]) -> str:
    """
    top_topics item format:
      {
        "topic_id": int,
        "pct": float,          # 0..1
        "count": int,
        "keywords": List[str],
        "examples": List[str]  # optional representative snippets
      }
    Returns a short structured summary string.
    """
    if not top_topics:
        return ""

    topic_count = len(top_topics)
    payload = {
        "cluster": cluster_label,
        "top_topics": [
            {
                "topic_id": t["topic_id"],
                "pct": round(float(t["pct"]) * 100, 1),
                "count": int(t["count"]),
                "keywords": t.get("keywords", [])[:12],
                "examples": t.get("examples", [])[:3],
            }
            for t in top_topics
        ],
    }

    prompt = f"""
You are writing a concise dashboard summary from topic keywords.

Rules:
- Do NOT mention model names, cluster names, or numeric topic IDs anywhere.
- Use ONLY the provided keywords, percentages, and examples (if present).
- If examples are provided, use them as grounding context for tone and intent.
- Do NOT invent product features, tools, or facts.
- Keep language plain and non-technical.
- Keep every topic explanation short (max 18 words).
- Generate exactly {topic_count} topic lines (one per provided topic, in given order).

Output format (follow exactly):
Overview: <1-2 short sentences about overall patterns>
Topics:
1. <short topic title> ({'{'}pct{'}'}%): <short explanation>
2. ...

Data (JSON):
{json.dumps(payload, ensure_ascii=False)}

Write the summary now.
""".strip()
    client = _get_client()
    if client is None:
        return "Overview: GROQ_API_KEY is not configured. Topics: keyword summaries are unavailable."

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=260,
    )

    return resp.choices[0].message.content.strip()


def _parse_json_summary_payload(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    candidates: List[str] = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend([c.strip() for c in fenced if c and c.strip()])

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
        summary = str(obj.get("summary") or "").strip()
        if summary:
            return summary

    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        if line.lower().startswith("summary:"):
            return line.split(":", 1)[1].strip()
    return ""


def _fallback_review_keyword_summary(topic_name: str, keywords: List[Dict[str, Any]]) -> str:
    terms: List[str] = []
    for row in keywords[:3]:
        term = str((row or {}).get("keyword") or "").strip()
        if term:
            terms.append(term)
    if not terms:
        return "The review discusses general app experience without a clear specific issue."

    topic = str(topic_name or "").strip()
    if topic:
        return f"For {topic}, this review mainly highlights {', '.join(terms)}."
    return f"This review mainly highlights {', '.join(terms)}."


@lru_cache(maxsize=4096)
def _llm_review_keyword_summary_cached(topic_name: str, keyword_payload_json: str) -> str:
    try:
        keywords = json.loads(keyword_payload_json)
    except Exception:
        keywords = []
    if not isinstance(keywords, list):
        keywords = []

    fallback = _fallback_review_keyword_summary(topic_name, keywords)
    if not keywords:
        return fallback

    lines = []
    for row in keywords:
        term = str((row or {}).get("keyword") or "").strip()
        try:
            score = float((row or {}).get("score") or 0.0)
        except Exception:
            score = 0.0
        if not term:
            continue
        lines.append(f"- {term} ({score:.4f})")

    if not lines:
        return fallback

    prompt = f"""
You are a product review analyst.

Task:
Write a short summary for ONE review using only the provided keywords and scores.

Rules:
- Use only the keyword evidence given. Do not invent details.
- Prioritize higher-score keywords when deciding the main point.
- Combine related keywords into natural phrasing.
- Keep it neutral and concise (1 sentence, max 25 words).
- Do not mention scores, confidence, or model names.
- If keywords are too weak/vague, output:
  "The review discusses general app experience without a clear specific issue."

Input:
Topic (optional): {topic_name or "N/A"}
Keywords (high to low relevance):
{chr(10).join(lines)}

Output format (JSON only):
{{"summary":"..."}}
""".strip()

    client = _get_client()
    if client is None:
        return fallback

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        raw = str(resp.choices[0].message.content or "").strip()
    except Exception:
        return fallback

    summary = _parse_json_summary_payload(raw)
    if not summary:
        return fallback

    summary = " ".join(summary.split())
    if len(summary.split()) > 32:
        summary = " ".join(summary.split()[:32]).rstrip(".,;:") + "."
    return summary


def llm_review_keyword_summary(
    topic_name: str,
    keyword_rows: List[Dict[str, Any]],
    max_keywords: int = 8,
) -> str:
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in keyword_rows or []:
        term = str((row or {}).get("keyword") or "").strip()
        if not term:
            continue
        try:
            score = float((row or {}).get("score") or 0.0)
        except Exception:
            score = 0.0
        key = term.lower()
        prev = dedup.get(key)
        if prev is None or score > float(prev.get("score") or 0.0):
            dedup[key] = {"keyword": term, "score": score}

    keywords = sorted(
        dedup.values(),
        key=lambda x: (-float(x.get("score") or 0.0), str(x.get("keyword") or "").lower()),
    )[: max(1, int(max_keywords or 8))]

    payload_json = json.dumps(
        [
            {
                "keyword": str(k.get("keyword") or ""),
                "score": round(float(k.get("score") or 0.0), 6),
            }
            for k in keywords
        ],
        ensure_ascii=False,
        sort_keys=True,
    )

    return _llm_review_keyword_summary_cached(str(topic_name or "").strip(), payload_json)
