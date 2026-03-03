from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
