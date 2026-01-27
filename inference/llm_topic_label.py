from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

    # Minimal safe parsing without assuming perfect JSON
    text = resp.choices[0].message.content.strip()
    # If it returns JSON, great; otherwise treat as plain text
    # Keep it simple for now:
    return {"label": text, "explanation": ""} if not text.startswith("{") else {"label": text, "explanation": ""}
