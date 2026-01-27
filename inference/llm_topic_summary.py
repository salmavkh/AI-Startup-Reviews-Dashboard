from __future__ import annotations
from typing import Dict, Any, List
import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def llm_topic_summary(cluster_label: str, top_topics: List[Dict[str, Any]]) -> str:
    """
    top_topics item format:
      {
        "topic_id": int,
        "pct": float,          # 0..1
        "count": int,
        "keywords": List[str]
      }
    Returns 1–3 sentence summary string.
    """
    payload = {
        "cluster": cluster_label,
        "top_topics": [
            {
                "topic_id": t["topic_id"],
                "pct": round(float(t["pct"]) * 100, 1),
                "count": int(t["count"]),
                "keywords": t.get("keywords", [])[:12],
            }
            for t in top_topics
        ],
    }

    prompt = f"""
You are writing a brief dashboard summary for BERTopic topic modelling results
from reviews from an AI startup.

Rules:
- Use high-level sentences that will make sense to the user.
- Don't mention the model and cluster name.
- Write 1–3 sentences total.
- Use ONLY the provided topic keywords and percentages.
- Do NOT invent product features, tools, or facts.
- Refer to themes, not specific companies.

Data (JSON):
{json.dumps(payload, ensure_ascii=False)}

Write the summary now.
""".strip()

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )

    return resp.choices[0].message.content.strip()
