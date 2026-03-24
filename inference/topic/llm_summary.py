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

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=260,
    )

    return resp.choices[0].message.content.strip()
