"""Minimal G2 fetcher wrapper for the dashboard.

- Uses `apify_client` and requires `APIFY_API_TOKEN` and `APIFY_ACTOR_ID` in environment.
- Exposes:
    - fetch_g2_reviews(slug, limit) -> list[dict]
    - extract_slug_from_g2_url(url_or_slug) -> slug or None

The returned review dicts are lightweight and safe to render in the UI.
"""
from typing import Any, Dict, List, Optional
import os
import re
import json

import requests
from bs4 import BeautifulSoup

try:
    from apify_client import ApifyClient
except Exception:  # pragma: no cover - library may be missing in CI/dev
    ApifyClient = None

# Default public actor used when the user supplies only APIFY_API_TOKEN.
# This is a convenience fallback; for production use you should set APIFY_ACTOR_ID
# to an actor you control.
DEFAULT_PUBLIC_ACTOR = "balmasi/g2-reviews"


def extract_slug_from_g2_url(url_or_slug: str) -> Optional[str]:
    """Accept either a full G2 URL or a bare slug and return the slug or None."""
    if not url_or_slug:
        return None
    s = url_or_slug.strip()
    # common full URL: https://www.g2.com/products/<slug>/reviews
    m = re.search(r"g2\.com/products/([a-z0-9\-_.]+)/?", s, re.IGNORECASE)
    if m:
        return m.group(1)
    # tolerate plain slug
    if re.fullmatch(r"[a-z0-9\-_.]+", s, re.IGNORECASE):
        return s
    # tolerate URLs like 'elevenlabs.io' (sometimes CSVs include domain); try to map to slug-ish token
    m2 = re.search(r"^([a-z0-9\-_.]+)\.(com|io|ai|co)$", s, re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None


class FetchError(RuntimeError):
    pass


def _ensure_client(actor_id: Optional[str] = None):
    """Return (client, actor_id).

    Behavior:
    - If APIFY_ACTOR_ID or APIFY_ACTOR_NAME is set, use it.
    - If only APIFY_API_TOKEN is present, fall back to DEFAULT_PUBLIC_ACTOR (convenience).
    - Raise FetchError when the apify-client library is missing or token is absent.
    """
    if ApifyClient is None:
        raise FetchError("missing dependency: install `apify-client` (pip install apify-client)")
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise FetchError("missing APIFY_API_TOKEN environment variable")

    # Prefer explicit actor id, then actor name, then a default public actor if only token is set
    actor = actor_id or os.getenv("APIFY_ACTOR_ID") or os.getenv("APIFY_ACTOR_NAME")
    if not actor:
        # Convenience: allow token-only by using a well-known public actor name
        actor = DEFAULT_PUBLIC_ACTOR

    return ApifyClient(token), actor


def _normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
    answers = it.get("answers")
    title = (answers[0] if isinstance(answers, list) and answers else "") or (it.get("headline") or "")
    content = "\n\n".join([a or "" for a in answers[:4]]) if isinstance(answers, list) else (it.get("content") or it.get("text") or "")
    rating = it.get("score")
    review_date = (
        (it.get("date") or {}).get("published")
        or (it.get("date") or {}).get("submitted")
        or (it.get("date") or {}).get("updated")
        or it.get("createdAt")
        or it.get("submissionTime")
        or it.get("date")
    )
    reviewer = it.get("name") or it.get("author") or ""
    platform_review_id = str(it.get("id") or "")
    product_slug = it.get("product_slug") or None
    review_url = f"https://www.g2.com/products/{product_slug}/reviews" if product_slug else None

    return {
        "id": platform_review_id or None,
        "title": title or None,
        "content": content or None,
        "rating": float(rating) if rating is not None else None,
        "date": review_date,
        "reviewer": reviewer,
        "review_url": review_url,
        "raw": json.dumps(it, ensure_ascii=False),
    }


def _safe_get(url: str, **kwargs):
    try:
        return requests.get(url, timeout=10, **kwargs)
    except Exception:
        return None


def _try_g2_html_fallback(slug: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Best-effort HTML/JSON-LD fallback for G2 product pages (may be blocked frequently)."""
    out: List[Dict[str, Any]] = []
    url = f"https://www.g2.com/products/{slug}/reviews"
    resp = _safe_get(url, headers={"User-Agent": "Mozilla/5.0"})
    if not resp or resp.status_code != 200:
        return out

    # try JSON-LD first
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup.find_all("script", type="application/ld+json"):
            txt = s.string or s.text or ""
            try:
                jd = json.loads(txt)
            except Exception:
                continue
            # look for Review objects
            if isinstance(jd, dict) and jd.get("@type") in ("Review", "Product"):
                # Product may contain 'review' array
                if jd.get("@type") == "Product" and isinstance(jd.get("review"), list):
                    items = jd.get("review")
                else:
                    items = [jd]
                for it in items[:limit]:
                    rid = str((it.get("@id") or it.get("identifier") or it.get("reviewId") or ""))
                    content = it.get("reviewBody") or it.get("description") or it.get("body") or None
                    rating = None
                    try:
                        rating = float((it.get("reviewRating") or {}).get("ratingValue")) if it.get("reviewRating") else None
                    except Exception:
                        rating = None
                    author = (it.get("author") or {}).get("name") if isinstance(it.get("author"), dict) else it.get("author")
                    out.append({
                        "id": rid or None,
                        "title": (it.get("name") or None),
                        "content": content,
                        "rating": rating,
                        "date": it.get("datePublished") or None,
                        "reviewer": author or None,
                        "review_url": url,
                        "raw": json.dumps(it, ensure_ascii=False),
                    })
                if out:
                    return out
    except Exception:
        pass

    # last-resort: try to scrape visible snippets (very best-effort)
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = soup.select("div[data-review-id], div.g2-review")
        for b in blocks[:limit]:
            content = (b.get_text(separator="\n", strip=True) or None)
            rid = b.get("data-review-id") or None
            out.append({
                "id": str(rid) if rid else None,
                "title": None,
                "content": content,
                "rating": None,
                "date": None,
                "reviewer": None,
                "review_url": url,
                "raw": None,
            })
    except Exception:
        return out

    return out


def fetch_g2_reviews(slug: str, limit: int = 20, actor_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch reviews for a G2 product `slug` using an Apify actor.

    Returns a list of normalized dicts. If Apify is not configured, attempts a best-effort HTML fallback.
    """
    if not slug:
        raise FetchError("empty slug")
    if limit <= 0:
        return []

    # prefer Apify actor when available
    try:
        client, actor = _ensure_client(actor_id=actor_id)
    except FetchError:
        # attempt a lightweight public fallback (may be blocked by G2)
        return _try_g2_html_fallback(slug, limit=min(limit, 6))

    run_input = {"query": slug, "mode": "review", "limit": int(limit)}
    run = client.actor(actor).call(run_input=run_input)
    dsid = run.get("defaultDatasetId")
    if not dsid:
        # if the actor ran but produced no dataset, try fallback
        return _try_g2_html_fallback(slug, limit=min(limit, 6))

    items = list(client.dataset(dsid).iterate_items())
    out = []
    for it in items[:int(limit)]:
        try:
            out.append(_normalize_item(it))
        except Exception:
            # skip malformed item
            continue
    return out


def search_g2(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Minimal G2 "search" for the UI (slug suggestion only)."""
    out: List[Dict[str, Any]] = []
    if not query:
        return out

    slug = extract_slug_from_g2_url(query)
    if slug:
        is_url = query.strip().lower().startswith("http")
        out.append({
            "name": (query if is_url else query),
            "subtitle": "G2 product",
            "logo": None,
            "g2_slug": slug,
            "unverified": not is_url,
        })
        return out

    suggestion = re.sub(r"[^a-z0-9]+", "-", query.strip().lower())
    suggestion = re.sub(r"-+", "-", suggestion).strip("-")[:80]
    if suggestion:
        out.append({
            "name": f"{query} (suggested)",
            "subtitle": "G2 product (suggested)",
            "logo": None,
            "g2_slug": suggestion,
            "unverified": True,
        })
    return out
