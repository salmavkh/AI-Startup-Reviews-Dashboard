"""Trustpilot fetcher based on Next.js __NEXT_DATA__ extraction."""

from typing import List, Dict, Optional, Any
import hashlib
import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from fetchers.language_filter import is_english_review

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def _safe_get(url: str, **kwargs) -> Optional[requests.Response]:
    """Small wrapper that never throws."""
    try:
        return requests.get(url, timeout=20, allow_redirects=True, **kwargs)
    except Exception:
        return None


def _normalize_review(
    platform: str,
    platform_id: str,
    review_id: Optional[str],
    title: Optional[str],
    content: Optional[str],
    rating: Optional[float],
    date: Optional[str],
    reviewer: Optional[str],
    url: Optional[str],
) -> Dict[str, Any]:
    return {
        "platform": platform,
        "platform_id": platform_id,
        "id": review_id or None,
        "title": title or None,
        "content": content or None,
        "rating": float(rating) if rating is not None else None,
        "date": date,
        "reviewer": reviewer,
        "review_url": url,
        "raw": None,
    }


def extract_slug_from_trustpilot_url(url_or_slug: str) -> Optional[str]:
    """Extract company slug from Trustpilot URL or accept raw slug/domain."""
    if not url_or_slug:
        return None
    s = url_or_slug.strip()
    if not s:
        return None

    if s.lower().startswith("http"):
        try:
            p = urlparse(s)
            parts = [x for x in p.path.split("/") if x]
            if len(parts) >= 2 and parts[0].lower() == "review":
                slug = parts[1].strip().lower().replace("www.", "")
                return slug or None
        except Exception:
            pass

    m = re.search(r"/review/([a-z0-9\-_.]+)", s, re.IGNORECASE)
    if m:
        slug = m.group(1).strip().lower().replace("www.", "")
        return slug or None

    # raw domain/slug fallback
    if re.fullmatch(r"[a-z0-9\-_.]+", s, re.IGNORECASE):
        slug = s.lower().replace("www.", "")
        return slug or None

    return None


def _extract_next_data(html_text: str) -> Optional[Dict[str, Any]]:
    """
    Pull Next.js data blob.
    Trustpilot review pages typically render review data in __NEXT_DATA__.
    """
    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception:
        return None

    for s in soup.find_all("script"):
        t = (s.get("type") or "").lower()
        sid = s.get("id") or ""
        js = s.string or s.text or ""
        if not js:
            continue
        js = js.strip()
        if not js.startswith("{"):
            continue
        if sid == "__NEXT_DATA__" or "application/json" in t:
            if "pageProps" not in js:
                continue
            try:
                return json.loads(js)
            except Exception:
                continue
    return None


def _get_path(obj: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _flatten_reviews(next_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find reviews in common Next.js payload shapes and normalize core fields.
    """
    paths = [
        ["props", "pageProps", "reviews"],
        ["props", "pageProps", "reviewsList", "reviews"],
        ["props", "pageProps", "businessUnit", "reviews"],
        ["props", "pageProps", "pageStore", "reviews"],
    ]

    reviews: List[Dict[str, Any]] = []
    for p in paths:
        cur = _get_path(next_data, p)
        if isinstance(cur, list):
            reviews = cur
            break
        if isinstance(cur, dict) and isinstance(cur.get("reviews"), list):
            reviews = cur["reviews"]
            break

    out: List[Dict[str, Any]] = []
    for r in reviews:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "id": str(r.get("id") or r.get("reviewId") or ""),
                "title": r.get("title") or r.get("headline") or "",
                "text": r.get("text") or r.get("content") or r.get("body") or "",
                "stars": r.get("stars") or r.get("rating") or None,
                "published": (r.get("dates") or {}).get("publishedAt")
                or (r.get("dates") or {}).get("publishedDate")
                or r.get("createdAt")
                or r.get("date"),
                "displayName": (r.get("consumer") or {}).get("displayName")
                or r.get("author")
                or r.get("userName")
                or "",
                "country": (r.get("consumer") or {}).get("country")
                or (r.get("location") or {}).get("country"),
            }
        )
    return out


def _make_review_id(
    slug: str,
    rid: Optional[str],
    published: Optional[str],
    who: str,
    title: str,
    text: str,
) -> str:
    """Generate a stable unique ID when Trustpilot review ID is missing."""
    if rid:
        rid_s = str(rid).strip()
        if rid_s:
            return f"tp_{rid_s}"
    key = f"{slug}||{published or ''}||{who or ''}||{title or ''}||{text or ''}"
    return "tp_" + hashlib.sha1(key.encode("utf-8")).hexdigest()


def fetch_trustpilot_reviews(slug: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch Trustpilot reviews page-by-page from Next.js __NEXT_DATA__."""
    out: List[Dict[str, Any]] = []
    if not slug:
        return out

    base_url = f"https://www.trustpilot.com/review/{slug}"
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9",
    }

    seen = set()
    page = 1
    max_pages = 25

    while len(out) < int(limit) and page <= max_pages:
        url = base_url if page == 1 else f"{base_url}?page={page}"
        resp = _safe_get(url, headers=headers)
        if not resp or resp.status_code != 200:
            break

        next_data = _extract_next_data(resp.text)
        if not next_data:
            break

        items = _flatten_reviews(next_data)
        if not items:
            break

        page_added = 0
        for it in items:
            title = str(it.get("title") or "").strip()
            text = str(it.get("text") or "").strip()
            if not is_english_review(title=title, content=text):
                continue

            rid = _make_review_id(
                slug=slug,
                rid=it.get("id"),
                published=it.get("published"),
                who=str(it.get("displayName") or ""),
                title=title,
                text=text,
            )
            if rid in seen:
                continue
            seen.add(rid)

            stars = it.get("stars")
            try:
                stars_f = float(stars) if stars is not None else None
            except Exception:
                stars_f = None

            out.append(
                _normalize_review(
                    platform="Trustpilot",
                    platform_id=slug,
                    review_id=rid,
                    title=title or None,
                    content=text or None,
                    rating=stars_f,
                    date=it.get("published"),
                    reviewer=str(it.get("displayName") or "") or None,
                    url=base_url,
                )
            )
            page_added += 1
            if len(out) >= int(limit):
                break

        if page_added == 0:
            break
        page += 1

    return out[: int(limit)]


def search_trustpilot(query: str, limit: int = 5) -> List[Dict]:
    """
    Minimal compatibility search:
    if query is a Trustpilot URL/domain slug, return one candidate.
    """
    if not query:
        return []
    slug = extract_slug_from_trustpilot_url(query)
    if not slug:
        return []
    return [
        {
            "name": query.strip(),
            "subtitle": slug,
            "logo": None,
            "tp_slug": slug,
        }
    ][: max(1, int(limit or 1))]
