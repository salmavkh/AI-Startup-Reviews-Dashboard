"""iOS / App Store search and review fetching."""

from typing import List, Dict, Optional, Any
import re
import urllib.parse

import requests

from fetchers.language_filter import is_english_review

def _safe_get(url: str, **kwargs) -> Optional[requests.Response]:
    """Small wrapper that never throws."""
    try:
        return requests.get(url, timeout=12, allow_redirects=True, **kwargs)
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


def search_ios(query: str, limit: int = 5, country: str = "us") -> List[Dict]:
    """Search iOS App Store for apps."""
    out: List[Dict] = []
    if not query:
        return out
    url = (
        "https://itunes.apple.com/search?"
        + urllib.parse.urlencode({"term": query, "entity": "software", "limit": limit, "country": country})
    )
    resp = _safe_get(url)
    if not resp or resp.status_code != 200:
        return out
    try:
        jd = resp.json()
    except Exception:
        return out
    for r in (jd.get("results") or [])[:limit]:
        out.append({
            "name": r.get("trackName"),
            "subtitle": r.get("sellerName") or r.get("primaryGenreName") or "App Store",
            "logo": r.get("artworkUrl100"),
            "app_id": str(r.get("trackId")) if r.get("trackId") else None,
        })
    return out


def extract_app_id_from_ios_url(url_or_id: str) -> Optional[str]:
    """Extract app ID from URL or return if already an ID."""
    if not url_or_id:
        return None
    s = url_or_id.strip()
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]
    if re.fullmatch(r"\d+", s):
        return s
    m = re.search(r"/id(\d+)", s)
    if m:
        return m.group(1)
    return None


def fetch_ios_reviews(app_id: str, limit: int = 20, country: str = "us") -> List[Dict[str, Any]]:
    """Fetch iOS reviews via iTunes RSS feed."""
    out: List[Dict[str, Any]] = []
    if not app_id:
        return out
    try:
        seen = set()
        max_pages = 10
        for page in range(1, max_pages + 1):
            if len(out) >= limit:
                break

            urls = [
                f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json",
                f"https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json?cc={country}",
            ]
            if page == 1:
                urls.append(f"https://itunes.apple.com/rss/customerreviews/id={app_id}/json")

            jd = None
            for url in urls:
                resp = _safe_get(url)
                if not resp or resp.status_code != 200:
                    continue
                try:
                    jd = resp.json()
                    break
                except Exception:
                    jd = None
                    continue

            if jd is None:
                continue

            entries = jd.get("feed", {}).get("entry") or []
            if not entries:
                continue

            page_added = 0
            for e in (entries or [])[1:]:
                if len(out) >= limit:
                    break
                title = (e.get("title") or {}).get("label") if isinstance(e.get("title"), dict) else e.get("title")
                content = (e.get("content") or {}).get("label") if isinstance(e.get("content"), dict) else e.get("content")
                rating = None
                try:
                    rating = float((e.get("im:rating") or {}).get("label"))
                except Exception:
                    rating = None
                author = (e.get("author") or {}).get("name", {}).get("label") if isinstance((e.get("author") or {}).get("name"), dict) else None
                date = (e.get("updated") or {}).get("label") if isinstance(e.get("updated"), dict) else None
                if not is_english_review(title=title, content=content):
                    continue

                key = (str(title or "").strip(), str(content or "").strip(), str(date or "").strip(), str(author or "").strip())
                if key in seen:
                    continue
                seen.add(key)

                out.append(_normalize_review(
                    platform="iOS App Store",
                    platform_id=app_id,
                    review_id=None,
                    title=title,
                    content=content,
                    rating=rating,
                    date=date,
                    reviewer=author,
                    url=None,
                ))
                page_added += 1

            if page_added == 0 and page > 1:
                # Stop early if additional pages stop yielding new English items.
                break
    except Exception:
        return out
    return out
