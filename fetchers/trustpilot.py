"""Trustpilot search and review fetching via Next.js JSON extraction."""

from typing import List, Dict, Optional, Any
import re
import json
import urllib.parse
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

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


def extract_slug_from_trustpilot_url(url_or_slug: str) -> Optional[str]:
    """Extract company slug from Trustpilot URL or validate slug."""
    if not url_or_slug:
        return None
    s = url_or_slug.strip()

    # If it's a URL, parse /review/<slug>
    if s.lower().startswith("http"):
        try:
            p = urlparse(s)
            parts = [x for x in p.path.split("/") if x]
            if len(parts) >= 2 and parts[0].lower() == "review":
                slug = parts[1].strip()
                slug = slug.replace("www.", "")
                return slug or None
        except Exception:
            pass

    # Otherwise accept a raw slug
    if re.fullmatch(r"[a-z0-9\-_.]+", s, re.IGNORECASE):
        s = s.replace("www.", "")
        return s or None

    # last-chance regex
    m = re.search(r"/review/([a-z0-9\-_.]+)", s, re.IGNORECASE)
    if m:
        slug = m.group(1).replace("www.", "")
        return slug or None
    return None


def _extract_next_data_from_html(html: str) -> Optional[dict]:
    """Extract Trustpilot's __NEXT_DATA__ JSON blob from HTML (Next.js)."""
    # Fast regex is usually most reliable
    try:
        m = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        if m:
            js = m.group(1)
            if js and js.strip().startswith("{"):
                return json.loads(js)
    except Exception:
        pass

    # Fallback: BeautifulSoup
    try:
        soup = BeautifulSoup(html, "html.parser")
        s = soup.find("script", attrs={"id": "__NEXT_DATA__"})
        if s:
            js = s.string or s.get_text(strip=False) or ""
            if js and js.strip().startswith("{"):
                return json.loads(js)
    except Exception:
        pass

    return None


def _walk_find_first_list(obj, predicate) -> Optional[list]:
    """Recursive finder: returns the first list where predicate(list) is True."""
    if isinstance(obj, list):
        if predicate(obj):
            return obj
        for it in obj:
            found = _walk_find_first_list(it, predicate)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for v in obj.values():
            found = _walk_find_first_list(v, predicate)
            if found is not None:
                return found
    return None


def _flatten_trustpilot_next_data(jd: dict, slug: str) -> List[Dict[str, Any]]:
    """Locate reviews in __NEXT_DATA__ JSON and normalize."""
    reviews = []
    try:
        reviews = jd.get("props", {}).get("pageProps", {}).get("reviews") or []
    except Exception:
        reviews = []

    # fallback paths
    if not reviews:
        paths = [
            ["props", "pageProps", "reviewsList", "reviews"],
            ["props", "pageProps", "businessUnit", "reviews"],
            ["props", "pageProps", "pageStore", "reviews"],
        ]
        for p in paths:
            cur = jd
            ok = True
            for k in p:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if ok and isinstance(cur, list):
                reviews = cur
                break
            if ok and isinstance(cur, dict) and "reviews" in cur and isinstance(cur["reviews"], list):
                reviews = cur["reviews"]
                break

    out = []
    for r in (reviews or []):
        if not isinstance(r, dict):
            continue
        rid = r.get("id") or r.get("reviewId") or None
        title = r.get("title") or r.get("headline") or None
        text = r.get("text") or r.get("content") or r.get("body") or None
        rating = r.get("rating") or r.get("stars") or None
        published = (r.get("dates") or {}).get("publishedDate") or r.get("createdAt") or r.get("date")
        author = (r.get("consumer") or {}).get("displayName") or r.get("author") or r.get("userName") or None
        language_hint = r.get("language") or r.get("locale") or (r.get("consumer") or {}).get("locale")
        if not is_english_review(title=title, content=text, language_hint=language_hint):
            continue

        out.append(_normalize_review(
            platform="Trustpilot",
            platform_id=slug,
            review_id=str(rid) if rid else None,
            title=title,
            content=text,
            rating=(float(rating) if rating is not None else None),
            date=published,
            reviewer=author,
            url=None,
        ))
    return out


def fetch_trustpilot_reviews(slug: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch Trustpilot reviews via Next.js JSON or HTML fallback."""
    out: List[Dict[str, Any]] = []
    if not slug:
        return out
    try:
        page = 1
        seen_ids = set()
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        }

        while len(out) < limit and page < 8:
            url = f"https://www.trustpilot.com/review/{slug}?page={page}"
            resp = _safe_get(url, headers=headers)
            if not resp or resp.status_code != 200:
                break

            jd = _extract_next_data_from_html(resp.text)
            if jd:
                items = _flatten_trustpilot_next_data(jd, slug=slug)
                for it in items:
                    rid = it.get("id") or None
                    if rid and rid in seen_ids:
                        continue
                    if rid:
                        seen_ids.add(rid)
                    out.append(it)
                    if len(out) >= limit:
                        break
                if len(out) >= limit:
                    break

            # legacy fallback (if JSON missing)
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = soup.find_all("article", attrs={"data-service-review-id": True})
            if not articles:
                articles = soup.select("div.review-card")

            page_added = 0
            for a in articles:
                if len(out) >= limit:
                    break
                rid = a.get("data-service-review-id") or None
                if rid and rid in seen_ids:
                    continue

                rating = None
                try:
                    star = a.select_one("div.star-rating, img[alt*='stars']")
                    if star and star.get("alt"):
                        mm = re.search(r"(\d+(?:\.\d+)?)", star.get("alt"))
                        if mm:
                            rating = float(mm.group(1))
                except Exception:
                    rating = None

                cont = None
                p = a.find("p")
                if p:
                    cont = p.get_text(strip=True)

                author = None
                auth = a.select_one("a.link[href*='/profile/']") or a.select_one("span.consumer-name")
                if auth:
                    author = auth.get_text(strip=True)

                date = None
                d = a.select_one("time")
                if d and d.get("datetime"):
                    date = d.get("datetime")

                review = _normalize_review(
                    platform="Trustpilot",
                    platform_id=slug,
                    review_id=rid,
                    title=None,
                    content=cont,
                    rating=rating,
                    date=date,
                    reviewer=author,
                    url=None,
                )
                if not is_english_review(title=review.get("title"), content=review.get("content")):
                    continue
                if rid:
                    seen_ids.add(rid)
                out.append(review)
                page_added += 1

            if page_added == 0 and not jd:
                break

            page += 1

    except Exception:
        return out
    return out


def search_trustpilot(query: str, limit: int = 5) -> List[Dict]:
    """Search Trustpilot businesses via Next.js JSON."""
    out: List[Dict] = []
    if not query:
        return out

    q = query.strip()
    url = f"https://www.trustpilot.com/search?query={urllib.parse.quote(q)}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = _safe_get(url, headers=headers)
    if not resp or resp.status_code != 200:
        return out

    jd = _extract_next_data_from_html(resp.text)
    if not jd:
        return out

    # find a list that looks like search results / business units
    candidates = _walk_find_first_list(
        jd,
        predicate=lambda L: len(L) > 0 and isinstance(L[0], dict) and (
            ("businessUnitId" in L[0]) or ("websiteUrl" in L[0]) or ("displayName" in L[0])
        ),
    ) or []

    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue

        name = item.get("displayName") or item.get("name") or item.get("title")
        website = item.get("websiteUrl") or item.get("website") or ""
        logo = None
        for k in ("logoUrl", "imageUrl", "profileImageUrl", "avatarUrl"):
            if item.get(k):
                logo = item.get(k)
                break

        tp_slug = None
        # some candidates include a /review/<slug> style url
        for k in ("profileUrl", "url", "businessUnitUrl", "link"):
            v = item.get(k)
            if isinstance(v, str) and "/review/" in v:
                tp_slug = extract_slug_from_trustpilot_url(
                    ("https://www.trustpilot.com" + v) if v.startswith("/") else v
                )
                break

        # fallback: Trustpilot commonly uses domain-style slugs
        if not tp_slug and isinstance(website, str) and website:
            dom = website.replace("https://", "").replace("http://", "").split("/")[0]
            tp_slug = dom.replace("www.", "")

        if not tp_slug or tp_slug in seen:
            continue
        seen.add(tp_slug)

        out.append({
            "name": name or tp_slug,
            "subtitle": website or "Trustpilot",
            "logo": logo,
            "tp_slug": tp_slug,
        })
        if len(out) >= limit:
            break

    return out
