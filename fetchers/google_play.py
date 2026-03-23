"""Google Play Store search and review fetching."""

from typing import List, Dict, Optional, Any
import re
import urllib.parse

try:
    from google_play_scraper import search as gp_search
except Exception:
    gp_search = None

import requests
from bs4 import BeautifulSoup

from fetchers.language_filter import is_english_review

ENGLISH_STOREFRONTS = ["us", "gb", "au", "ie", "nz", "sg"]
PACKAGE_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)+", re.IGNORECASE)


def _safe_get(url: str, **kwargs) -> Optional[requests.Response]:
    """Small wrapper that never throws."""
    try:
        return requests.get(url, timeout=12, allow_redirects=True, **kwargs)
    except Exception:
        return None


def _is_valid_package(package: str) -> bool:
    return bool(package and PACKAGE_RE.fullmatch(package.strip()))


def _extract_package_from_search_hit(hit: Dict[str, Any]) -> Optional[str]:
    for key in ("appId", "app_id", "package", "packageName"):
        value = hit.get(key)
        if isinstance(value, str) and _is_valid_package(value):
            return value.strip()

    for key in ("url", "appUrl", "storeUrl", "link"):
        value = hit.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        pkg = extract_package_from_google_play_url(value)
        if pkg and _is_valid_package(pkg):
            return pkg

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


def search_google_play(query: str, limit: int = 5, country: str = "us", lang: str = "en") -> List[Dict]:
    """Search Google Play Store for apps."""
    out: List[Dict] = []
    if not query:
        return out

    # Preferred: use google_play_scraper if installed
    if gp_search is not None:
        try:
            hits = gp_search(query, lang=lang, country=country, n_hits=limit)
            seen_keys = set()
            for h in hits:
                title = (h.get("title") or "").strip()
                subtitle = (h.get("developer") or h.get("genre") or "").strip()
                pkg = _extract_package_from_search_hit(h or {})

                if pkg:
                    dedupe_key = ("pkg", pkg.lower())
                else:
                    dedupe_key = ("name", title.lower(), subtitle.lower())

                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                out.append({
                    "name": title or pkg or query,
                    "subtitle": subtitle,
                    "logo": h.get("icon"),
                    "package": pkg,
                })
                if len(out) >= limit:
                    break
            return out
        except Exception:
            pass

    # Fallback: try the Play Store web lookup
    url = f"https://play.google.com/store/search?q={urllib.parse.quote(query)}&c=apps"
    resp = _safe_get(url, headers={"User-Agent": "Mozilla/5.0"})
    if not resp or resp.status_code != 200:
        return out
    soup = BeautifulSoup(resp.text, "html.parser")
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"details\?id=([\w\.\-]+)", href)
        if not m:
            continue
        pkg = m.group(1)
        if pkg in seen or not _is_valid_package(pkg):
            continue
        seen.add(pkg)
        title = a.get("aria-label") or (a.find("div") or {}).get_text(strip=True) or pkg
        out.append({"name": title, "subtitle": "Google Play app", "logo": None, "package": pkg})
        if len(out) >= limit:
            break
    return out


def extract_package_from_google_play_url(url_or_pkg: str) -> Optional[str]:
    """Extract package name from URL or return if already a package."""
    if not url_or_pkg:
        return None
    s = url_or_pkg.strip()
    # direct package id
    if re.fullmatch(r"[a-z0-9_\.]+", s, re.IGNORECASE):
        return s
    # try parse id= in url
    m = re.search(r"[?&]id=([a-z0-9_\.]+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def fetch_google_play_reviews(package: str, limit: int = 20, country: str = "us", lang: str = "en") -> List[Dict[str, Any]]:
    """Fetch Google Play reviews for a package."""
    out: List[Dict[str, Any]] = []
    if not package:
        return out

    # try the library first (preferred)
    try:
        from google_play_scraper import reviews as gp_reviews, Sort
    except Exception:
        gp_reviews = None

    if gp_reviews is not None:
        seq = [country.lower()] + [c for c in ENGLISH_STOREFRONTS if c.lower() != country.lower()]
        seen_ids = set()
        need = int(limit)
        for cc in seq:
            if need <= 0:
                break
            attempts = 0
            continuation_token = None
            while attempts < 4 and need > 0:
                attempts += 1
                try:
                    count = min(200, max(need * 4, need + 40))
                    resp, continuation_token = gp_reviews(
                        package,
                        lang=lang,
                        country=cc,
                        sort=Sort.NEWEST,
                        count=count,
                        continuation_token=continuation_token,
                    )
                    if not resp:
                        break
                    for r in (resp or []):
                        rid = r.get("reviewId") or r.get("userName") or None
                        if rid and rid in seen_ids:
                            continue
                        title = r.get("title") or None
                        content = r.get("content") or None
                        language_hint = r.get("reviewLanguage") or r.get("language") or r.get("lang")
                        if not is_english_review(title=title, content=content, language_hint=language_hint):
                            continue
                        seen_ids.add(rid)
                        out.append(
                            _normalize_review(
                                platform="Google Play Store",
                                platform_id=package,
                                review_id=rid,
                                title=title,
                                content=content,
                                rating=r.get("score"),
                                date=(r.get("at") or r.get("updated")) and str(r.get("at") or r.get("updated")),
                                reviewer=r.get("userName") or None,
                                url=None,
                            )
                        )
                        if len(out) >= limit:
                            break
                    need = limit - len(out)
                    if need <= 0 or not continuation_token:
                        break
                except Exception:
                    if attempts >= 4:
                        break
                    continue
        return out

    # library not available — brittle web fallback
    try:
        url = f"https://play.google.com/store/apps/details?id={package}&hl={lang}&gl={country}"
        resp = _safe_get(url, headers={"User-Agent": "Mozilla/5.0"})
        if not resp or resp.status_code != 200:
            return out
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = soup.select("div[jscontroller] div[data-review-id], div[class*='single-review']")
        for b in blocks[:limit]:
            text = b.get_text(separator="\n", strip=True)
            if not is_english_review(title=None, content=text):
                continue
            out.append(_normalize_review(
                platform="Google Play Store",
                platform_id=package,
                review_id=None,
                title=None,
                content=text,
                rating=None,
                date=None,
                reviewer=None,
                url=None,
            ))
    except Exception:
        return out
    return out
