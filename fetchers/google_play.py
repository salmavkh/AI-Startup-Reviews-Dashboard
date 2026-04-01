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


def _iter_nested_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield v
            yield from _iter_nested_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield v
            yield from _iter_nested_values(v)


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

    # Deep scan nested strings in case appId/package lives in nested payload.
    for value in _iter_nested_values(hit):
        if not isinstance(value, str):
            continue
        s = value.strip()
        if not s:
            continue

        if _is_valid_package(s):
            return s

        pkg = extract_package_from_google_play_url(s)
        if pkg and _is_valid_package(pkg):
            return pkg

        m = re.search(r"(?:[?&]id=|/details\\?id=)([a-z0-9_]+(?:\\.[a-z0-9_]+)+)", s, re.IGNORECASE)
        if m:
            pkg2 = m.group(1)
            if _is_valid_package(pkg2):
                return pkg2

    return None


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _parse_install_count(raw_value: Any) -> int:
    """Parse installs values like 500000, '5,000,000+', or '1.2M+'."""
    if raw_value is None:
        return 0

    if isinstance(raw_value, (int, float)):
        try:
            return max(0, int(raw_value))
        except Exception:
            return 0

    s = str(raw_value or "").strip().lower()
    if not s:
        return 0

    cleaned = s.replace(",", "").replace("+", "").strip()
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([kmb])$", cleaned)
    if m:
        try:
            base = float(m.group(1))
            suffix = m.group(2)
            mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
            return max(0, int(base * mult))
        except Exception:
            return 0

    digits = re.sub(r"[^0-9]", "", cleaned)
    if not digits:
        return 0
    try:
        return max(0, int(digits))
    except Exception:
        return 0


def _extract_hit_install_count(hit: Dict[str, Any]) -> int:
    for key in ("realInstalls", "minInstalls", "installs"):
        parsed = _parse_install_count(hit.get(key))
        if parsed > 0:
            return parsed
    return 0


def _search_google_play_web_hits(
    query: str,
    limit: int = 20,
    country: str = "us",
    lang: str = "en",
) -> List[Dict[str, str]]:
    """Search Play Store web page and return package/title hits."""
    out: List[Dict[str, str]] = []
    if not query:
        return out

    url = (
        "https://play.google.com/store/search?"
        f"q={urllib.parse.quote(query)}&c=apps&hl={urllib.parse.quote(lang)}&gl={urllib.parse.quote(country.upper())}"
    )
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

        title = (a.get("aria-label") or "").strip()
        if not title:
            div = a.find("div")
            if div is not None:
                title = div.get_text(strip=True)

        out.append({"title": title or pkg, "package": pkg})
        if len(out) >= limit:
            break
    return out


def _resolve_package_by_title_developer(
    title: str,
    developer_or_genre: str,
    country: str,
    lang: str,
) -> Optional[str]:
    """
    Try to recover missing package IDs for search hits by querying with title/developer
    and matching near-identical names.
    """
    if gp_search is None:
        return None

    t = str(title or "").strip()
    d = str(developer_or_genre or "").strip()
    if not t:
        return None

    queries = [f"{t} {d}".strip(), t]
    target_name = _norm_name(t)
    for q in queries:
        try:
            hits = gp_search(q, lang=lang, country=country, n_hits=15)
        except Exception:
            continue

        exact_with_pkg: List[tuple[int, str]] = []
        any_with_pkg: List[tuple[int, str]] = []
        saw_exact_without_pkg = False

        for h in hits or []:
            row = h or {}
            cand_name = _norm_name(str(row.get("title") or ""))
            pkg = _extract_package_from_search_hit(row)
            installs = _extract_hit_install_count(row)

            if target_name and cand_name == target_name and not pkg:
                saw_exact_without_pkg = True

            if not pkg:
                continue

            rec = (installs, pkg)
            any_with_pkg.append(rec)
            if target_name and cand_name == target_name:
                exact_with_pkg.append(rec)

        if exact_with_pkg:
            best_exact_pkg = max(exact_with_pkg, key=lambda x: (x[0], x[1]))[1]
            if not saw_exact_without_pkg:
                return best_exact_pkg

            # If the exact-title winner from gp_search is missing appId, use web
            # ranking to recover the canonical package (e.g., com.instagram.android).
            web_hits = _search_google_play_web_hits(q, limit=20, country=country, lang=lang)
            for wh in web_hits:
                web_pkg = wh.get("package")
                web_title = _norm_name(str(wh.get("title") or ""))
                if web_pkg and target_name and web_title == target_name:
                    return web_pkg
            return best_exact_pkg

        if saw_exact_without_pkg:
            web_hits = _search_google_play_web_hits(q, limit=20, country=country, lang=lang)
            for wh in web_hits:
                web_pkg = wh.get("package")
                web_title = _norm_name(str(wh.get("title") or ""))
                if web_pkg and target_name and web_title == target_name:
                    return web_pkg
            if web_hits:
                first_pkg = web_hits[0].get("package")
                if first_pkg:
                    return first_pkg

        if any_with_pkg:
            return max(any_with_pkg, key=lambda x: (x[0], x[1]))[1]

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
                if not pkg:
                    pkg = _resolve_package_by_title_developer(
                        title=title,
                        developer_or_genre=subtitle,
                        country=country,
                        lang=lang,
                    )

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
    web_hits = _search_google_play_web_hits(query, limit=limit, country=country, lang=lang)
    for hit in web_hits:
        out.append(
            {
                "name": hit.get("title") or hit.get("package") or query,
                "subtitle": "Google Play app",
                "logo": None,
                "package": hit.get("package"),
            }
        )
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
                        if not is_english_review(title=title, content=content):
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
