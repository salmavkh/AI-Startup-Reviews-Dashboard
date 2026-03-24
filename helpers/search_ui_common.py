"""UI helper functions for search page."""

import html
import hashlib
import math
import os
import pandas as pd
import random
import re
import streamlit as st
from fetchers.language_filter import filter_english_reviews

try:
    import altair as alt
except Exception:  # pragma: no cover - optional dependency
    alt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

TOPIC_DEBUG_MESSAGES_ENABLED = False
TOPIC_DEBUG_EXAMPLES_PER_THEME = 2

EMOTWEET_28 = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

NRC_VAD_LEXICON_PATH = os.path.join(
    "lexicon", "NRC-VAD-Lexicon-v2.1", "NRC-VAD-Lexicon-v2.1.txt"
)
EMOTION_ANCHOR_RBF_SIGMA = 0.40


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _rating_to_stars(rating) -> str:
    """Convert a numeric rating to a 5-star string."""
    try:
        r = float(rating)
    except Exception:
        return ""
    r = max(0.0, min(5.0, r))
    filled = int(round(r))
    return ("★" * filled) + ("☆" * (5 - filled))


def _short_text(text: str, limit: int = 220) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else (s[:limit] + "...")


def _find_key_in_obj(obj, key_name: str):
    if isinstance(obj, dict):
        if key_name in obj:
            return obj.get(key_name)
        for v in obj.values():
            found = _find_key_in_obj(v, key_name)
            if found is not None:
                return found
        return None
    if isinstance(obj, list):
        for item in obj:
            found = _find_key_in_obj(item, key_name)
            if found is not None:
                return found
    return None


def _extract_selected_review_idx(event_state) -> int | None:
    """Extract clicked review index from an Altair selection event."""
    if event_state is None:
        return None

    selection = getattr(event_state, "selection", None)
    if selection is None and isinstance(event_state, dict):
        selection = event_state.get("selection")
    if selection is None:
        return None

    direct = selection.get("review_pick") if isinstance(selection, dict) else None
    if isinstance(direct, dict):
        if "review_idx" in direct:
            try:
                return int(direct["review_idx"])
            except Exception:
                pass
        value = direct.get("value")
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict) and "review_idx" in first:
                try:
                    return int(first["review_idx"])
                except Exception:
                    pass
    elif isinstance(direct, list) and direct:
        first = direct[0]
        if isinstance(first, dict) and "review_idx" in first:
            try:
                return int(first["review_idx"])
            except Exception:
                pass

    value = _find_key_in_obj(selection, "review_idx")
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("review_idx")

    try:
        return int(value)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_emotweet28_va_points():
    if not os.path.exists(NRC_VAD_LEXICON_PATH):
        return [], list(EMOTWEET_28)

    lookup: dict[str, tuple[float, float]] = {}
    try:
        df = pd.read_csv(NRC_VAD_LEXICON_PATH, sep="\t")
    except Exception:
        return [], list(EMOTWEET_28)

    required = {"term", "valence", "arousal"}
    if not required.issubset(set(df.columns)):
        return [], list(EMOTWEET_28)

    for _, row in df.iterrows():
        term = str(row["term"]).strip().lower()
        if term and term not in lookup:
            lookup[term] = (_safe_float(row["valence"]), _safe_float(row["arousal"]))

    points = []
    missing = []
    for emotion in EMOTWEET_28:
        if emotion in lookup:
            v, a = lookup[emotion]
            points.append({"emotion": emotion, "valence": v, "arousal": a})
        else:
            missing.append(emotion)

    return points, missing


def _build_emotion_distance_rows(valence: float, arousal: float):
    points, missing = _load_emotweet28_va_points()
    rows = []
    sigma2 = max(1e-12, float(EMOTION_ANCHOR_RBF_SIGMA) ** 2)
    denom = 2.0 * sigma2
    for p in points:
        ev = _safe_float(p.get("valence"))
        ea = _safe_float(p.get("arousal"))
        d2 = (valence - ev) ** 2 + (arousal - ea) ** 2
        d = math.sqrt(d2)
        similarity = math.exp(-d2 / denom)
        rows.append(
            {
                "emotion": p.get("emotion"),
                "emotion_valence": ev,
                "emotion_arousal": ea,
                "review_valence": valence,
                "review_arousal": arousal,
                "distance": d,
                "similarity": similarity,
            }
        )

    rows.sort(key=lambda x: x["distance"])
    for i, row in enumerate(rows):
        row["rank"] = i + 1
        row["is_top10"] = i < 10

    return rows, missing

# ===== Fetch helpers =====

@st.cache_data(ttl=60 * 30)
def fetch_reviews_cached_non_tp(platform: str, identifier: dict, limit: int = 5):
    """Cached fetch for stable platforms (GP/iOS/G2)."""
    from fetchers.google_play import fetch_google_play_reviews
    from fetchers.ios import fetch_ios_reviews
    from fetchers.g2 import fetch_g2_reviews
    
    if not identifier or not platform:
        return []
    try:
        if platform == "Google Play Store":
            pkg = identifier.get("package")
            return fetch_google_play_reviews(pkg, limit=limit)
        if platform == "iOS App Store":
            appid = identifier.get("app_id")
            return fetch_ios_reviews(appid, limit=limit)
        if platform == "G2":
            slug = identifier.get("g2_slug")
            try:
                return fetch_g2_reviews(slug, limit=limit)
            except Exception:
                return []
    except Exception:
        return []
    return []


def fetch_reviews_uncached_tp(identifier: dict, limit: int = 5):
    """Always uncached Trustpilot fetch (Next.js JSON)."""
    from fetchers.trustpilot import fetch_trustpilot_reviews
    
    if not identifier:
        return []
    try:
        tp = identifier.get("tp_slug")
        return fetch_trustpilot_reviews(tp, limit=limit)
    except Exception:
        return []


def _review_identity(review: dict) -> tuple:
    rid = str(review.get("id") or "").strip()
    if rid:
        return ("id", rid)
    title = str(review.get("title") or "").strip().lower()
    content = str(review.get("content") or "").strip().lower()
    date = str(review.get("date") or "").strip()
    reviewer = str(review.get("reviewer") or "").strip().lower()
    return ("txt", title[:120], content[:280], date, reviewer)


def _fetch_reviews_raw(platform: str, identifier: dict, limit: int = 5):
    if platform == "Trustpilot":
        return fetch_reviews_uncached_tp(identifier, limit=limit)
    return fetch_reviews_cached_non_tp(platform, identifier, limit=limit)


def fetch_reviews_for_ui(platform: str, identifier: dict, limit: int = 5):
    """Unified fetch used by preview + full fetch."""
    target = max(1, int(limit or 1))
    # Escalating fetch sizes to top up English reviews.
    attempt_limits = []
    size = target
    for _ in range(4):
        capped = min(400, max(target, size))
        if capped not in attempt_limits:
            attempt_limits.append(capped)
        size = int(size * 1.8) + 10

    combined = []
    seen = set()

    for fetch_limit in attempt_limits:
        rows = _fetch_reviews_raw(platform, identifier, limit=fetch_limit)
        english_rows = filter_english_reviews(rows or [], limit=None)

        for r in english_rows:
            if not isinstance(r, dict):
                continue
            key = _review_identity(r)
            if key in seen:
                continue
            seen.add(key)
            combined.append(r)
            if len(combined) >= target:
                return combined[:target]

    return combined[:target]


# ===== Search result processing =====

def process_search_results(candidates: list, platform: str) -> list:
    """Convert raw search candidates to UI result format."""
    ui_candidates = []
    for idx, c in enumerate((candidates or [])[:5], start=1):
        stable_seed = (
            c.get("package")
            or c.get("app_id")
            or c.get("g2_slug")
            or c.get("tp_slug")
            or c.get("name")
            or c.get("title")
            or f"row-{idx}"
        )
        stable_hash = hashlib.sha1(f"{platform}|{stable_seed}|{idx}".encode("utf-8")).hexdigest()[:12]
        ui = {
            "id": f"r_{stable_hash}",
            "platform": platform,
            "name": c.get("name") or c.get("title") or "(unknown)",
            "subtitle": c.get("subtitle") or "",
            "logo": c.get("logo") or None,
        }
        if c.get("package"):
            ui["package"] = c.get("package")
            if platform == "Google Play Store":
                pkg = str(c.get("package") or "").strip()
                if pkg:
                    subtitle = str(ui.get("subtitle") or "").strip()
                    ui["subtitle"] = f"{subtitle} · {pkg}" if subtitle else pkg
        if c.get("app_id"):
            ui["app_id"] = c.get("app_id")
        if c.get("g2_slug"):
            ui["g2_slug"] = c.get("g2_slug")
        if c.get("tp_slug"):
            ui["tp_slug"] = c.get("tp_slug")
        ui_candidates.append(ui)
    return ui_candidates


# ===== Rendering helpers =====

def logo_html(url: str) -> str:
    """Generate HTML for logo image or placeholder."""
    if not url:
        return (
            "<svg width=48 height=48 viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'"
            " style='border-radius:8px;background:#f2f2f2;display:block'>"
            "<rect width='24' height='24' fill='#f2f2f2' rx='4'></rect>"
            "</svg>"
        )
    return f"<img src=\"{url}\" style=\"width:48px;height:48px;border-radius:8px;object-fit:cover;\"/>"


def render_result_card(r: dict, ix: int, cols: list):
    """Render a single result card in a column."""
    with cols[ix]:
        rid = r.get("id") or f"r_{ix}"
        is_selected = st.session_state.get("search3_selected_result") == rid
        selected_class = " selected" if is_selected else ""
        st.markdown(
            f"<div class=\"result-card{selected_class}\">"
            f"<div style='display:flex;gap:10px;align-items:center;'>"
            f"  <div style='flex:0 0 48px'>{logo_html(r.get('logo'))}</div>"
            f"  <div style='flex:1;min-width:0'>"
            f"    <div class='name' style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{r.get('name') or '(unknown)'}</div>"
            f"    <div class='subtitle' style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{r.get('subtitle') or ''}</div>"
            f"  </div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("Select", key=f"select_{rid}"):
            st.session_state.search3_selected_result = rid


def render_review_preview(fetched: list, platform: str):
    """Render preview section of fetched reviews."""
    st.markdown("---\n### Review preview (showing up to 5)")

    for it in fetched:
        title = it.get("title") or "(no title)"
        content = (it.get("content") or "").strip()
        snippet = content[:320] + ("…" if len(content) > 320 else "")
        rating = it.get("rating")
        date = it.get("date") or ""
        platform_id = it.get("platform") or platform

        st.markdown(
            f"<div class=\"result-card\">\n<strong>{title}</strong> — {rating or ''} ⭐<br/>\n<small>{platform_id} · {date}</small>\n<p style='margin-top:6px'>{snippet}</p>\n</div>",
            unsafe_allow_html=True,
        )



# ===== Confirmation & identifier extraction =====

def extract_identifier_info(confirmed: dict) -> tuple:
    """Extract identifier label and value from confirmed company dict."""
    if confirmed.get("package"):
        return ("Package", confirmed.get("package"))
    elif confirmed.get("app_id"):
        return ("App Store ID", confirmed.get("app_id"))
    elif confirmed.get("g2_slug"):
        return ("G2 slug", confirmed.get("g2_slug"))
    elif confirmed.get("tp_slug"):
        return ("Trustpilot slug", confirmed.get("tp_slug"))
    return None


def render_confirmed_company(confirmed: dict):
    """Render confirmed company card."""
    name = confirmed.get("name") or "(name not available)"
    platform = confirmed.get("platform") or ""
    source = confirmed.get("source") or ""

    identifier = extract_identifier_info(confirmed)
    id_html = f"<div><strong>{identifier[0]}:</strong> {identifier[1]}</div>" if identifier else ""

    st.markdown(
        f"<div class=\"result-card\">\n<h3 style=\"margin:0\">{name}</h3>\n<small>{platform} — confirmed via {source}</small>\n{id_html}\n</div>",
        unsafe_allow_html=True,
    )
