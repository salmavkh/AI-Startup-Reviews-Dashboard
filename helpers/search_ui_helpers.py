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
        ui = {
            "id": f"s{idx}",
            "platform": platform,
            "name": c.get("name") or c.get("title") or "(unknown)",
            "subtitle": c.get("subtitle") or "",
            "logo": c.get("logo") or None,
        }
        if c.get("package"):
            ui["package"] = c.get("package")
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


def render_analysis_results(
    analysis: dict,
    show_overall: bool = True,
    show_per_review: bool = True,
    show_topic_assignment: bool = True,
    show_section_heading: bool = True,
    show_topic_title_before_keywords: bool = False,
    show_review_preview: bool = True,
    compact_top_spacing: bool = False,
):
    """Render topic + sentiment analysis results."""
    if show_section_heading:
        st.markdown("---\n### Preview analysis")
    elif compact_top_spacing:
        st.markdown(
            """
            <style>
              div[data-testid="stTabs"] { margin-top: -8px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <style>
          .analysis-title {
            font-size: 22px;
            font-weight: 600;
            margin: 10px 0 4px 0;
          }
          .analysis-subtitle {
            color: #2f2f2f;
            margin-bottom: 8px;
          }
          .topic-summary-card {
            background: #f7f8fa;
            border: 1px solid #d6dbe3;
            border-radius: 12px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 8px;
          }
          .topic-summary-overview {
            color: #2f3442;
            line-height: 1.5;
            margin-bottom: 10px;
          }
          .topic-summary-subtitle {
            font-size: 15px;
            font-weight: 600;
            color: #2b3140;
            margin-bottom: 4px;
          }
          .topic-summary-list {
            margin: 0 0 8px 18px;
            color: #2f3442;
          }
          .topic-summary-list li {
            margin-bottom: 4px;
            line-height: 1.45;
          }
          .topic-summary-foot {
            color: #7b818c;
            font-size: 12px;
          }
          .topic-assign-card {
            background: #f7f8fa;
            border: 1px solid #d6dbe3;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 14px;
            min-height: 150px;
          }
          .topic-assign-row {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 12px;
          }
          .topic-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid #c9d1dc;
            background: #ffffff;
            color: #2f3442;
            border-radius: 999px;
            padding: 7px 13px;
            font-size: 15px;
            line-height: 1.25;
            font-weight: 500;
          }
          .topic-chip-primary {
            border-color: #86a8d9;
            background: #eaf2ff;
            color: #21406b;
            font-weight: 600;
          }
          .topic-chip-metric {
            margin-left: auto;
            border-color: #b9c2cf;
            background: #f3f5f8;
            color: #2a3240;
          }
          .topic-keyword-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
          }
          .topic-keyword-pill {
            display: inline-flex;
            align-items: center;
            background: #edf0f5;
            border: 1px solid #d5dbe4;
            color: #3a4150;
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 14px;
            line-height: 1.2;
          }
          .topic-assign-foot {
            color: #7b818c;
            font-size: 13px;
            margin-top: 12px;
          }
          .analysis-comment-label {
            font-size: 14px;
            font-weight: 700;
            margin-top: 6px;
            margin-bottom: 6px;
            color: #343845;
          }
          .analysis-comment-box {
            background: #f2f4f7;
            border-left: 4px solid #c5ccd6;
            border-radius: 10px;
            min-height: 44px;
            padding: 14px 14px 14px 40px;
            font-size: 15px;
            line-height: 1.45;
            color: #252933;
            margin-bottom: 12px;
            position: relative;
          }
          .analysis-comment-box::before {
            content: "“";
            position: absolute;
            left: 12px;
            top: 6px;
            font-size: 26px;
            line-height: 1;
            color: #9aa3af;
          }
          .emotion-pill {
            background: #cfcfcf;
            border-radius: 6px;
            text-align: center;
            padding: 10px 6px;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
          }
          .review-box {
            border: 1px solid #d6d6d6;
            border-radius: 12px;
            background: #f7f8fa;
            min-height: 120px;
            padding: 14px 16px 14px 40px;
            position: relative;
            margin-bottom: 14px;
          }
          .review-box::before {
            content: "“";
            position: absolute;
            left: 12px;
            top: 8px;
            font-size: 26px;
            line-height: 1;
            color: #9aa3af;
          }
          .review-count-label {
            color: #8d8d8d;
            font-size: 11px;
            margin-bottom: 6px;
          }
          .review-meta {
            color: #727982;
            font-size: 12px;
            margin-top: 10px;
          }
          .sentiment-rule {
            height: 4px;
            border-radius: 999px;
            margin-top: 6px;
            margin-bottom: 16px;
            width: 100%;
          }
          div[data-testid="stPopover"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            min-height: 0 !important;
            line-height: 1 !important;
            color: #262626 !important;
          }
          div[data-testid="stPopover"] button:hover {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
          }
          div[data-testid="stPopover"] button:focus {
            outline: none !important;
            box-shadow: none !important;
          }
          div[data-testid="stPopover"] button svg {
            display: none !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not show_overall and not show_per_review:
        return

    overall_placeholder = None
    per_review_placeholder = None
    has_tabbed_view = bool(show_overall and show_per_review)

    if has_tabbed_view:
        tabs = st.tabs(["Overall Results", "Insights per Review"])
        overall_tab = tabs[0]
        per_review_tab = tabs[1]
    elif show_per_review:
        per_review_tab = st.container()
        overall_placeholder = st.empty()
        overall_tab = overall_placeholder.container()
    else:
        overall_tab = st.container()
        per_review_placeholder = st.empty()
        per_review_tab = per_review_placeholder.container()

    reviews = analysis.get("reviews") or []
    sentiments = analysis.get("sentiment") or []

    per_review = analysis.get("emotion_by_review") or []
    if isinstance(per_review, dict):
        per_review_discrete = per_review.get("discrete") or []
        per_review_va = per_review.get("va") or []
    else:
        per_review_discrete = per_review or []
        per_review_va = []

    has_discrete = bool(reviews and len(per_review_discrete) == len(reviews))
    has_va = bool(reviews and len(per_review_va) == len(reviews))

    keyword_payload = analysis.get("keywords") or {}
    if isinstance(keyword_payload, dict):
        overall_keywords = keyword_payload.get("overall") or []
        per_review_keywords = keyword_payload.get("per_review") or []
    else:
        overall_keywords = []
        per_review_keywords = []

    topic_payload = analysis.get("topic") or {}
    if isinstance(topic_payload, dict):
        topics_per_review = topic_payload.get("topics") or []
        topic_keywords_by_topic = topic_payload.get("keywords_by_topic") or {}
        topic_labels_by_topic = topic_payload.get("topic_labels") or {}
        topic_probs = topic_payload.get("probs")
        topic_coherence = topic_payload.get("coherence") or {}
        raw_topic_rows = topic_payload.get("raw_topic_rows") or []
        raw_review_rows = topic_payload.get("raw_review_rows") or []
    else:
        topics_per_review = []
        topic_keywords_by_topic = {}
        topic_labels_by_topic = {}
        topic_probs = None
        topic_coherence = {}
        raw_topic_rows = []
        raw_review_rows = []

    def _normalize_sentiment_label(label) -> str:
        t = str(label or "").strip().lower()
        if t.startswith("pos"):
            return "Positive"
        if t.startswith("neg"):
            return "Negative"
        return "Uncertain"

    def _best_comment_for(label_name: str) -> str:
        best_idx = None
        best_conf = -1.0
        for i, item in enumerate(sentiments):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            s_label, s_conf = item[0], item[1]
            if _normalize_sentiment_label(s_label) != label_name:
                continue
            conf = _safe_float(s_conf, default=0.0)
            if conf > best_conf:
                best_conf = conf
                best_idx = i
        if best_idx is None or best_idx >= len(reviews):
            return "No matching review in current set."
        row = reviews[best_idx] or {}
        text = str(row.get("content") or row.get("title") or "").strip()
        if not text:
            return "Review text unavailable."
        return _short_text(text, limit=180)

    def _sentiment_palette(label: str) -> tuple[str, str]:
        l = str(label or "").strip().lower()
        if l.startswith("pos"):
            return ("#dbe8ff", "#1f77ff")
        if l.startswith("neg"):
            return ("#ffdada", "#e53935")
        return ("#fff3cc", "#f2c94c")

    def _render_emotion_intensity_circles(
        top_items: list[tuple[str, float]],
        key_prefix: str,
        sentiment_label: str = "Uncertain",
        title_text: str = "Top emotion",
    ):
        if not top_items:
            return
        st.markdown(title_text)
        ordered = [(str(name).title(), _safe_float(score)) for name, score in top_items]
        light_color, dark_color = _sentiment_palette(sentiment_label)
        df_bubble = pd.DataFrame(
            [
                {"emotion": name, "intensity": score, "order": idx}
                for idx, (name, score) in enumerate(ordered, start=1)
            ]
        )
        if alt is not None:
            bubble = (
                alt.Chart(df_bubble)
                .mark_circle(stroke="#7d8ca5", strokeWidth=1)
                .encode(
                    x=alt.X(
                        "emotion:N",
                        sort=[name for name, _ in ordered],
                        axis=alt.Axis(title=None, labelAngle=0, labelLimit=130),
                    ),
                    y=alt.value(85),
                    size=alt.Size("intensity:Q", scale=alt.Scale(range=[700, 6500]), legend=None),
                    color=alt.Color(
                        "intensity:Q",
                        scale=alt.Scale(range=[light_color, dark_color]),
                        legend=None,
                    ),
                    tooltip=[
                        "emotion:N",
                        alt.Tooltip("intensity:Q", format=".3f"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(
                bubble.configure_view(strokeOpacity=0),
                use_container_width=True,
                key=f"{key_prefix}_emotion_bubbles",
            )
        else:
            # Fallback when Altair is unavailable: draw circle-like cards in two rows.
            per_row = 5
            rows = [ordered[i:i + per_row] for i in range(0, len(ordered), per_row)]
            for ridx, row in enumerate(rows):
                cols = st.columns(per_row, gap="small")
                for cidx in range(per_row):
                    with cols[cidx]:
                        if cidx >= len(row):
                            st.write("")
                            continue
                        name, score = row[cidx]
                        diameter = int(max(62, min(120, 62 + (score * 58))))
                        st.markdown(
                            "<div style='display:flex;justify-content:center;'>"
                            f"<div style='width:{diameter}px;height:{diameter}px;border-radius:50%;"
                            f"background:{light_color};border:1px solid #7d8ca5;display:flex;align-items:center;"
                            "justify-content:center;text-align:center;padding:8px;font-size:11px;font-weight:600;'>"
                            f"{html.escape(name)}</div></div>"
                            f"<div style='text-align:center;font-size:11px;margin-top:4px;'>"
                            f"{score:.3f}</div>",
                            unsafe_allow_html=True,
                        )

    def _top_n_emotions(source: dict | None, n: int = 10) -> list[tuple[str, float]]:
        base = source or {}
        rows = sorted(
            ((str(k), _safe_float(v)) for k, v in base.items()),
            key=lambda kv: -kv[1],
        )
        if len(rows) >= n:
            return rows[:n]

        used = {name.lower() for name, _ in rows}
        for name in EMOTWEET_28:
            lname = str(name).strip().lower()
            if not lname or lname in used:
                continue
            rows.append((name, 0.0))
            used.add(lname)
            if len(rows) >= n:
                break
        return rows[:n]

    def _normalize_topic_id(value) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _topic_label(topic_id: int) -> str:
        if int(topic_id) == -1:
            return "Unassigned / Misc"
        return f"Topic {int(topic_id)}"

    def _build_raw_topic_cloud_rows(
        counts_map: dict | None,
        raw_rows: list | None,
        topic_labels_map: dict | None = None,
        topic_keywords_map: dict | None = None,
    ) -> list[dict]:
        topic_counts = {}
        if isinstance(counts_map, dict):
            for key, value in counts_map.items():
                tid = _normalize_topic_id(key)
                if tid is None:
                    continue
                try:
                    cnt = int(value)
                except Exception:
                    cnt = 0
                if cnt > 0:
                    topic_counts[tid] = cnt

        row_by_topic = {}
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                tid = _normalize_topic_id(row.get("topic_id"))
                if tid is None:
                    continue
                row_by_topic[tid] = row
                if tid not in topic_counts:
                    try:
                        cnt = int(row.get("count") or 0)
                    except Exception:
                        cnt = 0
                    if cnt > 0:
                        topic_counts[tid] = cnt

        total = max(1, sum(topic_counts.values()))
        rows = []
        for tid, cnt in sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            share = None
            src = row_by_topic.get(tid) or {}
            try:
                share = float(src.get("share"))
            except Exception:
                share = None
            if share is None:
                share = float(cnt) / float(total)
            topic_name = ""
            if isinstance(topic_labels_map, dict):
                topic_name = str(
                    topic_labels_map.get(tid)
                    or topic_labels_map.get(str(tid))
                    or ""
                ).strip()
            if not topic_name:
                topic_name = _topic_label(tid)

            top_words = str(src.get("words") or "").strip()
            if not top_words and isinstance(topic_keywords_map, dict):
                kws = topic_keywords_map.get(tid) or topic_keywords_map.get(str(tid)) or []
                if isinstance(kws, list):
                    top_words = ", ".join(str(k).strip() for k in kws[:10] if str(k).strip())

            if top_words:
                word_parts = [p.strip() for p in top_words.split(",") if p.strip()]
                words_short = ", ".join(word_parts[:6])
                legend_label = f"{topic_name}: {words_short}"
            else:
                legend_label = topic_name

            rows.append(
                {
                    "topic_id": tid,
                    "label": _topic_label(tid),
                    "topic_name": topic_name,
                    "count": int(cnt),
                    "share": max(0.0, min(1.0, float(share))),
                    "top_words": top_words,
                    "legend_label": legend_label,
                }
            )
        return rows

    def _render_topic_share_pie(topic_rows: list[dict], key_prefix: str):
        if not topic_rows:
            st.caption("No topics to visualize.")
            return

        chart_rows = []
        for row in topic_rows:
            cnt = int(row.get("count") or 0)
            if cnt <= 0:
                continue
            chart_rows.append(
                {
                    "topic_id": int(row.get("topic_id")),
                    "topic_name": str(row.get("topic_name") or row.get("label") or ""),
                    "top_words": str(row.get("top_words") or ""),
                    "legend_label": str(row.get("legend_label") or row.get("label") or ""),
                    "count": cnt,
                    "share_pct": _safe_float(row.get("share")) * 100.0,
                }
            )

        if not chart_rows:
            st.caption("No topics to visualize.")
            return

        df_pie = pd.DataFrame(chart_rows).sort_values(["count", "topic_id"], ascending=[False, True])

        if alt is not None:
            legend_domain = df_pie["legend_label"].tolist()
            legend_colors = [
                "#4E79A7",
                "#F28E2B",
                "#E15759",
                "#76B7B2",
                "#59A14F",
                "#EDC948",
                "#B07AA1",
                "#FF9DA7",
                "#9C755F",
                "#BAB0AC",
            ]
            color_range = [legend_colors[i % len(legend_colors)] for i in range(len(legend_domain))]
            color_map = {legend_domain[i]: color_range[i] for i in range(len(legend_domain))}

            pie_cols = st.columns([1.25, 1.0], gap="large")
            with pie_cols[0]:
                pie = (
                    alt.Chart(df_pie)
                    .mark_arc(outerRadius=120)
                    .encode(
                        theta=alt.Theta("count:Q", title="Share"),
                        color=alt.Color(
                            "legend_label:N",
                            legend=None,
                            scale=alt.Scale(domain=legend_domain, range=color_range),
                        ),
                        tooltip=[
                            "topic_name:N",
                            "top_words:N",
                            alt.Tooltip("count:Q", title="Reviews"),
                            alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(pie, use_container_width=True, key=f"{key_prefix}_topic_share_pie")

            with pie_cols[1]:
                st.markdown("**Topic name + top words**")
                for _, row in df_pie.iterrows():
                    label = str(row.get("legend_label") or "")
                    topic_name = str(row.get("topic_name") or "").strip()
                    top_words = str(row.get("top_words") or "").strip()
                    dot_color = color_map.get(label, "#4E79A7")
                    if topic_name and top_words:
                        legend_html = (
                            f"<strong>{html.escape(topic_name)}</strong>: {html.escape(top_words)}"
                        )
                    elif topic_name:
                        legend_html = f"<strong>{html.escape(topic_name)}</strong>"
                    else:
                        legend_html = html.escape(label)
                    st.markdown(
                        "<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;'>"
                        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                        f"margin-top:6px;background:{dot_color};flex:0 0 10px;'></span>"
                        f"<span style='font-size:14px;line-height:1.35;color:#2f3340;'>{legend_html}</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
        else:
            fallback_rows = []
            for row in chart_rows:
                fallback_rows.append(
                    {
                        "topic": row["topic_name"],
                        "top_words": row["top_words"],
                        "reviews": row["count"],
                        "share": f"{row['share_pct']:.1f}%",
                    }
                )
            st.dataframe(pd.DataFrame(fallback_rows), hide_index=True, use_container_width=True)

    def _build_topic_word_weights(raw_rows: list | None) -> dict[str, float]:
        weights: dict[str, float] = {}
        if not isinstance(raw_rows, list):
            return weights

        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            try:
                topic_count = max(1, int(row.get("count") or 0))
            except Exception:
                topic_count = 1
            words = str(row.get("words") or "").strip()
            if not words:
                continue
            parts = [p.strip().lower() for p in words.split(",") if p and p.strip()]
            for rank, token in enumerate(parts[:20]):
                cleaned = re.sub(r"[^a-z0-9\+\-\s]", " ", token)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if len(cleaned) < 2:
                    continue
                rank_weight = 1.0 / (1.0 + 0.35 * float(rank))
                weights[cleaned] = weights.get(cleaned, 0.0) + (float(topic_count) * rank_weight)
        return weights

    def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 4) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (
            (ax1 + pad) < bx0
            or (bx1 + pad) < ax0
            or (ay1 + pad) < by0
            or (by1 + pad) < ay0
        )

    def _render_weighted_wordcloud(
        weights: dict[str, float],
        key_prefix: str,
        width: int = 1200,
        height: int = 560,
        max_words: int = 120,
    ):
        clean_weights = {
            str(k).strip(): float(v)
            for k, v in (weights or {}).items()
            if str(k).strip() and _safe_float(v, 0.0) > 0
        }
        if not clean_weights:
            st.caption("No words available for word cloud.")
            return
        if Image is None or ImageDraw is None or ImageFont is None:
            st.caption("Word cloud rendering unavailable in this runtime.")
            return

        items = sorted(clean_weights.items(), key=lambda kv: -kv[1])[: max(10, int(max_words or 120))]
        if not items:
            st.caption("No words available for word cloud.")
            return

        canvas = Image.new("RGB", (width, height), "#f6f8fb")
        draw = ImageDraw.Draw(canvas)

        vals = [float(v) for _, v in items]
        min_v, max_v = min(vals), max(vals)

        def _font_size(value: float) -> int:
            if max_v <= min_v:
                return 42
            ratio = (float(value) - min_v) / max(1e-9, (max_v - min_v))
            return int(18 + ratio * 150)

        font_candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        ]
        font_cache: dict[int, object] = {}

        def _get_font(size: int):
            key = max(10, int(size))
            if key in font_cache:
                return font_cache[key]
            for path in font_candidates:
                try:
                    f = ImageFont.truetype(path, key)
                    font_cache[key] = f
                    return f
                except Exception:
                    continue
            f = ImageFont.load_default()
            font_cache[key] = f
            return f

        seed_src = "|".join(f"{w}:{v:.4f}" for w, v in items[:60])
        seed = int(hashlib.sha1(seed_src.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        palette = ["#1f77b4", "#1a9a8c", "#38a169", "#c5c90b", "#3f5aa9", "#1f6f8b", "#2f855a"]

        placed: list[tuple[int, int, int, int]] = []
        cx, cy = width // 2, height // 2
        max_radius = int(min(width, height) * 0.47)

        for idx_item, (word, weight) in enumerate(items):
            base_size = _font_size(float(weight))
            rendered = False
            for scale in (1.0, 0.88, 0.76, 0.66):
                font = _get_font(int(base_size * scale))
                try:
                    bbox0 = draw.textbbox((0, 0), word, font=font)
                except Exception:
                    continue
                tw = int(bbox0[2] - bbox0[0])
                th = int(bbox0[3] - bbox0[1])
                if tw <= 0 or th <= 0:
                    continue

                for attempt in range(450):
                    spiral = (attempt / 450.0) ** 1.35
                    radius = int(spiral * max_radius)
                    angle = rng.random() * 2.0 * math.pi + (idx_item * 0.17)
                    x = cx + int(radius * math.cos(angle)) - (tw // 2)
                    y = cy + int(radius * math.sin(angle)) - (th // 2)
                    rect = (x, y, x + tw, y + th)

                    if rect[0] < 8 or rect[1] < 8 or rect[2] > (width - 8) or rect[3] > (height - 8):
                        continue
                    if any(_rects_overlap(rect, other, pad=5) for other in placed):
                        continue

                    color = palette[(idx_item + attempt) % len(palette)]
                    draw.text((x, y), word, font=font, fill=color)
                    placed.append(rect)
                    rendered = True
                    break
                if rendered:
                    break

        st.image(canvas, use_container_width=True)

    def _render_topic_wordcloud_from_top_words(raw_rows: list | None, key_prefix: str):
        weights = _build_topic_word_weights(raw_rows)
        if not weights:
            st.caption("No topic top-words available for word cloud.")
            return
        _render_weighted_wordcloud(
            weights,
            key_prefix=key_prefix,
            width=1200,
            height=560,
            max_words=120,
        )

    def _build_one_review_per_topic(
        reviews_list: list,
        topics_list: list,
        counts_map: dict | None,
        topic_labels_map: dict | None,
        topic_keywords_map: dict | None,
    ) -> list[dict]:
        if not isinstance(reviews_list, list) or not isinstance(topics_list, list):
            return []
        if len(reviews_list) != len(topics_list):
            return []

        # Rank topics by frequency (outlier topic -1 shown last).
        topic_counts: dict[int, int] = {}
        if isinstance(counts_map, dict):
            for k, v in counts_map.items():
                tid = _normalize_topic_id(k)
                if tid is None:
                    continue
                try:
                    topic_counts[tid] = max(0, int(v))
                except Exception:
                    topic_counts[tid] = 0
        if not topic_counts:
            for t in topics_list:
                tid = _normalize_topic_id(t)
                if tid is None:
                    continue
                topic_counts[tid] = topic_counts.get(tid, 0) + 1

        best_by_topic: dict[int, dict] = {}
        for idx_row, topic_value in enumerate(topics_list):
            tid = _normalize_topic_id(topic_value)
            if tid is None or idx_row >= len(reviews_list):
                continue
            rv = reviews_list[idx_row] or {}
            title = str(rv.get("title") or "").strip()
            content = str(rv.get("content") or "").strip()
            text = content or title
            if not text:
                continue

            words = []
            if isinstance(topic_keywords_map, dict):
                raw_words = topic_keywords_map.get(tid) or topic_keywords_map.get(str(tid)) or []
                if isinstance(raw_words, list):
                    words = [str(w).strip().lower() for w in raw_words[:10] if str(w).strip()]
            overlap = sum(1 for w in words if w and w in text.lower())
            score = float(overlap) * 3.0 + min(1.0, len(text) / 220.0)

            stars = _rating_to_stars(rv.get("rating"))
            meta = " · ".join(
                [p for p in [str(rv.get("platform") or "").strip(), str(rv.get("date") or "").strip(), (stars if stars else None)] if p]
            )
            topic_name = ""
            if isinstance(topic_labels_map, dict):
                topic_name = str(
                    topic_labels_map.get(tid)
                    or topic_labels_map.get(str(tid))
                    or ""
                ).strip()
            if not topic_name:
                topic_name = _topic_label(tid)

            candidate = {
                "topic_id": tid,
                "topic_name": topic_name,
                "review_idx": idx_row,
                "title": title or f"Review {idx_row + 1}",
                "content": _short_text(content or title or "(no review text)", limit=260),
                "meta": meta,
                "score": score,
                "top_words": ", ".join(words[:6]) if words else "",
            }
            prev = best_by_topic.get(tid)
            if prev is None or candidate["score"] > prev["score"]:
                best_by_topic[tid] = candidate

        ordered_topic_ids = sorted(
            list(topic_counts.keys()),
            key=lambda tid: (1 if int(tid) == -1 else 0, -int(topic_counts.get(tid, 0)), int(tid)),
        )
        return [best_by_topic[tid] for tid in ordered_topic_ids if tid in best_by_topic]

    def _render_raw_topic_cloud(
        topic_rows: list[dict],
        key_prefix: str,
        active_topic_id: int | None = None,
    ):
        if not topic_rows:
            st.caption("No topics to visualize.")
            return

        sorted_rows = sorted(topic_rows, key=lambda r: (-int(r.get("count", 0)), int(r.get("topic_id", 0))))
        chart_rows = []
        labels = []
        for idx_row, row in enumerate(sorted_rows, start=1):
            tid = _normalize_topic_id(row.get("topic_id"))
            if tid is None:
                continue
            label = str(row.get("label") or _topic_label(tid))
            labels.append(label)
            chart_rows.append(
                {
                    "topic_id": tid,
                    "label": label,
                    "count": int(row.get("count") or 0),
                    "share_pct": _safe_float(row.get("share")) * 100.0,
                    "is_active": bool(active_topic_id is not None and int(active_topic_id) == tid),
                    "status": "active" if (active_topic_id is not None and int(active_topic_id) == tid) else "other",
                    "row_order": idx_row,
                }
            )

        if not chart_rows:
            st.caption("No topics to visualize.")
            return

        df_cloud = pd.DataFrame(chart_rows)
        if alt is not None:
            base = alt.Chart(df_cloud).encode(
                x=alt.X(
                    "label:N",
                    sort=labels,
                    axis=alt.Axis(title=None, labelAngle=0, labelLimit=220),
                ),
                y=alt.value(95),
                tooltip=[
                    "label:N",
                    alt.Tooltip("count:Q", title="Reviews"),
                    alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                ],
            )
            bubbles = base.mark_circle().encode(
                size=alt.Size(
                    "count:Q",
                    scale=alt.Scale(range=[1000, 9000]),
                    legend=None,
                ),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(domain=["other", "active"], range=["#8fb7ff", "#1f77ff"]),
                    legend=None,
                ),
                stroke=alt.condition(
                    "datum.is_active",
                    alt.value("#1b1f2a"),
                    alt.value("#6e7b8f"),
                ),
                strokeWidth=alt.condition(
                    "datum.is_active",
                    alt.value(2.0),
                    alt.value(1.0),
                ),
                opacity=alt.value(0.95),
            )
            labels_layer = base.mark_text(
                baseline="middle",
                align="center",
                dy=0,
                fontSize=11,
                color="#1f2430",
                fontWeight=600,
            ).encode(text="label:N")
            st.altair_chart(
                (bubbles + labels_layer).properties(height=240).configure_view(strokeOpacity=0),
                use_container_width=True,
                key=f"{key_prefix}_raw_topic_cloud",
            )
        else:
            cols_per_row = 4
            for row_start in range(0, len(chart_rows), cols_per_row):
                row_items = chart_rows[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row, gap="small")
                for cidx in range(cols_per_row):
                    with cols[cidx]:
                        if cidx >= len(row_items):
                            st.write("")
                            continue
                        item = row_items[cidx]
                        cnt = int(item.get("count") or 0)
                        dia = int(max(72, min(132, 72 + cnt * 6)))
                        bg = "#1f77ff" if item.get("is_active") else "#dfeaff"
                        fg = "#ffffff" if item.get("is_active") else "#243142"
                        border = "#1b1f2a" if item.get("is_active") else "#8ea1ba"
                        st.markdown(
                            "<div style='display:flex;justify-content:center;'>"
                            f"<div style='width:{dia}px;height:{dia}px;border-radius:50%;"
                            f"background:{bg};border:1px solid {border};display:flex;align-items:center;"
                            "justify-content:center;text-align:center;padding:8px;'>"
                            f"<div style='font-size:12px;font-weight:600;color:{fg};line-height:1.25;'>"
                            f"{html.escape(str(item.get('label') or 'Topic'))}<br/>"
                            f"({cnt})"
                            "</div></div></div>",
                            unsafe_allow_html=True,
                        )

    def _emotion_bar_height(num_items: int) -> int:
        # Keep enough vertical room so all top-10 labels are visible.
        return max(360, int(num_items) * 38 + 40)

    def _overall_distance_scores_from_va(va_rows: list | None) -> dict[str, float]:
        """Average distance-derived emotion scores over all reviews with VA points."""
        if not isinstance(va_rows, list) or not va_rows:
            return {}

        sums: dict[str, float] = {}
        counts_local: dict[str, int] = {}
        for point in va_rows:
            if not isinstance(point, dict):
                continue
            v = _safe_float(point.get("valence", 0.0))
            a = _safe_float(point.get("arousal", 0.0))
            rows, _missing = _build_emotion_distance_rows(v, a)
            for row in rows:
                emo = str(row.get("emotion") or "").strip().lower()
                if not emo:
                    continue
                d = _safe_float(row.get("distance"), 0.0)
                score = 1.0 / (1.0 + max(0.0, d))
                sums[emo] = sums.get(emo, 0.0) + score
                counts_local[emo] = counts_local.get(emo, 0) + 1

        out = {}
        for emo, total in sums.items():
            n_local = max(1, int(counts_local.get(emo, 0)))
            out[emo] = float(total) / float(n_local)
        return out

    def _overall_model_top_emotion_confidence(per_review_dist_rows: list | None) -> float | None:
        """Average top emotion probability across reviews (model-based)."""
        if not isinstance(per_review_dist_rows, list) or not per_review_dist_rows:
            return None
        vals = []
        for row in per_review_dist_rows:
            if not isinstance(row, dict) or not row:
                continue
            probs = []
            for v in row.values():
                try:
                    probs.append(float(v))
                except Exception:
                    continue
            if probs:
                vals.append(max(0.0, min(1.0, max(probs))))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _overall_distance_top_emotion_confidence(va_rows: list | None) -> float | None:
        """Average top distance-derived emotion score across reviews."""
        if not isinstance(va_rows, list) or not va_rows:
            return None
        vals = []
        for point in va_rows:
            if not isinstance(point, dict):
                continue
            v = _safe_float(point.get("valence", 0.0))
            a = _safe_float(point.get("arousal", 0.0))
            rows, _missing = _build_emotion_distance_rows(v, a)
            if not rows:
                continue
            d = _safe_float(rows[0].get("distance"), 0.0)
            score = 1.0 / (1.0 + max(0.0, d))
            vals.append(max(0.0, min(1.0, score)))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _overall_topic_assignment_confidence(topic_prob_rows, raw_rows: list | None) -> float | None:
        """Average per-review topic assignment confidence."""
        vals = []

        if topic_prob_rows is not None:
            try:
                for row in topic_prob_rows:
                    if hasattr(row, "__len__") and not isinstance(row, (str, bytes)):
                        probs = []
                        for x in row:
                            try:
                                probs.append(float(x))
                            except Exception:
                                continue
                        if probs:
                            vals.append(max(0.0, min(1.0, max(probs))))
                    else:
                        vals.append(max(0.0, min(1.0, float(row))))
            except Exception:
                vals = []

        if not vals and isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                conf = row.get("confidence")
                if conf is None:
                    continue
                vals.append(max(0.0, min(1.0, _safe_float(conf, 0.0))))

        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _parse_topic_llm_summary(summary_text: str) -> tuple[str, list[str]]:
        text = str(summary_text or "").strip()
        if not text:
            return "", []

        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if str(ln).strip()]
        overview = ""
        bullets: list[str] = []

        for ln in lines:
            low = ln.lower()
            if low in {"llm summary", "summary"}:
                continue
            if low.startswith("overview:"):
                candidate = ln.split(":", 1)[1].strip()
                if candidate and not overview:
                    overview = candidate
                continue
            if low.startswith("topics:") or low.startswith("top themes:"):
                continue

            numbered = re.match(r"^\d+[\.\)]\s*(.+)$", ln)
            dashed = re.match(r"^[-*]\s+(.+)$", ln)
            if numbered:
                item = numbered.group(1).strip()
                if item:
                    bullets.append(item)
                continue
            if dashed:
                item = dashed.group(1).strip()
                if item:
                    bullets.append(item)
                continue

            if not overview:
                overview = ln
            elif not bullets and len(ln.split()) >= 5:
                bullets.append(ln)

        if not bullets:
            for m in re.finditer(r"(?:^|\n)\s*\d+[\.\)]\s*(.+)", text):
                item = re.sub(r"\s+", " ", str(m.group(1) or "")).strip()
                if item:
                    bullets.append(item)

        cleaned_bullets = []
        seen = set()
        for item in bullets:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_bullets.append(item)
            if len(cleaned_bullets) >= 5:
                break

        if not overview and lines:
            for ln in lines:
                low = ln.lower()
                if low.startswith("topics:"):
                    continue
                overview = ln
                break

        return overview, cleaned_bullets

    def _render_topic_summary_card(summary_text: str):
        overview, bullets = _parse_topic_llm_summary(summary_text)
        fallback = _short_text(str(summary_text or "").strip(), limit=650)
        if not overview:
            overview = fallback or "Summary unavailable for this sample."

        bullets_html = "".join(
            f"<li>{html.escape(str(item))}</li>" for item in bullets if str(item).strip()
        )
        card_html = (
            "<div class='topic-summary-card'>"
            f"<div class='topic-summary-overview'><strong>Overview:</strong> {html.escape(overview)}</div>"
        )
        if bullets_html:
            card_html += (
                "<div class='topic-summary-subtitle'>Top themes</div>"
                f"<ul class='topic-summary-list'>{bullets_html}</ul>"
            )
        card_html += "<div class='topic-summary-foot'>Generated from current fetched sample.</div></div>"
        st.markdown(card_html, unsafe_allow_html=True)

    distance_info_text = (
        "Distance calculation: this ranks emotions by directly measuring how close the review's "
        "valence-arousal point is to each emotion anchor in VA space. Smaller distance means higher score."
    )
    prediction_info_text = (
        "Model prediction: this uses the trained discrete-emotion model that outputs emotion scores directly "
        "from the review text. It learns multi-emotion patterns beyond pure VA distance."
    )

    def _render_graph_subheader(title: str, info_text: str):
        cols = st.columns([20, 1], gap="small")
        with cols[0]:
            st.markdown(
                f"<div style='font-size:16px;font-weight:400;line-height:1.2;color:#31333F;'>{html.escape(title)}</div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            if hasattr(st, "popover"):
                with st.popover("ⓘ"):
                    st.write(info_text)
            else:
                st.caption("ⓘ")
        st.markdown(
            "<div style='border-top:1px solid #2f2f2f;margin:4px 0 14px 0;'></div>",
            unsafe_allow_html=True,
        )

    # Emotions: supports both legacy and nested format
    emo = analysis.get("emotion") or {}
    if isinstance(emo, dict) and isinstance(emo.get("discrete"), dict):
        emo_discrete = emo.get("discrete") or {}
        emo_va = emo.get("va") or {}
    else:
        emo_discrete = emo if isinstance(emo, dict) else {}
        emo_va = {}

    # --------------------
    # Overall tab
    # --------------------
    with overall_tab:
        st.markdown("<div class='analysis-title'>Sentiment analysis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='analysis-subtitle'>Quickly see whether overall sentiment leans positive or negative.</div>",
            unsafe_allow_html=True,
        )
        pos = neg = unc = 0
        for s, _conf in sentiments:
            norm = _normalize_sentiment_label(s)
            if norm == "Positive":
                pos += 1
            elif norm == "Negative":
                neg += 1
            else:
                unc += 1

        total = pos + neg + unc
        sentiment_conf_vals = []
        for item in sentiments:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            sentiment_conf_vals.append(max(0.0, min(1.0, _safe_float(item[1], 0.0))))
        overall_sentiment_conf = (
            float(sum(sentiment_conf_vals) / len(sentiment_conf_vals))
            if sentiment_conf_vals
            else None
        )
        overall_sentiment_label = "Uncertain"
        if pos >= neg and pos >= unc:
            overall_sentiment_label = "Positive"
        elif neg >= pos and neg >= unc:
            overall_sentiment_label = "Negative"

        if total > 0:
            df_sent = pd.DataFrame(
                [
                    {"label": "Positive", "count": pos, "percent": (pos / total) * 100.0},
                    {"label": "Negative", "count": neg, "percent": (neg / total) * 100.0},
                    {"label": "Uncertain", "count": unc, "percent": (unc / total) * 100.0},
                ]
            )
            sent_cols = st.columns([1, 1], gap="large")
            with sent_cols[0]:
                st.caption(
                    f"Positive {pos/total:.1%} · Negative {neg/total:.1%} · Uncertain {unc/total:.1%}"
                )
                if overall_sentiment_conf is not None:
                    st.caption(f"Overall sentiment confidence: {overall_sentiment_conf:.1%}")
                if alt is not None:
                    pie = (
                        alt.Chart(df_sent)
                        .mark_arc(outerRadius=120)
                        .encode(
                            theta=alt.Theta(field="count", type="quantitative"),
                            color=alt.Color(
                                field="label",
                                type="nominal",
                                scale=alt.Scale(
                                    domain=["Positive", "Negative", "Uncertain"],
                                    range=["#1f77ff", "#e53935", "#f2c94c"],
                                ),
                                legend=alt.Legend(title="Label"),
                            ),
                            tooltip=["label", "count", alt.Tooltip("percent:Q", format=".1f")],
                        )
                    )
                    st.altair_chart(pie, use_container_width=True)
                else:
                    st.bar_chart(df_sent, x="label", y="count", use_container_width=True)

            with sent_cols[1]:
                pos_comment = html.escape(_best_comment_for("Positive"))
                neg_comment = html.escape(_best_comment_for("Negative"))
                unc_comment = html.escape(_best_comment_for("Uncertain"))
                st.markdown("<div class='analysis-comment-label'>Most positive comment</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='analysis-comment-box'>{pos_comment}</div>", unsafe_allow_html=True)
                st.markdown("<div class='analysis-comment-label'>Most negative comment</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='analysis-comment-box'>{neg_comment}</div>", unsafe_allow_html=True)
                st.markdown("<div class='analysis-comment-label'>Most neutral comment</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='analysis-comment-box'>{unc_comment}</div>", unsafe_allow_html=True)
        else:
            st.write("Positive: 0 — Negative: 0 — Uncertain: 0")

        emo_pct = emo_discrete.get("percentages") or {}
        has_emo_overall = bool(emo_va or emo_pct)
        if has_emo_overall:
            st.markdown("<div class='analysis-title'>Emotion analysis</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='analysis-subtitle'>Discover the specific emotions behind reviews.</div>",
                unsafe_allow_html=True,
            )
            if emo_pct:
                top_items = _top_n_emotions(emo_pct, n=10)
                if top_items:
                    _render_emotion_intensity_circles(
                        top_items,
                        key_prefix="overall",
                        sentiment_label=overall_sentiment_label,
                        title_text="Top emotion (model)",
                    )

                with st.expander("See how we calculate the emotion", expanded=True):
                    st.markdown(
                    "1. Valence → measures whether a review leans positive or negative "
                    "(pleasant ↔ unpleasant).\n"
                    "2. Arousal → measures emotional intensity, from activated to calm "
                    "(excited ↔ calm).\n\n"
                    "Together, Valence and Arousal place each review in emotional space, "
                    "so we capture both emotion type and strength."
                )

            if emo_va:
                mean_val = _safe_float(emo_va.get("mean_valence", 0.0))
                mean_aro = _safe_float(emo_va.get("mean_arousal", 0.0))
                std_val = _safe_float(emo_va.get("std_valence", 0.0))
                std_aro = _safe_float(emo_va.get("std_arousal", 0.0))
                iqr_val = _safe_float(emo_va.get("iqr_valence", 0.0))
                iqr_aro = _safe_float(emo_va.get("iqr_arousal", 0.0))
                min_val = _safe_float(emo_va.get("min_valence", 0.0))
                max_val = _safe_float(emo_va.get("max_valence", 0.0))
                min_aro = _safe_float(emo_va.get("min_arousal", 0.0))
                max_aro = _safe_float(emo_va.get("max_arousal", 0.0))
                mean_dist = _safe_float(emo_va.get("mean_distance", 0.0))

                st.caption("Explore how reviews map across emotional space. Hover over points to read the review.")
                if has_va:
                    va_rows = []
                    for i, review in enumerate(reviews):
                        point = per_review_va[i] or {}
                        q = str(point.get("quadrant") or "").strip().upper()
                        if q not in {"HVHA", "HVLA", "LVHA", "LVLA"}:
                            q = "REVIEW"
                        va_rows.append(
                            {
                                "review_idx": i,
                                "review_label": f"Review {i + 1}",
                                "title": str(review.get("title") or "(no title)"),
                                "review": _short_text(review.get("content") or ""),
                                "valence": _safe_float(point.get("valence", 0.0)),
                                "arousal": _safe_float(point.get("arousal", 0.0)),
                                "quadrant": str(point.get("quadrant") or ""),
                                "kind": "review",
                                "color_key": q,
                            }
                        )
                    guide_rows = []
                    for step in range(-100, 101):
                        t = step / 100.0
                        guide_rows.append(
                            {
                                "review_idx": -1,
                                "review_label": "",
                                "title": "",
                                "review": "",
                                "valence": t,
                                "arousal": 0.0,
                                "quadrant": "",
                                "kind": "guide",
                                "color_key": "GUIDE",
                            }
                        )
                        guide_rows.append(
                            {
                                "review_idx": -1,
                                "review_label": "",
                                "title": "",
                                "review": "",
                                "valence": 0.0,
                                "arousal": t,
                                "quadrant": "",
                                "kind": "guide",
                                "color_key": "GUIDE",
                            }
                        )
                    df_va = pd.DataFrame(guide_rows + va_rows)
                    if alt is not None:
                        pick = alt.selection_point(
                            name="review_pick",
                            fields=["review_idx"],
                            on="click",
                            nearest=True,
                            empty=False,
                        )
                        points_chart = (
                            alt.Chart(df_va)
                            .mark_circle()
                            .encode(
                                x=alt.X("valence:Q", scale=alt.Scale(domain=[-1, 1]), title="Valence"),
                                y=alt.Y("arousal:Q", scale=alt.Scale(domain=[-1, 1]), title="Arousal"),
                                color=alt.Color(
                                    "color_key:N",
                                    title="Quadrant",
                                    scale=alt.Scale(
                                        domain=["GUIDE", "HVHA", "HVLA", "LVHA", "LVLA", "REVIEW"],
                                        range=[
                                            "#c8c8c8",
                                            "#1f77b4",
                                            "#2ca02c",
                                            "#d62728",
                                            "#9467bd",
                                            "#111111",
                                        ],
                                    ),
                                    legend=alt.Legend(values=["HVHA", "HVLA", "LVHA", "LVLA", "REVIEW"]),
                                ),
                                size=alt.condition(
                                    "datum.kind == 'review'",
                                    alt.value(170),
                                    alt.value(12),
                                ),
                                opacity=alt.condition(
                                    "datum.kind == 'review'",
                                    alt.value(0.95),
                                    alt.value(0.55),
                                ),
                                stroke=alt.condition(
                                    "datum.kind == 'review'",
                                    alt.value("#111111"),
                                    alt.value("#c8c8c8"),
                                ),
                                strokeWidth=alt.condition(
                                    "datum.kind == 'review'",
                                    alt.value(0.6),
                                    alt.value(0.0),
                                ),
                                tooltip=[
                                    "review_label:N",
                                    "title:N",
                                    "review:N",
                                    alt.Tooltip("valence:Q", format=".3f"),
                                    alt.Tooltip("arousal:Q", format=".3f"),
                                ],
                            )
                            .add_params(pick)
                            .properties(height=320)
                        )
                        event = st.altair_chart(
                            points_chart,
                            use_container_width=True,
                            on_select="rerun",
                            selection_mode=["review_pick"],
                            key="search3_va_overall_scatter",
                        )
                        picked_idx = _extract_selected_review_idx(event)
                        if picked_idx is not None and 0 <= int(picked_idx) < len(reviews):
                            if picked_idx != st.session_state.get("search3_review_idx"):
                                st.session_state.search3_review_idx = picked_idx
                                if has_tabbed_view:
                                    st.success(
                                        f"Selected Review {picked_idx + 1}. Open 'Insights per Review' tab to view details."
                                    )
                                else:
                                    st.success(f"Selected Review {picked_idx + 1}.")
                    else:
                        st.scatter_chart(df_va, x="valence", y="arousal", color="quadrant")

                with st.expander("See more details breakdown", expanded=True):
                    mcols = st.columns(2)
                    with mcols[0]:
                        st.markdown("**VALENCE**")
                        st.caption(f"[{min_val:.3f}, {max_val:.3f}]")
                        vm = st.columns(3)
                        vm[0].metric("Mean valence", f"{mean_val:.3f}")
                        vm[1].metric("Valence std", f"{std_val:.3f}")
                        vm[2].metric("Valence IQR", f"{iqr_val:.3f}")
                    with mcols[1]:
                        st.markdown("**AROUSAL**")
                        st.caption(f"[{min_aro:.3f}, {max_aro:.3f}]")
                        am = st.columns(3)
                        am[0].metric("Mean arousal", f"{mean_aro:.3f}")
                        am[1].metric("Arousal std", f"{std_aro:.3f}")
                        am[2].metric("Arousal IQR", f"{iqr_aro:.3f}")

                    st.info(
                        "Overall review trend profile. Distance intensity is high on average."
                        if mean_dist >= 0.55
                        else "Overall review trend profile. Distance intensity is moderate to low on average."
                    )

                    quadrant_defs = {
                        "HVHA": "High Valence, High Arousal",
                        "HVLA": "High Valence, Low Arousal",
                        "LVHA": "Low Valence, High Arousal",
                        "LVLA": "Low Valence, Low Arousal",
                    }
                    quadrant_order = ["HVHA", "HVLA", "LVHA", "LVLA"]
                    quadrant_counts = {q: 0 for q in quadrant_order}

                    if has_va and per_review_va:
                        for point in per_review_va:
                            q = str((point or {}).get("quadrant") or "").strip().upper()
                            if q in quadrant_counts:
                                quadrant_counts[q] += 1

                    if sum(quadrant_counts.values()) == 0:
                        quad_pct = emo_va.get("quadrant_percentages") or {}
                        total_rows = max(0, len(reviews))
                        for q in quadrant_order:
                            pct = _safe_float(quad_pct.get(q, 0.0))
                            quadrant_counts[q] = int(round((pct / 100.0) * total_rows))

                    total_quadrant = max(1, sum(quadrant_counts.values()))
                    q_df = pd.DataFrame(
                        [
                            {
                                "quadrant": q,
                                "definition": quadrant_defs[q],
                                "count": int(quadrant_counts[q]),
                                "share": (float(quadrant_counts[q]) / float(total_quadrant)) * 100.0,
                            }
                            for q in quadrant_order
                        ]
                    )

                    st.markdown("**VA quadrants (count of reviews)**")
                    st.caption(
                        "HVHA = High Valence + High Arousal · "
                        "HVLA = High Valence + Low Arousal · "
                        "LVHA = Low Valence + High Arousal · "
                        "LVLA = Low Valence + Low Arousal"
                    )

                    q_cols = st.columns(4, gap="small")
                    for idx_q, q in enumerate(quadrant_order):
                        q_cols[idx_q].metric(q, f"{int(quadrant_counts[q])}")

                    if alt is not None:
                        q_chart = (
                            alt.Chart(q_df)
                            .mark_bar(color="#1976d2")
                            .encode(
                                x=alt.X("quadrant:N", sort=quadrant_order, title=None),
                                y=alt.Y("count:Q", title="Number of reviews"),
                                tooltip=[
                                    "quadrant:N",
                                    "definition:N",
                                    "count:Q",
                                    alt.Tooltip("share:Q", format=".1f"),
                                ],
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(q_chart, use_container_width=True)
                    else:
                        st.bar_chart(q_df, x="quadrant", y="count", use_container_width=True)

            if emo_pct or has_va:
                detail_cols = st.columns(2, gap="large")
                with detail_cols[0]:
                    st.markdown("Emotions intensity")
                    _render_graph_subheader("By distance calculation", distance_info_text)
                    overall_distance_conf = _overall_distance_top_emotion_confidence(
                        per_review_va if has_va else []
                    )
                    if overall_distance_conf is not None:
                        st.caption(f"Overall emotion confidence (distance top score): {overall_distance_conf:.1%}")
                    overall_dist = _overall_distance_scores_from_va(per_review_va if has_va else [])
                    if overall_dist:
                        top_dist = _top_n_emotions(overall_dist, n=10)
                        df_dist = pd.DataFrame(top_dist, columns=["emotion", "score"])
                        if alt is not None:
                            d_chart = (
                                alt.Chart(df_dist)
                                .mark_bar(color="#1976d2")
                                .encode(
                                    y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                    x=alt.X("score:Q", title="Score"),
                                    tooltip=["emotion", alt.Tooltip("score:Q", format=".3f")],
                                )
                                .properties(height=_emotion_bar_height(len(df_dist)))
                            )
                            st.altair_chart(d_chart, use_container_width=True)
                        else:
                            st.bar_chart(df_dist, x="emotion", y="score", use_container_width=True)
                    else:
                        st.caption("Distance-based view unavailable for this set.")

                with detail_cols[1]:
                    st.markdown("Emotions intensity")
                    _render_graph_subheader("By model prediction", prediction_info_text)
                    overall_model_conf = _overall_model_top_emotion_confidence(
                        per_review_discrete if has_discrete else []
                    )
                    if overall_model_conf is not None:
                        st.caption(f"Overall emotion confidence (model top score): {overall_model_conf:.1%}")
                    top_pred = _top_n_emotions(emo_pct, n=10) if emo_pct else []
                    if top_pred:
                        df_pred = pd.DataFrame(top_pred, columns=["emotion", "score"])
                        if alt is not None:
                            p_chart = (
                                alt.Chart(df_pred)
                                .mark_bar(color="#1976d2")
                                .encode(
                                    y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                    x=alt.X("score:Q", title="Score (0-1)"),
                                    tooltip=["emotion", alt.Tooltip("score:Q", format=".3f")],
                                )
                                .properties(height=_emotion_bar_height(len(df_pred)))
                            )
                            st.altair_chart(p_chart, use_container_width=True)
                        else:
                            st.bar_chart(df_pred, x="emotion", y="score", use_container_width=True)
                    else:
                        st.caption("Model-based view unavailable for this set.")
        else:
            st.caption("No overall emotion analysis to show yet.")

        # Topic (overall)
        counts = topic_payload.get("counts") or {}
        keywords = topic_keywords_by_topic or {}
        topic_summary = (analysis.get("topic_summary") or "").strip()
        if show_topic_assignment and (counts or topic_summary):
            st.markdown("<div class='analysis-title'>Topic Modelling</div>", unsafe_allow_html=True)
            c_v_overall = topic_coherence.get("c_v_overall")
            c_v_available = bool(topic_coherence.get("available"))
            c_v_error = str(topic_coherence.get("error") or "").strip()
            proxy_overall = _safe_float(topic_coherence.get("proxy_overall", 0.0))
            topic_assign_conf = _overall_topic_assignment_confidence(topic_probs, raw_review_rows)
            topic_cloud_rows = _build_raw_topic_cloud_rows(
                counts,
                raw_topic_rows,
                topic_labels_map=topic_labels_by_topic,
                topic_keywords_map=topic_keywords_by_topic,
            )
            if topic_cloud_rows:
                st.markdown("**Topic share**")
                st.caption("Pie chart of raw topic share. Legend shows topic name and top words.")
                pie_section_cols = st.columns([0.45, 1.55], gap="large")
                with pie_section_cols[0]:
                    st.metric("Topics found", f"{len([t for t in counts.keys() if int(t) != -1])}")
                    if c_v_overall is not None:
                        st.metric("Coherence c_v", f"{_safe_float(c_v_overall):.3f}")
                    else:
                        st.metric("Coherence c_v", "N/A")
                    st.metric("Coherence proxy", f"{proxy_overall:.3f}")
                    if topic_assign_conf is not None:
                        st.metric("Assignment confidence", f"{topic_assign_conf:.1%}")
                    else:
                        st.metric("Assignment confidence", "N/A")
                with pie_section_cols[1]:
                    _render_topic_share_pie(
                        topic_cloud_rows,
                        key_prefix="overall",
                    )
                st.markdown("**Topic word cloud (from all top words)**")
                st.caption("Built from raw topic top words weighted by topic size and term rank.")
                _render_topic_wordcloud_from_top_words(
                    raw_topic_rows,
                    key_prefix="overall",
                )
                topic_examples = _build_one_review_per_topic(
                    reviews_list=reviews,
                    topics_list=topics_per_review,
                    counts_map=counts,
                    topic_labels_map=topic_labels_by_topic,
                    topic_keywords_map=topic_keywords_by_topic,
                )
                if topic_examples:
                    st.markdown("**Reviews sample**")
                    st.caption("Representative example from each discovered topic.")
                    topic_page_key = "search3_topic_examples_page"
                    per_page = 4
                    total_examples = len(topic_examples)
                    total_pages = max(1, int(math.ceil(total_examples / float(per_page))))
                    if topic_page_key not in st.session_state:
                        st.session_state[topic_page_key] = 0
                    st.session_state[topic_page_key] = max(
                        0,
                        min(int(st.session_state[topic_page_key]), total_pages - 1),
                    )

                    page_idx = int(st.session_state[topic_page_key])
                    start = page_idx * per_page
                    end = min(total_examples, start + per_page)
                    page_examples = topic_examples[start:end]

                    for row_start in range(0, len(page_examples), 2):
                        row_items = page_examples[row_start:row_start + 2]
                        cols = st.columns(2, gap="large")
                        for cidx in range(2):
                            with cols[cidx]:
                                if cidx >= len(row_items):
                                    st.write("")
                                    continue
                                ex = row_items[cidx]
                                tid = int(ex.get("topic_id", -1))
                                tname = str(ex.get("topic_name") or "").strip() or f"Topic {tid}"
                                topic_head = f"Topic {tid}: {tname}"
                                st.markdown(f"**{html.escape(topic_head)}**")
                                if ex.get("top_words"):
                                    st.caption(f"Top words: {ex.get('top_words')}")
                                title_html = html.escape(str(ex.get("title") or "")).strip()
                                content_html = html.escape(str(ex.get("content") or "(no review text)")).strip()
                                meta_html = html.escape(str(ex.get("meta") or "")).strip()
                                st.markdown(
                                    "<div class='analysis-comment-box'>"
                                    f"<div style='font-size:14px;font-weight:600;margin-bottom:6px;'>{title_html}</div>"
                                    f"<div style='font-size:15px;line-height:1.45;'>{content_html}</div>"
                                    + (
                                        f"<div class='review-meta' style='margin-top:8px;'>{meta_html}</div>"
                                        if meta_html
                                        else ""
                                    )
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )

                    if total_examples > per_page:
                        nav_cols = st.columns([1, 1, 3], gap="small")
                        with nav_cols[0]:
                            if st.button("Prev", key="overall_topic_examples_prev", use_container_width=True):
                                st.session_state[topic_page_key] = (page_idx - 1) % total_pages
                                st.rerun()
                        with nav_cols[1]:
                            if st.button("Next", key="overall_topic_examples_next", use_container_width=True):
                                st.session_state[topic_page_key] = (page_idx + 1) % total_pages
                                st.rerun()
                        with nav_cols[2]:
                            status_text = (
                                f"Showing topics {start + 1}-{end} of {total_examples} "
                                f"(page {page_idx + 1} of {total_pages})"
                            )
                            st.markdown(
                                f"<div style='text-align:right;color:#7b818c;font-size:0.9rem;"
                                f"padding-top:6px;'>{html.escape(status_text)}</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption(f"Showing topics {start + 1}-{end} of {total_examples}")

            if TOPIC_DEBUG_MESSAGES_ENABLED:
                topic_examples = {}
                if isinstance(topics_per_review, list) and len(topics_per_review) == len(reviews):
                    scored_examples = {}
                    for idx, topic_id in enumerate(topics_per_review):
                        try:
                            tid = int(topic_id)
                        except Exception:
                            continue
                        review_obj = reviews[idx] or {}
                        review_text = str((review_obj.get("content") or review_obj.get("title") or "")).strip()
                        if not review_text:
                            continue
                        if tid == -1:
                            continue
                        topic_keywords = [str(k).strip().lower() for k in keywords.get(tid, [])[:10] if str(k).strip()]
                        overlap = sum(1 for kw in topic_keywords if kw and kw in review_text.lower())
                        snippet = _short_text(review_text, limit=180)
                        scored_examples.setdefault(tid, []).append((overlap, snippet))

                    for tid, rows in scored_examples.items():
                        rows.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
                        picked = []
                        seen = set()
                        for overlap, snippet in rows:
                            if overlap <= 0:
                                continue
                            if snippet in seen:
                                continue
                            seen.add(snippet)
                            picked.append(snippet)
                            if len(picked) >= TOPIC_DEBUG_EXAMPLES_PER_THEME:
                                break
                        if picked:
                            topic_examples[tid] = picked

                theme_rank = 1
                for topic_id, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
                    if topic_id == -1:
                        lbl = "Outliers"
                    else:
                        top_words = ", ".join(keywords.get(topic_id, [])[:5])
                        if top_words:
                            lbl = f"Theme {theme_rank} ({cnt} reviews): {top_words}"
                        else:
                            lbl = f"Theme {theme_rank} ({cnt} reviews)"
                        theme_rank += 1

                    if topic_id == -1:
                        st.write(f"- {lbl}: {cnt}")
                    else:
                        st.write(f"- {lbl}")

                    examples = topic_examples.get(topic_id, [])
                    if topic_id != -1:
                        if examples:
                            for i, ex in enumerate(examples, start=1):
                                st.caption(f"Example {i}: {ex}")
                        else:
                            st.caption("No strong keyword-matching examples in this sample.")

            if topic_summary:
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                st.markdown("**LLM Summary**")
                _render_topic_summary_card(topic_summary)

    # --------------------
    # Per-review tab
    # --------------------
    with per_review_tab:
        if not reviews:
            st.caption("No per-review details to show yet.")
        else:
            if "search3_review_idx" not in st.session_state:
                st.session_state.search3_review_idx = 0

            n = len(reviews)
            st.session_state.search3_review_idx = max(0, min(st.session_state.search3_review_idx, n - 1))

            idx = st.session_state.search3_review_idx
            r = reviews[idx]
            title = str(r.get("title") or f"Review {idx + 1}").strip()
            content = str(r.get("content") or "").strip()
            rating = r.get("rating")
            date = r.get("date") or ""
            platform = r.get("platform") or ""
            stars = _rating_to_stars(rating)
            meta = " · ".join(
                [p for p in [platform, date, (stars if stars else None)] if p]
            )
            if show_review_preview:
                safe_body = html.escape(content or "(no review text)").replace("\n", "<br/>")

                st.markdown(
                    "<div class='review-box'>"
                    f"<div class='review-count-label'>Review {idx + 1} of {n}</div>"
                    f"<div style='font-size:14px;font-weight:600;margin-bottom:6px;'>{html.escape(title)}</div>"
                    f"<div style='font-size:14px;line-height:1.45;'>{safe_body}</div>"
                    f"<div class='review-meta'>{html.escape(meta)}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                nav_cols = st.columns([1, 1, 1, 3], gap="small")
                with nav_cols[0]:
                    if st.button("Prev", key="review_prev", use_container_width=True):
                        st.session_state.search3_review_idx = (idx - 1) % n
                        st.rerun()
                with nav_cols[1]:
                    if st.button("Next", key="review_next", use_container_width=True):
                        st.session_state.search3_review_idx = (idx + 1) % n
                        st.rerun()
                with nav_cols[2]:
                    if st.button("Random review", key="review_rand", use_container_width=True):
                        st.session_state.search3_review_idx = random.randrange(0, n)
                        st.rerun()
                with nav_cols[3]:
                    review_status = f"Showing review {st.session_state.search3_review_idx + 1} of {n}"
                    st.markdown(
                        f"<div style='text-align:right;color:#7b818c;font-size:0.9rem;padding-top:6px;'>"
                        f"{html.escape(review_status)}</div>",
                        unsafe_allow_html=True,
                    )

            idx = st.session_state.search3_review_idx
            r = reviews[idx]
            dist = (per_review_discrete[idx] or {}) if has_discrete else {}
            va_point = (per_review_va[idx] or {}) if has_va else {}

            if idx < len(sentiments):
                s_label, s_conf = sentiments[idx]
                s_label = _normalize_sentiment_label(s_label)
                s_conf = _safe_float(s_conf)
                sentiment_color = "#f2c94c"
                if s_label == "Positive":
                    sentiment_color = "#1f77ff"
                elif s_label == "Negative":
                    sentiment_color = "#e53935"
                st.markdown("<div class='analysis-title'>Sentiment analysis</div>", unsafe_allow_html=True)
                st.write(f"{s_label}: {s_conf:.1%}")
                st.caption(f"Sentiment confidence: {max(0.0, min(1.0, s_conf)):.1%}")
                st.markdown(
                    f"<div class='sentiment-rule' style='background:{sentiment_color};'></div>",
                    unsafe_allow_html=True,
                )
            else:
                s_label = "Uncertain"

            if dist or va_point:
                st.markdown("<div class='analysis-title'>Emotion analysis</div>", unsafe_allow_html=True)
                rendered_prediction_chart = False
                model_emotion_conf = None
                distance_emotion_conf = None
                if dist:
                    top_emotions = _top_n_emotions(dist, n=10)
                    if top_emotions:
                        _render_emotion_intensity_circles(
                            top_emotions,
                            key_prefix="per_review",
                            sentiment_label=s_label,
                            title_text="Top emotion (model)",
                        )
                        model_emotion_conf = max(0.0, min(1.0, _safe_float(top_emotions[0][1], 0.0)))

                with st.expander("See how we calculate the emotion", expanded=True):
                    st.markdown(
                        "1. Valence → measures whether a review leans positive or negative "
                        "(pleasant ↔ unpleasant).\n"
                        "2. Arousal → measures emotional intensity, from activated to calm "
                        "(excited ↔ calm).\n\n"
                        "Together, Valence and Arousal place each review in emotional space, "
                        "so we capture both emotion type and strength."
                    )

                if va_point:
                    v = _safe_float(va_point.get("valence", 0.0))
                    a = _safe_float(va_point.get("arousal", 0.0))
                    q = str(va_point.get("quadrant", ""))
                    st.markdown("Explore how a review distance is calculated to 28 emotions.")
                    vcols = st.columns(3)
                    vcols[0].metric("Valence", f"{v:.3f}")
                    vcols[1].metric("Arousal", f"{a:.3f}")
                    vcols[2].metric("Quadrant", q or "N/A")

                    rows, missing = _build_emotion_distance_rows(v, a)
                    if rows:
                        distance_emotion_conf = max(
                            0.0,
                            min(1.0, 1.0 / (1.0 + max(0.0, _safe_float(rows[0].get("distance"), 0.0)))),
                        )
                        df_e = pd.DataFrame(rows)
                        if alt is not None:
                            axis_df = pd.DataFrame([{"x": 0.0, "y": 0.0}])
                            hline = alt.Chart(axis_df).mark_rule(color="#d0d0d0").encode(y="y:Q")
                            vline = alt.Chart(axis_df).mark_rule(color="#d0d0d0").encode(x="x:Q")

                            lines = (
                                alt.Chart(df_e)
                                .mark_rule(strokeWidth=1.3)
                                .encode(
                                    x=alt.X("review_valence:Q", scale=alt.Scale(domain=[-1, 1]), title="Valence"),
                                    y=alt.Y("review_arousal:Q", scale=alt.Scale(domain=[-1, 1]), title="Arousal"),
                                    x2="emotion_valence:Q",
                                    y2="emotion_arousal:Q",
                                    color=alt.condition(
                                        "datum.is_top10",
                                        alt.value("#cf5c36"),
                                        alt.value("#c9c9c9"),
                                    ),
                                    opacity=alt.condition(
                                        "datum.is_top10",
                                        alt.value(0.95),
                                        alt.value(0.25),
                                    ),
                                    tooltip=[
                                        "emotion:N",
                                        alt.Tooltip("distance:Q", format=".3f"),
                                        alt.Tooltip("similarity:Q", format=".3f"),
                                        "rank:Q",
                                    ],
                                )
                            )

                            emotion_points = (
                                alt.Chart(df_e)
                                .mark_circle(size=70)
                                .encode(
                                    x=alt.X("emotion_valence:Q", scale=alt.Scale(domain=[-1, 1])),
                                    y=alt.Y("emotion_arousal:Q", scale=alt.Scale(domain=[-1, 1])),
                                    color=alt.condition(
                                        "datum.is_top10",
                                        alt.value("#cf5c36"),
                                        alt.value("#8c8c8c"),
                                    ),
                                    opacity=alt.condition(
                                        "datum.is_top10",
                                        alt.value(1.0),
                                        alt.value(0.55),
                                    ),
                                    tooltip=[
                                        "emotion:N",
                                        alt.Tooltip("emotion_valence:Q", format=".3f"),
                                        alt.Tooltip("emotion_arousal:Q", format=".3f"),
                                        alt.Tooltip("distance:Q", format=".3f"),
                                        alt.Tooltip("similarity:Q", format=".3f"),
                                        "rank:Q",
                                    ],
                                )
                            )

                            top10_labels = (
                                alt.Chart(df_e[df_e["is_top10"]])
                                .mark_text(dx=6, dy=-6, fontSize=10, color="#b23a48")
                                .encode(
                                    x=alt.X("emotion_valence:Q", scale=alt.Scale(domain=[-1, 1])),
                                    y=alt.Y("emotion_arousal:Q", scale=alt.Scale(domain=[-1, 1])),
                                    text="emotion:N",
                                )
                            )

                            review_df = pd.DataFrame([{"label": "Review", "valence": v, "arousal": a}])
                            review_point = (
                                alt.Chart(review_df)
                                .mark_point(shape="diamond", size=180, color="#1f1f1f")
                                .encode(
                                    x=alt.X("valence:Q", scale=alt.Scale(domain=[-1, 1])),
                                    y=alt.Y("arousal:Q", scale=alt.Scale(domain=[-1, 1])),
                                    tooltip=[
                                        "label:N",
                                        alt.Tooltip("valence:Q", format=".3f"),
                                        alt.Tooltip("arousal:Q", format=".3f"),
                                    ],
                                )
                            )

                            chart = (hline + vline + lines + emotion_points + review_point + top10_labels).properties(
                                height=300
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            scatter_df = df_e[["emotion_valence", "emotion_arousal", "emotion"]]
                            st.scatter_chart(scatter_df, x="emotion_valence", y="emotion_arousal")

                        top10 = df_e[df_e["is_top10"]][["rank", "emotion", "distance", "similarity"]]
                        detail_cols = st.columns(2, gap="large")
                        with detail_cols[0]:
                            st.markdown("Emotions intensity")
                            _render_graph_subheader("By distance calculation", distance_info_text)
                            if distance_emotion_conf is not None:
                                st.caption(
                                    f"Emotion confidence (distance top score): {distance_emotion_conf:.1%}"
                                )
                            top10_distance = top10.copy()
                            if not top10_distance.empty:
                                top10_distance["distance_score"] = 1.0 / (1.0 + top10_distance["distance"].astype(float))
                                if alt is not None:
                                    d_chart = (
                                        alt.Chart(top10_distance)
                                        .mark_bar(color="#1976d2")
                                        .encode(
                                            y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                            x=alt.X("distance_score:Q", title="Score"),
                                            tooltip=[
                                                "emotion",
                                                alt.Tooltip("distance:Q", format=".3f"),
                                                alt.Tooltip("distance_score:Q", format=".3f"),
                                            ],
                                        )
                                        .properties(height=_emotion_bar_height(len(top10_distance)))
                                    )
                                    st.altair_chart(d_chart, use_container_width=True)
                                else:
                                    st.bar_chart(
                                        top10_distance.rename(columns={"distance_score": "score"}),
                                        x="emotion",
                                        y="score",
                                        use_container_width=True,
                                    )
                        with detail_cols[1]:
                            st.markdown("Emotions intensity")
                            _render_graph_subheader("By model prediction", prediction_info_text)
                            if model_emotion_conf is not None:
                                st.caption(f"Emotion confidence (model top score): {model_emotion_conf:.1%}")
                            if dist:
                                pred_items = _top_n_emotions(dist, n=10)
                                pred_df = pd.DataFrame(pred_items, columns=["emotion", "score"])
                                if alt is not None:
                                    p_chart = (
                                        alt.Chart(pred_df)
                                        .mark_bar(color="#1976d2")
                                        .encode(
                                            y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                            x=alt.X("score:Q", title="Score (0-1)"),
                                            tooltip=["emotion", alt.Tooltip("score:Q", format=".3f")],
                                        )
                                        .properties(height=_emotion_bar_height(len(pred_df)))
                                    )
                                    st.altair_chart(p_chart, use_container_width=True)
                                else:
                                    st.bar_chart(pred_df, x="emotion", y="score", use_container_width=True)
                                rendered_prediction_chart = True
                    if missing:
                        st.caption(f"Missing lexicon entries: {', '.join(missing)}")

                if dist and not rendered_prediction_chart:
                    st.markdown("Emotions intensity")
                    _render_graph_subheader("By model prediction", prediction_info_text)
                    if model_emotion_conf is not None:
                        st.caption(f"Emotion confidence (model top score): {model_emotion_conf:.1%}")
                    pred_items = _top_n_emotions(dist, n=10)
                    pred_df = pd.DataFrame(pred_items, columns=["emotion", "score"])
                    if alt is not None:
                        p_chart = (
                            alt.Chart(pred_df)
                            .mark_bar(color="#1976d2")
                            .encode(
                                y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                x=alt.X("score:Q", title="Score (0-1)"),
                                tooltip=["emotion", alt.Tooltip("score:Q", format=".3f")],
                            )
                            .properties(height=_emotion_bar_height(len(pred_df)))
                        )
                        st.altair_chart(p_chart, use_container_width=True)
                    else:
                        st.bar_chart(pred_df, x="emotion", y="score", use_container_width=True)

            # Topic details (after emotion, before keywords)
            topic_id = None
            try:
                if isinstance(topics_per_review, list) and idx < len(topics_per_review):
                    topic_id = int(topics_per_review[idx])
            except Exception:
                topic_id = None

            if show_topic_assignment and topic_id is not None:
                st.markdown("<div class='analysis-title'>Topic Modelling</div>", unsafe_allow_html=True)
                blocked_terms = {
                    "idk",
                    "tab",
                    "useful",
                    "features",
                    "pay",
                    "sucks",
                    "list",
                    "force",
                    "date",
                }

                topic_confidence = None
                try:
                    if topic_probs is not None and idx < len(topic_probs):
                        row = topic_probs[idx]
                        if hasattr(row, "__len__") and not isinstance(row, (str, bytes)):
                            vals = []
                            for x in row:
                                try:
                                    vals.append(float(x))
                                except Exception:
                                    continue
                            if vals:
                                topic_confidence = max(vals)
                        else:
                            topic_confidence = float(row)
                except Exception:
                    topic_confidence = None

                if topic_id == -1:
                    topic_tag = "Unassigned / Misc"
                    topic_name = "Outlier review"
                    topic_words = []
                else:
                    topic_tag = f"Topic {topic_id}"
                    topic_name = str(
                        topic_labels_by_topic.get(topic_id)
                        or topic_labels_by_topic.get(str(topic_id))
                        or ""
                    ).strip() or f"Topic {topic_id}"
                    raw_topic_words = (
                        topic_keywords_by_topic.get(topic_id, [])
                        if isinstance(topic_keywords_by_topic, dict)
                        else []
                    )
                    topic_words = [
                        w for w in raw_topic_words
                        if str(w).strip().lower() not in blocked_terms
                    ]

                if topic_confidence is not None:
                    conf_text = f"Confidence {max(0.0, min(1.0, float(topic_confidence))):.3f}"
                else:
                    conf_text = "Confidence N/A"

                topic_word_pills = "".join(
                    f"<span class='topic-keyword-pill'>{html.escape(str(w))}</span>"
                    for w in topic_words[:10]
                    if str(w).strip()
                )
                if not topic_word_pills:
                    topic_word_pills = "<span class='topic-keyword-pill'>No topic words</span>"

                st.markdown("**Topic Assignment**")
                st.markdown(
                    "<div class='topic-assign-card'>"
                    "<div class='topic-assign-row'>"
                    f"<span class='topic-chip topic-chip-primary'>{html.escape(topic_tag)}</span>"
                    f"<span class='topic-chip'>{html.escape(topic_name)}</span>"
                    f"<span class='topic-chip topic-chip-metric'>{html.escape(conf_text)}</span>"
                    "</div>"
                    f"<div class='topic-keyword-wrap'>{topic_word_pills}</div>"
                    "<div class='topic-assign-foot'>Assigned from overall topic model.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Keywords below topic details
            review_keyword_rows = []
            if isinstance(per_review_keywords, list) and idx < len(per_review_keywords):
                raw_items = per_review_keywords[idx] or []
                if isinstance(raw_items, list):
                    blocked_terms = {
                        "idk",
                        "tab",
                        "useful",
                        "features",
                        "pay",
                        "sucks",
                        "list",
                        "force",
                        "date",
                    }
                    for item in raw_items:
                        if isinstance(item, dict):
                            term = str(item.get("keyword") or "").strip()
                            score = float(item.get("score") or 0.0)
                        else:
                            term = str(item).strip()
                            score = 0.0
                        if not term:
                            continue
                        if term.lower() in blocked_terms:
                            continue
                        review_keyword_rows.append({"keyword": term, "score": score})

            if review_keyword_rows:
                if show_topic_title_before_keywords:
                    st.markdown("<div class='analysis-title'>Topic Modelling</div>", unsafe_allow_html=True)
                st.markdown("**Keywords**")
                keyword_weights = {}
                for rank_idx, row in enumerate(review_keyword_rows):
                    kw = str(row.get("keyword") or "").strip()
                    if not kw:
                        continue
                    score = _safe_float(row.get("score"), 0.0)
                    if score <= 0:
                        score = max(0.1, 1.0 - (rank_idx * 0.08))
                    keyword_weights[kw] = max(keyword_weights.get(kw, 0.0), score)
                if keyword_weights:
                    _render_weighted_wordcloud(
                        keyword_weights,
                        key_prefix=f"per_review_keywords_{idx}",
                        width=1000,
                        height=360,
                        max_words=50,
                    )
                st.dataframe(pd.DataFrame(review_keyword_rows), hide_index=True, use_container_width=True)

    if overall_placeholder is not None:
        overall_placeholder.empty()
    if per_review_placeholder is not None:
        per_review_placeholder.empty()

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
