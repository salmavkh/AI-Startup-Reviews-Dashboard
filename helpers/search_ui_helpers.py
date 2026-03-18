"""UI helper functions for search page."""

import html
import math
import os
import pandas as pd
import random
import streamlit as st
from fetchers.language_filter import filter_english_reviews

try:
    import altair as alt
except Exception:  # pragma: no cover - optional dependency
    alt = None

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


def render_analysis_results(analysis: dict):
    """Render topic + sentiment analysis results."""
    st.markdown("---\n### Preview analysis")
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
          .analysis-comment-label {
            font-size: 13px;
            font-weight: 600;
            margin-top: 2px;
            margin-bottom: 4px;
          }
          .analysis-comment-box {
            background: #d9d9d9;
            border-radius: 6px;
            min-height: 44px;
            padding: 10px 12px;
            font-size: 13px;
            color: #1d1d1d;
            margin-bottom: 10px;
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
            border-radius: 8px;
            background: #ffffff;
            min-height: 88px;
            padding: 10px 12px;
          }
          .review-count-label {
            color: #8d8d8d;
            font-size: 11px;
            margin-bottom: 6px;
          }
          .sentiment-rule {
            height: 4px;
            border-radius: 999px;
            margin-top: 6px;
            margin-bottom: 16px;
            width: 100%;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Overall Results", "Insights per Review"])

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
    ):
        if not top_items:
            return
        st.markdown("Top emotion")
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
    with tabs[0]:
        st.markdown("<div class='analysis-title'>Sentiment analysis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='analysis-subtitle'>See whether people feel positively or negatively about your AI startup at a glance.</div>",
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
                    )

            with st.expander("See how we calculate the emotion", expanded=True):
                st.markdown(
                    "- Valence -> how positive or negative the review is\n"
                    "- Arousal -> how intense or emotionally charged the review is\n\n"
                    "These help us understand not just what users feel, but also how strongly they feel it."
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
                                st.success(f"Selected Review {picked_idx + 1}. Open 'Insights per Review' tab to view details.")
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

            if emo_pct:
                st.markdown("Emotions intensity")
                top_items = _top_n_emotions(emo_pct, n=10)
                if top_items:
                    df = pd.DataFrame(top_items, columns=["emotion", "score"])
                    if alt is not None:
                        chart = (
                            alt.Chart(df)
                            .mark_bar(color="#1976d2")
                            .encode(
                                y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                x=alt.X("score:Q", title="Avg intensity"),
                                tooltip=["emotion", alt.Tooltip("score:Q", format=".3f")],
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.bar_chart(df, x="emotion", y="score", use_container_width=True)
        else:
            st.caption("No overall emotion analysis to show yet.")

        # Topic (overall)
        counts = topic_payload.get("counts") or {}
        keywords = topic_keywords_by_topic or {}
        topic_summary = (analysis.get("topic_summary") or "").strip()
        if counts or topic_summary:
            st.markdown("**Topic (overall)**")
            if topic_summary:
                st.markdown("**LLM Summary**")
                st.write(topic_summary)

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

            st.markdown("**Raw topic outputs (overall)**")
            c_v_overall = topic_coherence.get("c_v_overall")
            c_v_available = bool(topic_coherence.get("available"))
            c_v_error = str(topic_coherence.get("error") or "").strip()
            proxy_overall = _safe_float(topic_coherence.get("proxy_overall", 0.0))
            mc = st.columns(3)
            mc[0].metric("Topics found", f"{len([t for t in counts.keys() if int(t) != -1])}")
            if c_v_overall is not None:
                mc[1].metric("Coherence c_v", f"{_safe_float(c_v_overall):.3f}")
            else:
                mc[1].metric("Coherence c_v", "N/A")
            mc[2].metric("Coherence proxy", f"{proxy_overall:.3f}")

            if (not c_v_available) and c_v_error:
                st.caption(f"c_v coherence unavailable in runtime: {c_v_error}")

            if raw_topic_rows:
                df_raw_topics = pd.DataFrame(raw_topic_rows)
                # Primary overall topic table (clean UI): topic_id, size, top_words, coherence_c_v
                primary_rows = []
                for row in raw_topic_rows:
                    topic_id = row.get("topic_id")
                    size = int(row.get("count") or 0)
                    top_words = str(row.get("words") or "").strip()
                    topic_name = ""
                    if topic_id is not None:
                        topic_name = str(
                            topic_labels_by_topic.get(int(topic_id))
                            or topic_labels_by_topic.get(str(topic_id))
                            or ""
                        ).strip()
                    if not topic_name and int(topic_id or -1) == -1:
                        topic_name = "Unassigned / Misc"
                    coherence_c_v = row.get("coherence_c_v")
                    if coherence_c_v is not None:
                        try:
                            coherence_c_v = round(float(coherence_c_v), 3)
                        except Exception:
                            pass
                    primary_rows.append(
                        {
                            "topic_id": topic_id,
                            "size": size,
                            "topic_name": topic_name,
                            "top_words": top_words,
                            "coherence_c_v": coherence_c_v,
                        }
                    )
                st.dataframe(pd.DataFrame(primary_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No topic rows to display.")

            with st.expander("Topic diagnostics (confidence/proxy/details)", expanded=False):
                if raw_topic_rows:
                    df_debug_topics = pd.DataFrame(raw_topic_rows)
                    if "share" in df_debug_topics.columns:
                        try:
                            df_debug_topics["share"] = df_debug_topics["share"].astype(float).map(lambda x: f"{x:.1%}")
                        except Exception:
                            pass
                    st.dataframe(df_debug_topics, hide_index=True, use_container_width=True)
                else:
                    st.caption("No topic diagnostics to display.")

            with st.expander("Raw per-review topic assignments (overall set)", expanded=False):
                if raw_review_rows:
                    df_raw_reviews = pd.DataFrame(raw_review_rows)
                    if "topic_id" in df_raw_reviews.columns:
                        def _name_for_topic(value):
                            try:
                                tid = int(value)
                            except Exception:
                                return ""
                            if tid == -1:
                                return "Unassigned / Misc"
                            return str(
                                topic_labels_by_topic.get(tid)
                                or topic_labels_by_topic.get(str(tid))
                                or ""
                            ).strip()
                        df_raw_reviews["topic_name"] = df_raw_reviews["topic_id"].map(_name_for_topic)
                    if "confidence" in df_raw_reviews.columns:
                        try:
                            df_raw_reviews["confidence"] = df_raw_reviews["confidence"].map(
                                lambda x: (f"{float(x):.3f}" if x is not None else "")
                            )
                        except Exception:
                            pass
                    st.dataframe(df_raw_reviews, hide_index=True, use_container_width=True)
                else:
                    st.caption("No per-review topic assignment rows to display.")

        if overall_keywords:
            st.markdown("**Keywords (overall)**")
            keyword_rows = []
            for item in overall_keywords:
                if isinstance(item, dict):
                    term = str(item.get("keyword") or "").strip()
                    review_count = int(item.get("review_count") or 0)
                    mentions = int(item.get("mentions") or 0)
                    avg_score = float(item.get("avg_score") or 0.0)
                else:
                    term = str(item).strip()
                    review_count = 0
                    mentions = 0
                    avg_score = 0.0

                if not term:
                    continue
                keyword_rows.append(
                    {
                        "keyword": term,
                        "reviews": review_count,
                        "mentions": mentions,
                        "avg_score": avg_score,
                    }
                )

            if keyword_rows:
                st.write("Top terms:", ", ".join(r["keyword"] for r in keyword_rows[:10]))
                st.dataframe(pd.DataFrame(keyword_rows), hide_index=True, use_container_width=True)

    # --------------------
    # Per-review tab
    # --------------------
    with tabs[1]:
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
            review_preview = _short_text(content or title, limit=220)

            nav_cols = st.columns([1, 10, 1], gap="small")
            with nav_cols[0]:
                if st.button("<", key="review_prev", use_container_width=True):
                    st.session_state.search3_review_idx = (idx - 1) % n
                    st.rerun()
            with nav_cols[1]:
                st.markdown(
                    "<div class='review-box'>"
                    f"<div class='review-count-label'>Review {idx + 1} of {n}</div>"
                    f"<div style='font-size:13px;font-weight:600;margin-bottom:5px;'>{html.escape(title)}</div>"
                    f"<div style='font-size:13px;'>{html.escape(review_preview)}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with nav_cols[2]:
                if st.button(">", key="review_next", use_container_width=True):
                    st.session_state.search3_review_idx = (idx + 1) % n
                    st.rerun()

            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("Random review", key="review_rand"):
                    st.session_state.search3_review_idx = random.randrange(0, n)
                    st.rerun()
            with c2:
                st.caption(f"Showing review {st.session_state.search3_review_idx + 1} of {n}")

            idx = st.session_state.search3_review_idx
            r = reviews[idx]
            dist = (per_review_discrete[idx] or {}) if has_discrete else {}
            va_point = (per_review_va[idx] or {}) if has_va else {}

            content = (r.get("content") or "").strip()
            rating = r.get("rating")
            date = r.get("date") or ""
            platform = r.get("platform") or ""

            meta = " · ".join(
                [p for p in [platform, date, (f"{rating} ★" if rating is not None else None)] if p]
            )
            if meta:
                st.caption(meta)

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
                st.markdown(
                    f"<div class='sentiment-rule' style='background:{sentiment_color};'></div>",
                    unsafe_allow_html=True,
                )
            else:
                s_label = "Uncertain"

            if dist or va_point:
                st.markdown("<div class='analysis-title'>Emotion analysis</div>", unsafe_allow_html=True)
                rendered_prediction_chart = False
                if dist:
                    top_emotions = _top_n_emotions(dist, n=10)
                    if top_emotions:
                        _render_emotion_intensity_circles(
                            top_emotions,
                            key_prefix="per_review",
                            sentiment_label=s_label,
                        )

                with st.expander("See how we calculate the emotion", expanded=True):
                    st.markdown(
                        "- Valence -> how positive or negative the review is\n"
                        "- Arousal -> how intense or emotionally charged the review is\n\n"
                        "These help us understand not just what users feel, but also how strongly they feel it."
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
                            st.caption("By distance calculation")
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
                                        .properties(height=220)
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
                            st.caption("By model prediction")
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
                                        .properties(height=220)
                                    )
                                    st.altair_chart(p_chart, use_container_width=True)
                                else:
                                    st.bar_chart(pred_df, x="emotion", y="score", use_container_width=True)
                                rendered_prediction_chart = True
                    if missing:
                        st.caption(f"Missing lexicon entries: {', '.join(missing)}")

                if dist and not rendered_prediction_chart:
                    st.markdown("Emotions intensity")
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
                            .properties(height=220)
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

            if topic_id is not None:
                st.markdown("**Topic (this review)**")
                if topic_id == -1:
                    st.write("Topic: Unassigned / Misc (outlier)")
                    topic_words = []
                else:
                    st.write(f"Topic ID: {topic_id}")
                    topic_name = str(
                        topic_labels_by_topic.get(topic_id)
                        or topic_labels_by_topic.get(str(topic_id))
                        or ""
                    ).strip()
                    if topic_name:
                        st.write(f"Topic name: {topic_name}")
                    topic_words = topic_keywords_by_topic.get(topic_id, []) if isinstance(topic_keywords_by_topic, dict) else []
                    if topic_words:
                        st.write("Topic words: " + ", ".join(topic_words[:10]))

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

                if topic_confidence is not None:
                    st.write(f"Topic confidence: {max(0.0, min(1.0, float(topic_confidence))):.3f}")

            # Keywords below topic details
            review_keyword_rows = []
            if isinstance(per_review_keywords, list) and idx < len(per_review_keywords):
                raw_items = per_review_keywords[idx] or []
                if isinstance(raw_items, list):
                    for item in raw_items:
                        if isinstance(item, dict):
                            term = str(item.get("keyword") or "").strip()
                            score = float(item.get("score") or 0.0)
                        else:
                            term = str(item).strip()
                            score = 0.0
                        if not term:
                            continue
                        review_keyword_rows.append({"keyword": term, "score": score})

            if review_keyword_rows:
                st.markdown("**Keywords (this review)**")
                st.write(", ".join(row["keyword"] for row in review_keyword_rows[:8]))
                st.dataframe(pd.DataFrame(review_keyword_rows), hide_index=True, use_container_width=True)

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
