"""UI helper functions for search page."""

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


def fetch_reviews_for_ui(platform: str, identifier: dict, limit: int = 5):
    """Unified fetch used by preview + full fetch."""
    rows = []
    if platform == "Trustpilot":
        rows = fetch_reviews_uncached_tp(identifier, limit=limit)
    else:
        rows = fetch_reviews_cached_non_tp(platform, identifier, limit=limit)
    # Final hard gate: never return non-English reviews to page UI.
    return filter_english_reviews(rows or [], limit=limit)


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
        topic_probs = topic_payload.get("probs")
    else:
        topics_per_review = []
        topic_keywords_by_topic = {}
        topic_probs = None

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
        st.markdown("**Sentiment (overall)**")
        pos = neg = unc = 0
        for s, _conf in sentiments:
            if s == "Positive":
                pos += 1
            elif s == "Negative":
                neg += 1
            else:
                unc += 1

        total = pos + neg + unc
        if total > 0:
            st.write(
                f"Positive: {pos} ({pos/total:.1%}) — "
                f"Negative: {neg} ({neg/total:.1%}) — "
                f"Uncertain: {unc} ({unc/total:.1%})"
            )
            df_sent = pd.DataFrame(
                [
                    {"label": "Positive", "count": pos, "percent": (pos / total) * 100.0},
                    {"label": "Negative", "count": neg, "percent": (neg / total) * 100.0},
                    {"label": "Uncertain", "count": unc, "percent": (unc / total) * 100.0},
                ]
            )
            if alt is not None:
                pie = (
                    alt.Chart(df_sent)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta(field="count", type="quantitative"),
                        color=alt.Color(field="label", type="nominal"),
                        tooltip=["label", "count", alt.Tooltip("percent:Q", format=".1f")],
                    )
                )
                st.altair_chart(pie, use_container_width=True)
            else:
                st.bar_chart(df_sent, x="label", y="count", use_container_width=True)
        else:
            st.write("Positive: 0 — Negative: 0 — Uncertain: 0")

        emo_pct = emo_discrete.get("percentages") or {}
        has_emo_overall = bool(emo_va or emo_pct)
        if has_emo_overall:
            st.markdown("**Emotion (overall)**")

            if emo_va:
                st.markdown("VA")
                mean_val = _safe_float(emo_va.get("mean_valence", 0.0))
                mean_aro = _safe_float(emo_va.get("mean_arousal", 0.0))
                std_val = _safe_float(emo_va.get("std_valence", 0.0))
                std_aro = _safe_float(emo_va.get("std_arousal", 0.0))
                mcols = st.columns(4)
                mcols[0].metric("Mean valence", f"{mean_val:.3f}")
                mcols[1].metric("Mean arousal", f"{mean_aro:.3f}")
                mcols[2].metric("Valence std", f"{std_val:.3f}")
                mcols[3].metric("Arousal std", f"{std_aro:.3f}")

                iqr_val = _safe_float(emo_va.get("iqr_valence", 0.0))
                iqr_aro = _safe_float(emo_va.get("iqr_arousal", 0.0))
                min_val = _safe_float(emo_va.get("min_valence", 0.0))
                max_val = _safe_float(emo_va.get("max_valence", 0.0))
                min_aro = _safe_float(emo_va.get("min_arousal", 0.0))
                max_aro = _safe_float(emo_va.get("max_arousal", 0.0))
                mean_dist = _safe_float(emo_va.get("mean_distance", 0.0))
                scols = st.columns(3)
                scols[0].metric("Valence IQR", f"{iqr_val:.3f}")
                scols[1].metric("Arousal IQR", f"{iqr_aro:.3f}")
                scols[2].metric("Mean intensity", f"{mean_dist:.3f}")
                st.caption(
                    f"Valence range: [{min_val:.3f}, {max_val:.3f}] · "
                    f"Arousal range: [{min_aro:.3f}, {max_aro:.3f}]"
                )

                polarity = emo_va.get("polarity_split") or {}
                activation = emo_va.get("activation_split") or {}
                p_df = pd.DataFrame(
                    [
                        {"bucket": "Positive", "percent": _safe_float(polarity.get("positive_pct", 0.0))},
                        {"bucket": "Neutral", "percent": _safe_float(polarity.get("neutral_pct", 0.0))},
                        {"bucket": "Negative", "percent": _safe_float(polarity.get("negative_pct", 0.0))},
                    ]
                )
                a_df = pd.DataFrame(
                    [
                        {"bucket": "High arousal", "percent": _safe_float(activation.get("high_pct", 0.0))},
                        {"bucket": "Mid arousal", "percent": _safe_float(activation.get("mid_pct", 0.0))},
                        {"bucket": "Calm", "percent": _safe_float(activation.get("calm_pct", 0.0))},
                    ]
                )
                split_cols = st.columns(2)
                with split_cols[0]:
                    st.markdown("Polarity split")
                    if alt is not None:
                        p_chart = (
                            alt.Chart(p_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("bucket:N", title=None),
                                y=alt.Y("percent:Q", title="Share (%)"),
                                tooltip=["bucket", alt.Tooltip("percent:Q", format=".1f")],
                            )
                        )
                        st.altair_chart(p_chart, use_container_width=True)
                    else:
                        st.bar_chart(p_df, x="bucket", y="percent", use_container_width=True)
                with split_cols[1]:
                    st.markdown("Activation split")
                    if alt is not None:
                        a_chart = (
                            alt.Chart(a_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("bucket:N", title=None),
                                y=alt.Y("percent:Q", title="Share (%)"),
                                tooltip=["bucket", alt.Tooltip("percent:Q", format=".1f")],
                            )
                        )
                        st.altair_chart(a_chart, use_container_width=True)
                    else:
                        st.bar_chart(a_df, x="bucket", y="percent", use_container_width=True)

                quad_pct = emo_va.get("quadrant_percentages") or {}
                if quad_pct:
                    q_df = pd.DataFrame(list(quad_pct.items()), columns=["quadrant", "percent"])
                    if alt is not None:
                        q_chart = (
                            alt.Chart(q_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("quadrant:N", title="VA quadrant"),
                                y=alt.Y("percent:Q", title="Share (%)"),
                                tooltip=["quadrant", alt.Tooltip("percent:Q", format=".1f")],
                            )
                        )
                        st.altair_chart(q_chart, use_container_width=True)
                    else:
                        st.bar_chart(q_df, x="quadrant", y="percent", use_container_width=True)

                # Overall VA scatter with click-to-review.
                if has_va:
                    st.markdown("VA space (reviews)")
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
                    # Draw guides first, reviews on top.
                    df_va = pd.DataFrame(guide_rows + va_rows)
                    st.caption("Hover for review text. Click a point, then open 'Insights per Review' to inspect it.")
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
                            .properties(height=360)
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

                # Short interpretation to prevent "mean near 0 = neutral" misread.
                interp = []
                if abs(mean_val) <= 0.10 and std_val >= 0.30:
                    interp.append("Valence mean is near 0, but spread is high, indicating mixed positive and negative reactions.")
                elif mean_val > 0.10:
                    interp.append("Overall valence trends positive.")
                elif mean_val < -0.10:
                    interp.append("Overall valence trends negative.")
                else:
                    interp.append("Overall valence appears close to neutral with modest spread.")

                if mean_dist >= 0.55:
                    interp.append("Emotional intensity is high on average.")
                elif mean_dist >= 0.30:
                    interp.append("Emotional intensity is moderate on average.")
                else:
                    interp.append("Emotional intensity is generally low.")

                high_pct = _safe_float(activation.get("high_pct", 0.0))
                if high_pct >= 40.0:
                    interp.append("A large share of reviews are high-arousal.")
                st.info(" ".join(interp))

            if emo_pct:
                st.markdown("Discrete emotion")
                top_items = list(emo_pct.items())[:10]
                if top_items:
                    df = pd.DataFrame(top_items, columns=["emotion", "score"])
                    if alt is not None:
                        chart = (
                            alt.Chart(df)
                            .mark_bar()
                            .encode(
                                y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                x=alt.X("score:Q", title="Avg intensity"),
                            )
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

            nav_cols = st.columns([1, 1, 1, 2])
            with nav_cols[0]:
                if st.button("Prev review", key="review_prev"):
                    st.session_state.search3_review_idx = (st.session_state.search3_review_idx - 1) % n
            with nav_cols[1]:
                if st.button("Next review", key="review_next"):
                    st.session_state.search3_review_idx = (st.session_state.search3_review_idx + 1) % n
            with nav_cols[2]:
                if st.button("Random review", key="review_rand"):
                    st.session_state.search3_review_idx = random.randrange(0, n)
            with nav_cols[3]:
                st.caption(f"Review {st.session_state.search3_review_idx + 1} of {n}")

            idx = st.session_state.search3_review_idx
            r = reviews[idx]
            dist = (per_review_discrete[idx] or {}) if has_discrete else {}
            va_point = (per_review_va[idx] or {}) if has_va else {}

            title = r.get("title") or "(no title)"
            content = (r.get("content") or "").strip()
            rating = r.get("rating")
            date = r.get("date") or ""
            platform = r.get("platform") or ""

            st.markdown(f"**{title}**")
            meta = " · ".join(
                [p for p in [platform, date, (f"{rating} ★" if rating is not None else None)] if p]
            )
            if meta:
                st.caption(meta)
            if content:
                st.write(content)

            if idx < len(sentiments):
                s_label, s_conf = sentiments[idx]
                s_conf = _safe_float(s_conf)
                st.markdown("**Sentiment (this review)**")
                st.write(f"{s_label}: {s_conf:.1%}")
                st.progress(max(0.0, min(1.0, s_conf)))

            if dist or va_point:
                st.markdown("**Emotion (this review)**")

                if va_point:
                    st.markdown("VA")
                    v = _safe_float(va_point.get("valence", 0.0))
                    a = _safe_float(va_point.get("arousal", 0.0))
                    q = str(va_point.get("quadrant", ""))
                    vcols = st.columns(3)
                    vcols[0].metric("Valence", f"{v:.3f}")
                    vcols[1].metric("Arousal", f"{a:.3f}")
                    vcols[2].metric("Quadrant", q or "N/A")

                    rows, missing = _build_emotion_distance_rows(v, a)
                    if rows:
                        st.markdown("VA space vs. emotion anchors")
                        st.caption("All 28 emotions are shown; top 10 closest are highlighted.")
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
                                height=420
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            scatter_df = df_e[["emotion_valence", "emotion_arousal", "emotion"]]
                            st.scatter_chart(scatter_df, x="emotion_valence", y="emotion_arousal")

                        top10 = df_e[df_e["is_top10"]][["rank", "emotion", "distance", "similarity"]]
                        st.dataframe(top10, hide_index=True, use_container_width=True)
                    if missing:
                        st.caption(f"Missing lexicon entries: {', '.join(missing)}")

                if dist:
                    top_items = sorted(((k, _safe_float(v)) for k, v in dist.items()), key=lambda kv: -kv[1])[:10]
                    st.markdown("Discrete emotion")
                    df = pd.DataFrame(top_items, columns=["emotion", "score"])
                    if alt is not None:
                        chart = (
                            alt.Chart(df)
                            .mark_bar()
                            .encode(
                                y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                                x=alt.X("score:Q", title="Score (0-1)"),
                            )
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.bar_chart(df, x="emotion", y="score", use_container_width=True)

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

                # LLM rationale (best-effort; cached by review index)
                cluster_label = str(analysis.get("cluster_label") or "").strip()
                llm_cache = analysis.get("topic_llm_by_review")
                if not isinstance(llm_cache, dict):
                    llm_cache = {}
                    analysis["topic_llm_by_review"] = llm_cache

                if idx not in llm_cache and cluster_label:
                    with st.spinner("Generating topic rationale..."):
                        try:
                            from inference.llm_topic_label import llm_label_topic

                            llm_cache[idx] = llm_label_topic(
                                cluster_label=cluster_label,
                                topic_id=int(topic_id),
                                keywords=topic_words[:10],
                                review_text=(content or title or "")[:600],
                            )
                        except Exception as exc:
                            llm_cache[idx] = {
                                "label": "Topic rationale unavailable",
                                "explanation": f"LLM error: {exc}",
                            }

                llm_item = llm_cache.get(idx) if isinstance(llm_cache, dict) else None
                if isinstance(llm_item, dict):
                    label = str(llm_item.get("label") or "").strip()
                    explanation = str(llm_item.get("explanation") or "").strip()
                    if label:
                        st.write(f"LLM topic label: {label}")
                    if explanation:
                        st.write(f"Why: {explanation}")

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
