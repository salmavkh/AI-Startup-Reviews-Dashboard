"""UI helper functions for search page."""

import pandas as pd
import random
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover - optional dependency
    alt = None

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
    if platform == "Trustpilot":
        return fetch_reviews_uncached_tp(identifier, limit=limit)
    return fetch_reviews_cached_non_tp(platform, identifier, limit=limit)


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

    # --------------------
    # Overall tab
    # --------------------
    with tabs[0]:
        # Sentiment (overall)
        st.markdown("**Sentiment (overall)**")
        pos = neg = unc = 0
        for s, conf in (analysis.get("sentiment") or []):
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

        # Emotions (overall)
        emo = analysis.get("emotion") or {}
        emo_pct = emo.get("percentages") or {}
        if emo_pct:
            st.markdown("**Emotion (overall)**")
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
            st.caption("No overall emotion distribution to show yet.")

        # Topic (overall)
        t = analysis.get("topic") or {}
        counts = t.get("counts") or {}
        keywords = t.get("keywords_by_topic") or {}
        if counts:
            st.markdown("**Topic (overall)**")
            for topic_id, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
                if topic_id == -1:
                    lbl = "(outliers)"
                else:
                    lbl = f"Topic {topic_id} — {', '.join(keywords.get(topic_id, [])[:5])}"
                st.write(f"- {lbl}: {cnt}")

    # --------------------
    # Per-review tab
    # --------------------
    with tabs[1]:
        reviews = analysis.get("reviews") or []
        per_review = analysis.get("emotion_by_review") or []
        sentiments = analysis.get("sentiment") or []

        if not reviews or not per_review or len(reviews) != len(per_review):
            st.caption("No per-review emotion details to show yet.")
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
            dist = per_review[idx] or {}

            title = r.get("title") or "(no title)"
            content = (r.get("content") or "").strip()
            rating = r.get("rating")
            date = r.get("date") or ""
            platform = r.get("platform") or ""

            st.markdown(f"**{title}**")
            meta = " · ".join([p for p in [platform, date, (f"{rating} ★" if rating is not None else None)] if p])
            if meta:
                st.caption(meta)
            if content:
                st.write(content)

            # Per-review sentiment viz
            if idx < len(sentiments):
                s_label, s_conf = sentiments[idx]
                s_conf = float(s_conf)
                st.markdown("**Sentiment (this review)**")
                st.write(f"{s_label}: {s_conf:.1%}")
                st.progress(max(0.0, min(1.0, s_conf)))

            # Per-review emotion viz
            if dist:
                top_items = sorted(((k, float(v)) for k, v in dist.items()), key=lambda kv: -kv[1])[:10]

                st.markdown("**Emotion (this review)**")
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

    st.markdown("\nYou can fetch the full review set and re-run analysis, or proceed to the next step in the pipeline.")


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
