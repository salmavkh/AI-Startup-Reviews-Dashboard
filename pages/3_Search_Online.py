import streamlit as st
import hashlib
import json

from fetchers.google_play import search_google_play
from fetchers.ios import search_ios
from fetchers.g2 import search_g2
from fetchers.trustpilot import search_trustpilot
from inference.topic import discover_topics_batch
from inference.sentiment import predict_single as predict_sentiment_single
from inference.emotion import emotion_percentages, predict_proba_single
from inference.va import predict_va_single, summarize_va
from inference.llm_topic_summary import llm_topic_summary
from inference.llm_topic_label import llm_label_topics_from_keywords
from inference.keywords import extract_keywords_batch
from fetchers.language_filter import filter_english_reviews

# Import refactored helpers
from helpers.search_ui_helpers import (
    fetch_reviews_for_ui, process_search_results, logo_html, render_result_card,
    render_review_preview, render_analysis_results, extract_identifier_info, render_confirmed_company
)
from helpers.search_validation import validate_search_inputs, validate_submit_inputs, parse_pasted_link

st.set_page_config(page_title="Search Online Reviews", page_icon="🔎", layout="wide")

TOPIC_EXAMPLES_PER_THEME = 2
KEYWORDS_PER_REVIEW = 5
KEYWORDS_OVERALL = 20

# --- CSS (same vibe as your other pages) ---
st.markdown(
    """
    <style>
      .field-title {
        font-size: 18px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      /* tighten spacing under our custom titles */
      div[data-testid="stTextInput"], div[data-testid="stRadio"] {
        margin-top: 0px;
      }

      /* results heading (smaller than primary headings) */
      .results-heading {
        font-size: 15px;
        font-weight: 600;
        margin: 0 0 10px 0;
        color: #222;
      }

      /* compact card style for result tiles */
      .result-card {
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 10px;
        padding: 8px 10px;
        background: white;
        margin-bottom: 8px;
      }

      /* selected state: stronger border + subtle lift */
      .result-card.selected {
        border: 2px solid #000000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        padding: 7px 9px; /* account for thicker border */
      }

      /* small circular indicator shown for selected cards */
      .result-card.selected::before {
        content: '';
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #000;
        display: inline-block;
        margin-right: 10px;
        vertical-align: middle;
      }

      .result-card img, .result-card svg {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        object-fit: cover;
        display: block;
      }

      .result-card .name {
        font-size: 14px;
        font-weight: 600;
        margin: 0;
      }

      .result-card .subtitle {
        font-size: 12px;
        color: #666;
        margin-top: 2px;
      }

      /* smaller select buttons inside cards */
      .result-card button {
        padding: 6px 10px !important;
        font-size: 13px !important;
        border-radius: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Back button
if st.button("← Back"):
    st.switch_page("app.py")

st.header("Search Online Reviews")

# ---------------------------
# Session state initialization
# ---------------------------
defaults = {
    "search3_errors": [],
    "search3_search_clicked": False,
    "search3_submit_clicked": False,
    "search3_results": [],
    "search3_selected_result": None,
    "search3_pasted_link": "",
    "search3_num_reviews": 20,
    "search3_fetched_reviews": [],
    "search3_confirmed_company": None,
    "search3_fetched_for": {},
    "search3_preview_analysis": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _collect_topic_examples_for_payload(
    topic_res: dict,
    review_texts: list[str],
    per_topic_limit: int = TOPIC_EXAMPLES_PER_THEME,
) -> dict:
    examples_by_topic = {}
    topics = topic_res.get("topics") or []
    keywords_by_topic = topic_res.get("keywords_by_topic") or {}
    if not isinstance(topics, list) or len(topics) != len(review_texts):
        return examples_by_topic

    per_topic_limit = max(1, min(5, int(per_topic_limit or 2)))
    scored = {}

    for idx, topic_id in enumerate(topics):
        try:
            tid = int(topic_id)
        except Exception:
            continue
        if tid == -1:
            continue

        text = str(review_texts[idx] or "").strip()
        if not text:
            continue

        text_lower = text.lower()
        topic_keywords = [str(k).strip().lower() for k in keywords_by_topic.get(tid, [])[:10] if str(k).strip()]
        overlap = sum(1 for kw in topic_keywords if kw and kw in text_lower)
        if overlap <= 0:
            continue

        snippet = text[:180] + ("..." if len(text) > 180 else "")
        scored.setdefault(tid, []).append((overlap, snippet))

    for tid, rows in scored.items():
        rows.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        picked = []
        seen = set()
        for _overlap, snippet in rows:
            if snippet in seen:
                continue
            seen.add(snippet)
            picked.append(snippet)
            if len(picked) >= per_topic_limit:
                break
        if picked:
            examples_by_topic[tid] = picked

    return examples_by_topic


def _build_top_topics_payload(
    topic_res: dict,
    total_reviews: int,
    review_texts: list[str] | None = None,
    examples_per_topic: int = TOPIC_EXAMPLES_PER_THEME,
) -> list:
    if not topic_res or total_reviews <= 0:
        return []
    counts = topic_res.get("counts") or {}
    keywords_by_topic = topic_res.get("keywords_by_topic") or {}
    examples_by_topic = {}
    if review_texts:
        examples_by_topic = _collect_topic_examples_for_payload(
            topic_res=topic_res,
            review_texts=review_texts,
            per_topic_limit=examples_per_topic,
        )
    items = [(tid, c) for tid, c in counts.items() if tid != -1]
    items.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            "topic_id": tid,
            "count": c,
            "pct": c / total_reviews,
            "keywords": keywords_by_topic.get(tid, []),
            "examples": examples_by_topic.get(tid, []),
        }
        for tid, c in items[:5]
    ]


def _topic_summary_or_empty(
    topic_res: dict,
    total_reviews: int,
    cluster_label: str,
    review_texts: list[str] | None = None,
    examples_per_topic: int = TOPIC_EXAMPLES_PER_THEME,
) -> str:
    top_topics_payload = _build_top_topics_payload(
        topic_res=topic_res,
        total_reviews=total_reviews,
        review_texts=review_texts,
        examples_per_topic=examples_per_topic,
    )
    if not top_topics_payload:
        return ""
    return llm_topic_summary(cluster_label, top_topics_payload)


def _reviews_signature(rows: list[dict] | None) -> str:
    payload = []
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        payload.append(
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "content": r.get("content"),
                "date": r.get("date"),
                "platform": r.get("platform"),
                "reviewer": r.get("reviewer"),
            }
        )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

# ---------------------------
# LEFT: inputs + Search button
# ---------------------------
with left:
    st.markdown('<div class="field-title">Search your company name</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Search your company name",
        placeholder="Search",
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown('<div class="field-title">What platform?</div>', unsafe_allow_html=True)
    platform = st.radio(
        "What platform?",
        options=["Google Play Store", "iOS App Store", "G2", "Trustpilot"],
        index=None,  # IMPORTANT: no default selection
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown('<div class="field-title">How many reviews?</div>', unsafe_allow_html=True)
    num_reviews = st.number_input(
        "How many reviews?",
        min_value=1,
        max_value=100,
        step=1,
        value=st.session_state.get("search3_num_reviews", 20),
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown('<div class="field-title">What cluster is your company?</div>', unsafe_allow_html=True)
    cluster = st.radio(
        "What cluster is your company?",
        options=[
            "Cluster 1 (AI-Charged Product/Service Providers)",
            "Cluster 2 (AI Development Facilitators)",
            "Cluster 3 (Data Analytics Providers)",
            "Cluster 4 (Deep Tech Researchers)",
        ],
        index=None,  # IMPORTANT: no default selection
        label_visibility="collapsed",
    )

    st.write("")
    if st.button("Search", type="primary"):
        st.session_state.search3_errors = []
        st.session_state.search3_num_reviews = num_reviews

        # Validate inputs
        errors = validate_search_inputs(query, platform, num_reviews, cluster)
        st.session_state.search3_errors = errors

        if st.session_state.search3_errors:
            st.session_state.search3_search_clicked = False
            st.session_state.search3_results = []
            st.session_state.search3_selected_result = None
            st.session_state.search3_pasted_link = ""
        else:
            st.session_state.search3_search_clicked = True
            st.session_state["cluster"] = cluster

            st.session_state.search3_selected_result = None
            st.session_state.search3_pasted_link = ""

            candidates = []
            try:
                if platform == "Google Play Store":
                    candidates = search_google_play(query, limit=5)
                elif platform == "iOS App Store":
                    candidates = search_ios(query, limit=5)
                elif platform == "G2":
                    candidates = search_g2(query, limit=5)
                elif platform == "Trustpilot":
                    candidates = search_trustpilot(query, limit=5)
            except Exception as e:
                st.session_state.search3_errors.append(f"Search failed: {e}")
                candidates = []

            ui_candidates = process_search_results(candidates, platform)
            st.session_state.search3_results = ui_candidates

            if platform == "G2" and not any(r.get("g2_slug") and not r.get("unverified") for r in ui_candidates):
                st.session_state.search3_errors.append(
                    "G2 search is often JS-protected — results may be blocked.\n"
                    "If you have the product page, paste the G2 product URL on the right. "
                    "You can also set APIFY_API_TOKEN in your environment — the app will use a public actor by default if you don't set APIFY_ACTOR_ID."
                )
                st.info("G2 often blocks direct scraping. Paste the G2 product URL on the right (recommended). If you set APIFY_API_TOKEN the app will try a public Apify actor by default; set APIFY_ACTOR_ID to use your own actor.")
                st.write("Example: https://www.g2.com/products/<product-slug>/reviews")

# Validation warnings
for msg in st.session_state.search3_errors:
    st.warning(msg)

# ---------------------------
# RIGHT: results + Submit flow
# ---------------------------
with right:
    if st.session_state.search3_search_clicked:
        st.markdown('<div class="results-heading">Select your company from these results:</div>', unsafe_allow_html=True)

        results = st.session_state.search3_results

        cols_per_row = 2
        rows = [results[i:i+cols_per_row] for i in range(0, len(results), cols_per_row)]
        for row in rows:
            cols = st.columns(len(row), gap="large")
            for ix, r in enumerate(row):
                render_result_card(r, ix, cols)

        none_choice = st.radio("", options=["None of those"], index=None, label_visibility="collapsed")
        none_selected = (none_choice == "None of those") or (not results)

        if none_selected:
            st.markdown('<div class="field-title">Paste the app link here:</div>', unsafe_allow_html=True)
            st.session_state.search3_pasted_link = st.text_input(
                "Paste the app link here",
                placeholder="e.g. https://www.trustpilot.com/review/spotify.com",
                label_visibility="collapsed",
                value=st.session_state.search3_pasted_link,
            )
            st.session_state.search3_selected_result = None

        st.write("")
        if st.button("Submit", type="primary"):
            submit_errors = validate_submit_inputs(
                st.session_state.search3_selected_result is not None,
                st.session_state.search3_pasted_link
            )

            if submit_errors:
                for e in submit_errors:
                    st.warning(e)
                st.session_state.search3_submit_clicked = False
            else:
                st.session_state.search3_confirmed_company = None

                def _find_selected_result():
                    sel = st.session_state.search3_selected_result
                    if not sel:
                        return None
                    for rr in st.session_state.search3_results:
                        if rr.get("id") == sel:
                            return rr
                    return None

                confirmed = None
                picked_result = st.session_state.search3_selected_result is not None
                
                if not picked_result:
                    pasted = st.session_state.search3_pasted_link
                    confirmed, errors = parse_pasted_link(platform, pasted)

                    if errors:
                        for e in errors:
                            st.warning(e)
                        st.session_state.search3_submit_clicked = False
                        confirmed = None
                else:
                    sel = _find_selected_result()
                    if not sel:
                        st.warning("Selected result not found — please choose a result or paste a link.")
                        st.session_state.search3_submit_clicked = False
                        confirmed = None
                    else:
                        confirmed = sel.copy()
                        confirmed["source"] = "selection"

                if confirmed:
                    st.session_state.search3_confirmed_company = confirmed
                    st.session_state.search3_submit_clicked = True
                else:
                    st.session_state.search3_submit_clicked = False


# ---------------------------
# Confirmed company + preview fetch
# ---------------------------
if st.session_state.search3_submit_clicked:
    st.markdown("## Confirmed company")

    confirmed = st.session_state.get("search3_confirmed_company")
    if not confirmed:
        st.info("No company confirmed — please select a result or paste a valid link and Submit.")
    else:
        render_confirmed_company(confirmed)

        preview_limit = min(5, max(1, int(st.session_state.get("search3_num_reviews", 5))))

        prev = st.session_state.get("search3_fetched_reviews") or []
        prev_key = st.session_state.get("search3_fetched_for") or {}

        identifier = extract_identifier_info(confirmed)
        should_fetch_preview = True
        if prev and prev_key.get("platform") == confirmed.get("platform") and prev_key.get("identifier") == (identifier[1] if identifier else None):
            should_fetch_preview = False

        if should_fetch_preview:
            with st.spinner("Fetching a short review preview (up to 5)..."):
                preview = fetch_reviews_for_ui(confirmed.get("platform"), confirmed, limit=preview_limit)

            st.session_state.search3_fetched_reviews = preview or []
            st.session_state.search3_fetched_for = {"platform": confirmed.get("platform"), "identifier": (identifier[1] if identifier else None)}
            st.session_state.search3_preview_analysis = None

        st.info("A short preview of reviews is shown below. You can run analysis on the preview, or fetch the full set of reviews you requested.")

        fetched = st.session_state.get("search3_fetched_reviews") or []
        # Defensive cleanup for previously cached/session results.
        fetched = filter_english_reviews(fetched, limit=None)
        st.session_state.search3_fetched_reviews = fetched
        platform = confirmed.get("platform")
        
        if not fetched:
            if platform == "Google Play Store":
                try:
                    import google_play_scraper  # type: ignore
                    gp_ok = True
                except Exception:
                    gp_ok = False

                if not gp_ok:
                    st.warning(
                        "No reviews fetched — the `google_play_scraper` Python package is not available in this environment. "
                        "Install it (`pip install google-play-scraper`) for reliable Play Store access, or paste a direct Play Store package/link."
                    )
                else:
                    st.warning(
                        "No reviews fetched — the Play Store likely returned no public reviews for that package from the requested storefront, "
                        "or Google is blocking access from this host (rate-limiting / geo restrictions). Try a different storefront/package or paste a direct Play Store link."
                    )

            elif platform == "G2":
                st.warning("No reviews fetched from G2. G2 often requires a renderer (Apify) for reliable scraping.")
                st.caption("If you have an Apify actor, set APIFY_API_TOKEN + APIFY_ACTOR_ID in your environment and restart the app.")
                if st.button("Try fallback scraping (best-effort)", key="g2_try_fallback"):
                    with st.spinner("Attempting a best-effort G2 HTML/JSON-LD scrape..."):
                        try:
                            preview = fetch_reviews_for_ui(platform, confirmed, limit=preview_limit)
                        except Exception:
                            preview = []
                        st.session_state.search3_fetched_reviews = preview or []
                        st.session_state.search3_preview_analysis = None

            elif platform == "Trustpilot":
                st.warning("No reviews fetched from Trustpilot (likely blocked/no public reviews). Try a different slug like the domain shown in the Trustpilot URL.")
                st.caption("Example: paste a full Trustpilot page like https://www.trustpilot.com/review/<domain>")

                if st.button("Retry Trustpilot fetch", key="tp_retry"):
                    with st.spinner("Retrying Trustpilot fetch (Next.js JSON)..."):
                        try:
                            preview = fetch_reviews_for_ui(platform, confirmed, limit=preview_limit)
                        except Exception:
                            preview = []
                        st.session_state.search3_fetched_reviews = preview or []
                        st.session_state.search3_preview_analysis = None

            else:
                st.warning("No reviews could be fetched for this company (platform may block scraping). Try pasting a direct product link or check your environment.")
        else:
            render_review_preview(fetched, platform)

            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button("Run topic + sentiment + emotion on current set", key="run_preview_analysis"):
                    texts = [(r.get("content") or "").strip() for r in fetched]
                    cluster_label = st.session_state.get("cluster") or "All reviews"

                    topic_res = None
                    topic_summary_text = ""
                    topic_labels = {}
                    with st.spinner("Running topic model on preview..."):
                        try:
                            topic_res = discover_topics_batch(
                                texts,
                                top_k_words=10,
                                min_topic_size=5,
                            )
                        except Exception as exc:
                            st.error(f"Topic model failed: {exc}")
                            topic_res = None

                    if topic_res:
                        with st.spinner("Generating high-level topic names..."):
                            try:
                                topic_labels = llm_label_topics_from_keywords(
                                    cluster_label=cluster_label,
                                    keywords_by_topic=topic_res.get("keywords_by_topic") or {},
                                )
                            except Exception as exc:
                                st.warning(f"Topic naming failed: {exc}")
                                topic_labels = {}
                            topic_res["topic_labels"] = topic_labels

                        with st.spinner("Generating LLM topic summary..."):
                            try:
                                topic_summary_text = _topic_summary_or_empty(
                                    topic_res=topic_res,
                                    total_reviews=len(texts),
                                    cluster_label=cluster_label,
                                    review_texts=texts,
                                    examples_per_topic=TOPIC_EXAMPLES_PER_THEME,
                                )
                            except Exception as exc:
                                st.warning(f"LLM summary failed: {exc}")
                                topic_summary_text = ""

                    with st.spinner("Running sentiment on preview..."):
                        sentiments = [predict_sentiment_single(t or "") for t in texts]

                    with st.spinner("Extracting keywords on current set..."):
                        if KEYWORDS_PER_REVIEW <= 0 and KEYWORDS_OVERALL <= 0:
                            keyword_res = {"per_review": [[] for _ in texts], "overall": []}
                        else:
                            try:
                                keyword_res = extract_keywords_batch(
                                    texts=texts,
                                    per_review_top_n=KEYWORDS_PER_REVIEW,
                                    overall_top_n=KEYWORDS_OVERALL,
                                )
                            except Exception as exc:
                                st.warning(f"Keyword extraction failed: {exc}")
                                keyword_res = {"per_review": [[] for _ in texts], "overall": []}

                    with st.spinner("Running emotion analysis on current set..."):
                        # 1) VA regression
                        va_by_review = [predict_va_single(t or "") for t in texts]
                        va_overall = summarize_va(va_by_review)

                        # 2) Discrete emotion distribution
                        discrete_overall = emotion_percentages(texts, method="prob")
                        discrete_by_review = [predict_proba_single(t or "") for t in texts]

                    st.session_state.search3_preview_analysis = {
                        "cluster_label": cluster_label,
                        "topic": topic_res,
                        "topic_summary": topic_summary_text,
                        "keywords": keyword_res,
                        "reviews_signature": _reviews_signature(fetched),
                        "review_count": len(fetched),
                        "sentiment": sentiments,
                        "emotion": {
                            "va": va_overall,
                            "discrete": discrete_overall,
                        },
                        "emotion_by_review": {
                            "va": va_by_review,
                            "discrete": discrete_by_review,
                        },
                        "reviews": fetched,
                    }

            with cols[1]:
                if st.button(f"Fetch {st.session_state.get('search3_num_reviews', 20)} reviews", key="fetch_full_reviews"):
                    full_n = int(st.session_state.get("search3_num_reviews", 20))
                    with st.spinner(f"Fetching up to {full_n} reviews (this may take a moment)..."):
                        full = fetch_reviews_for_ui(platform, confirmed, limit=full_n)
                    st.session_state.search3_fetched_reviews = full or []
                    st.session_state.search3_preview_analysis = None
                    st.success(f"Fetched {len(full or [])} reviews. You can now run analysis on the full set.")

            with cols[2]:
                st.write("\n")
                st.caption("Reviews are cached for 30 minutes in the app session (except Trustpilot). No inference is run without your explicit action.")

        analysis = st.session_state.get("search3_preview_analysis")
        if analysis:
            current_sig = _reviews_signature(st.session_state.get("search3_fetched_reviews") or [])
            analysis_sig = str(analysis.get("reviews_signature") or "")
            if not analysis_sig or analysis_sig != current_sig:
                st.warning("Review set changed since last analysis. Please rerun analysis.")
                st.session_state.search3_preview_analysis = None
                analysis = None
        if analysis:
            render_analysis_results(analysis)
