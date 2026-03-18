import hashlib
import html
import json

import streamlit as st

from fetchers.g2 import search_g2
from fetchers.google_play import search_google_play
from fetchers.ios import search_ios
from fetchers.trustpilot import search_trustpilot
from fetchers.language_filter import filter_english_reviews
from helpers.search_ui_helpers import (
    extract_identifier_info,
    fetch_reviews_for_ui,
    logo_html,
    process_search_results,
    render_analysis_results,
)
from helpers.search_validation import validate_search_inputs, validate_submit_inputs
from inference.emotion import emotion_percentages, predict_proba_single
from inference.keywords import extract_keywords_batch
from inference.llm_topic_label import llm_label_topics_from_keywords
from inference.llm_topic_summary import llm_topic_summary
from inference.sentiment import predict_single as predict_sentiment_single
from inference.topic import discover_topics_batch
from inference.va import predict_va_single, summarize_va

st.set_page_config(page_title="Search Online Reviews", page_icon="🔎", layout="wide")

TOPIC_EXAMPLES_PER_THEME = 2
KEYWORDS_PER_REVIEW = 5
KEYWORDS_OVERALL = 20

st.markdown(
    """
    <style>
      .field-title {
        font-size: 34px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      .section-title {
        font-size: 34px;
        font-weight: 400;
        margin: 0 0 10px 0;
      }

      .result-option-card {
        border: 1px solid rgba(0, 0, 0, 0.12);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        background: #ffffff;
      }

      .result-option-card.selected {
        border: 2px solid #000000;
        padding: 11px;
      }

      .result-option-row {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .result-option-logo {
        width: 48px;
        height: 48px;
        flex: 0 0 48px;
      }

      .result-option-name {
        font-size: 18px;
        font-weight: 500;
        line-height: 1.2;
      }

      .result-option-subtitle {
        font-size: 13px;
        color: #6f6f6f;
        margin-top: 4px;
      }

      .none-option {
        font-size: 32px;
        line-height: 1.2;
        margin-top: 5px;
      }

      .company-header {
        font-size: 50px;
        font-weight: 700;
        margin: 26px 0 4px 0;
      }

      .fetched-count {
        text-align: right;
        font-size: 32px;
        margin-top: 30px;
      }

      .preview-card {
        border: 1px solid rgba(0, 0, 0, 0.12);
        border-radius: 12px;
        min-height: 200px;
        padding: 14px;
        background: #ffffff;
      }

      .preview-card-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
      }

      .preview-card-content {
        font-size: 15px;
        color: #232323;
      }

      .divider-line {
        border-top: 1px solid #d9d9d9;
        margin: 12px 0 22px 0;
      }

      div[data-testid="stButton"] button[kind="primary"] {
        background-color: #000000;
        border: 1px solid #000000;
        color: #ffffff;
      }

      div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #171717;
        border-color: #171717;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Back"):
    st.switch_page("app.py")

st.header("Search Online Reviews")


defaults = {
    "search3_errors": [],
    "search3_search_clicked": False,
    "search3_submit_clicked": False,
    "search3_results": [],
    "search3_selected_result": None,
    "search3_none_selected": False,
    "search3_num_reviews": 20,
    "search3_fetched_reviews": [],
    "search3_confirmed_company": None,
    "search3_fetched_for": {},
    "search3_preview_analysis": None,
    "search3_review_carousel_start": 0,
    "search3_query_input": "",
    "search3_platform": None,
    "search3_num_reviews_input": 20,
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


def _run_analysis_on_rows(rows: list[dict]) -> dict:
    texts = [(r.get("content") or "").strip() for r in rows]
    cluster_label = "General online review feedback"

    topic_res = None
    topic_summary_text = ""
    topic_labels = {}

    with st.spinner("Running topic model..."):
        try:
            topic_res = discover_topics_batch(
                texts,
                top_k_words=10,
                min_topic_size=5,
            )
        except Exception as exc:
            st.warning(f"Topic model failed: {exc}")
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

    with st.spinner("Running sentiment..."):
        sentiments = [predict_sentiment_single(t or "") for t in texts]

    with st.spinner("Extracting keywords..."):
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

    with st.spinner("Running emotion analysis..."):
        va_by_review = [predict_va_single(t or "") for t in texts]
        va_overall = summarize_va(va_by_review)
        discrete_overall = emotion_percentages(texts, method="prob")
        discrete_by_review = [predict_proba_single(t or "") for t in texts]

    return {
        "cluster_label": cluster_label,
        "topic": topic_res,
        "topic_summary": topic_summary_text,
        "keywords": keyword_res,
        "reviews_signature": _reviews_signature(rows),
        "review_count": len(rows),
        "sentiment": sentiments,
        "emotion": {
            "va": va_overall,
            "discrete": discrete_overall,
        },
        "emotion_by_review": {
            "va": va_by_review,
            "discrete": discrete_by_review,
        },
        "reviews": rows,
    }


def _render_option_card(r: dict):
    rid = str(r.get("id") or "")
    is_selected = st.session_state.get("search3_selected_result") == rid

    card_cols = st.columns([1, 11], gap="small")
    with card_cols[0]:
        if st.button("●" if is_selected else "○", key=f"search3_pick_dot_{rid}", use_container_width=True):
            st.session_state.search3_selected_result = rid
            st.session_state.search3_none_selected = False

    with card_cols[1]:
        selected_class = " selected" if is_selected else ""
        st.markdown(
            "<div class='result-option-card{}'>"
            "<div class='result-option-row'>"
            "<div class='result-option-logo'>{}</div>"
            "<div>"
            "<div class='result-option-name'>{}</div>"
            "<div class='result-option-subtitle'>{}</div>"
            "</div>"
            "</div>"
            "</div>".format(
                selected_class,
                logo_html(r.get("logo")),
                html.escape(str(r.get("name") or "(unknown)")),
                html.escape(str(r.get("subtitle") or "")),
            ),
            unsafe_allow_html=True,
        )


search_cols = st.columns(2, gap="large")

with search_cols[0]:
    st.markdown('<div class="field-title">Search your company name</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Search your company name",
        placeholder="Search",
        label_visibility="collapsed",
        key="search3_query_input",
    )

    st.write("")
    st.markdown('<div class="field-title">What platform?</div>', unsafe_allow_html=True)
    platform = st.radio(
        "What platform?",
        options=["Google Play Store", "iOS App Store", "G2", "Trustpilot"],
        index=None,
        label_visibility="collapsed",
        key="search3_platform",
    )

    st.write("")
    st.markdown('<div class="field-title">How many reviews?</div>', unsafe_allow_html=True)
    num_reviews = int(
        st.number_input(
            "How many reviews?",
            min_value=1,
            max_value=100,
            step=1,
            label_visibility="collapsed",
            key="search3_num_reviews_input",
        )
    )

    st.write("")
    if st.button("Search", type="primary", use_container_width=True):
        st.session_state.search3_errors = []
        st.session_state.search3_num_reviews = num_reviews
        st.session_state.search3_submit_clicked = False
        st.session_state.search3_confirmed_company = None
        st.session_state.search3_fetched_reviews = []
        st.session_state.search3_fetched_for = {}
        st.session_state.search3_preview_analysis = None
        st.session_state.search3_review_carousel_start = 0

        errors = validate_search_inputs(query, platform, num_reviews)
        st.session_state.search3_errors = errors

        if st.session_state.search3_errors:
            st.session_state.search3_search_clicked = False
            st.session_state.search3_results = []
            st.session_state.search3_selected_result = None
            st.session_state.search3_none_selected = False
        else:
            st.session_state.search3_search_clicked = True
            st.session_state.search3_selected_result = None
            st.session_state.search3_none_selected = False

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
            except Exception as exc:
                st.session_state.search3_errors.append(f"Search failed: {exc}")
                candidates = []

            st.session_state.search3_results = process_search_results(candidates, platform)

with search_cols[1]:
    st.markdown('<div class="section-title">Select your company from this possible results:</div>', unsafe_allow_html=True)

    if st.session_state.search3_search_clicked:
        results = st.session_state.search3_results or []

        if results:
            rows = [results[i:i + 2] for i in range(0, len(results), 2)]
            for row in rows:
                row_cols = st.columns(2, gap="large")
                for ix, item in enumerate(row):
                    with row_cols[ix]:
                        _render_option_card(item)
        else:
            st.info("No results found. Try a different company/app name.")

        none_selected = bool(st.session_state.get("search3_none_selected"))
        none_cols = st.columns([1, 11], gap="small")
        with none_cols[0]:
            if st.button("●" if none_selected else "○", key="search3_pick_none", use_container_width=True):
                st.session_state.search3_selected_result = None
                st.session_state.search3_none_selected = True
        with none_cols[1]:
            selected_class = " selected" if none_selected else ""
            st.markdown(
                f"<div class='result-option-card{selected_class}'><div class='none-option'>None of those</div></div>",
                unsafe_allow_html=True,
            )

        st.write("")
        if st.button("Submit", type="primary", use_container_width=True):
            submit_errors = validate_submit_inputs(
                st.session_state.search3_selected_result is not None,
                "",
            )
            if submit_errors:
                for err in submit_errors:
                    st.warning(err)
                st.session_state.search3_submit_clicked = False
            else:
                selected_id = st.session_state.search3_selected_result
                selected = None
                for rr in st.session_state.search3_results:
                    if rr.get("id") == selected_id:
                        selected = rr
                        break

                if not selected:
                    st.warning("Selected result not found. Please pick a result and submit again.")
                    st.session_state.search3_submit_clicked = False
                else:
                    st.session_state.search3_confirmed_company = {**selected, "source": "selection"}
                    st.session_state.search3_submit_clicked = True
                    st.session_state.search3_preview_analysis = None
                    st.session_state.search3_review_carousel_start = 0

for msg in st.session_state.search3_errors:
    st.warning(msg)


if st.session_state.search3_submit_clicked:
    confirmed = st.session_state.get("search3_confirmed_company")
    if confirmed:
        identifier = extract_identifier_info(confirmed)
        fetch_key = {
            "platform": confirmed.get("platform"),
            "identifier": (identifier[1] if identifier else None),
            "limit": int(st.session_state.get("search3_num_reviews", 20)),
        }

        if st.session_state.get("search3_fetched_for") != fetch_key:
            with st.spinner(f"Fetching up to {fetch_key['limit']} reviews..."):
                fetched = fetch_reviews_for_ui(
                    confirmed.get("platform"),
                    confirmed,
                    limit=fetch_key["limit"],
                )
            fetched = filter_english_reviews(fetched or [], limit=None)
            st.session_state.search3_fetched_reviews = fetched
            st.session_state.search3_fetched_for = fetch_key
            st.session_state.search3_preview_analysis = None
            st.session_state.search3_review_carousel_start = 0

        fetched = st.session_state.get("search3_fetched_reviews") or []

        header_cols = st.columns([5, 2])
        with header_cols[0]:
            st.markdown(
                f"<div class='company-header'>{html.escape(str(confirmed.get('name') or '(name not available)'))}</div>",
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown(
                f"<div class='fetched-count'>Fetched: {len(fetched)} reviews</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

        if not fetched:
            st.warning("No reviews were fetched for this company. Try another result or lower the review count.")
        else:
            total = len(fetched)
            start = int(st.session_state.get("search3_review_carousel_start", 0))
            start = max(0, min(start, max(0, total - 1)))
            st.session_state.search3_review_carousel_start = start

            max_cards = min(3, total)
            show_idx = [((start + i) % total) for i in range(max_cards)]

            carousel_cols = st.columns([1, 10, 1], gap="small")
            with carousel_cols[0]:
                if st.button("<", key="search3_prev_reviews", use_container_width=True):
                    st.session_state.search3_review_carousel_start = (start - 1) % total
                    st.rerun()

            with carousel_cols[1]:
                review_cols = st.columns(max_cards, gap="large")
                for ix, review_idx in enumerate(show_idx):
                    row = fetched[review_idx] or {}
                    title = (row.get("title") or f"Review {review_idx + 1}").strip()
                    content = (row.get("content") or "").strip()
                    snippet = content[:180] + ("..." if len(content) > 180 else "")

                    with review_cols[ix]:
                        st.markdown(
                            "<div class='preview-card'>"
                            f"<div class='preview-card-title'>{html.escape(title)}</div>"
                            f"<div class='preview-card-content'>{html.escape(snippet or '(no text)')}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

            with carousel_cols[2]:
                if st.button(">", key="search3_next_reviews", use_container_width=True):
                    st.session_state.search3_review_carousel_start = (start + 1) % total
                    st.rerun()

            st.write("")
            center_cols = st.columns([2, 3, 2])
            with center_cols[1]:
                if st.button("Analyze", type="primary", use_container_width=True):
                    st.session_state.search3_preview_analysis = _run_analysis_on_rows(fetched)

analysis = st.session_state.get("search3_preview_analysis")
if analysis:
    current_sig = _reviews_signature(st.session_state.get("search3_fetched_reviews") or [])
    analysis_sig = str(analysis.get("reviews_signature") or "")
    if not analysis_sig or analysis_sig != current_sig:
        st.warning("Review set changed since last analysis. Please run Analyze again.")
        st.session_state.search3_preview_analysis = None
        analysis = None

if analysis:
    render_analysis_results(analysis)
