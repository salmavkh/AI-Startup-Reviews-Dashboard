"""Main renderer for Search Online (Page 3)."""

import html

import streamlit as st

from fetchers.language_filter import filter_english_reviews
from helpers.search_ui_helpers import (
    extract_identifier_info,
    fetch_reviews_for_ui,
    render_analysis_results,
)
from helpers.sidebar_nav import render_sidebar_nav
from pages.page3.analysis import is_analysis_stale, run_page_analysis
from pages.page3.components import render_option_card
from pages.page3.constants import DIRECT_LINK_PLATFORMS, PAGE_CSS
from pages.page3.state import initialize_state
from pages.page3.workflow import handle_primary_action, handle_submit_action


def _render_left_panel() -> None:
    st.markdown('<div class="field-title">What platform?</div>', unsafe_allow_html=True)
    platform = st.radio(
        "What platform?",
        options=["Google Play Store", "iOS App Store", "G2", "Trustpilot"],
        index=None,
        label_visibility="collapsed",
        key="search3_platform",
    )

    is_direct_link_mode = platform in DIRECT_LINK_PLATFORMS

    st.write("")
    if is_direct_link_mode:
        st.markdown('<div class="field-title">Paste company link</div>', unsafe_allow_html=True)
        placeholder = (
            "e.g. https://www.g2.com/products/openai/reviews"
            if platform == "G2"
            else "e.g. https://www.trustpilot.com/review/spotify.com"
        )
        query = st.text_input(
            "Paste company link",
            placeholder=placeholder,
            label_visibility="collapsed",
            key="search3_query_input",
        )
    else:
        st.markdown('<div class="field-title">Search your company name</div>', unsafe_allow_html=True)
        query = st.text_input(
            "Search your company name",
            placeholder="Search",
            label_visibility="collapsed",
            key="search3_query_input",
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
    primary_label = "Fetch Reviews" if is_direct_link_mode else "Search"
    if st.button(primary_label, type="primary", use_container_width=True):
        handle_primary_action(
            platform=platform,
            query=query,
            num_reviews=num_reviews,
            is_direct_link_mode=is_direct_link_mode,
        )

    if st.session_state.search3_search_clicked:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Select your company from this possible results:</div>',
            unsafe_allow_html=True,
        )
        results = st.session_state.search3_results or []

        if results:
            rows = [results[i:i + 2] for i in range(0, len(results), 2)]
            for row in rows:
                row_cols = st.columns(2, gap="large")
                for ix, item in enumerate(row):
                    with row_cols[ix]:
                        render_option_card(item)
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

        if none_selected:
            st.markdown('<div class="field-title">Paste the app/company link</div>', unsafe_allow_html=True)
            st.text_input(
                "Paste the app/company link",
                placeholder="e.g. https://www.trustpilot.com/review/spotify.com",
                label_visibility="collapsed",
                key="search3_pasted_link",
            )

        st.write("")
        if st.button("Submit", type="primary", use_container_width=True):
            handle_submit_action()


def _render_right_panel() -> None:
    if not st.session_state.search3_submit_clicked:
        st.caption("Select and submit a company to preview fetched reviews here.")
        return

    confirmed = st.session_state.get("search3_confirmed_company")
    if not confirmed:
        return

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

    header_cols = st.columns([3, 2])
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

    if not fetched:
        st.warning("No reviews were fetched for this company. Try another result or lower the review count.")
        return

    total_reviews = len(fetched)
    window_size = 5
    max_start = max(0, total_reviews - window_size)
    start = int(st.session_state.get("search3_review_carousel_start", 0))
    start = max(0, min(start, max_start))

    nav_cols = st.columns([1, 1, 4], gap="small")
    with nav_cols[0]:
        prev_clicked = st.button(
            "Prev",
            key="search3_prev_page",
            use_container_width=True,
            disabled=(start <= 0),
        )
    with nav_cols[1]:
        next_clicked = st.button(
            "Next",
            key="search3_next_page",
            use_container_width=True,
            disabled=(start >= max_start),
        )

    if prev_clicked:
        start = max(0, start - window_size)
    if next_clicked:
        start = min(max_start, start + window_size)
    st.session_state.search3_review_carousel_start = start

    end = min(total_reviews, start + window_size)
    with nav_cols[2]:
        st.caption(f"Showing {start + 1}-{end} of {total_reviews} fetched reviews.")

    for review_idx, row in enumerate(fetched[start:end], start=start + 1):
        row = row or {}
        title = (row.get("title") or f"Review {review_idx}").strip()
        content = (row.get("content") or "").strip()
        snippet = content[:220] + ("..." if len(content) > 220 else "")
        st.markdown(
            "<div class='preview-card'>"
            f"<div class='preview-card-title'>{html.escape(title)}</div>"
            f"<div class='preview-card-content'>{html.escape(snippet or '(no text)')}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    analyze_cols = st.columns([2, 3, 2])
    with analyze_cols[1]:
        if st.button("Analyze", type="primary", use_container_width=True):
            st.session_state.search3_preview_analysis = run_page_analysis(fetched)


def _render_analysis() -> None:
    analysis = st.session_state.get("search3_preview_analysis")
    if is_analysis_stale(analysis, st.session_state.get("search3_fetched_reviews") or []):
        st.warning("Review set changed since last analysis. Please run Analyze again.")
        st.session_state.search3_preview_analysis = None
        analysis = None

    if analysis:
        render_analysis_results(
            analysis,
            show_section_heading=False,
            compact_top_spacing=True,
        )


def render_page() -> None:
    st.set_page_config(page_title="Search Online Reviews", page_icon="🔎", layout="wide")
    render_sidebar_nav()
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    if st.button("← Back"):
        st.switch_page("app.py")

    st.header("Search Online Reviews")

    initialize_state()

    layout_cols = st.columns([1.0, 0.03, 1.1], gap="large")
    left_panel, divider_panel, right_panel = layout_cols

    with divider_panel:
        st.markdown("<div class='vertical-divider'></div>", unsafe_allow_html=True)

    with left_panel:
        _render_left_panel()

    for msg in st.session_state.search3_errors:
        st.warning(msg)

    with right_panel:
        _render_right_panel()

    _render_analysis()
