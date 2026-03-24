"""State transitions and actions for Search Online page."""

import streamlit as st

from fetchers.g2 import search_g2
from fetchers.google_play import search_google_play
from fetchers.ios import search_ios
from fetchers.trustpilot import search_trustpilot
from helpers.search_ui_helpers import process_search_results
from helpers.search_validation import (
    parse_pasted_link,
    validate_search_inputs,
    validate_submit_inputs,
)
from pages.page3.state import clear_search_results_state, reset_after_primary_action


def _search_candidates(platform: str | None, query: str) -> list:
    if platform == "Google Play Store":
        return search_google_play(query, limit=5)
    if platform == "iOS App Store":
        return search_ios(query, limit=5)
    if platform == "G2":
        return search_g2(query, limit=5)
    if platform == "Trustpilot":
        return search_trustpilot(query, limit=5)
    return []


def handle_primary_action(
    platform: str | None,
    query: str,
    num_reviews: int,
    *,
    is_direct_link_mode: bool,
) -> None:
    reset_after_primary_action(num_reviews=num_reviews)

    if is_direct_link_mode:
        errors = []
        if not (query and query.strip()):
            errors.append("Please paste the company link before fetching.")
        if platform is None:
            errors.append("Please select a platform before fetching.")
        if not (isinstance(num_reviews, int) and 1 <= num_reviews <= 100):
            errors.append("Please enter a number of reviews between 1 and 100.")
        st.session_state.search3_errors = errors
    else:
        st.session_state.search3_errors = validate_search_inputs(query, platform, num_reviews)

    if st.session_state.search3_errors:
        clear_search_results_state()
        return

    if is_direct_link_mode:
        confirmed, parse_errors = parse_pasted_link(platform, query)
        if parse_errors:
            st.session_state.search3_errors = parse_errors
            clear_search_results_state()
            return

        if not confirmed:
            st.session_state.search3_errors = ["Could not confirm company from the pasted link."]
            clear_search_results_state()
            return

        if not confirmed.get("name"):
            confirmed_name = confirmed.get("g2_slug") or confirmed.get("tp_slug") or query.strip()
            confirmed["name"] = confirmed_name

        st.session_state.search3_search_clicked = False
        st.session_state.search3_selected_result = None
        st.session_state.search3_none_selected = False
        st.session_state.search3_pasted_link = query.strip()
        st.session_state.search3_results = []
        st.session_state.search3_confirmed_company = confirmed
        st.session_state.search3_submit_clicked = True
        st.session_state.search3_preview_analysis = None
        st.session_state.search3_review_carousel_start = 0
        return

    st.session_state.search3_search_clicked = True
    st.session_state.search3_selected_result = None
    st.session_state.search3_none_selected = False
    st.session_state.search3_pasted_link = ""

    candidates = []
    try:
        candidates = _search_candidates(platform, query)
    except Exception as exc:
        st.session_state.search3_errors.append(f"Search failed: {exc}")
        candidates = []

    st.session_state.search3_results = process_search_results(candidates, platform)


def handle_submit_action() -> None:
    picked_result = st.session_state.search3_selected_result is not None
    none_selected = bool(st.session_state.get("search3_none_selected"))
    pasted_link = st.session_state.get("search3_pasted_link", "") if none_selected else ""

    submit_errors = validate_submit_inputs(
        picked_result,
        pasted_link,
    )
    if submit_errors:
        for err in submit_errors:
            st.warning(err)
        st.session_state.search3_submit_clicked = False
        return

    if picked_result:
        selected_id = st.session_state.search3_selected_result
        selected = None
        for row in st.session_state.search3_results:
            if row.get("id") == selected_id:
                selected = row
                break

        if not selected:
            st.warning("Selected result not found. Please pick a result and submit again.")
            st.session_state.search3_submit_clicked = False
            return

        if selected.get("platform") == "Google Play Store" and not selected.get("package"):
            st.warning(
                "This Google Play result is missing a package ID, so reviews cannot be fetched. "
                "Please pick another result or paste the direct app link."
            )
            st.session_state.search3_submit_clicked = False
            st.session_state.search3_confirmed_company = None
            return

        st.session_state.search3_confirmed_company = {**selected, "source": "selection"}
        st.session_state.search3_submit_clicked = True
        st.session_state.search3_preview_analysis = None
        st.session_state.search3_review_carousel_start = 0
        return

    confirmed, parse_errors = parse_pasted_link(
        st.session_state.get("search3_platform"),
        st.session_state.get("search3_pasted_link", ""),
    )
    if parse_errors:
        for err in parse_errors:
            st.warning(err)
        st.session_state.search3_submit_clicked = False
    elif not confirmed:
        st.warning("Could not confirm company from the pasted link.")
        st.session_state.search3_submit_clicked = False
    else:
        st.session_state.search3_confirmed_company = confirmed
        st.session_state.search3_submit_clicked = True
        st.session_state.search3_preview_analysis = None
        st.session_state.search3_review_carousel_start = 0
