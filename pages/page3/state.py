"""Session-state helpers for Search Online page."""

import streamlit as st

from pages.page3.constants import DEFAULTS


def initialize_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_after_primary_action(num_reviews: int) -> None:
    st.session_state.search3_errors = []
    st.session_state.search3_num_reviews = int(num_reviews)
    st.session_state.search3_submit_clicked = False
    st.session_state.search3_confirmed_company = None
    st.session_state.search3_fetched_reviews = []
    st.session_state.search3_fetched_for = {}
    st.session_state.search3_preview_analysis = None
    st.session_state.search3_review_carousel_start = 0
    st.session_state.search3_pasted_link = ""


def clear_search_results_state() -> None:
    st.session_state.search3_search_clicked = False
    st.session_state.search3_results = []
    st.session_state.search3_selected_result = None
    st.session_state.search3_none_selected = False
    st.session_state.search3_pasted_link = ""
    st.session_state.search3_submit_clicked = False
