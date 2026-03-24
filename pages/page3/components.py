"""UI components for Search Online page."""

import html

import streamlit as st

from helpers.search_ui_helpers import logo_html


def render_option_card(candidate: dict) -> None:
    rid = str(candidate.get("id") or "")
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
                logo_html(candidate.get("logo")),
                html.escape(str(candidate.get("name") or "(unknown)")),
                html.escape(str(candidate.get("subtitle") or "")),
            ),
            unsafe_allow_html=True,
        )
