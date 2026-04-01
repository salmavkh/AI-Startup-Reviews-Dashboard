"""UI components for Search Online page."""

import json
import streamlit as st


def _key_suffix(value: str) -> str:
    return "".join(ch for ch in str(value or "") if (ch.isalnum() or ch in {"_", "-"})) or "unknown"


def _inject_card_logo_css(card_key_suffix: str, logo_url: str) -> None:
    logo = str(logo_url or "").strip()
    logo_css = (
        "background-image: none;"
        if not logo
        else f"background-image: url({json.dumps(logo)});"
    )
    st.markdown(
        f"""
        <style>
          div[class*="st-key-search3_pick_card_{card_key_suffix}"] button {{
            position: relative !important;
            padding-left: var(--search3-left-pad) !important;
          }}

          div[class*="st-key-search3_pick_card_{card_key_suffix}"] button::before {{
            content: "";
            position: absolute;
            left: 12px;
            top: 50%;
            width: var(--search3-logo-sz);
            height: var(--search3-logo-sz);
            border-radius: 8px;
            transform: translateY(-50%);
            background-color: #f2f2f2;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            {logo_css}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_option_card(candidate: dict) -> None:
    rid = str(candidate.get("id") or "")
    key_suffix = _key_suffix(rid)
    raw_name = str(candidate.get("name") or "(unknown)")
    subtitle = str(candidate.get("subtitle") or "").strip()
    is_selected = st.session_state.get("search3_selected_result") == rid

    def _select_card() -> None:
        st.session_state.search3_selected_result = rid
        st.session_state.search3_none_selected = False

    card_cols = st.columns([1, 11], gap="small")
    with card_cols[0]:
        st.button(
            "●" if is_selected else "○",
            key=f"search3_pick_dot_{key_suffix}",
            use_container_width=True,
            on_click=_select_card,
        )

    with card_cols[1]:
        label = raw_name if not subtitle else f"{raw_name}\n`{subtitle}`"
        st.button(
            label,
            key=f"search3_pick_card_{key_suffix}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
            on_click=_select_card,
        )
        _inject_card_logo_css(key_suffix, str(candidate.get("logo") or ""))


def render_none_option() -> None:
    none_selected = bool(st.session_state.get("search3_none_selected"))

    def _select_none() -> None:
        st.session_state.search3_selected_result = None
        st.session_state.search3_none_selected = True

    card_cols = st.columns([1, 11], gap="small")
    with card_cols[0]:
        st.button(
            "●" if none_selected else "○",
            key="search3_pick_none",
            use_container_width=True,
            on_click=_select_none,
        )
    with card_cols[1]:
        st.button(
            "None of those",
            key="search3_pick_none_card",
            use_container_width=True,
            type="primary" if none_selected else "secondary",
            on_click=_select_none,
        )
