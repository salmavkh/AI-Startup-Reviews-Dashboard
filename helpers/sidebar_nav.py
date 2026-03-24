"""Custom sidebar navigation links."""

import streamlit as st


def render_sidebar_nav() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            display: none;
        }
        section[data-testid="stSidebar"] hr {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.page_link("app.py", label="Home")
        st.page_link("pages/1_Analyze_Single_Review.py", label="Analyze Single Review")
        st.page_link("pages/2_Analyze_Multiple_Reviews.py", label="Analyze Multiple Reviews")
        st.page_link("pages/3_Search_Online_Reviews.py", label="Search Online Reviews")
