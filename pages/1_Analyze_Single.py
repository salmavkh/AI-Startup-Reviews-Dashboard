import streamlit as st

from helpers.review_analysis_pipeline import run_review_analysis
from helpers.search_ui_helpers import render_analysis_results

st.set_page_config(page_title="Analyze a Single Review", page_icon="📝", layout="wide")

TOPIC_EXAMPLES_PER_THEME = 2
KEYWORDS_PER_REVIEW = 5
KEYWORDS_OVERALL = 20

st.markdown(
    """
    <style>
      .field-title {
        font-size: 16px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      textarea {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
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

      div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #c3c8d0;
        color: #2e3340;
      }

      div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #f7f9fc;
        border-color: #aeb6c2;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze a Single Review")

if "single_errors" not in st.session_state:
    st.session_state.single_errors = []
if "single_result" not in st.session_state:
    st.session_state.single_result = None

st.markdown('<div class="field-title">Enter a review</div>', unsafe_allow_html=True)
review_text = st.text_area(
    "Enter a review",
    placeholder="Type or paste a user review here...",
    height=130,
    label_visibility="collapsed",
)

if st.button("Analyze", type="primary"):
    st.session_state.single_errors = []
    st.session_state.single_result = None

    if not review_text or not review_text.strip():
        st.session_state.single_errors.append("Please enter a review before submitting.")
    else:
        try:
            text = review_text.strip()
            review_rows = [
                {
                    "id": "manual_1",
                    "title": "Review 1",
                    "content": text,
                    "rating": None,
                    "date": "",
                    "platform": "Manual Input",
                    "reviewer": "",
                }
            ]
            analysis = run_review_analysis(
                review_rows,
                keywords_per_review=KEYWORDS_PER_REVIEW,
                keywords_overall=KEYWORDS_OVERALL,
                topic_examples_per_theme=TOPIC_EXAMPLES_PER_THEME,
            )
            st.session_state.single_result = {"analysis": analysis}
        except Exception as exc:
            st.session_state.single_errors.append(f"Error: {exc}")

for msg in st.session_state.single_errors:
    st.warning(msg)

if st.session_state.single_result:
    res = st.session_state.single_result
    render_analysis_results(
        res["analysis"],
        show_overall=False,
        show_per_review=True,
        show_topic_assignment=False,
        show_section_heading=False,
        show_topic_title_before_keywords=True,
        show_review_preview=False,
    )
