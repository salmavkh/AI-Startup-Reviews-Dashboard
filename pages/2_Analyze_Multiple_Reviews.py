import pandas as pd
import streamlit as st

from helpers.review_analysis_pipeline import run_review_analysis
from helpers.search_ui_helpers import render_analysis_results
from helpers.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Analyze Multiple Reviews", page_icon="📄", layout="wide")
render_sidebar_nav()

TOPIC_EXAMPLES_PER_THEME = 2
KEYWORDS_PER_REVIEW = 5
KEYWORDS_OVERALL = 20
CSV_UPLOAD_MAX_MB = 300

st.markdown(
    """
    <style>
      .field-title {
        font-size: 16px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      div[data-testid="stFileUploader"] {
        margin-top: 6px;
      }

      div[data-testid="stFileUploader"] section {
        padding: 14px 16px !important;
        min-height: 100px !important;
        margin-bottom: 6px;
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

st.header("Analyze Multiple Reviews")

if "multi_errors" not in st.session_state:
    st.session_state.multi_errors = []
if "multi_result" not in st.session_state:
    st.session_state.multi_result = None


def _read_reviews_csv(file) -> list[str]:
    df = pd.read_csv(file, engine="python", sep=",", quotechar='"', escapechar="\\", on_bad_lines="skip")
    if df.shape[1] < 1:
        raise ValueError("CSV has no columns.")

    col = df.columns[0]
    reviews = df[col].dropna().astype(str).map(lambda s: s.strip())
    reviews = [r for r in reviews.tolist() if r]
    if not reviews:
        raise ValueError("No non-empty reviews found in the CSV.")
    return reviews


st.markdown(
    '<div class="field-title">Upload a CSV with one header and one review per row.</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload a CSV with one header and one review per row.",
    type=["csv"],
    accept_multiple_files=False,
    max_upload_size=CSV_UPLOAD_MAX_MB,
    label_visibility="collapsed",
)

if st.button("Analyze", type="primary"):
    st.session_state.multi_errors = []
    st.session_state.multi_result = None

    if uploaded_file is None:
        st.session_state.multi_errors.append("Please upload a CSV file before submitting.")
    else:
        try:
            reviews = _read_reviews_csv(uploaded_file)
            review_rows = [
                {
                    "id": f"csv_{i + 1}",
                    "title": f"Review {i + 1}",
                    "content": r,
                    "rating": None,
                    "date": "",
                    "platform": "Uploaded CSV",
                    "reviewer": "",
                }
                for i, r in enumerate(reviews)
            ]
            analysis = run_review_analysis(
                review_rows,
                keywords_per_review=KEYWORDS_PER_REVIEW,
                keywords_overall=KEYWORDS_OVERALL,
                topic_examples_per_theme=TOPIC_EXAMPLES_PER_THEME,
            )

            pos = neg = unc = 0
            for label, _conf in analysis.get("sentiment") or []:
                llow = str(label or "").lower()
                if llow.startswith("pos"):
                    pos += 1
                elif llow.startswith("neg"):
                    neg += 1
                else:
                    unc += 1

            st.session_state.multi_result = {
                "total": len(reviews),
                "sentiment_counts": {
                    "positive": pos,
                    "negative": neg,
                    "uncertain": unc,
                },
                "analysis": analysis,
            }

        except Exception as exc:
            st.session_state.multi_errors.append(f"Error: {exc}")

for msg in st.session_state.multi_errors:
    st.warning(msg)

if st.session_state.multi_result:
    res = st.session_state.multi_result
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    render_analysis_results(
        res["analysis"],
        show_section_heading=False,
    )
