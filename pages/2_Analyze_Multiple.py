import streamlit as st
import pandas as pd
from inference.sentiment import predict_single  # uses your local model

st.set_page_config(page_title="Analyze Multiple Reviews", page_icon="📄", layout="wide")

# --- CSS (same pattern as Page 1) ---
st.markdown(
    """
    <style>
      .field-title {
        font-size: 20px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      div[data-testid="stFileUploader"], div[data-testid="stRadio"] {
        margin-top: 6px;
      }

      div[data-testid="stFileUploader"] section {
        padding: 14px 16px !important;
        min-height: 100px !important;
        margin-bottom: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Back button
if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze Multiple Reviews")

# --- Session state ---
if "multi_submitted" not in st.session_state:
    st.session_state.multi_submitted = False
if "multi_errors" not in st.session_state:
    st.session_state.multi_errors = []
if "multi_result" not in st.session_state:
    st.session_state.multi_result = None  # dict with summary

# --- Layout ---
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(
        '<div class="field-title">Upload a CSV with one header and one review per row.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload a CSV with one header and one review per row.",
        type=["csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

with right:
    st.markdown('<div class="field-title">What cluster is your company?</div>', unsafe_allow_html=True)
    cluster = st.radio(
        "What cluster is your company?",
        options=[
            "Cluster 1 (AI-Charged Product/Service Providers)",
            "Cluster 2 (AI Development Facilitators)",
            "Cluster 3 (Data Analytics Providers)",
            "Cluster 4 (Deep Tech Researchers)",
        ],
        index=None,
        label_visibility="collapsed",
    )

def _read_reviews_csv(file) -> list[str]:
    """Assumes: 1 review column with a header. Handles commas inside reviews."""
    # Try robust parsing first
    try:
        df = pd.read_csv(
            file,
            engine="python",          # more tolerant than C engine
            sep=",",
            quotechar='"',            # reviews with commas should be wrapped in quotes
            escapechar="\\",
        )
    except Exception:
        # Fallback: keep going even if some rows are malformed
        file.seek(0)
        df = pd.read_csv(
            file,
            engine="python",
            sep=",",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="skip",      # skip broken rows instead of crashing
        )

    if df.shape[1] != 1:
        # If multiple columns exist, user likely has extra columns.
        # We'll use the FIRST column as the review column.
        df = df.iloc[:, [0]]

    col = df.columns[0]
    reviews = (
        df[col]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )
    reviews = [r for r in reviews.tolist() if r]

    if len(reviews) == 0:
        raise ValueError("No non-empty reviews found in the CSV.")

    return reviews

# --- Submit + validation + inference (simple loop for now) ---
with left:
    if st.button("Submit", type="primary"):
        st.session_state.multi_errors = []
        st.session_state.multi_submitted = False
        st.session_state.multi_result = None

        file_ok = uploaded_file is not None
        cluster_ok = cluster is not None

        if not file_ok and not cluster_ok:
            st.session_state.multi_errors.append(
                "Please upload a CSV and select your AI startup cluster before submitting."
            )
        elif not file_ok:
            st.session_state.multi_errors.append("Please upload a CSV file before submitting.")
        elif not cluster_ok:
            st.session_state.multi_errors.append("Please select your AI startup cluster before submitting.")
        else:
            try:
                reviews = _read_reviews_csv(uploaded_file)

                pos = neg = unc = 0
                for r in reviews:
                    label, _conf = predict_single(r)  # "Positive" / "Negative" / "Uncertain"
                    if label.lower().startswith("pos"):
                        pos += 1
                    elif label.lower().startswith("neg"):
                        neg += 1
                    else:
                        unc += 1

                total = len(reviews)
                st.session_state.multi_result = {
                    "cluster": cluster,
                    "total": total,
                    "positive": pos,
                    "negative": neg,
                    "uncertain": unc,
                }
                st.session_state.multi_submitted = True

            except Exception as e:
                st.session_state.multi_errors.append(str(e))

# --- Validation messages ---
for msg in st.session_state.multi_errors:
    st.warning(msg)

# --- Results (simple summary for now) ---
if st.session_state.multi_submitted and st.session_state.multi_result:
    res = st.session_state.multi_result
    total = res["total"]

    st.markdown("## Results")
    st.write(f"**Cluster:** {res['cluster']}")
    st.write(f"**Total reviews:** {total}")
    st.write(f"**Positive:** {res['positive']} ({res['positive']/total:.1%})")
    st.write(f"**Negative:** {res['negative']} ({res['negative']/total:.1%})")
    st.write(f"**Uncertain:** {res['uncertain']} ({res['uncertain']/total:.1%})")
