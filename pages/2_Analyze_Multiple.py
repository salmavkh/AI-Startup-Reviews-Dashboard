import streamlit as st

st.set_page_config(page_title="Analyze Multiple Reviews", page_icon="📄", layout="wide")

# --- CSS (same pattern as Page 1: controlled titles + tight spacing) ---
st.markdown(
    """
    <style>
      .field-title {
        font-size: 20px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      /* tighten spacing under our custom titles */
      div[data-testid="stFileUploader"], div[data-testid="stRadio"] {
        margin-top: 6px;
      }

      /* Reduce drag-and-drop upload box height */
     div[data-testid="stFileUploader"] section {
        padding: 14px 16px !important;   /* overall box height */
        min-height: 100px !important;    /* key line: controls height */
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

# --- Layout like your mock: uploader left, cluster right ---
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
        label_visibility="collapsed",  # we show our own title
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

# --- Submit button (left-aligned under uploader like your screenshot) ---
with left:
    if st.button("Submit", type="primary"):
        st.session_state.multi_errors = []

        file_ok = uploaded_file is not None
        cluster_ok = cluster is not None

        if not file_ok and not cluster_ok:
            st.session_state.multi_errors.append(
                "Please upload a CSV and select your AI startup cluster before submitting."
            )
            st.session_state.multi_submitted = False
        elif not file_ok:
            st.session_state.multi_errors.append("Please upload a CSV file before submitting.")
            st.session_state.multi_submitted = False
        elif not cluster_ok:
            st.session_state.multi_errors.append("Please select your AI startup cluster before submitting.")
            st.session_state.multi_submitted = False
        else:
            st.session_state.multi_submitted = True

# --- Validation messages ---
for msg in st.session_state.multi_errors:
    st.warning(msg)

# --- Output (placeholder) ---
if st.session_state.multi_submitted:
    st.markdown("## Results")
