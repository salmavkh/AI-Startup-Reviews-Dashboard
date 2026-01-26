import streamlit as st

st.set_page_config(page_title="Analyze a Single Review", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
      /* Smaller, not-bold titles */
      .field-title {
        font-size: 20px;
        font-weight: 400;
        margin: 0 0 6px 0;   /* controls gap under title */
      }

      /* Remove extra spacing above widgets inside columns */
      div[data-testid="stTextArea"], div[data-testid="stRadio"] {
        margin-top: 0px;    /* controls gap between title and widget */
      }

      /* Slightly reduce textarea “empty feel” */
      textarea {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze a Single Review")

if "single_submitted" not in st.session_state:
    st.session_state.single_submitted = False
if "single_errors" not in st.session_state:
    st.session_state.single_errors = []

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="field-title">Enter a review</div>', unsafe_allow_html=True)
    review_text = st.text_area(
        "Enter a review",
        placeholder="Type or paste a user review here...",
        height=130,
        label_visibility="collapsed",  # <-- hide Streamlit label (reliable)
    )

with right:
    st.markdown('<div class="field-title">What cluster is AI Startup?</div>', unsafe_allow_html=True)
    cluster = st.radio(
        "What cluster is your company?",
        options=[
            "Cluster 1 (AI-Charged Product/Service Providers)",
            "Cluster 2 (AI Development Facilitators)",
            "Cluster 3 (Data Analytics Providers)",
            "Cluster 4 (Deep Tech Researchers)",
        ],
        index=None,
        label_visibility="collapsed",  # <-- hide Streamlit label (reliable)
    )

with left:
    if st.button("Submit", type="primary"):
        st.session_state.single_errors = []

        review_ok = bool(review_text and review_text.strip())
        cluster_ok = cluster is not None

        if not review_ok and not cluster_ok:
            st.session_state.single_errors.append(
                "Please enter a review and select your AI startup cluster before submitting."
            )
            st.session_state.single_submitted = False
        elif not review_ok:
            st.session_state.single_errors.append("Please enter a review before submitting.")
            st.session_state.single_submitted = False
        elif not cluster_ok:
            st.session_state.single_errors.append("Please select your AI startup cluster before submitting.")
            st.session_state.single_submitted = False
        else:
            st.session_state.single_submitted = True

for msg in st.session_state.single_errors:
    st.warning(msg)

if st.session_state.single_submitted:
    st.markdown("## Results")
