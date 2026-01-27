import streamlit as st
from inference.sentiment import predict_single  # <-- uses your local model

st.set_page_config(page_title="Analyze a Single Review", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
      .field-title {
        font-size: 20px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      div[data-testid="stTextArea"], div[data-testid="stRadio"] {
        margin-top: 0px;
      }

      textarea {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Back button
if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze a Single Review")

# --- Session state ---
if "single_submitted" not in st.session_state:
    st.session_state.single_submitted = False
if "single_errors" not in st.session_state:
    st.session_state.single_errors = []
if "single_result" not in st.session_state:
    st.session_state.single_result = None  # (label, confidence, cluster)

# --- Layout ---
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="field-title">Enter a review</div>', unsafe_allow_html=True)
    review_text = st.text_area(
        "Enter a review",
        placeholder="Type or paste a user review here...",
        height=130,
        label_visibility="collapsed",
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
        label_visibility="collapsed",
    )

# --- Submit + validation + inference ---
with left:
    if st.button("Submit", type="primary"):
        st.session_state.single_errors = []
        st.session_state.single_submitted = False
        st.session_state.single_result = None

        review_ok = bool(review_text and review_text.strip())
        cluster_ok = cluster is not None

        if not review_ok and not cluster_ok:
            st.session_state.single_errors.append(
                "Please enter a review and select your AI startup cluster before submitting."
            )
        elif not review_ok:
            st.session_state.single_errors.append("Please enter a review before submitting.")
        elif not cluster_ok:
            st.session_state.single_errors.append("Please select your AI startup cluster before submitting.")
        else:
            try:
                label, conf = predict_single(review_text.strip())  # <-- sentiment inference
                st.session_state.single_result = (label, conf, cluster)
                st.session_state.single_submitted = True
            except Exception as e:
                st.session_state.single_errors.append(f"Sentiment model error: {e}")

# --- Warnings ---
for msg in st.session_state.single_errors:
    st.warning(msg)

# --- Results ---
if st.session_state.single_submitted and st.session_state.single_result:
    label, conf, chosen_cluster = st.session_state.single_result

    st.markdown("## Results")
    st.write(f"**Cluster:** {chosen_cluster}")
    st.write(f"**Sentiment:** {label}")
    st.write(f"**Confidence:** {conf:.3f}")
