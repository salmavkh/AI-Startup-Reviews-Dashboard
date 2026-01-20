import streamlit as st

st.set_page_config(page_title="Analyze a Single Review", page_icon="📝", layout="wide")

# Back button
if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze a Single Review")

# --- Input: type/paste review ---
review_text = st.text_area(
    "Enter a review",
    placeholder="Type or paste a user review here...",
    height=180,
)

# --- Click state ---
if "single_submitted" not in st.session_state:
    st.session_state.single_submitted = False

# --- Submit button ---
if st.button("Submit", type="primary"):
    st.session_state.single_submitted = True

# --- Output (placeholder) ---
if st.session_state.single_submitted:
    st.markdown("## Results")