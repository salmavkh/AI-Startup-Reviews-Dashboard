import streamlit as st

st.set_page_config(page_title="Analyze Multiple Reviews", page_icon="📄", layout="wide")

# Back button
if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze Multiple Reviews")

# --- Upload CSV ---
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    accept_multiple_files=False,
)

# --- Click state ---
if "multi_submitted" not in st.session_state:
    st.session_state.multi_submitted = False

# --- Submit button ---
submit_disabled = uploaded_file is None
if st.button("Submit", type="primary", disabled=submit_disabled):
    st.session_state.multi_submitted = True

# --- Output (placeholder) ---
if st.session_state.multi_submitted:
    st.markdown("## Results")
