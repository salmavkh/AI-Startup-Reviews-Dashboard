import streamlit as st

st.set_page_config(page_title="Search Online Reviews", page_icon="🔎", layout="wide")

# --- CSS (same vibe as your other pages) ---
st.markdown(
    """
    <style>
      .field-title {
        font-size: 20px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      /* tighten spacing under our custom titles */
      div[data-testid="stTextInput"], div[data-testid="stRadio"] {
        margin-top: 0px;
      }

      /* simple “card” look for results */
      .result-card {
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 12px;
        padding: 12px 14px;
        background: white;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Back button
if st.button("← Back"):
    st.switch_page("app.py")

st.header("Search Online Reviews")

# ---------------------------
# Session state initialization
# ---------------------------
defaults = {
    "search3_errors": [],
    "search3_search_clicked": False,
    "search3_submit_clicked": False,
    "search3_results": [],
    "search3_selected_result": None,
    "search3_pasted_link": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------
# Layout: left inputs, right results
# ---------------------------
left, right = st.columns([3, 2], gap="large")

# ---------------------------
# LEFT: inputs + Search button
# ---------------------------
with left:
    st.markdown('<div class="field-title">Search your company name</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Search your company name",
        placeholder="Search",
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown('<div class="field-title">What platform?</div>', unsafe_allow_html=True)
    platform = st.radio(
        "What platform?",
        options=["Google Play Store", "iOS App Store", "G2", "Trustpilot"],
        index=None,  # IMPORTANT: no default selection
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown('<div class="field-title">What cluster is your company?</div>', unsafe_allow_html=True)
    cluster = st.radio(
        "What cluster is your company?",
        options=[
            "Cluster 1 (AI-Charged Product/Service Providers)",
            "Cluster 2 (AI Development Facilitators)",
            "Cluster 3 (Data Analytics Providers)",
            "Cluster 4 (Deep Tech Researchers)",
        ],
        index=None,  # IMPORTANT: no default selection
        label_visibility="collapsed",
    )

    st.write("")
    if st.button("Search", type="primary"):
        st.session_state.search3_errors = []

        query_ok = bool(query and query.strip())
        platform_ok = platform is not None
        cluster_ok = cluster is not None

        if not query_ok:
            st.session_state.search3_errors.append("Please enter a company/app name before searching.")
        if not platform_ok:
            st.session_state.search3_errors.append("Please select a platform before searching.")
        if not cluster_ok:
            st.session_state.search3_errors.append("Please select your AI startup cluster before searching.")

        if st.session_state.search3_errors:
            st.session_state.search3_search_clicked = False
            st.session_state.search3_results = []
            st.session_state.search3_selected_result = None
            st.session_state.search3_pasted_link = ""
        else:
            # Mark that Search succeeded
            st.session_state.search3_search_clicked = True

            # Reset selection state for a fresh search
            st.session_state.search3_selected_result = None
            st.session_state.search3_pasted_link = ""

            # Placeholder results (replace later with real search)
            # Keep this as up to 5 items to match your wireframe.
            st.session_state.search3_results = [
                {"id": "r1", "platform": platform, "name": "Company1", "subtitle": "Developer/Domain/Category", "logo": None},
                {"id": "r2", "platform": platform, "name": "Company2", "subtitle": "Developer/Domain/Category", "logo": None},
                {"id": "r3", "platform": platform, "name": "Company3", "subtitle": "Developer/Domain/Category", "logo": None},
                {"id": "r4", "platform": platform, "name": "Company4", "subtitle": "Developer/Domain/Category", "logo": None},
                {"id": "r5", "platform": platform, "name": "Company5", "subtitle": "Developer/Domain/Category", "logo": None},
            ]

# Validation warnings (same style as your other pages)
for msg in st.session_state.search3_errors:
    st.warning(msg)

# ---------------------------
# RIGHT: results + Submit flow
# ---------------------------
with right:
    if st.session_state.search3_search_clicked:
        st.markdown("### Select your company from this possible results:")

        results = st.session_state.search3_results

        # Build radio labels that include platform explicitly
        # (Even though platform is chosen, this makes the UI self-explanatory.)
        option_labels = []
        id_by_label = {}
        for r in results:
            label = f"{r['platform']} — {r['name']} ({r['subtitle']})"
            option_labels.append(label)
            id_by_label[label] = r["id"]

        option_labels.append("None of those")

        selection = st.radio(
            "Select a result",
            options=option_labels,
            index=None,  # IMPORTANT: no default selection
            label_visibility="collapsed",
        )

        # If user chose “None of those”, show paste link input
        none_selected = (selection == "None of those")

        if none_selected:
            st.markdown('<div class="field-title">Paste the app link here:</div>', unsafe_allow_html=True)
            st.session_state.search3_pasted_link = st.text_input(
                "Paste the app link here",
                placeholder="e.g. https://ca.trustpilot.com/review/ziplines.com",
                label_visibility="collapsed",
                value=st.session_state.search3_pasted_link,
            )
            st.session_state.search3_selected_result = None
        else:
            # Store selected result id (if any)
            if selection is None:
                st.session_state.search3_selected_result = None
            else:
                st.session_state.search3_selected_result = id_by_label.get(selection)

        st.write("")
        if st.button("Submit", type="primary"):
            # Submit validation
            submit_errors = []

            picked_result = st.session_state.search3_selected_result is not None
            pasted_link_ok = bool(st.session_state.search3_pasted_link and st.session_state.search3_pasted_link.strip())

            if selection is None:
                submit_errors.append("Please select one of the results, or choose 'None of those'.")
            elif none_selected and not pasted_link_ok:
                submit_errors.append("Please paste the app/company link before submitting.")
            elif (not none_selected) and (not picked_result):
                submit_errors.append("Please select a valid result before submitting.")

            if submit_errors:
                for e in submit_errors:
                    st.warning(e)
                st.session_state.search3_submit_clicked = False
            else:
                st.session_state.search3_submit_clicked = True

# Placeholder output
if st.session_state.search3_submit_clicked:
    st.markdown("## Results")
