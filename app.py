# streamlit run app.py

import streamlit as st

st.set_page_config(
    page_title="AI Review Insights",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        background-color: #f3f3f3;
        padding: 48px 24px;
        border-radius: 14px;
        text-align: center;
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HERO ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("## Reveal sentiment, emotion, and key topics shaping your AI startup.")
st.caption("See more information")
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

btn1, btn2, btn3 = st.columns(3)

with btn1:
    if st.button("Analyze a Single Review", use_container_width=True):
        st.switch_page("pages/1_Analyze_Single.py")

with btn2:
    if st.button("Analyze Multiple Reviews", use_container_width=True):
        st.switch_page("pages/2_Analyze_Multiple.py")

with btn3:
    if st.button("Search Online Reviews", use_container_width=True):
        st.switch_page("pages/3_Search_Online.py")

st.write("")
st.write("")

# ---------- MAIN CONTENT ----------
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown("### Models We Use")
    st.markdown(
        """
        **Trained on 33k AI startup reviews**, we selected the best-performing models:

        - **RoBERTa** — sentiment classification  
        - **DistilBERT** — emotion detection  
        - **BERTopic** — topic modeling  

        *(Metrics and model details will be shown here later.)*
        """
    )

    st.write("")

    st.markdown("### AI Startup Cluster")
    st.markdown(
        """
        In the analysis, you will be asked to select which AI startup cluster best
        represents your product.

        There are **4 AI startup clusters**:
        - Cluster 1
        - Cluster 2
        - Cluster 3
        - Cluster 4

        *(Cluster definitions will be added later.)*
        """
    )

with right:
    st.markdown('<div class="card" style="height:240px; text-align:center;">[ image ]</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card" style="height:240px; text-align:center;">[ image ]</div>', unsafe_allow_html=True)

st.write("")
st.divider()
st.caption("© 2026 by Salma — Honours Thesis Prototype")
