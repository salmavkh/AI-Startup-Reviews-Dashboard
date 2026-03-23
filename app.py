# streamlit run app.py

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env (restart required after editing .env)
load_dotenv()

st.set_page_config(
    page_title="AI Review Insights",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --hero-bg: #d4deea;
        --footer-bg: #d4deea;
        --text: #111111;
        --button: #000000;
        --button-hover: #1a1a1a;
    }

    .stApp {
        background-color: #ffffff;
        color: var(--text);
    }

    .block-container {
        max-width: 100%;
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
    }

    .hero-title {
        text-align: center;
        font-size: 32px;
        line-height: 1.2;
        font-weight: 700;
        max-width: 760px;
        margin: 0 auto 36px auto;
    }

    .hero-shell-anchor,
    .hero-inner-anchor,
    .content-anchor {
        display: none;
    }

    /* Full-bleed hero background */
    div[data-testid="stVerticalBlock"]:has(.hero-shell-anchor):not(:has(div[data-testid="stVerticalBlock"] .hero-shell-anchor)) {
        background: var(--hero-bg);
        border-radius: 0;
        width: 100%;
        margin-left: 0;
        margin-right: 0;
        margin-top: 26px;
        margin-bottom: 12px;
        padding-top: 72px;
        padding-bottom: 52px;
        padding-left: 0;
        padding-right: 0;
    }

    /* Centered hero content (title + buttons) */
    div[data-testid="stVerticalBlock"]:has(.hero-inner-anchor):not(:has(div[data-testid="stVerticalBlock"] .hero-inner-anchor)) {
        max-width: 960px;
        margin: 0 auto;
        padding: 0 24px;
    }

    /* Keep regular page content centered while allowing full-width hero background */
    div[data-testid="stVerticalBlock"]:has(.content-anchor):not(:has(div[data-testid="stVerticalBlock"] .content-anchor)) {
        max-width: 960px;
        margin: 0 auto;
        padding: 0 24px;
    }

    .section-spacer {
        height: 10px;
    }

    .body-header {
        font-size: 24px;
        margin-bottom: 8px;
    }

    .body-copy {
        font-size: 16px;
        line-height: 1.4;
    }

    .body-copy ul,
    .body-copy ol {
        margin-top: 10px;
    }

    .body-copy li {
        margin-bottom: 8px;
    }

    .reference-link {
        color: var(--text);
        text-decoration: underline;
    }

    .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: var(--text);
        border: 1px solid rgba(0, 0, 0, 0.45);
        border-radius: 8px;
        min-height: 42px;
        font-size: 16px;
        font-weight: 500;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: rgba(0, 0, 0, 0.7);
        color: var(--text);
    }

    .stButton > button[kind="primary"] {
        background-color: var(--button);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        min-height: 42px;
        font-size: 16px;
        font-weight: 500;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: var(--button-hover);
    }

    .footer-wrap {
        width: 100vw;
        margin-left: calc(50% - 50vw);
        margin-right: calc(50% - 50vw);
        background: var(--footer-bg);
        margin-top: 84px;
    }

    .footer-content {
        max-width: 960px;
        margin: 0 auto;
        text-align: center;
        padding: 18px 16px 22px 16px;
        font-size: 14px;
    }

    .footer-content a {
        color: var(--text);
        font-weight: 700;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="hero-shell-anchor"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="hero-inner-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">Reveal sentiment, emotion, and key topics shaping your AI startup.</div>',
            unsafe_allow_html=True,
        )

        btn1, btn2, btn3 = st.columns(3, gap="small")

        with btn1:
            if st.button("Analyze a Single Review", type="secondary", use_container_width=True):
                st.switch_page("pages/1_Analyze_Single.py")

        with btn2:
            if st.button("Analyze Multiple Reviews", type="secondary", use_container_width=True):
                st.switch_page("pages/2_Analyze_Multiple.py")

        with btn3:
            if st.button("Search Online Reviews", type="primary", use_container_width=True):
                st.switch_page("pages/3_Search_Online.py")

with st.container():
    st.markdown('<div class="content-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    st.markdown('<div class="body-header"><strong>Models We Use</strong></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="body-copy">
        Using a dataset of 33,000 AI startup reviews, we evaluated multiple approaches and selected the best-performing models for each task.
        <ul>
            <li>For sentiment analysis, we use RoBERTa to classify reviews as positive or negative. This model achieved an F1-score of 97% and an accuracy of 97.2%.</li>
            <li>For emotion analysis, we use DistilBERT in two ways. One model estimates Valence (R<sup>2</sup> = 0.8237, MAE = 0.1197) and Arousal (R<sup>2</sup> = 0.6291, MAE = 0.1048). Another model predicts discrete 28 emotions and identifies the top 10 emotions with their intensity in each review (R<sup>2</sup> = 0.7480, MAE = 0.0092).</li>
            <li>For topic modeling, we use BERTopic. The model achieved a coherence score of 0.840 (how semantically consistent each topic is), topic diversity of 0.428 (how distinct topics are from one another), and an outlier rate of 14.8% (the share of reviews not confidently assigned to a topic).</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="body-header"><strong>Dataset</strong></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="body-copy">
        Our dataset contains AI startup reviews grouped into four business-model clusters adapted from Weber et al. (2022):
        <ol>
            <li>Cluster 1 (AI-Charged Product/Service Providers): embed pre-trained AI models into a product or service to solve a specific task, with offerings are often relatively standardized and commonly sold to business customers.</li>
            <li>Cluster 2 (AI Development Facilitators): support others in building AI by offering development infrastructure such as APIs, SDKs, platforms, and no-code tools.</li>
            <li>Cluster 3 (Data Analytics Providers): create value by integrating and analyzing data to deliver insights, monitoring, anomaly detection, and predictive decision support.</li>
            <li>Cluster 4 (Deep Tech Researchers): focus on advancing frontier AI technologies and foundational capabilities, often targeting specialized technical applications rather than mass-market standardized products.</li>
        </ol>
        Reference:<br>
        Weber, M., Beutter, M., Weking, J., et al. (2022). AI Startup Business Models. Business &amp; Information Systems Engineering, 64, 91-109.
        <a class="reference-link" href="https://doi.org/10.1007/s12599-021-00732-w" target="_blank">https://doi.org/10.1007/s12599-021-00732-w</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer-wrap">
        <div class="footer-content">
            &copy; 2026 by <a href="https://www.linkedin.com/in/salmavkh/" target="_blank">salma</a> &#9829; as part of her undergraduate honours thesis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
