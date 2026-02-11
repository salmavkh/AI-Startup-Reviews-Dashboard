import streamlit as st
import pandas as pd

from inference.sentiment import predict_single
from inference.emotion import predict_proba_single
from inference.topic import predict_topic_single
from inference.llm_topic_label import llm_label_topic

st.set_page_config(page_title="Analyze a Single Review", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
      .field-title {
        font-size: 20px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }
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

# --- Session state ---
if "single_errors" not in st.session_state:
    st.session_state.single_errors = []
if "single_result" not in st.session_state:
    st.session_state.single_result = None

clusters = [
    "Cluster 1 (AI-Charged Product/Service Providers)",
    "Cluster 2 (AI Development Facilitators)",
    "Cluster 3 (Data Analytics Providers)",
    "Cluster 4 (Deep Tech Researchers)",
]

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
        options=clusters,
        index=None,
        label_visibility="collapsed",
    )

with left:
    if st.button("Submit", type="primary"):
        st.session_state.single_errors = []
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
                text = review_text.strip()

                # 1) Sentiment
                sentiment_label, sentiment_conf = predict_single(text)

                # 2) Emotion
                emotion_dist = predict_proba_single(text)

                # 3) Topic (cluster-specific BERTopic)
                topic_res = predict_topic_single(text, cluster_label=cluster, top_k_words=10)

                # 4) LLM human label + explanation (Groq)
                llm_res = llm_label_topic(
                    cluster_label=cluster,
                    topic_id=topic_res["topic_id"],
                    keywords=topic_res["keywords"],
                    review_text=text[:600],  # keep short
                )

                st.session_state.single_result = {
                    "cluster": cluster,
                    "sentiment": {"label": sentiment_label, "confidence": sentiment_conf},
                    "emotion": emotion_dist,
                    "topic": topic_res,
                    "topic_llm": llm_res,
                }

            except Exception as e:
                st.session_state.single_errors.append(f"Error: {e}")

for msg in st.session_state.single_errors:
    st.warning(msg)

if st.session_state.single_result:
    res = st.session_state.single_result
    st.markdown("## Results")
    st.write(f"**Cluster:** {res['cluster']}")

    st.subheader("Sentiment")
    st.write(f"**Sentiment:** {res['sentiment']['label']}")
    st.write(f"**Confidence:** {res['sentiment']['confidence']:.3f}")
    st.progress(max(0.0, min(1.0, float(res["sentiment"]["confidence"]))))

    st.subheader("Emotion")
    dist = res.get("emotion") or {}
    if dist:
        top_items = sorted(((k, float(v)) for k, v in dist.items()), key=lambda kv: -kv[1])[:10]
        df = pd.DataFrame(top_items, columns=["emotion", "score"])
        st.bar_chart(df, x="emotion", y="score", use_container_width=True)

    st.subheader("Topic")
    topic = res["topic"]
    if topic["is_outlier"]:
        st.write("**Topic:** Unassigned / Misc (outlier)")
    else:
        st.write(f"**Topic ID:** {topic['topic_id']}")
        if topic["keywords"]:
            st.write("**Keywords:** " + ", ".join(topic["keywords"]))
        if topic["confidence"] is not None:
            st.write(f"**Topic confidence:** {topic['confidence']:.3f}")

    st.write("**LLM topic label / explanation:**")
    st.write(res["topic_llm"]["label"])
    if res["topic_llm"].get("explanation"):
        st.write(res["topic_llm"]["explanation"])
