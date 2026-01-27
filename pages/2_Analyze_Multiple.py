import streamlit as st
import pandas as pd

from inference.sentiment import predict_single
from inference.topic import predict_topic_batch
from inference.llm_topic_summary import llm_topic_summary

st.set_page_config(page_title="Analyze Multiple Reviews", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .field-title { font-size: 20px; font-weight: 400; margin: 0 0 6px 0; }
      div[data-testid="stFileUploader"], div[data-testid="stRadio"] { margin-top: 6px; }
      div[data-testid="stFileUploader"] section {
        padding: 14px 16px !important;
        min-height: 100px !important;
        margin-bottom: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze Multiple Reviews")

clusters = [
    "Cluster 1 (AI-Charged Product/Service Providers)",
    "Cluster 2 (AI Development Facilitators)",
    "Cluster 3 (Data Analytics Providers)",
    "Cluster 4 (Deep Tech Researchers)",
]

if "multi_errors" not in st.session_state:
    st.session_state.multi_errors = []
if "multi_result" not in st.session_state:
    st.session_state.multi_result = None

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="field-title">Upload a CSV with one header and one review per row.</div>', unsafe_allow_html=True)
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
        options=clusters,
        index=None,
        label_visibility="collapsed",
    )

def _read_reviews_csv(file) -> list[str]:
    df = pd.read_csv(file, engine="python", sep=",", quotechar='"', escapechar="\\", on_bad_lines="skip")
    if df.shape[1] < 1:
        raise ValueError("CSV has no columns.")
    col = df.columns[0]
    reviews = df[col].dropna().astype(str).map(lambda s: s.strip())
    reviews = [r for r in reviews.tolist() if r]
    if not reviews:
        raise ValueError("No non-empty reviews found in the CSV.")
    return reviews

with left:
    if st.button("Submit", type="primary"):
        st.session_state.multi_errors = []
        st.session_state.multi_result = None

        if uploaded_file is None and cluster is None:
            st.session_state.multi_errors.append("Please upload a CSV and select your AI startup cluster before submitting.")
        elif uploaded_file is None:
            st.session_state.multi_errors.append("Please upload a CSV file before submitting.")
        elif cluster is None:
            st.session_state.multi_errors.append("Please select your AI startup cluster before submitting.")
        else:
            try:
                reviews = _read_reviews_csv(uploaded_file)

                # ---- Sentiment (simple loop for now) ----
                pos = neg = unc = 0
                for r in reviews:
                    label, _conf = predict_single(r)
                    if label.lower().startswith("pos"):
                        pos += 1
                    elif label.lower().startswith("neg"):
                        neg += 1
                    else:
                        unc += 1

                # ---- Topic modelling (batch) ----
                topic_batch = predict_topic_batch(reviews, cluster_label=cluster, top_k_words=10)

                # Build top topics payload (top 5, excluding -1)
                counts = topic_batch["counts"]
                items = [(tid, c) for tid, c in counts.items() if tid != -1]
                items.sort(key=lambda x: x[1], reverse=True)

                top_topics_payload = []
                for tid, c in items[:5]:
                    top_topics_payload.append({
                        "topic_id": tid,
                        "count": c,
                        "pct": c / len(reviews),
                        "keywords": topic_batch["keywords_by_topic"].get(tid, []),
                    })

                topic_summary_text = ""
                if top_topics_payload:
                    topic_summary_text = llm_topic_summary(cluster, top_topics_payload)

                st.session_state.multi_result = {
                    "cluster": cluster,
                    "total": len(reviews),
                    "sentiment": {"positive": pos, "negative": neg, "uncertain": unc},
                    "topics": topic_batch,  # includes counts + keywords_by_topic
                    "topic_summary": topic_summary_text,
                }

            except Exception as e:
                st.session_state.multi_errors.append(f"Error: {e}")

for msg in st.session_state.multi_errors:
    st.warning(msg)

if st.session_state.multi_result:
    res = st.session_state.multi_result
    total = res["total"]

    st.markdown("## Results")
    st.write(f"**Cluster:** {res['cluster']}")
    st.write(f"**Total reviews:** {total}")

    st.subheader("Sentiment Analysis")
    s = res["sentiment"]
    st.write(f"**Positive:** {s['positive']} ({s['positive']/total:.1%})")
    st.write(f"**Negative:** {s['negative']} ({s['negative']/total:.1%})")
    st.write(f"**Uncertain:** {s['uncertain']} ({s['uncertain']/total:.1%})")

    st.subheader("Topic Modelling")
    t = res["topics"]
    counts = t["counts"]

    # show top 5 topics by count (excluding -1 first)
    items = [(tid, c) for tid, c in counts.items() if tid != -1]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        st.write("No assigned topics (all reviews were outliers / unassigned).")
    else:
        st.write("Top topics (by count):")
        for tid, c in items[:5]:
            kws = t["keywords_by_topic"].get(tid, [])
            st.write(f"- **Topic {tid}** — {c} ({c/total:.1%})")
            if kws:
                st.caption("Keywords: " + ", ".join(kws))

    outliers = t.get("outlier_count", counts.get(-1, 0))
    st.write(f"**Unassigned / Outliers (-1):** {outliers} ({outliers/total:.1%})")

    if res.get("topic_summary"):
        st.markdown("**LLM Summary**")
        st.write(res["topic_summary"])

