import pandas as pd
import streamlit as st

from helpers.search_ui_helpers import render_analysis_results
from inference.emotion import emotion_percentages, predict_proba_single
from inference.keywords import extract_keywords_batch
from inference.llm_topic_label import llm_label_topics_from_keywords
from inference.llm_topic_summary import llm_topic_summary
from inference.sentiment import predict_single as predict_sentiment_single
from inference.topic import discover_topics_batch
from inference.va import predict_va_single, summarize_va

st.set_page_config(page_title="Analyze Multiple Reviews", page_icon="📄", layout="wide")

TOPIC_EXAMPLES_PER_THEME = 2
KEYWORDS_PER_REVIEW = 5
KEYWORDS_OVERALL = 20
GENERAL_CLUSTER_LABEL = "General online review feedback"

st.markdown(
    """
    <style>
      .field-title {
        font-size: 16px;
        font-weight: 400;
        margin: 0 0 6px 0;
      }

      div[data-testid="stFileUploader"] {
        margin-top: 6px;
      }

      div[data-testid="stFileUploader"] section {
        padding: 14px 16px !important;
        min-height: 100px !important;
        margin-bottom: 6px;
      }

      div[data-testid="stButton"] button[kind="primary"] {
        background-color: #000000;
        border: 1px solid #000000;
        color: #ffffff;
      }

      div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #171717;
        border-color: #171717;
      }

      div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #c3c8d0;
        color: #2e3340;
      }

      div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #f7f9fc;
        border-color: #aeb6c2;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Back"):
    st.switch_page("app.py")

st.header("Analyze Multiple Reviews")

if "multi_errors" not in st.session_state:
    st.session_state.multi_errors = []
if "multi_result" not in st.session_state:
    st.session_state.multi_result = None


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


def _collect_topic_examples_for_payload(
    topic_res: dict,
    review_texts: list[str],
    per_topic_limit: int = TOPIC_EXAMPLES_PER_THEME,
) -> dict:
    examples_by_topic = {}
    topics = topic_res.get("topics") or []
    keywords_by_topic = topic_res.get("keywords_by_topic") or {}
    if not isinstance(topics, list) or len(topics) != len(review_texts):
        return examples_by_topic

    per_topic_limit = max(1, min(5, int(per_topic_limit or 2)))
    scored = {}

    for idx, topic_id in enumerate(topics):
        try:
            tid = int(topic_id)
        except Exception:
            continue
        if tid == -1:
            continue

        text = str(review_texts[idx] or "").strip()
        if not text:
            continue

        text_lower = text.lower()
        topic_keywords = [str(k).strip().lower() for k in keywords_by_topic.get(tid, [])[:10] if str(k).strip()]
        overlap = sum(1 for kw in topic_keywords if kw and kw in text_lower)
        if overlap <= 0:
            continue

        snippet = text[:180] + ("..." if len(text) > 180 else "")
        scored.setdefault(tid, []).append((overlap, snippet))

    for tid, rows in scored.items():
        rows.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        picked = []
        seen = set()
        for _overlap, snippet in rows:
            if snippet in seen:
                continue
            seen.add(snippet)
            picked.append(snippet)
            if len(picked) >= per_topic_limit:
                break
        if picked:
            examples_by_topic[tid] = picked

    return examples_by_topic


def _build_top_topics_payload(
    topic_res: dict,
    total_reviews: int,
    review_texts: list[str] | None = None,
    examples_per_topic: int = TOPIC_EXAMPLES_PER_THEME,
) -> list:
    if not topic_res or total_reviews <= 0:
        return []

    counts = topic_res.get("counts") or {}
    keywords_by_topic = topic_res.get("keywords_by_topic") or {}

    examples_by_topic = {}
    if review_texts:
        examples_by_topic = _collect_topic_examples_for_payload(
            topic_res=topic_res,
            review_texts=review_texts,
            per_topic_limit=examples_per_topic,
        )

    items = [(tid, c) for tid, c in counts.items() if tid != -1]
    items.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "topic_id": tid,
            "count": c,
            "pct": c / total_reviews,
            "keywords": keywords_by_topic.get(tid, []),
            "examples": examples_by_topic.get(tid, []),
        }
        for tid, c in items[:5]
    ]


def _topic_summary_or_empty(
    topic_res: dict,
    total_reviews: int,
    cluster_label: str,
    review_texts: list[str] | None = None,
    examples_per_topic: int = TOPIC_EXAMPLES_PER_THEME,
) -> str:
    top_topics_payload = _build_top_topics_payload(
        topic_res=topic_res,
        total_reviews=total_reviews,
        review_texts=review_texts,
        examples_per_topic=examples_per_topic,
    )
    if not top_topics_payload:
        return ""
    return llm_topic_summary(cluster_label, top_topics_payload)


st.markdown(
    '<div class="field-title">Upload a CSV with one header and one review per row.</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload a CSV with one header and one review per row.",
    type=["csv"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if st.button("Analyze", type="primary"):
    st.session_state.multi_errors = []
    st.session_state.multi_result = None

    if uploaded_file is None:
        st.session_state.multi_errors.append("Please upload a CSV file before submitting.")
    else:
        try:
            reviews = _read_reviews_csv(uploaded_file)
            cluster_label = GENERAL_CLUSTER_LABEL

            topic_res = None
            topic_summary_text = ""
            topic_labels = {}

            with st.spinner("Running topic model..."):
                try:
                    topic_res = discover_topics_batch(
                        reviews,
                        top_k_words=10,
                        min_topic_size=5,
                    )
                except Exception as exc:
                    st.warning(f"Topic model failed: {exc}")
                    topic_res = None

            if topic_res:
                with st.spinner("Generating high-level topic names..."):
                    try:
                        topic_labels = llm_label_topics_from_keywords(
                            cluster_label=cluster_label,
                            keywords_by_topic=topic_res.get("keywords_by_topic") or {},
                        )
                    except Exception as exc:
                        st.warning(f"Topic naming failed: {exc}")
                        topic_labels = {}
                    topic_res["topic_labels"] = topic_labels

                with st.spinner("Generating LLM topic summary..."):
                    try:
                        topic_summary_text = _topic_summary_or_empty(
                            topic_res=topic_res,
                            total_reviews=len(reviews),
                            cluster_label=cluster_label,
                            review_texts=reviews,
                            examples_per_topic=TOPIC_EXAMPLES_PER_THEME,
                        )
                    except Exception as exc:
                        st.warning(f"LLM summary failed: {exc}")
                        topic_summary_text = ""

            with st.spinner("Running sentiment..."):
                sentiments = [predict_sentiment_single(r or "") for r in reviews]

            with st.spinner("Extracting keywords..."):
                if KEYWORDS_PER_REVIEW <= 0 and KEYWORDS_OVERALL <= 0:
                    keyword_res = {
                        "per_review": [[] for _ in reviews],
                        "overall": [],
                    }
                else:
                    try:
                        keyword_res = extract_keywords_batch(
                            texts=reviews,
                            per_review_top_n=KEYWORDS_PER_REVIEW,
                            overall_top_n=KEYWORDS_OVERALL,
                        )
                    except Exception as exc:
                        st.warning(f"Keyword extraction failed: {exc}")
                        keyword_res = {
                            "per_review": [[] for _ in reviews],
                            "overall": [],
                        }

            with st.spinner("Running emotion analysis..."):
                va_by_review = [predict_va_single(r or "") for r in reviews]
                va_overall = summarize_va(va_by_review)
                discrete_overall = emotion_percentages(reviews, method="prob")
                discrete_by_review = [predict_proba_single(r or "") for r in reviews]

            pos = neg = unc = 0
            for label, _conf in sentiments:
                llow = str(label or "").lower()
                if llow.startswith("pos"):
                    pos += 1
                elif llow.startswith("neg"):
                    neg += 1
                else:
                    unc += 1

            review_rows = [
                {
                    "id": f"csv_{i + 1}",
                    "title": f"Review {i + 1}",
                    "content": r,
                    "rating": None,
                    "date": "",
                    "platform": "Uploaded CSV",
                    "reviewer": "",
                }
                for i, r in enumerate(reviews)
            ]

            st.session_state.multi_result = {
                "total": len(reviews),
                "sentiment_counts": {
                    "positive": pos,
                    "negative": neg,
                    "uncertain": unc,
                },
                "analysis": {
                    "topic": topic_res,
                    "topic_summary": topic_summary_text,
                    "keywords": keyword_res,
                    "sentiment": sentiments,
                    "emotion": {
                        "va": va_overall,
                        "discrete": discrete_overall,
                    },
                    "emotion_by_review": {
                        "va": va_by_review,
                        "discrete": discrete_by_review,
                    },
                    "reviews": review_rows,
                },
            }

        except Exception as exc:
            st.session_state.multi_errors.append(f"Error: {exc}")

for msg in st.session_state.multi_errors:
    st.warning(msg)

if st.session_state.multi_result:
    res = st.session_state.multi_result
    render_analysis_results(
        res["analysis"],
        show_section_heading=False,
    )
