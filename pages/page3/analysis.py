"""Analysis helpers for Search Online page."""

from helpers.review_analysis_pipeline import reviews_signature, run_review_analysis

from pages.page3.constants import (
    KEYWORDS_OVERALL,
    KEYWORDS_PER_REVIEW,
    TOPIC_EXAMPLES_PER_THEME,
)


def run_page_analysis(rows: list[dict]) -> dict:
    return run_review_analysis(
        rows,
        keywords_per_review=KEYWORDS_PER_REVIEW,
        keywords_overall=KEYWORDS_OVERALL,
        topic_examples_per_theme=TOPIC_EXAMPLES_PER_THEME,
        include_signature=True,
    )


def is_analysis_stale(analysis: dict | None, fetched_reviews: list[dict] | None) -> bool:
    if not analysis:
        return False
    current_sig = reviews_signature(fetched_reviews or [])
    analysis_sig = str(analysis.get("reviews_signature") or "")
    return (not analysis_sig) or (analysis_sig != current_sig)
