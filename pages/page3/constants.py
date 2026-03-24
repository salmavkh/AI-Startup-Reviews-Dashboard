"""Constants for Search Online page."""

TOPIC_EXAMPLES_PER_THEME = 2
KEYWORDS_PER_REVIEW = 5
KEYWORDS_OVERALL = 20
DIRECT_LINK_PLATFORMS = {"G2", "Trustpilot"}

DEFAULTS = {
    "search3_errors": [],
    "search3_search_clicked": False,
    "search3_submit_clicked": False,
    "search3_results": [],
    "search3_selected_result": None,
    "search3_none_selected": False,
    "search3_pasted_link": "",
    "search3_num_reviews": 20,
    "search3_fetched_reviews": [],
    "search3_confirmed_company": None,
    "search3_fetched_for": {},
    "search3_preview_analysis": None,
    "search3_review_carousel_start": 0,
    "search3_query_input": "",
    "search3_platform": None,
    "search3_num_reviews_input": 20,
}

PAGE_CSS = """
<style>
  .field-title {
    font-size: 16px;
    font-weight: 400;
    margin: 0 0 6px 0;
  }

  .section-title {
    font-size: 20px;
    font-weight: 500;
    margin: 0 0 10px 0;
  }

  .result-option-card {
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    background: #ffffff;
  }

  .result-option-card.selected {
    border: 2px solid #000000;
    padding: 9px;
  }

  .result-option-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .result-option-logo {
    width: 40px;
    height: 40px;
    flex: 0 0 40px;
  }

  .result-option-name {
    font-size: 14px;
    font-weight: 500;
    line-height: 1.2;
  }

  .result-option-subtitle {
    font-size: 12px;
    color: #6f6f6f;
    margin-top: 2px;
  }

  .none-option {
    font-size: 16px;
    line-height: 1.2;
    margin-top: 2px;
  }

  .company-header {
    font-size: 22px;
    font-weight: 700;
    margin: 2px 0 8px 0;
  }

  .fetched-count {
    text-align: right;
    font-size: 16px;
    margin-top: 6px;
  }

  .preview-card {
    border: 1px solid #d6d6d6;
    border-radius: 12px;
    min-height: 130px;
    padding: 14px 16px 14px 40px;
    background: #f7f8fa;
    margin-bottom: 14px;
    position: relative;
  }

  .preview-card::before {
    content: "“";
    position: absolute;
    left: 12px;
    top: 8px;
    font-size: 26px;
    line-height: 1;
    color: #9aa3af;
  }

  .preview-card-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 6px;
    color: #2f3340;
  }

  .preview-card-content {
    font-size: 15px;
    line-height: 1.45;
    color: #2a2f3a;
  }

  .divider-line {
    border-top: 1px solid #d9d9d9;
    margin: 12px 0 22px 0;
  }

  .vertical-divider {
    width: 1px;
    background: #d9d9d9;
    min-height: 980px;
    margin: 0 auto;
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
"""
