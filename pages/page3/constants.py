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
  :root {
    --search3-card-h: 76px;
    --search3-logo-sz: 32px;
    --search3-left-pad: 56px;
    --search3-gap-y: -10px;
  }

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

  /* Dot-only selectors: no button border/background box */
  div[class*="st-key-search3_pick_dot_"],
  div[class*="st-key-search3_pick_none"] {
    height: var(--search3-card-h);
    margin: 0 0 var(--search3-gap-y) 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  div[class*="st-key-search3_pick_dot_"] button,
  div[class*="st-key-search3_pick_none"] button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 0 !important;
    color: #2f3442 !important;
    font-size: 20px !important;
    line-height: 1 !important;
  }

  div[class*="st-key-search3_pick_dot_"] button:hover,
  div[class*="st-key-search3_pick_dot_"] button:focus,
  div[class*="st-key-search3_pick_none"] button:hover,
  div[class*="st-key-search3_pick_none"] button:focus {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
  }

  /* Clickable result cards (single click, uniform size). */
  div[class*="st-key-search3_pick_card_"],
  div[class*="st-key-search3_pick_none_card"] {
    margin: 0 0 var(--search3-gap-y) 0 !important;
  }

  div[class*="st-key-search3_pick_dot_"] > div[data-testid="stButton"],
  div[class*="st-key-search3_pick_none"] > div[data-testid="stButton"],
  div[class*="st-key-search3_pick_card_"] > div[data-testid="stButton"],
  div[class*="st-key-search3_pick_none_card"] > div[data-testid="stButton"] {
    margin: 0 !important;
    padding: 0 !important;
  }

  div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-search3_pick_card_"]),
  div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-search3_pick_none_card"]) {
    margin: 0 !important;
    padding: 0 !important;
  }

  div[class*="st-key-search3_pick_card_"] button,
  div[class*="st-key-search3_pick_none_card"] button {
    text-align: left !important;
    white-space: pre-line !important;
    line-height: 1.1 !important;
    font-size: 12px !important;
    min-height: var(--search3-card-h) !important;
    height: var(--search3-card-h) !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
    background: #ffffff !important;
    color: #2f3442 !important;
    box-shadow: none !important;
  }

  div[class*="st-key-search3_pick_none_card"] button {
    position: relative !important;
    padding-left: var(--search3-left-pad) !important;
  }

  div[class*="st-key-search3_pick_none_card"] button::before {
    content: "";
    position: absolute;
    left: 12px;
    top: 50%;
    width: var(--search3-logo-sz);
    height: var(--search3-logo-sz);
    border-radius: 8px;
    transform: translateY(-50%);
    background: transparent;
  }

  div[class*="st-key-search3_pick_card_"] button[kind="primary"],
  div[class*="st-key-search3_pick_none_card"] button[kind="primary"] {
    border: 2px solid #000000 !important;
    background: #ffffff !important;
    color: #2f3442 !important;
  }

  div[class*="st-key-search3_pick_card_"] button:hover,
  div[class*="st-key-search3_pick_none_card"] button:hover {
    border-color: rgba(0, 0, 0, 0.24) !important;
    background: #ffffff !important;
  }

  div[class*="st-key-search3_pick_card_"] button[kind="primary"]:hover,
  div[class*="st-key-search3_pick_none_card"] button[kind="primary"]:hover {
    border-color: #000000 !important;
    background: #ffffff !important;
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
