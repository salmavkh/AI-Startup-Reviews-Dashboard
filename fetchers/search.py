"""
Search helpers for the dashboard — unified interface re-exporting from platform-specific modules.

For backward compatibility, import from platform-specific modules here.
"""

# Re-export Google Play functions
from .google_play import search_google_play, extract_package_from_google_play_url, fetch_google_play_reviews

# Re-export iOS functions
from .ios import search_ios, extract_app_id_from_ios_url, fetch_ios_reviews

# Re-export Trustpilot functions
from .trustpilot import search_trustpilot, extract_slug_from_trustpilot_url, fetch_trustpilot_reviews

# Re-export G2 functions
from .g2 import search_g2, extract_slug_from_g2_url, fetch_g2_reviews

__all__ = [
    # Google Play
    "search_google_play",
    "extract_package_from_google_play_url",
    "fetch_google_play_reviews",
    # iOS
    "search_ios",
    "extract_app_id_from_ios_url",
    "fetch_ios_reviews",
    # Trustpilot
    "search_trustpilot",
    "extract_slug_from_trustpilot_url",
    "fetch_trustpilot_reviews",
    # G2
    "search_g2",
    "extract_slug_from_g2_url",
    "fetch_g2_reviews",
]
