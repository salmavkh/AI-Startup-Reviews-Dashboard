"""Validation logic for search page."""

from fetchers.g2 import extract_slug_from_g2_url
from fetchers.google_play import extract_package_from_google_play_url
from fetchers.ios import extract_app_id_from_ios_url
from fetchers.trustpilot import extract_slug_from_trustpilot_url


def validate_search_inputs(query: str, platform: str, num_reviews: int, cluster: str | None = None) -> list:
    """Validate initial search form inputs. Returns list of error messages."""
    errors = []
    
    if not (query and query.strip()):
        errors.append("Please enter a company/app name before searching.")
    if platform is None:
        errors.append("Please select a platform before searching.")
    if not (isinstance(num_reviews, int) and 1 <= num_reviews <= 100):
        errors.append("Please enter a number of reviews between 1 and 100.")
    return errors


def validate_submit_inputs(picked_result: bool, pasted_link: str) -> list:
    """Validate submit form inputs. Returns list of error messages."""
    errors = []
    
    if not picked_result and not (pasted_link and pasted_link.strip()):
        errors.append("Please select one of the results, or paste the app/company link.")
    
    return errors


def parse_pasted_link(platform: str, pasted: str) -> tuple:
    """Parse pasted link and return (confirmed_dict, error_list)."""
    errors = []
    confirmed = None
    
    pasted = (pasted or "").strip()
    
    if platform == "G2":
        slug = extract_slug_from_g2_url(pasted)
        if not slug:
            errors.append("Couldn't parse a G2 product slug from the pasted link.")
        else:
            confirmed = {"platform": "G2", "name": None, "g2_slug": slug, "source": "paste"}
    
    elif platform == "Google Play Store":
        pkg = extract_package_from_google_play_url(pasted)
        if not pkg:
            errors.append("Couldn't parse a Google Play package name from the pasted link.")
        else:
            confirmed = {"platform": "Google Play Store", "name": None, "package": pkg, "source": "paste"}
    
    elif platform == "iOS App Store":
        appid = extract_app_id_from_ios_url(pasted)
        if not appid:
            errors.append("Couldn't parse an App Store id from the pasted link.")
        else:
            confirmed = {"platform": "iOS App Store", "name": None, "app_id": appid, "source": "paste"}
    
    elif platform == "Trustpilot":
        tp = extract_slug_from_trustpilot_url(pasted)
        if not tp:
            errors.append("Couldn't parse a Trustpilot company slug from the pasted link.")
        else:
            confirmed = {"platform": "Trustpilot", "name": None, "tp_slug": tp, "source": "paste"}
    
    return confirmed, errors
