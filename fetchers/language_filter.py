"""Lightweight English-language filtering for fetched review text."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

try:
    from langdetect import LangDetectException, detect as _detect_lang, detect_langs as _detect_langs
except Exception:  # pragma: no cover - optional dependency
    LangDetectException = Exception
    _detect_lang = None
    _detect_langs = None


_EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my", "not",
    "of", "on", "or", "our", "so", "that", "the", "their", "them", "there", "they", "this",
    "to", "too", "very", "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your",
}

_EN_HINT_WORDS = {
    "app", "great", "good", "bad", "slow", "fast", "support", "review", "service", "product",
    "easy", "hard", "issue", "problem", "works", "working", "update", "price", "value",
}

_NON_EN_STOPWORDS = {
    # Spanish
    "de", "la", "que", "el", "en", "los", "las", "por", "para", "una", "un", "muy", "pero", "con",
    # French
    "le", "les", "des", "pas", "est", "sur", "dans", "pour", "avec", "une", "tres", "tout",
    # German
    "und", "der", "die", "das", "nicht", "ist", "mit", "ein", "eine", "sehr", "aber",
    # Portuguese
    "não", "nao", "uma", "bom", "boa", "muito", "mais", "como", "com", "sem",
    # Indonesian/Malay
    "dan", "yang", "ini", "itu", "saya", "tidak", "ada", "untuk", "dengan",
}


def _clean_text(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            latin += 1
    return latin / len(letters)


def _english_heuristic(text: str) -> bool:
    s = _clean_text(text)
    if not s:
        return False

    # Quickly reject clearly non-Latin text.
    if _latin_ratio(s) < 0.75:
        return False

    tokens = re.findall(r"[a-z']+", s.lower())
    if not tokens:
        return False

    if len(tokens) <= 2:
        # Short reviews are ambiguous; be strict.
        if any(tok in _EN_HINT_WORDS for tok in tokens) or all(tok in _EN_STOPWORDS for tok in tokens):
            return True
        return False

    stop_hits = sum(1 for tok in tokens if tok in _EN_STOPWORDS)
    hint_hits = sum(1 for tok in tokens if tok in _EN_HINT_WORDS)
    non_en_hits = sum(1 for tok in tokens if tok in _NON_EN_STOPWORDS)

    if non_en_hits >= 2 and non_en_hits >= stop_hits:
        return False

    if stop_hits >= 2:
        return True
    if stop_hits >= 1 and (stop_hits / len(tokens)) >= 0.10 and non_en_hits == 0:
        return True
    if hint_hits >= 2 and non_en_hits == 0:
        return True
    return False


def _strong_non_english_signal(text: str) -> bool:
    """
    Conservative rejection rule for obvious non-English text.
    Used to override noisy language hints from data sources.
    """
    s = _clean_text(text)
    if not s:
        return False

    tokens = re.findall(r"[a-z']+", s.lower())
    if len(tokens) < 3:
        return False

    non_en_hits = sum(1 for tok in tokens if tok in _NON_EN_STOPWORDS)
    en_hits = sum(1 for tok in tokens if tok in _EN_STOPWORDS)

    # Clearly non-English when non-English stopwords dominate.
    if non_en_hits >= 3 and non_en_hits >= (en_hits + 1):
        return True
    if non_en_hits >= 2 and (non_en_hits / max(1, len(tokens))) >= 0.25 and en_hits == 0:
        return True
    return False


def is_english_review(title: str | None, content: str | None, language_hint: str | None = None) -> bool:
    """Return True when a review appears to be English."""
    lang = str(language_hint or "").strip().lower()
    hinted_english = False
    if lang:
        # Common normalized forms.
        if lang in {"en", "eng", "en-us", "en-gb", "en_au", "en-ca", "english"} or lang.startswith("en-"):
            hinted_english = True
        else:
            return False

    text = f"{title or ''} {content or ''}".strip()
    if not text:
        return False

    cleaned = _clean_text(text)
    if len(cleaned) < 6:
        return False

    # Reject strong non-English signals even if source hints "en".
    if _strong_non_english_signal(cleaned):
        return False

    # Prefer explicit language detector when available for longer content.
    if _detect_langs is not None and len(cleaned) >= 20:
        try:
            langs = _detect_langs(cleaned)
            if langs:
                top = langs[0]
                if str(top.lang).lower() == "en" and float(top.prob) >= 0.65:
                    return True
                if str(top.lang).lower() != "en" and float(top.prob) >= 0.65:
                    return False
        except LangDetectException:
            pass
        except Exception:
            pass

    if _detect_lang is not None and len(cleaned) >= 25 and not hinted_english:
        try:
            detected = _detect_lang(cleaned)
            if str(detected).lower() == "en":
                return True
            return False
        except LangDetectException:
            pass
        except Exception:
            pass

    heuristic_en = _english_heuristic(cleaned)
    if hinted_english:
        # English hint must still look English heuristically.
        return heuristic_en
    return heuristic_en


def filter_english_reviews(reviews: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    """Filter a review list to English-only entries."""
    out: List[Dict[str, Any]] = []
    target = None if limit is None else max(0, int(limit))
    for review in reviews or []:
        if not isinstance(review, dict):
            continue
        title = review.get("title")
        content = review.get("content")
        language_hint = review.get("language") or review.get("lang") or review.get("locale")
        if not is_english_review(title=title, content=content, language_hint=language_hint):
            continue
        out.append(review)
        if target is not None and len(out) >= target:
            break
    return out
