"""Lightweight English-language filtering for fetched review text."""

from __future__ import annotations

import re
import unicodedata

try:
    from langdetect import LangDetectException, detect as _detect_lang
except Exception:  # pragma: no cover - optional dependency
    LangDetectException = Exception
    _detect_lang = None


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
        return all(ord(ch) < 128 for ch in s)

    if len(tokens) <= 2:
        if any(tok in _EN_HINT_WORDS for tok in tokens):
            return True
        return all(ord(ch) < 128 for ch in s)

    stop_hits = sum(1 for tok in tokens if tok in _EN_STOPWORDS)
    hint_hits = sum(1 for tok in tokens if tok in _EN_HINT_WORDS)

    if stop_hits >= 2:
        return True
    if stop_hits >= 1 and (stop_hits / len(tokens)) >= 0.10:
        return True
    if hint_hits >= 2:
        return True
    return False


def is_english_review(title: str | None, content: str | None, language_hint: str | None = None) -> bool:
    """Return True when a review appears to be English."""
    lang = str(language_hint or "").strip().lower()
    if lang:
        # Common normalized forms.
        if lang in {"en", "eng", "en-us", "en-gb", "en_au", "en-ca", "english"}:
            return True
        if lang.startswith("en-"):
            return True
        return False

    text = f"{title or ''} {content or ''}".strip()
    if not text:
        return False

    # Prefer explicit language detector when available for longer content.
    if _detect_lang is not None and len(text) >= 25:
        try:
            detected = _detect_lang(text)
            if str(detected).lower() == "en":
                return True
            return False
        except LangDetectException:
            pass
        except Exception:
            pass

    return _english_heuristic(text)
