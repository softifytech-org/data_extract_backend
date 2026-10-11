"""Low-level helpers shared by field extractors: reading-order text
reconstruction (with a mapping back to source OCRWords for bounding boxes
and confidence), plus value normalization (dates, currency, enums).
"""
from __future__ import annotations

import difflib
from datetime import date

from app.models.document import BoundingBox
from app.services.ocr.base import OCRWord

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

WordSpan = tuple[int, int, OCRWord]


def build_indexed_text(words: list[OCRWord]) -> tuple[str, list[WordSpan]]:
    """Reconstruct reading-order text from OCR words, tracking each word's
    character offsets so regex matches can be traced back to source words."""
    ordered = sorted(words, key=lambda w: (w.page, w.line_num, w.x))
    parts: list[str] = []
    spans: list[WordSpan] = []
    pos = 0
    prev_line: tuple[int, int] | None = None
    for w in ordered:
        key = (w.page, w.line_num)
        if prev_line is not None:
            sep = "\n" if key != prev_line else " "
            parts.append(sep)
            pos += 1
        start = pos
        parts.append(w.text)
        pos += len(w.text)
        spans.append((start, pos, w))
        prev_line = key
    return "".join(parts), spans


def words_in_span(spans: list[WordSpan], start: int, end: int) -> list[OCRWord]:
    return [w for s, e, w in spans if s < end and e > start]


def bounding_box_for_words(words: list[OCRWord]) -> BoundingBox | None:
    if not words:
        return None
    min_x = min(w.x for w in words)
    min_y = min(w.y for w in words)
    max_right = max(w.right for w in words)
    max_bottom = max(w.bottom for w in words)
    return BoundingBox(
        page=words[0].page, x=min_x, y=min_y, width=max_right - min_x, height=max_bottom - min_y
    )


def average_confidence(words: list[OCRWord]) -> float:
    if not words:
        return 0.0
    return sum(w.confidence for w in words) / len(words)


def normalize_date_long(month_name: str, day: str, year: str) -> str | None:
    month = MONTHS.get(month_name.lower())
    if not month:
        return None
    try:
        return date(int(year), month, int(day)).isoformat()
    except ValueError:
        return None


def normalize_date_numeric(month: str, day: str, year: str) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def normalize_currency(raw: str) -> str | None:
    # Keeps the "$" - BN_WorkFlow.pdf's own valid examples ("$12", "$50.99")
    # require it; stripping it would silently change the source
    # representation rather than just cleaning up whitespace/commas.
    cleaned = raw.replace("$", "").replace(",", "").strip()
    try:
        return f"${float(cleaned):.2f}"
    except ValueError:
        return None


_DIGIT_CONFUSABLE_MAP = {
    "O": "0", "o": "0", "Q": "0", "D": "0", "@": "0",
    "I": "1", "L": "1", "l": "1", "|": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
}


def normalize_confusable_digits(raw: str) -> str:
    """Map common OCR digit-confusable characters (O/0, @/0, I/1, S/5, B/8...)
    back to digits. Only ever applied to substrings a regex has already
    anchored into a known digit position - never to free text - so it can't
    silently corrupt unrelated words."""
    return "".join(_DIGIT_CONFUSABLE_MAP.get(ch, ch) for ch in raw)


def closest_enum_match(raw: str, vocabulary: set[str], cutoff: float = 0.75) -> str | None:
    """Match OCR output against a closed vocabulary, tolerating the kind of
    single/double-character confusion Tesseract commonly produces (O/0, I/1...).
    Returns None rather than guessing when nothing is close enough."""
    candidate = " ".join(raw.strip().upper().split())
    if candidate in vocabulary:
        return candidate
    matches = difflib.get_close_matches(candidate, vocabulary, n=1, cutoff=cutoff)
    return matches[0] if matches else None
