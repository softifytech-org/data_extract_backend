"""Regression tests for record_splitter.py's anchor/boundary detection,
including the RECORD_ID_PREFIXED_NUMERIC shape added for the USER12 block-1
document family (a 2-character OCR-confusable prefix glued to 6 digits,
e.g. "S8360099"). The specific confusable variants tested below are the
real tokens dumped from Azure OCR output on 2 real USER12 images (40
real-world anchors; see patterns.py's RECORD_ID_PREFIXED_NUMERIC docstring)
- not invented examples.
"""
from __future__ import annotations

from app.services.extraction.record_splitter import (
    _find_anchor_words,
    choose_reference_boundaries,
)
from app.services.ocr.base import OCRWord


def _word(text: str, x: int, width: int, line_num: int, y: int, height: int = 13) -> OCRWord:
    return OCRWord(text=text, confidence=95.0, x=x, y=y, width=width, height=height,
                   page=1, line_num=line_num, word_num=0)


# --- RECORD_ID_PREFIXED_NUMERIC anchor recognition ---

REAL_OBSERVED_VARIANTS = ["SB360827", "$8360825", "58360828", "$6360832", "56360111", "S8360099"]


def test_prefixed_numeric_anchor_recognized_for_all_observed_variants():
    """Every real confusable variant actually observed in Azure OCR output
    must be recognized as an anchor when leftmost on its line."""
    for i, token in enumerate(REAL_OBSERVED_VARIANTS):
        words = [_word(token, x=0, width=70, line_num=i, y=i * 20)]
        found = _find_anchor_words(words)
        assert len(found) == 1, f"{token!r} was not recognized as an anchor"
        assert found[0][0] == token.upper()


def test_prefixed_numeric_anchor_requires_leftmost_of_line():
    """The same shape appearing anywhere but leftmost on its line must NOT
    be treated as a record anchor - this is the existing safeguard
    (already used by RECORD_ID_NUMERIC) extended to the new shape, so an
    incidental dollar figure mid-line is never mistaken for a record ID."""
    words = [
        _word("Total", x=0, width=40, line_num=0, y=0),
        _word("S8360099", x=50, width=70, line_num=0, y=0),
    ]
    assert _find_anchor_words(words) == []


def test_prefixed_numeric_anchor_rejects_five_digit_zip():
    """A bare 5-digit ZIP, even leftmost on its line, must not match - the
    pattern requires exactly 8 characters in the specific shape."""
    words = [_word("11725", x=0, width=50, line_num=0, y=0)]
    assert _find_anchor_words(words) == []


def test_prefixed_numeric_anchor_rejects_nine_digit_zip_plus_four():
    words = [_word("087654321", x=0, width=80, line_num=0, y=0)]
    assert _find_anchor_words(words) == []


def test_prefixed_numeric_anchor_rejects_real_dollar_amount_false_positive():
    """Confirmed real non-match from the calibration dump: a dollar amount
    with a decimal point does not match the 8-character fullmatch shape."""
    words = [_word("799.5670", x=0, width=80, line_num=0, y=0)]
    assert _find_anchor_words(words) == []


def test_choose_reference_boundaries_uses_prefixed_numeric_as_clean_hint():
    """choose_reference_boundaries must prefer a clean prefixed-numeric
    match as the record_id_hint (same treatment as the other two clean
    shapes), not fall back to some other noisy token in the cluster."""
    words = [
        _word("S8360099", x=0, width=70, line_num=0, y=0),
        _word("S8360100", x=0, width=70, line_num=5, y=19),
    ]
    boundaries = choose_reference_boundaries({"azure": words}, boundary_margin=2)
    assert len(boundaries) == 2
    assert boundaries[0].record_id_hint == "S8360099"
    assert boundaries[1].record_id_hint == "S8360100"


def test_choose_reference_boundaries_splits_two_adjacent_records():
    """Two prefixed-numeric anchors far enough apart vertically must
    produce two separate boundaries, each owning its own word."""
    words = [
        _word("S8360099", x=0, width=70, line_num=0, y=0),
        _word("Customer", x=0, width=70, line_num=1, y=19),
        _word("S8360100", x=0, width=70, line_num=2, y=38),
        _word("Customer", x=0, width=70, line_num=3, y=57),
    ]
    boundaries = choose_reference_boundaries({"azure": words}, boundary_margin=2)
    assert len(boundaries) == 2
    first_slice = [w for w in words if boundaries[0].y_start <= w.y < boundaries[0].y_end]
    second_slice = [w for w in words if boundaries[1].y_start <= w.y < boundaries[1].y_end]
    assert {w.text for w in first_slice} == {"S8360099", "Customer"}
    assert {w.text for w in second_slice} == {"S8360100", "Customer"}
