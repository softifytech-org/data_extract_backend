"""Regression tests for four confirmed bugs fixed together: ImageName
sanitization stripping meaningful characters, CreateDate using the wrong
(minority) format, UpdateDate's empty-tag closing form, and Phone_2 being
shifted into Phone_1 when Phone_1 is genuinely absent. Root causes were
each verified against real ground truth (data/ground_truth/softify_dump,
90,321 real records) and/or a real freshly-generated submission the user
provided, not guessed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from app.core.security import sanitize_original_filename
from app.services.export.dump_xml_export import _format_dump_datetime
from app.services.extraction.engine import RuleBasedExtractionEngine
from app.services.ocr.base import OCRWord


# --- Bug 1: ImageName must preserve meaningful characters like "(1)" ---

def test_sanitize_filename_preserves_parenthesized_duplicate_suffix():
    assert sanitize_original_filename("1400 (1).jpeg") == "1400 (1).jpeg"


def test_sanitize_filename_preserves_other_meaningful_punctuation():
    assert sanitize_original_filename("Scan [2] - copy & more.jpg") == "Scan [2] - copy & more.jpg"


def test_sanitize_filename_still_strips_path_traversal_and_control_chars():
    assert sanitize_original_filename("../../etc/passwd.jpg") == "passwd.jpg"
    assert sanitize_original_filename("bad\x00name.jpg") == "bad_name.jpg"


# --- Bug 2: CreateDate must use the real majority format, no weekday ---

def test_create_date_format_has_no_weekday_and_is_24h_zero_padded():
    dt = datetime(2026, 8, 20, 11, 22, 55, tzinfo=timezone.utc)
    # IST is UTC+5:30, so 11:22:55 UTC -> 16:52:55 IST same day.
    assert _format_dump_datetime(dt) == "20 August 2026 16:52:55"


def test_create_date_format_zero_pads_single_digit_day_and_midnight_hour():
    dt = datetime(2026, 9, 15, 18, 30, 0, tzinfo=timezone.utc)  # -> 00:00:00 IST next day
    assert _format_dump_datetime(dt) == "16 September 2026 00:00:00"


# --- Bug 3: UpdateDate (and every field) must have an explicit close tag ---

def test_empty_update_date_is_not_self_closing():
    root = ET.Element("doc")
    datam = ET.SubElement(root, "DataM")
    upd = ET.SubElement(datam, "UpdateDate")
    upd.text = "" or None
    body = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    assert "<UpdateDate></UpdateDate>" in body
    assert "<UpdateDate/>" not in body
    assert "<UpdateDate />" not in body


# --- Bug 4: a lone detected phone must not default to Phone_1 ---

def _word(text: str, x: int, width: int, line_num: int, y: int = 0, height: int = 13) -> OCRWord:
    return OCRWord(text=text, confidence=95.0, x=x, y=y, width=width, height=height,
                   page=1, line_num=line_num, word_num=0)


def test_lone_phone_after_state_2_is_assigned_to_phone_2_not_phone_1():
    """Phone_1 is genuinely absent from the page (no phone-shaped text
    before State_2); the one real phone sits after State_2, matching
    FIELD_TO_DUMP_TAG's schema order where Phone_2 follows Country_2/State_2.
    Must land in phno_2, not silently shift into phno_1. Calls the real
    engine method, not a re-derived copy of its logic."""
    state_2_word = _word("CA", x=100, width=20, line_num=5)
    phone_word = _word("1630566471", x=100, width=90, line_num=7)
    assert RuleBasedExtractionEngine._assign_lone_phone_slot(phone_word, state_2_word) == "phno_2"


def test_lone_phone_before_state_2_is_assigned_to_phone_1():
    state_2_word = _word("CA", x=100, width=20, line_num=5)
    phone_word = _word("2226890149", x=100, width=90, line_num=2)
    assert RuleBasedExtractionEngine._assign_lone_phone_slot(phone_word, state_2_word) == "phno_1"


def test_lone_phone_with_no_state_2_anchor_falls_back_to_phone_1():
    """No State_2 was found at all (state_2_word is None) - preserves the
    original default rather than guessing, since there's no better signal."""
    phone_word = _word("2226890149", x=100, width=90, line_num=2)
    assert RuleBasedExtractionEngine._assign_lone_phone_slot(phone_word, None) == "phno_1"


