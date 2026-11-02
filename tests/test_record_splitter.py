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


# --- RECORD_ID_NUMERIC: 5/7/8-digit shapes require a name+email companion
# on the same line (see patterns.py's RECORD_ID_NUMERIC and
# patterns.looks_like_name_start/line_has_email for the full rationale).
# The exact-6-digit shape is unchanged and needs no companion at all.

def _record_line(id_text, *companions, line_num=20, y=400):
    """A record-ID word followed by companion words on the same physical
    line, spaced out like a real OCR'd row."""
    words = [_word(id_text, x=0, width=70, line_num=line_num, y=y)]
    x = 80
    for c in companions:
        width = max(40, len(c) * 7)
        words.append(_word(c, x=x, width=width, line_num=line_num, y=y))
        x += width + 10
    return words


def test_five_digit_record_id_accepted_with_name_and_email_companions():
    """Real shape from the "100526" dataset (USER 25): a genuine 5-digit
    Record No followed by a two-word name and an email."""
    words = _record_line("22491", "Krzychika", "Dipen.Maun", "akire@adelphia.net")
    assert [t for t, _ in _find_anchor_words(words)] == ["22491"]


def test_eight_digit_plain_record_id_accepted_with_name_and_email_companions():
    """Real shape from the "100526" dataset (USER 13): an 8-digit Record
    No (not the letter-prefixed shape) followed by name and email."""
    words = _record_line("94401303", "Mblaney", "Luckybagcw2Simpson", "hebdrbill57noi@yahoo.com")
    assert [t for t, _ in _find_anchor_words(words)] == ["94401303"]


def test_six_digit_record_id_accepted_without_any_companion_unchanged():
    """The original, most-verified shape needs no companion check at all -
    must still match bare, exactly as before this change."""
    words = [_word("405400", x=0, width=50, line_num=0, y=0)]
    assert [t for t, _ in _find_anchor_words(words)] == ["405400"]


def test_five_digit_bare_zip_alone_on_line_still_rejected():
    """The historical false positive this safeguard exists for: a wrapped
    5-digit ZIP with nothing else on its line."""
    words = [_word("11725", x=0, width=50, line_num=0, y=0)]
    assert _find_anchor_words(words) == []


def test_five_digit_zip_followed_by_country_code_rejected():
    words = _record_line("11725", "US")
    assert _find_anchor_words(words) == []


def test_five_digit_zip_followed_by_phone_digits_rejected():
    words = _record_line("94316", "8640622926")
    assert _find_anchor_words(words) == []


def test_five_digit_wrapped_field_with_garbled_companion_rejected():
    """Confirmed real false positive found during evaluation: a wrapped
    Zip_2 followed by garbled OCR noise ("USTN") that passes a naive
    alphabetic-word check must still be rejected because no email appears
    anywhere on the line - this is why the email check exists in addition
    to the name-shape check, not instead of it."""
    words = _record_line("55060", "USTN", "Pi0.3-Ur_6780")
    assert _find_anchor_words(words) == []


def test_five_digit_record_id_rejected_when_companion_is_a_day_name():
    """Confirmed real false positive found during evaluation: "13284
    Monday, 07/31/1939..." is a wrapped mid-record field, not a record
    start. Also confirms punctuation is stripped before the closed-
    vocabulary check - "Monday," with its trailing comma must still be
    excluded the same as a bare "Monday" would be."""
    words = _record_line("13284", "Monday,", "07/31/1939")
    assert _find_anchor_words(words) == []


def test_numeric_record_id_companion_found_across_split_line_num():
    """Real reviewer-reported bug: a record silently missing from the
    output, its fields absorbed into the PRECEDING record instead. Traced
    to the OCR engine sometimes assigning two different line_num values to
    words on the SAME printed row (e.g. the email, a few pixels lower than
    the ID+name on the same row, landing in a different line_num bucket).
    The old by-line_num-only lookup never saw the email in that case and
    silently dropped the whole anchor. Companions here share the ID's Y
    (within _ROW_Y_TOLERANCE) but NOT its line_num, and must still be
    found."""
    words = [
        _word("9286578", x=0, width=70, line_num=5, y=400),
        _word("Ldiamond", x=80, width=70, line_num=5, y=400),
        _word("jblue@WIZREALM.COM", x=400, width=150, line_num=6, y=403),
    ]
    assert [t for t, _ in _find_anchor_words(words)] == ["9286578"]


def test_numeric_record_id_companion_not_pulled_from_next_printed_row():
    """The Y-proximity widening above must not reach far enough to borrow
    a genuinely DIFFERENT, next printed row's email - only a same-row
    fragment split across line_num. No email on the ID's own row means no
    anchor, even though an email exists a full row (19px, matching Azure's
    real row spacing) below."""
    words = [
        _word("9286578", x=0, width=70, line_num=5, y=400),
        _word("Ldiamond", x=80, width=70, line_num=5, y=400),
        _word("NextCust", x=0, width=70, line_num=6, y=419),
        _word("jblue@WIZREALM.COM", x=80, width=150, line_num=6, y=419),
    ]
    assert _find_anchor_words(words) == []


def test_numeric_record_id_companion_email_split_across_two_words():
    """Real reviewer-reported bug: two whole records (9286578 and 9286594,
    from real documents "MI 27_K-1 (1).jpeg" and "(3).jpeg") silently
    vanished from the output. Root cause, confirmed against the real
    Azure OCR response for these exact images: the record's own email
    got split into two OCR words right after the final '.' - "jblue@
    WIZREALM." + "COM" (a 2px gap) - so neither fragment alone was
    email-shaped and the companion check rejected a perfectly good
    anchor. Companions here reproduce that exact real split, including
    the real 2px gap - _record_line's default 10px companion spacing is
    deliberately NOT used here since it would be wider than the real
    split this test exists to reproduce."""
    words = [
        _word("9286578", x=0, width=70, line_num=20, y=400),
        _word("Ldiamond", x=80, width=70, line_num=20, y=400),
        _word("jblue@WIZREALM.", x=160, width=104, line_num=20, y=400),
        _word("COM", x=266, width=25, line_num=20, y=400),
    ]
    assert [t for t, _ in _find_anchor_words(words)] == ["9286578"]


def test_numeric_record_id_companion_email_split_not_glued_across_normal_gap():
    """The concatenation check above must not fire across an ordinary
    word-to-word gap - only a near-touching split. "jblue@WIZREALM" and
    "COM" here are far enough apart (comfortably more than the 2-3px real
    split gaps above) that they must be treated as two unrelated words,
    not one glued-together email - so the anchor is correctly rejected
    (neither fragment is email-shaped alone, same as any other
    no-email line)."""
    words = [
        _word("9286578", x=0, width=70, line_num=20, y=400),
        _word("Ldiamond", x=80, width=70, line_num=20, y=400),
        _word("jblue@WIZREALM", x=160, width=90, line_num=20, y=400),
        _word("COM", x=300, width=30, line_num=20, y=400),
    ]
    assert _find_anchor_words(words) == []


def test_eight_digit_token_matching_both_shapes_uses_prefixed_path():
    """"58360441" is shape-identical to BOTH RECORD_ID_NUMERIC (8 plain
    digits) and RECORD_ID_PREFIXED_NUMERIC ("5"+"8"+6 digits). It must
    still be recognized via the already-proven prefixed path even with no
    name/email companion on its line - the newer, stricter plain-numeric
    requirements must never suppress an anchor the prefixed shape alone
    would already have found."""
    words = [_word("58360441", x=0, width=70, line_num=0, y=0)]
    assert [t for t, _ in _find_anchor_words(words)] == ["58360441"]


# --- Alphanumeric vs numeric/prefixed anchor mutual exclusivity ---
# Real documents use exactly ONE anchor shape per page, never a mix -
# see _find_anchor_words' own docstring for the full evidence.

def test_numeric_anchor_suppressed_when_page_has_real_alnum_anchors():
    """Real, confirmed bug (MI_New T(288).jpeg, November dataset): a
    phone-number fragment "811880" sitting in the middle of an existing
    alphanumeric record's own body text coincidentally satisfied the
    plain-numeric shape's leftmost+name+email companion check and became
    its own phantom extra record. With 2+ genuine alphanumeric anchors on
    the page, a bare-digit token elsewhere must never be treated as a
    second, competing record family."""
    words = [
        _word("MI296T3949", x=17, width=70, line_num=0, y=0),
        _word("MI296T3950", x=17, width=70, line_num=10, y=200),
        # A stray leftmost 6-digit number elsewhere on the page, with a
        # name-shaped word and an email on its own row - satisfies every
        # requirement the plain-numeric path checks for, on its own.
        _word("811880", x=66, width=50, line_num=20, y=400),
        _word("Someone", x=150, width=60, line_num=20, y=400),
        _word("a@b.com", x=250, width=60, line_num=20, y=400),
    ]
    found = [t for t, _ in _find_anchor_words(words)]
    assert found == ["MI296T3949", "MI296T3950"]
    assert "811880" not in found


def test_numeric_anchors_unaffected_by_a_single_alnum_false_positive():
    """RECORD_ID_ANCHOR itself has rare false positives against ordinary
    text that happens to fit its confusable-character shape (confirmed
    real: "BALDORIOTY", a place name, and "ob90@aol.com", an email
    fragment, both found in real cached NUMERIC-family documents). A
    single stray match like this must not suppress that page's real
    numeric anchors - only 2+ alphanumeric matches are confident enough
    evidence that this is genuinely an alphanumeric-family page."""
    words = [
        _word("241001", x=0, width=50, line_num=0, y=0),
        _word("241002", x=0, width=50, line_num=10, y=200),
        # A single coincidental alphanumeric-shaped false positive,
        # unrelated to either real record, leftmost on neither line.
        _word("Somewhere", x=300, width=70, line_num=0, y=0),
        _word("BALDORIOTY", x=300, width=70, line_num=10, y=200),
    ]
    found = [t for t, _ in _find_anchor_words(words)]
    assert "241001" in found
    assert "241002" in found
