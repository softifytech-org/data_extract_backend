"""Regression tests for specific OCR extraction fixes validated against
real ground-truth data during development (see project history for the
full investigation of each case). Each test encodes a behavior that was
manually verified at the time the fix was made; this suite exists so none
of them can silently regress in a later change to engine.py.

These call the extraction engine's internal static helpers directly
(_smart_join, _extract_hash_codes, _extract_phones) rather than the full
extract() pipeline, because those helpers are the actual units the fixes
live in and can be exercised with minimal synthetic OCRWord input. The
stm_name "&"/initial pattern fix lives inline in extract() itself (not a
separate helper) and is not covered here - replicating a full synthetic
record layout to reach that code path was judged not worth the fragility
for one additional case; it remains covered by the manual verification
already on record (09104.jpeg records 241037/241039).
"""
from __future__ import annotations

from app.services.extraction.engine import RuleBasedExtractionEngine, _smart_join
from app.services.extraction.parser import build_indexed_text
from app.services.ocr.base import OCRWord


def _word(text: str, x: int, width: int, line_num: int = 0, y: int = 0, height: int = 13) -> OCRWord:
    return OCRWord(text=text, confidence=95.0, x=x, y=y, width=width, height=height,
                   page=1, line_num=line_num, word_num=0)


# --- _smart_join: defensive floor against mid-word OCR splits ---

def test_smart_join_merges_small_gap_word_split():
    """A 1px gap - well under the measured-real 4px minimum gap between two
    genuinely distinct words - must be treated as one OCR-split word."""
    words = [_word("GARLA", x=0, width=50), _word("ND", x=51, width=20)]
    assert _smart_join(words) == "GARLAND"


def test_smart_join_preserves_real_word_gap():
    """A 4px gap (the measured real gap between two distinct adjacent
    words, "David" next to "Lewis") must not be merged."""
    words = [_word("David", x=0, width=50), _word("Lewis", x=54, width=50)]
    assert _smart_join(words) == "David Lewis"


def test_smart_join_empty_input():
    assert _smart_join([]) == ""


# --- _extract_hash_codes: leading comma, density filter, separator spacing ---

def _run_hash_codes(words: list[OCRWord]):
    text, spans = build_indexed_text(words)
    return RuleBasedExtractionEngine._extract_hash_codes(text, spans, set())


def test_hash_code_recovers_glued_leading_comma():
    """Azure sometimes reads the leading comma as part of the same OCR word
    as the code (e.g. text=',BBaX_01aCm-9834' as one token, confirmed via
    raw OCR on 09104.jpeg/09107.jpeg) - HASH_CODE's own \\b boundary can't
    start on a comma, so the match must be extended back by one character
    to recover it, but only when adjacent with no whitespace."""
    words = [_word(",BBaX_01aCm-9834", x=0, width=140)]
    results = _run_hash_codes(words)
    assert len(results) == 1
    assert results[0][1] == ",BBaX_01aCm-9834"


def test_hash_code_collapses_separator_spacing_artifact():
    """build_indexed_text's single-space join between separately-OCR'd
    words lands next to the code's own separator punctuation when the
    source has no real space there - verified against 3 real records, e.g.
    ground truth "BaX_01acM,--5168" vs reconstructed "BaX_01acM, -- 5168"."""
    words = [_word("BaX_01acM,", x=0, width=80), _word("--", x=85, width=20), _word("5168", x=110, width=40)]
    results = _run_hash_codes(words)
    assert len(results) == 1
    assert results[0][1] == "BaX_01acM,--5168"


def test_hash_code_accepts_real_low_density_unpunctuated_code():
    """The lowest digit density among all 5,452 distinct real unpunctuated
    code values in the ground-truth corpus is 37.5% ("ZtA019vB") - must
    still be accepted."""
    words = [_word("ZtA019vB", x=0, width=80)]
    results = _run_hash_codes(words)
    assert len(results) == 1
    assert results[0][1] == "ZtA019vB"


def test_hash_code_rejects_low_density_false_positive_name():
    """Confirmed real false positives ("Faith2Gm", 12.5% digits) are
    synthetic customer names, not codes, and must be rejected by the 30%
    digit-density floor (which sits safely below the real 37.5% minimum)."""
    words = [_word("Faith2Gm", x=0, width=80)]
    assert _run_hash_codes(words) == []


def test_hash_code_rejects_period_joined_name_with_no_digit():
    """A period is a valid real code separator but also joins real person
    names (e.g. "Eric.Turcotte") - a match with no digit anywhere must be
    rejected regardless of shape."""
    words = [_word("Eric.Turcotte", x=0, width=120)]
    assert _run_hash_codes(words) == []


# --- _find_numeric_record_ids: prefixed-numeric shape (USER12 block 1) ---

def _run_numeric_record_ids(words: list[OCRWord]):
    text, spans = build_indexed_text(words)
    return RuleBasedExtractionEngine._find_numeric_record_ids(words, spans)


def test_numeric_record_ids_strips_prefix_for_prefixed_numeric_shape():
    """record_no's STORED value must be just the trailing 6 digits, not the
    full 8-character token - verified against DataM (2).xml: RecordNo
    "360099" matches the trailing digits of the anchor actually printed on
    that record's image, not the full glued token. The raw text keeps the
    full match (used for windowing/consumption), only the digits value is
    trimmed."""
    words = [_word("S8360099", x=0, width=70)]
    hits = _run_numeric_record_ids(words)
    assert len(hits) == 1
    _, digits, raw = hits[0]
    assert digits == "360099"
    assert raw == "S8360099"


def test_numeric_record_ids_recognizes_all_observed_confusable_variants():
    for token, expected_digits in [
        ("SB360827", "360827"),
        ("$8360825", "360825"),
        ("58360828", "360828"),
        ("$6360832", "360832"),
        ("56360111", "360111"),
    ]:
        hits = _run_numeric_record_ids([_word(token, x=0, width=70)])
        assert len(hits) == 1, f"{token!r} not recognized"
        assert hits[0][1] == expected_digits


def test_numeric_record_ids_prefixed_shape_requires_leftmost_of_line():
    words = [
        _word("Total", x=0, width=40, line_num=0),
        _word("S8360099", x=50, width=70, line_num=0),
    ]
    assert _run_numeric_record_ids(words) == []


def test_numeric_record_ids_prefixed_shape_rejects_zip_and_dollar_lookalikes():
    assert _run_numeric_record_ids([_word("11725", x=0, width=50)]) == []
    assert _run_numeric_record_ids([_word("087654321", x=0, width=80)]) == []
    assert _run_numeric_record_ids([_word("799.5670", x=0, width=80)]) == []


# --- _find_numeric_record_ids: RECORD_ID_NUMERIC's 5/7/8-digit shapes
# require a name+email companion on the same line, same as
# record_splitter.py's _find_anchor_words (this is the separate record_no
# FIELD-population path, not boundary detection - see patterns.py's
# looks_like_name_start/line_has_email for the shared logic). The
# exact-6-digit shape is unchanged and needs no companion at all.

def _record_line(id_text, *companions, line_num=0):
    words = [_word(id_text, x=0, width=70, line_num=line_num)]
    x = 80
    for c in companions:
        width = max(40, len(c) * 7)
        words.append(_word(c, x=x, width=width, line_num=line_num))
        x += width + 10
    return words


def test_numeric_record_ids_five_digit_accepted_with_name_and_email_companions():
    """Real shape from the "100526" dataset (USER 25)."""
    words = _record_line("22491", "Krzychika", "Dipen.Maun", "akire@adelphia.net")
    hits = _run_numeric_record_ids(words)
    assert [h[1] for h in hits] == ["22491"]


def test_numeric_record_ids_eight_digit_plain_accepted_with_name_and_email_companions():
    """Real shape from the "100526" dataset (USER 13) - an 8-digit plain
    Record No, distinct from the letter-prefixed 8-character shape."""
    words = _record_line("94401303", "Mblaney", "Luckybagcw2Simpson", "hebdrbill57noi@yahoo.com")
    hits = _run_numeric_record_ids(words)
    assert [h[1] for h in hits] == ["94401303"]


def test_numeric_record_ids_six_digit_unchanged_no_companion_needed():
    hits = _run_numeric_record_ids([_word("241001", x=0, width=50, line_num=0)])
    assert [h[1] for h in hits] == ["241001"]


def test_numeric_record_ids_five_digit_zip_rejected_without_companions():
    assert _run_numeric_record_ids([_word("11725", x=0, width=50, line_num=0)]) == []


def test_numeric_record_ids_five_digit_zip_rejected_with_country_companion():
    words = _record_line("11725", "US")
    assert _run_numeric_record_ids(words) == []


def test_numeric_record_ids_five_digit_rejected_when_companion_is_a_day_name():
    """Mirrors the real false positive found during evaluation: "13284
    Monday, 07/31/1939..." is a wrapped mid-record field, not a Record No."""
    words = _record_line("13284", "Monday,", "07/31/1939")
    assert _run_numeric_record_ids(words) == []


def test_numeric_record_ids_eight_digit_prefix_overlap_uses_trailing_six_digits():
    """"58360441" matches both RECORD_ID_NUMERIC (8 plain digits) and
    RECORD_ID_PREFIXED_NUMERIC ("5"+"8"+6 digits). It must still take the
    already-proven prefixed path (trailing-6-digit stored value) even with
    no name/email companion on its line - the newer plain-numeric
    requirements must never suppress an anchor the prefixed shape alone
    would already have found."""
    hits = _run_numeric_record_ids([_word("58360441", x=0, width=70, line_num=0)])
    assert len(hits) == 1
    assert hits[0][1] == "360441"
    assert hits[0][2] == "58360441"


def test_numeric_record_ids_plain_and_prefixed_shapes_coexist_sorted_by_position():
    """Both numeric record-ID shapes are distinct document families that
    never co-occur on a real page, but the function must not special-case
    one at the expense of recognizing the other if both are present."""
    words = [
        _word("241001", x=0, width=50, line_num=0),
        _word("S8360099", x=0, width=70, line_num=5),
    ]
    hits = _run_numeric_record_ids(words)
    assert [h[1] for h in hits] == ["241001", "360099"]


# --- _extract_phones: hyphen-adjacent space artifact ---

def test_phone_strips_hyphen_adjacent_join_space():
    """When a phone number is split across two OCR words, the
    build_indexed_text join space lands right next to the first hyphen -
    verified against 4 real records, always "NNN- NNN-NNNN", never
    elsewhere. Must be cleaned to the real "NNN-NNN-NNNN" shape."""
    words = [_word("907-", x=0, width=30), _word("722-5546", x=35, width=70)]
    results = RuleBasedExtractionEngine._extract_phones(*build_indexed_text(words), set())
    assert len(results) == 1
    _, value, raw = results[0]
    assert value == "907-722-5546"
    assert raw == "907- 722-5546"
