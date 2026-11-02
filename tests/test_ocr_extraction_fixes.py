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

from app.services.extraction import patterns as pat
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


def test_numeric_record_ids_companion_found_across_split_line_num():
    """Same real bug, same fix, as record_splitter.py's identical test -
    this is the separate record_no FIELD-population path, which needs the
    identical Y-proximity companion lookup or a record whose boundary WAS
    found correctly could still save with record_no blank."""
    words = [
        _word("9286578", x=0, width=70, line_num=5, y=400),
        _word("Ldiamond", x=80, width=70, line_num=5, y=400),
        _word("jblue@WIZREALM.COM", x=400, width=150, line_num=6, y=403),
    ]
    hits = _run_numeric_record_ids(words)
    assert [h[1] for h in hits] == ["9286578"]


def test_numeric_record_ids_companion_not_pulled_from_next_printed_row():
    words = [
        _word("9286578", x=0, width=70, line_num=5, y=400),
        _word("Ldiamond", x=80, width=70, line_num=5, y=400),
        _word("NextCust", x=0, width=70, line_num=6, y=419),
        _word("jblue@WIZREALM.COM", x=80, width=150, line_num=6, y=419),
    ]
    assert _run_numeric_record_ids(words) == []


def test_numeric_record_ids_companion_email_split_across_two_words():
    """Same real bug as record_splitter.py's identical test - this is the
    separate record_no FIELD-population path, which needs the identical
    fix or a record whose boundary WAS found correctly could still save
    with record_no blank."""
    words = [
        _word("9286578", x=0, width=70, line_num=20, y=400),
        _word("Ldiamond", x=80, width=70, line_num=20, y=400),
        _word("jblue@WIZREALM.", x=160, width=104, line_num=20, y=400),
        _word("COM", x=266, width=25, line_num=20, y=400),
    ]
    hits = _run_numeric_record_ids(words)
    assert [h[1] for h in hits] == ["9286578"]


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


# --- patterns.STREET_ADDRESS: suffix must be its own word, not a tail
# fragment of an unrelated longer word (real November-dataset bug) ---

def test_street_address_does_not_match_card_as_road_suffix():
    """Real, confirmed bug (not hypothetical): res_address came back as
    '1938 MALE UnionPay Card' (and 23 other near-identical cases across
    15 images in the November dataset's 44-image baseline sample) because
    the old regex had no word boundary BEFORE the suffix alternation, so
    the greedy {4,60} middle class could stop matching mid-word and let
    "Rd" (Road) match the tail of "Card"/"MasterCard" instead of a real,
    standalone road abbreviation. None of these should match at all - the
    only "address-shaped" content here is a stray date fragment, a
    gender, and a payment card name, not a street."""
    for text in [
        "1938 MALE UnionPay Card",
        "1982 MALE MasterCard",
        "04/05/1938 MALE UnionPay Card AMBIEN",
        "1956 27-Sep-82 FEMALE Maestro Card",
    ]:
        assert pat.STREET_ADDRESS.search(text) is None, text


def test_street_address_still_matches_genuine_addresses():
    """The added word boundary must not cost any real match - confirmed
    against real address shapes already relied upon elsewhere in this
    document family (see engine.py's _extract_gap/_extract_street_address
    comments)."""
    cases = {
        "12345 Main St": "12345 Main St",
        "269 Longliaf Circle": "269 Longliaf Circle",
        "18513 Grackli Way Rd Denair": "18513 Grackli Way Rd",
        "8131 240TH ST. SW.23630": "8131 240TH ST",
        "432 Chapman Dr 5728": "432 Chapman Dr",
    }
    for text, expected in cases.items():
        m = pat.STREET_ADDRESS.search(text)
        assert m is not None, text
        assert m.group(0) == expected


def test_res_address_fallback_no_longer_grabs_card_name_text():
    """End-to-end reproduction of the real failing record (787.jpeg,
    USER 4 69N, November dataset): with City_1 unavailable (so
    _extract_gap can't run) and a stray unconsumed digit near a Card Name
    value, res_address must come back as None (preserve uncertainty) or a
    real address fragment - never the old wrong value built from a date
    remnant + gender + card name."""
    words = [
        _word("1938", x=18, width=40, line_num=9, y=146),
        _word("MALE", x=114, width=40, line_num=9, y=147),
        _word("UnionPay", x=168, width=60, line_num=9, y=147),
        _word("Card", x=232, width=35, line_num=9, y=147),
    ]
    text, spans = build_indexed_text(words)
    result = RuleBasedExtractionEngine._extract_street_address(text, spans, set())
    assert result is None


# --- _find_numeric_record_ids: suppressed when the record's own
# alphanumeric ID is genuinely present (mirrors record_splitter.py's
# page-wide guard, scoped to one record's own word slice) ---

def test_numeric_record_id_suppressed_by_own_leftmost_alnum_id():
    """A record whose own ID is the alphanumeric shape must never have
    that overridden by some other, coincidentally numeric-shaped token
    elsewhere in its own word slice - the caller prefers whatever this
    function returns over the correct alphanumeric path, so a wrong
    non-empty result here would silently corrupt record_no."""
    words = [
        _word("MI296T3952", x=17, width=70, line_num=0, y=0),
        _word("811880", x=66, width=50, line_num=5, y=100),
        _word("Someone", x=150, width=60, line_num=5, y=100),
        _word("a@b.com", x=250, width=60, line_num=5, y=100),
    ]
    assert _run_numeric_record_ids(words) == []


def test_numeric_record_id_unaffected_by_non_leftmost_alnum_false_positive():
    """A coincidental alphanumeric-shaped match that is NOT leftmost on
    its own line (the same real false-positive shape as "BALDORIOTY"/
    "ob90@aol.com" - see record_splitter.py's identical test) must not
    suppress this record's own genuine numeric ID."""
    words = [
        _word("241001", x=0, width=50, line_num=0, y=0),
        _word("BALDORIOTY", x=300, width=70, line_num=0, y=0),
    ]
    hits = _run_numeric_record_ids(words)
    assert [h[1] for h in hits] == ["241001"]


# --- _find_flag_run: 5-answer YES/NO/N.A run for Alcoholic/Smoker/
# Past_Surg/Diabetic/Allergies - a SHORT run is dangerous, not just
# incomplete: the caller (zip(flag_fields, flags), front-aligned) maps
# a short run onto the WRONG field names whenever the run is missing its
# FIRST item, silently shifting every value down by one slot. ---

def test_flag_run_recovers_answer_glued_to_a_phone_number():
    """Real, confirmed bug (November dataset, MI_New T(132).jpeg record
    1): the flag run's first answer was glued with no space onto the
    end of a phone number OCR read as "880-6110YES", making it
    completely invisible as its own token - the run came back 4 long
    instead of 5, and the front-aligned zip then silently shifted every
    value to the wrong field (alcoholic got smoker's true answer, smoker
    got past_surg's, etc.), with allergies simply lost. All 5 must come
    back in the correct original order."""
    words = [
        _word("880-6110YES", x=497, width=100, line_num=2, y=48),
        _word("YES", x=594, width=30, line_num=2, y=48),
        _word("NO", x=642, width=25, line_num=2, y=48),
        _word("NO", x=690, width=25, line_num=2, y=48),
        _word("YES", x=738, width=30, line_num=2, y=48),
    ]
    found = RuleBasedExtractionEngine._find_flag_run(words, set(), "old")
    assert [v for _, v in found] == ["YES", "YES", "NO", "NO", "YES"]


def test_flag_run_does_not_glue_a_word_merely_ending_in_no():
    """The digit-before-suffix requirement must reject a genuine word
    that happens to end in "no"/"yes" (e.g. a real place name like
    "Reno") - only a DIGIT immediately before the suffix is the real
    signature of text glued onto a number, never a letter."""
    words = [
        _word("Reno", x=0, width=40, line_num=0, y=0),
        _word("YES", x=50, width=30, line_num=0, y=0),
    ]
    found = RuleBasedExtractionEngine._find_flag_run(words, set(), "old")
    assert [v for _, v in found] == ["YES"]


def test_flag_run_glued_suffix_recovered_even_when_word_already_consumed():
    """The glued word is normally claimed first by phone extraction
    (consumed before flags run) - the glued flag answer must still be
    recoverable from it, not silently lost because the word was already
    spoken for by a different field."""
    words = [
        _word("880-6110YES", x=497, width=100, line_num=2, y=48),
        _word("NO", x=594, width=25, line_num=2, y=48),
    ]
    already_consumed = {id(words[0])}
    found = RuleBasedExtractionEngine._find_flag_run(words, already_consumed, "old")
    assert [v for _, v in found] == ["YES", "NO"]


# --- _find_name_run / _trailing_contiguous_run: same-phrase gap
# threshold widened from 0.8x to 1.0x word-height (both functions,
# identical real-data-measured reason) ---

def test_find_name_run_recovers_repeated_name_with_12px_gap():
    """Real, confirmed bug (November dataset, MI_New T(132).jpeg record
    1): a genuinely repeated 2-word customer name ("tedjustok
    micheal246") printed with an 11-12px inter-word gap against an 11-
    12px word height (ratio ~0.92-1.0) in ALL 3 of its occurrences -
    just over the old 0.8x cutoff, so neither word ever joined into a
    2-word run and the repeat-detection never saw it at all. A
    completely unrelated name won by default instead, which then also
    blocked Shipper Name's own extraction (it depends on Billing Name's
    real position). Must now be found as the repeated name."""
    words = [
        _word("tedjustok", x=114, width=54, line_num=0, y=19, height=11),
        _word("micheal246", x=179, width=61, line_num=0, y=19, height=12),
        _word("tedjustok", x=642, width=53, line_num=1, y=34, height=12),
        _word("micheal246", x=706, width=62, line_num=1, y=34, height=12),
    ]
    result = RuleBasedExtractionEngine._find_name_run(words, set(), allow_alnum=True)
    assert result is not None
    value, is_repeated = result[1], result[2]
    assert value == "tedjustok micheal246"
    assert is_repeated is True


def test_trailing_contiguous_run_recovers_12px_gap():
    """Real, confirmed bug (same record as above): Shipper Name's own
    2-word value ("cherrylynne edikis") had a 12px gap against a 12px
    word height (ratio 1.0, exactly at the old cutoff) - each word was
    isolated instead of joined, so the >=2-token check downstream failed
    and Shipper Name came back blank even once its search window was
    otherwise computed correctly."""
    words = [
        _word("cherrylynne", x=66, width=65, line_num=2, y=48, height=12),
        _word("edikis", x=143, width=37, line_num=2, y=48, height=11),
    ]
    run = RuleBasedExtractionEngine._trailing_contiguous_run(words)
    assert [w.text for w in run] == ["cherrylynne", "edikis"]


def test_flag_run_treats_not_as_no():
    """Real, recurring pattern (62 occurrences across 19 of 44 real
    November-dataset images, confirmed against the actual source image
    for one case - MI_New T(132).jpeg record 2, which visually prints
    "YES NOT YES NO NO NO"): a token reading "NOT" sits where a flag
    answer belongs, one letter off from "NO" (raw difflib ratio 0.80,
    just under the 0.85 cutoff). Breaking the run here previously cost
    the run's LAST slot (Allergies) in every affected record - the exact
    same failure mode as the period/glued-suffix cases already fixed,
    not a new mechanism. All 6 real tokens here must resolve into one
    unbroken run, keeping the trailing 5 (see _find_flag_run's own
    comment on why the trailing slice, not the leading one, is correct)."""
    words = [
        _word("YES", x=355, width=19, line_num=7, y=118),
        _word("NOT", x=401, width=19, line_num=7, y=118),
        _word("YES", x=426, width=18, line_num=7, y=118),
        _word("NO", x=450, width=14, line_num=7, y=119),
        _word("NO", x=497, width=15, line_num=7, y=119),
        _word("NO", x=545, width=15, line_num=7, y=119),
    ]
    found = RuleBasedExtractionEngine._find_flag_run(words, set(), "old")
    assert [v for _, v in found] == ["NO", "YES", "NO", "NO", "NO"]


def test_extract_city_skips_street_suffix_to_reach_real_lowercase_city():
    """Real, confirmed bug found via manual verification against the
    actual source image (November dataset, MI_New T(132).jpeg record 1):
    the line above State_1 read "...WOOTiIN RD Gaithersburg" - the real
    city ("Gaithersburg") is lowercase, invisible to the ALL-CAPS
    line_offset=-1 tier, so "RD" (the all-caps street suffix right
    before it) won city_1 outright instead of being skipped past. Must
    now reach the real city on the require_upper=False tier instead of
    stopping at the suffix."""
    words = [
        _word("WOOTiIN", x=620, width=50, line_num=0, y=19),
        _word("RD", x=667, width=23, line_num=0, y=19),
        _word("Gaithersburg", x=690, width=90, line_num=0, y=20),
        _word("NE", x=66, width=14, line_num=1, y=34),
    ]
    state_word = words[-1]
    city = RuleBasedExtractionEngine._extract_city(words, set(), state_word, line_offset=-1)
    assert city is None  # the ALL-CAPS tier must not pick "RD"
    city = RuleBasedExtractionEngine._extract_city(
        words, set(), state_word, line_offset=-1, require_upper=False
    )
    assert city is not None
    assert city[1] == "Gaithersburg"
