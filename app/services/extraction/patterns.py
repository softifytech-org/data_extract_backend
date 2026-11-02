"""Regex patterns and closed vocabularies used by field extraction.

Kept isolated from the extraction logic itself so patterns can be tuned
(or a new document family's patterns added) without touching the engine.
"""
from __future__ import annotations

import re

# No trailing \b: the ID is frequently glued directly to the next word with
# no space at all (e.g. "MI296T23psychonurse") - a boundary requirement
# after the final digit would refuse to match there, since both a digit and
# a following lowercase letter are "word" characters in regex terms. The
# trailing digit group's own {2,5} upper bound is what keeps the match from
# over-matching, not a fixed width - see the comment on the trailing group
# below for why it's a range rather than exactly 2.
#
# Trailing digit group widened from a fixed 2 digits to 2-5: verified
# directly against a real document whose anchors print as "MI296T27312",
# "MI296T27313", ... "MI296T27353" - a genuine 5-digit trailing run, not 2.
# The old exactly-\d{2} pattern silently truncated these mid-match (a plain
# `.search()` with no length requirement just stops once its own pattern is
# satisfied), storing "29627" instead of the real "29627312" - the record
# was still found and split correctly (boundary detection only needs the
# anchor's grid POSITION, not its exact text), only the Record No VALUE
# came out wrong. 2-5 keeps the final digits-only value (3-digit middle
# group + this trailing group) inside the same 5-8 digit range already
# established and verified for the plain-numeric Record No family below
# (RECORD_ID_NUMERIC) - not a new, separately-invented bound. The original
# exactly-2-digit shape ("MI296T23") still matches unchanged, since {2,5}
# accepts 2 digits same as before.
RECORD_ID = re.compile(r"\b[A-Z]{2}\d{3}[A-Z]\d{2,5}")
# Characters Tesseract commonly substitutes for a digit in this document
# family - shared with parser.normalize_confusable_digits, which maps them
# back. Kept broad (I/l/L/| for 1, Z/z for 2, O/o/Q/D/@ for 0) because a
# false match here just costs a little extraction precision, whereas being
# too strict silently drops entire records from detection.
_DIGIT_CONFUSABLE_CLASS = r"0-9OoQD@IlL|Zz"
# Tolerant fallback: only tried when the strict pattern above finds nothing
# - see extraction/engine.py. Trailing group widened in lockstep with
# RECORD_ID above, same reason.
RECORD_ID_FUZZY = re.compile(
    rf"\b([A-Z]{{2}})([{_DIGIT_CONFUSABLE_CLASS}]{{3}})([A-Z])([{_DIGIT_CONFUSABLE_CLASS}]{{2,5}})"
)
# Deliberately looser still - used only to locate record BOUNDARIES on a
# densely multi-record page (see extraction/record_splitter.py), not to
# read the ID's actual text. On this document family, Tesseract sometimes
# misreads the leading two letters badly enough (M/I -> w/r) that even the
# fuzzy pattern above won't match - but the LL-DDD-L-DD shape is usually
# still recognizable, which is all boundary detection needs. Trailing group
# widened in lockstep with RECORD_ID above, same reason - this one doesn't
# strictly need it for boundary POSITIONING (that only uses the matched
# word's own x/y), but keeping all three patterns in sync avoids a
# surprise mismatch between the anchor hint text and the real value later.
RECORD_ID_ANCHOR = re.compile(
    rf"\b[A-Za-z]{{2}}[{_DIGIT_CONFUSABLE_CLASS}]{{3}}[A-Za-z][{_DIGIT_CONFUSABLE_CLASS}]{{2,5}}"
)

# A second, distinct document family in this schema prints Record No as a
# plain digit number with no letter prefix at all (e.g. "241001"), matching
# BN_Workflow.pdf's own rule - the alphanumeric "MI296T23" shape above
# turned out to be specific to one stress-test file, not the general case.
# Required to be the entire token (fullmatch, not search) since a bare run
# of digits is otherwise too easy to confuse with a ZIP/phone/other numeric
# field if only matched as a substring.
#
# Widened from an earlier exact-6-digit pin to 5-8 digits: a newer real
# dataset (the "100526" batch) genuinely uses 5-digit (e.g. "22491") and
# 8-digit (e.g. "94401303") Record Nos for several users, which the old
# exact-6 pin rejected outright ("No record identifiers could be found").
# The original 6-digit pin existed specifically because a bare 5-digit
# match is shape-identical to a wrapped ZIP that lands leftmost on its own
# line (e.g. "11725") - shape alone can't tell them apart. That
# disambiguation is NOT done here by digit count; see record_splitter.py's
# _find_anchor_words, which requires a non-6-digit match to also have a
# name-shaped word immediately after AND an email address later on the
# SAME line before accepting it - the structural signature of a genuine
# record-start line, confirmed present on every real record line
# inspected (both this new dataset and the original held-out/USER01/
# USER12 corpus) and absent from the historical wrapped-ZIP false
# positive, which was re-verified to still correctly reject under the new
# logic. The 6-digit case itself is unchanged/ungated - zero new risk to
# the already-proven majority of real data.
RECORD_ID_NUMERIC = re.compile(r"\A\d{5,8}\Z")

# A third document family prints Record No as a 2-character letter prefix
# glued to a 6-digit number (e.g. "S8360099"), where the real stored value
# is just the trailing 6 digits (verified against DataM (2).xml: its
# RecordNo "360099" matches the trailing digits of the OCR'd anchor on the
# corresponding image, not the full 8-character token). Azure reads the two
# prefix characters inconsistently - dumped raw OCR output for 40 real
# anchors across 2 real images and found position 1 only ever reads as
# 'S', '5', or '$', and position 2 only ever as '8', 'B', or '6' (confirmed
# 39/40 real anchors match this exact two-position confusable set; the one
# non-match, "799.5670", is a genuine dollar amount, correctly excluded).
# Required to be the entire token (fullmatch) and, like RECORD_ID_NUMERIC,
# only accepted when leftmost on its line - an 8-character run starting
# with one of these two character sets is not a shape expected to collide
# with a ZIP/phone/dollar field elsewhere on a line, and 3 full regression
# runs against the held-out set and USER01 (neither of which use this
# document family) measured zero false positives from adding it.
RECORD_ID_PREFIXED_NUMERIC = re.compile(r"\A[S5\$][8B6]\d{6}\Z")

EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Two phone grammars appear across document families: "(919) - 877-8368"
# (punctuated) and a bare 10-digit run with no separators at all
# ("4656094688"). Both are tried, punctuated first since it's unambiguous.
# The separator between the area code and the rest is intentionally a
# tolerant character class, not a literal "-": Tesseract commonly misreads
# that dash as "=", "~", "_" or similar on this document family, and a
# strict literal would refuse to match through it.
PHONE_PUNCTUATED = re.compile(r"\(?\d{3}\)?[\s\-=~_.]*\d{3}\s*-\s*\d{4}")
# Optional leading "1" (US country code) before the 10-digit number, e.g.
# "13741116994" - without this, an 11-digit run matches nowhere at all
# under a strict \b\d{10}\b (there's no internal word boundary to anchor on).
PHONE_BARE = re.compile(r"\b1?(\d{10})\b")

# Verified against real ground-truth data: this document family's zip codes
# are sometimes stored/printed with a leading zero dropped (e.g. a real
# New Jersey zip "08095" appearing as "8095", 4 digits) - a strict 5-digit
# requirement made the REAL zip invisible as a candidate, so _extract_zip's
# fallback picked a nearby unrelated 5-digit number instead (usually the
# other zip slot's own value, producing a systematic Zip_1/Zip_2 swap).
ZIP_CODE = re.compile(r"\b\d{3,5}(?:-\d{4})?\b")

# Digit-confusable class reused (not just \d) so an OCR-garbled amount like
# "$2.o0.00" is captured whole as ONE candidate rather than the regex
# silently stopping at the first non-digit character and only matching
# "$2" - the garbled text must survive as evidence, not disappear.
CURRENCY = re.compile(
    rf"\$\s?[{_DIGIT_CONFUSABLE_CLASS}]{{1,3}}(?:,[{_DIGIT_CONFUSABLE_CLASS}]{{3}})*"
    rf"(?:\.[{_DIGIT_CONFUSABLE_CLASS}]{{1,3}})?"
)

# Optional decimal portion (37.5 MG, 0.25 MG are common real doses) -
# without it, "\b\d{1,4}\s?MG\b" skips straight past the "37." prefix (a
# literal "." isn't \s and isn't a digit) and matches just the "5 MG" tail,
# silently dropping most of the actual dose. Verified against real
# ground-truth data, not the earlier stress-test file, which had no
# decimal doses to expose this.
DOSAGE = re.compile(r"\b\d{1,4}(?:\.\d{1,2})?\s?MG\b", re.IGNORECASE)
DOSAGE_FUZZY = re.compile(r"\b([0-9OQD@]{1,4}(?:\.[0-9OQD@]{1,2})?)\s?MG\b", re.IGNORECASE)

# Two date grammars appear in the source documents: a long form with a
# leading weekday, and a numeric mm/dd/yyyy form. Both normalize to ISO
# (YYYY-MM-DD) downstream - see extraction/parser.py.
#
# No trailing \b, same reasoning as RECORD_ID above: the year is frequently
# glued directly to the next field with no space (e.g. "...1999162 173" is
# really "...1999" + a stray/garbled "162" + "173") - a boundary requirement
# right after the 4th digit refuses to match there, silently dropping the
# date entirely. The exact-width \d{4} group already bounds the match.
DATE_LONG = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)
# Optional leading day-name prefix (e.g. "Thursday, 01/22/1998" / BN_Workflow.
# pdf's own "Friday,11/03/1589" example, comma with no space) - non-capturing
# so it doesn't disturb the month/day/year groups normalize_date_numeric
# expects, but included in the overall match so the day name survives as
# part of the preserved source text rather than being silently dropped.
DATE_NUMERIC = re.compile(
    r"\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*)?"
    r"(\d{1,2})/(\d{1,2})/(\d{4})",
    re.IGNORECASE,
)

DAY_MONTH_NAMES = {
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
    "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
}

# The \b BEFORE the suffix group (not just after) is deliberate, not
# redundant - confirmed real, systemic false-positive bug: without it,
# the greedy {4,60} middle class can stop matching mid-word, letting the
# suffix alternation match a TAIL FRAGMENT of an unrelated longer word
# instead of a real, standalone abbreviation. "Rd" (Road) matched inside
# "UnionPay Card"/"MasterCard"/"Maestro Card" (the suffix is "...Ca" +
# "rd", not an actual road abbreviation) in 24 separate real records
# across 15 images in the November dataset, every one of them a Card
# Name value sitting near a stray unconsumed digit (confirmed via
# res_address values like "1938 MALE UnionPay Card" - see
# evaluation/november_baseline_report.json). The same tail-collision risk
# exists for "St" (test, August, almost...), "Ter" (water, letter,
# master...), "Way" (highway, away...) and others - \b requires an actual
# word boundary immediately before the suffix starts, which a mid-word
# position between two letters never satisfies, so none of these can
# match as a substring of a longer, unrelated word anymore.
STREET_ADDRESS = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\-'#\s]{4,60}"
    r"\b(?:St\.?|Street|Ave\.?|Avenue|Dr\.?|Drive|Rd\.?|Road|Ln\.?|Lane|Blvd\.?|Cir\.?|Circle|"
    r"Ct\.?|Court|Way|Pkwy\.?|Parkway|Terrace|Ter\.?|Trail|Place|Pl\.?|Loop|Square|Sq\.?)\b",
    re.IGNORECASE,
)

# Street-suffix words, used as a STOP SIGNAL when walking backward from a
# state to find the city (see _extract_city): when there is no physical
# gap at all between Res Address and City on one line (verified against
# real ground-truth data - "13070 PFilFLi AVi AK STEEL" has uniformly
# small ~4px gaps throughout), a street-suffix word marks the end of the
# address and the start of whatever comes after it (the city), and
# everything at or before it belongs to the address, not the city.
STREET_SUFFIX_WORDS = {
    "ST", "STREET", "AVE", "AVENUE", "DR", "DRIVE", "RD", "ROAD", "LN", "LANE",
    "BLVD", "CIR", "CIRCLE", "CT", "COURT", "WAY", "PKWY", "PARKWAY",
    "TERRACE", "TER", "TRAIL", "PLACE", "PL", "LOOP", "SQUARE", "SQ", "VIEW",
}
# This document consistently OCRs a printed "E" as "I" in these suffix
# words specifically (confirmed against real ground truth: "Drivi"=Drive,
# "Circli"=Circle, "Lani"=Lane, "Striit"=Street, "Viiw"=View, "Avi"=Ave -
# the position of the substitution varies, so this isn't generic fuzzy
# edit-distance, which can't safely separate real cases like "AVI"~"AVE"
# (ratio 0.667) from false positives like "STEEL"~"STREET" (ratio 0.727,
# a higher ratio than a real case). Checking exact equality against each
# suffix's own "every E replaced by I" form is precise instead of fuzzy.
STREET_SUFFIX_WORDS_OCR_VARIANTS = {s.replace("E", "I") for s in STREET_SUFFIX_WORDS}

US_STATE_ABBREVIATIONS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
    # USPS territory/military codes, confirmed by a full ground-truth scan
    # (3597 records): PR (283 occurrences across State_1/State_2), AE (37),
    # AP (16), VI (6), AA (5) - all real, standard USPS abbreviations this
    # vocabulary was simply missing, not an edge case.
    "PR", "VI", "AE", "AP", "AA",
}

GENDER_VALUES = {"MALE", "FEMALE"}

YES_NO_NA_VALUES = {"YES", "NO", "N.A", "NA"}

BLOOD_GROUP_VALUES = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}

# Values that mean "the source document explicitly says this is not
# applicable/unavailable" - distinct from OCR simply failing to find
# anything. See services/validation/remark.py: only an EXPLICIT NA reading
# contributes to the generated Remark field; a blank extraction does not.
EXPLICIT_NA_VALUES = {"N.A", "N.A.", "NA", "N/A", "NONE"}

CARD_TYPES = {
    "AMERICAN EXPRESS",
    "AMERICAN EXPRESS GREEN CARD",
    "AMERICAN EXPRESS PLATINUM CARD",
    "MASTER CARD",
    "MASTERCARD",
    "VISA",
    "DISCOVER",
    "BANK",
    "RETAIL",
    # Verified against real ground-truth data: "Other" and "Gas" are two of
    # the most common real Card Name values in this document family and
    # were entirely missing from this vocabulary, so they could never be
    # recognized no matter how cleanly OCR read them.
    "OTHER",
    "GAS",
}

# Known drug names appearing in the source samples. In production this would
# be backed by a real formulary lookup - a closed list here still lets us
# validate OCR output against a known vocabulary rather than trusting raw
# characters, which is the whole point of enum-style validation.
DRUG_NAMES = {
    "XANAX", "VALIUM", "ZOLOFT", "PHENTERMINE", "AMBIEN", "DIAZEPAM", "ADIPEX",
    "LORAZEPAM", "CLONAZAPAM", "CLONAZEPAM", "DIDREX",
}

COUNTRY_VALUES = {"US", "USA"}

# Content-based disambiguation for RECORD_ID_NUMERIC's non-6-digit shapes
# (5, 7, 8 digits) - shared by record_splitter.py (record-boundary
# detection) and extraction/engine.py's _find_numeric_record_ids (the
# record_no FIELD-population path), which both need the identical check so
# a wrapped ZIP/other numeric field can't be mistaken for a Record No by
# either consumer. See RECORD_ID_NUMERIC's own comment above for the full
# history of why this exists - shape (digit count) alone can't tell a
# genuine Record No from a wrapped field that happens to land leftmost on
# its own line at the same length.
_NON_NAME_VOCAB = COUNTRY_VALUES | GENDER_VALUES | US_STATE_ABBREVIATIONS | DAY_MONTH_NAMES


def looks_like_name_start(text: str) -> bool:
    letters = "".join(c for c in text if c.isalpha())
    if len(letters) < 3:
        return False
    # Strip surrounding punctuation before the vocab check - "Monday,"
    # (e.g. a wrapped field reading "13284 Monday, 07/31/1939...") must be
    # excluded the same as a bare "Monday" would be; comparing the raw
    # text with its comma still attached would silently miss this -
    # confirmed as a real false positive during evaluation before this
    # strip was added.
    stripped = text.strip(".,;:!?()[]{}\"'").upper()
    return stripped not in _NON_NAME_VOCAB


# How close two adjacent words' boxes must be (end of one to start of the
# next, in pixels) to be tried as one OCR-split email - see line_has_email
# below. Confirmed real, from two actual reviewer-reported missing-record
# documents: Azure's layout OCR split "jblue@WIZREALM.COM" into
# "jblue@WIZREALM." + "COM" (a 2px gap) and "jwong@SLIP.NET" into
# "jwong@SLIP." + "NET" (a 3px gap) - both right after the final '.',
# where the TLD landed as its own token. A genuine word-to-word gap on the
# same real rows measured 6-40px, so 4px cleanly separates "one token
# split in two" from "two different words" without widening the net.
_EMAIL_SPLIT_MAX_GAP = 4


def line_has_email(line_words) -> bool:
    """A genuine record-start line carries a CustomerName/ShipperName AND
    an email address shortly after the ID, confirmed present on every real
    record-start line inspected (both a newer 5/8-digit-ID dataset and the
    original held-out/USER01/USER12 corpus this engine was tuned against).
    A wrapped mid-record field never does - its line continues with policy
    numbers, dates, dollar amounts or flags, none of which are
    email-shaped. Confirmed by direct inspection this is a materially
    stronger signal than "the next word looks alphabetic" alone, which a
    garbled OCR token (e.g. a corrupted "US" read as "USTN") can slip
    past.

    Also tries joining each pair of immediately-adjacent words (sorted
    left to right, touching or nearly touching - see
    _EMAIL_SPLIT_MAX_GAP) before giving up - real bug, not a hypothetical:
    two whole records silently vanished from reviewer-reported documents
    because Azure split their one printed email into two OCR words right
    after the final '.', and neither fragment alone is email-shaped, so
    the single-word check failed the companion test and the anchor (and
    everything _find_anchor_words/_find_numeric_record_ids needed it for)
    was silently dropped."""
    if any(EMAIL.search(w.text) for w in line_words):
        return True
    ordered = sorted(line_words, key=lambda w: w.x)
    for a, b in zip(ordered, ordered[1:]):
        gap = b.x - (a.x + a.width)
        if gap <= _EMAIL_SPLIT_MAX_GAP and EMAIL.search(a.text + b.text):
            return True
    return False
