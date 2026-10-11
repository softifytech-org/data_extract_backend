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
# fixed-width groups already bound the match to exactly 8 characters, so
# dropping the trailing \b doesn't risk over-matching.
RECORD_ID = re.compile(r"\b[A-Z]{2}\d{3}[A-Z]\d{2}")
# Characters Tesseract commonly substitutes for a digit in this document
# family - shared with parser.normalize_confusable_digits, which maps them
# back. Kept broad (I/l/L/| for 1, Z/z for 2, O/o/Q/D/@ for 0) because a
# false match here just costs a little extraction precision, whereas being
# too strict silently drops entire records from detection.
_DIGIT_CONFUSABLE_CLASS = r"0-9OoQD@IlL|Zz"
# Tolerant fallback: only tried when the strict pattern above finds nothing
# - see extraction/engine.py.
RECORD_ID_FUZZY = re.compile(
    rf"\b([A-Z]{{2}})([{_DIGIT_CONFUSABLE_CLASS}]{{3}})([A-Z])([{_DIGIT_CONFUSABLE_CLASS}]{{2}})"
)
# Deliberately looser still - used only to locate record BOUNDARIES on a
# densely multi-record page (see extraction/record_splitter.py), not to
# read the ID's actual text. On this document family, Tesseract sometimes
# misreads the leading two letters badly enough (M/I -> w/r) that even the
# fuzzy pattern above won't match - but the LL-DDD-L-DD *shape* is usually
# still recognizable, which is all boundary detection needs.
RECORD_ID_ANCHOR = re.compile(
    rf"\b[A-Za-z]{{2}}[{_DIGIT_CONFUSABLE_CLASS}]{{3}}[A-Za-z][{_DIGIT_CONFUSABLE_CLASS}]{{2}}"
)

# A second, distinct document family in this schema prints Record No as a
# plain digit number with no letter prefix at all (e.g. "241001"), matching
# BN_Workflow.pdf's own rule - the alphanumeric "MI296T23" shape above
# turned out to be specific to one stress-test file, not the general case.
# Required to be the entire token (fullmatch, not search) since a bare run
# of digits is otherwise too easy to confuse with a ZIP/phone/other numeric
# field if only matched as a substring. Pinned to exactly 6 digits rather
# than BN_Workflow's full "5 or 6" range: verified directly against a real
# multi-record page that a wrapped line starting with a 5-digit ZIP (e.g.
# "11725") is otherwise indistinguishable from a genuine 5-digit Record No
# by shape alone, and every real Record No seen in this family is 6 digits.
RECORD_ID_NUMERIC = re.compile(r"\A\d{6}\Z")

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

ZIP_CODE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

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

STREET_ADDRESS = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\-'\s]{4,60}?"
    r"(?:St\.?|Street|Ave\.?|Avenue|Dr\.?|Drive|Rd\.?|Road|Ln\.?|Lane|Blvd\.?|Cir\.?|Circle|"
    r"Ct\.?|Court|Way|Pkwy\.?|Parkway|Terrace|Ter\.?|Trail|Place|Pl\.?|Loop|Square|Sq\.?)\b",
    re.IGNORECASE,
)

US_STATE_ABBREVIATIONS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
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
    "LORAZEPAM", "CLONAZAPAM", "DIDREX",
}

COUNTRY_VALUES = {"US", "USA"}
