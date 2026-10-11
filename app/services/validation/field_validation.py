"""Validation A (partially) and Validation C: format/type validation.

Answers "does this value make sense for what it claims to be?" - independent
of how confident OCR was. A value can have 99% OCR confidence and still fail
format validation (e.g. an "email" with no @ sign), and that mismatch is
exactly the kind of signal this system is built to surface rather than hide.

An explicit "N.A" reading is always treated as format-valid regardless of
field type - it's a legitimate answer, not malformed data. See
services/validation/remark.py for how confirmed-NA fields feed the derived
Remark field.
"""
from __future__ import annotations

import re
from datetime import date

from app.services.extraction import patterns as pat

_STRICT_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_ZIP = re.compile(r"^\d{5}(?:-\d{4})?$")
_CURRENCY = re.compile(r"^\$\d+(?:\.\d{2})?$")  # "$" required per BN_WorkFlow.pdf ("Given as $")


def _is_explicit_na(value: str) -> bool:
    return value.strip().upper().rstrip(".") in {v.rstrip(".") for v in pat.EXPLICIT_NA_VALUES}


def _valid_email(value: str) -> bool:
    return bool(_STRICT_EMAIL.match(value))


def _valid_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    # 10 digits, or 11 with a leading US country-code "1" - both appear as
    # valid in BN_WorkFlow.pdf's own examples ("1234567890" and
    # "+1 123-456-7890"). The value itself is stored exactly as read from
    # the image (see _extract_phones) - this only checks the digit count.
    return len(digits) == 10 or (len(digits) == 11 and digits[0] == "1")


def _valid_record_no(value: str) -> bool:
    # BN_Workflow.pdf rule 2: strictly numeric, 5 or 6 digits.
    return value.isdigit() and len(value) in (5, 6)


def _valid_name(value: str) -> bool:
    parts = value.split()
    return len(parts) >= 2 and all(p.isalpha() and len(p) >= 2 for p in parts)


def _valid_two_word_name(value: str) -> bool:
    # shipper_name/stm_name specifically: measured against 250 held-out
    # records (round 7), word count is a strong discriminator ONLY for
    # these two fields - false-accept rate climbs from 18.7% (2 words) to
    # 56.2% (3 words) to 88.9% (4 words) for shipper_name, and 41.0% to
    # 92.3% to 100.0% for stm_name, because their dominant extraction error
    # is an adjacent field's leftover word getting glued onto the front of
    # an otherwise-correct 2-word name (e.g. "Matschek Fredrick Beckman"
    # instead of "Fredrick Beckman"). The same word-count split shows NO
    # such relationship for customer_name/billing_name/name_p_holder (FAR
    # stays ~27-42% regardless of word count there), so this stricter rule
    # is intentionally scoped to just these two fields rather than folded
    # into the shared _valid_name used by the others.
    parts = value.split()
    return len(parts) == 2 and all(p.isalpha() and len(p) >= 2 for p in parts)


def _valid_street_address(value: str) -> bool:
    return bool(re.match(r"^\d{1,6}\s+\S", value)) and len(value) >= 6


def _valid_city(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z .'\-]{2,40}", value))


def _valid_state(value: str) -> bool:
    return value.upper() in pat.US_STATE_ABBREVIATIONS


def _valid_zip(value: str) -> bool:
    return bool(_ZIP.match(value))


def _valid_country(value: str) -> bool:
    return value.upper() in pat.COUNTRY_VALUES


def _valid_iso_date(value: str) -> bool:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return 1900 <= parsed.year <= 2100


def _valid_source_date(value: str) -> bool:
    """Date fields keep the exact source text (e.g. "Sunday, October 10,
    1999") rather than being rewritten to ISO format - this validates that
    text directly by requiring it to fully match one of the two date
    grammars this document family uses, anchored start-to-end (the pattern's
    own leading \\b is harmless under fullmatch - it's satisfied at position
    0 same as anywhere else)."""
    from app.services.extraction.parser import normalize_date_long, normalize_date_numeric

    m = re.fullmatch(pat.DATE_LONG.pattern, value, re.IGNORECASE)
    if m:
        return normalize_date_long(*m.groups()) is not None
    m = re.fullmatch(pat.DATE_NUMERIC.pattern, value)
    if m:
        return normalize_date_numeric(*m.groups()) is not None
    return False


def _valid_gender(value: str) -> bool:
    return value.upper() in pat.GENDER_VALUES


def _valid_card_name(value: str) -> bool:
    return value.upper() in pat.CARD_TYPES


def _valid_medicine(value: str) -> bool:
    return value.upper() in pat.DRUG_NAMES


def _valid_dosage(value: str) -> bool:
    return bool(pat.DOSAGE.fullmatch(value.strip()))


def _valid_count(value: str) -> bool:
    return value.isdigit() and 0 < int(value) <= 100_000


def _valid_currency(value: str) -> bool:
    if not _CURRENCY.match(value):
        return False
    return float(value[1:]) >= 0


def _valid_height(value: str) -> bool:
    return value.isdigit() and 40 <= int(value) <= 300  # cm, generous range


def _valid_weight(value: str) -> bool:
    return value.isdigit() and 20 <= int(value) <= 400  # lb/kg, generous range


def _valid_blood_group(value: str) -> bool:
    return value.upper() in pat.BLOOD_GROUP_VALUES


def _valid_yes_no_na(value: str) -> bool:
    return value.strip().upper().rstrip(".") in {v.rstrip(".") for v in pat.YES_NO_NA_VALUES}


def _valid_yes_no(value: str) -> bool:
    # Allergies (BN_WorkFlow.pdf rule 27) is explicitly YES/NO only -
    # unlike the other health-detail flags, an N.A answer here is invalid,
    # not a confirmed non-answer.
    return value.strip().upper().rstrip(".") in {"YES", "NO"}


FIELD_VALIDATORS: dict[str, callable] = {
    "record_no": _valid_record_no,
    "customer_name": _valid_name,
    "email_address": _valid_email,
    "res_address": _valid_street_address,
    "city_1": _valid_city,
    "state_1": _valid_state,
    "zip_1": _valid_zip,
    "phno_1": _valid_phone,
    "country_1": _valid_country,
    "sex_1": _valid_gender,
    "d_birth": _valid_source_date,
    "height": _valid_height,
    "weight": _valid_weight,
    "blood_group": _valid_blood_group,
    "billing_name": _valid_name,
    "shipper_name": _valid_two_word_name,
    "city_2": _valid_city,
    "state_2": _valid_state,
    "zip_2": _valid_zip,
    "country_2": _valid_country,
    "phno_2": _valid_phone,
    "alcoholic": _valid_yes_no_na,
    "smoker": _valid_yes_no_na,
    "past_surg": _valid_yes_no_na,
    "diabetic": _valid_yes_no_na,
    "allergies": _valid_yes_no,
    "policy_no": lambda v: len(v) >= 4,
    # D_B_Life_Assure is a date field (e.g. "Friday,11/03/1589") despite
    # sitting next to the currency fields on the legacy form - confirmed
    # against BN_WorkFlow.pdf's field spec, not a currency amount as
    # previously assumed.
    "d_b_life_assure": _valid_source_date,
    "p_inst": _valid_currency,
    "name_p_holder": _valid_name,
    "stm_name": _valid_two_word_name,  # BN_Workflow.pdf rule 31: letters + dot only, no "_"/"-"
    "stm_code": lambda v: len(v) >= 3,
    "dob": _valid_source_date,
    "sex_2": _valid_gender,
    "card_name": _valid_card_name,
    "medicine": _valid_medicine,
    "dosage": _valid_dosage,
    "tablets": _valid_count,
    "pill_rate": _valid_currency,
    "cost": _valid_currency,
    "shipping_cost": _valid_currency,
    "total_amount": _valid_currency,
}


def validate_field_format(field_name: str, value: str | None) -> bool | None:
    """Returns None (not applicable) when there's no value or no validator
    registered for the field - never conflate that with False (invalid)."""
    if not value:
        return None
    if _is_explicit_na(value):
        return True
    validator = FIELD_VALIDATORS.get(field_name)
    if validator is None:
        return None
    try:
        return bool(validator(value))
    except (ValueError, TypeError):
        return False
