"""Regression tests for four real, user-reported issues found auditing the
derived Remark field (services/validation/remark.py) and field-format
validation (services/validation/field_validation.py) against the real
November-dump ground truth:

1. A flag field (Alcoholic/Smoker/Past Surg/Diabetic/Allergies) whose OCR
   token matched the bare "NA" spelling in YES_NO_NA_VALUES used to keep
   that literal spelling as its claimed value - everywhere else in this
   system (remark.py's own "N.A" return, dump_xml_export.py, the real
   ground truth itself) the canonical spelling is always "N.A".
2. Remark named invalid fields using FieldSpec.label (the UI's human-
   readable label, e.g. "City 1"), but the real legacy system's own Remark
   convention - verified by scanning 2,176 real Remarks sentences across
   all 17 November-dump users - uses a different spelling for many fields
   (e.g. "City_1", "PHNo_2", "CARD Name", "Allergiesd").
3. _valid_name/_valid_two_word_name rejected any name containing a period
   within a word (e.g. "C.Dreszer", "Jose.Magana") as a format error, even
   though this is a real, valid shape in the source documents (an initial
   or title glued on with a period instead of a space).
4. _valid_phone required exactly 10 (or 11 with a leading "1") digits,
   flagging a correctly-read phone number as invalid purely for being a
   different length.
"""
from __future__ import annotations

from app.models.document import DataType, ExtractedField, FieldValidation
from app.services.validation.field_validation import validate_field_format
from app.services.validation.remark import generate_remark


def _field(name, label, value, *, format_valid=None, marked_invalid=False):
    return ExtractedField(
        field_name=name,
        label=label,
        value=value,
        data_type=DataType.TEXT,
        validation=FieldValidation(format_valid=format_valid),
        marked_invalid=marked_invalid,
    )


# --- 1. Flag value normalization (NA -> N.A) ---------------------------


def test_bare_na_flag_value_is_never_claimed_literally():
    from app.services.extraction.engine import RuleBasedExtractionEngine
    from app.services.ocr.base import OCRWord

    words = [
        OCRWord(text=t, page=1, line_num=1, word_num=i, x=i * 10, y=0, width=8, height=8, confidence=99.0)
        for i, t in enumerate(["YES", "NO", "NA", "NO", "YES"])
    ]
    flags = RuleBasedExtractionEngine._find_flag_run(words, set())
    values = [v for _, v in flags]
    assert "NA" not in values
    assert values == ["YES", "NO", "N.A", "NO", "YES"]


# --- 2. Remark field labels match the real legacy convention -----------


def test_remark_uses_legacy_label_not_ui_label():
    fields = {
        "phno_2": _field("phno_2", "Ph No 2", "5551234567", format_valid=False),
    }
    assert generate_remark(fields) == "PHNo_2 is invalid"


def test_remark_underscore_labels():
    cases = {
        "city_1": "City_1", "city_2": "City_2", "zip_1": "Zip_1", "zip_2": "Zip_2",
        "sex_1": "Sex_1", "sex_2": "Sex_2", "country_1": "Country_1", "country_2": "Country_2",
        "state_1": "State_1", "state_2": "State_2", "stm_code": "STM_Code",
        "d_b_life_assure": "D_B_Life_Assure", "d_birth": "D_Birth", "p_inst": "P_Inst",
        "name_p_holder": "Name_P_Holder",
    }
    for field_name, expected_label in cases.items():
        fields = {field_name: _field(field_name, "irrelevant ui label", "bogus", format_valid=False)}
        assert generate_remark(fields) == f"{expected_label} is invalid"


def test_remark_card_name_and_legacy_spelling_quirks():
    assert generate_remark({"card_name": _field("card_name", "Card Name", "x", format_valid=False)}) == (
        "CARD Name is invalid"
    )
    assert generate_remark({"allergies": _field("allergies", "Allergies", "x", format_valid=False)}) == (
        "Allergiesd is invalid"
    )
    assert generate_remark({"alcoholic": _field("alcoholic", "Alcoholic", "x", format_valid=False)}) == (
        "Alcohlic is invalid"
    )


def test_remark_all_valid_fields_is_na():
    fields = {"customer_name": _field("customer_name", "Customer Name", "John Smith", format_valid=True)}
    assert generate_remark(fields) == "N.A"


# --- 3. Names with an internal period are format-valid ------------------


def test_name_with_period_initial_is_valid():
    assert validate_field_format("customer_name", "C.Dreszer Jose.Magana") is True
    assert validate_field_format("customer_name", "Markfarchi Dr.Cm") is True
    assert validate_field_format("billing_name", "Christine.Fukui Kapshiv") is True
    assert validate_field_format("name_p_holder", "B.Grossmanma Jd") is True


def test_two_word_name_with_period_is_valid():
    assert validate_field_format("shipper_name", "Jose.Magana House") is True
    assert validate_field_format("stm_name", "Rogertmoorec21 Smith") is True


def test_name_without_letters_still_invalid():
    assert validate_field_format("customer_name", "1234 5678") is False


# --- 4. Phone fields are never format-invalidated by digit count -------


def test_phone_any_digit_count_is_not_format_invalid():
    assert validate_field_format("phno_1", "5551234567") is None
    assert validate_field_format("phno_1", "123456789012345") is None
    assert validate_field_format("phno_2", "555") is None


def test_phone_explicit_na_still_valid():
    assert validate_field_format("phno_1", "N.A") is True
