"""Derives the legacy form's Remark field per BN_WorkFlow.pdf rule 43
("Remarks Rule (Most Important)"): if every field is valid, Remark = "N.A";
otherwise Remark names the invalid field(s) ("X is invalid" / "X and Y are
invalid" / "X, Y, and Z are invalid").

A field counts as invalid here when any of:
  - its value is an explicit N.A marker in the source ("N.A", "NA", ...) -
    the spec's "Common Rule (N.A Handling)" treats a confirmed-blank source
    field as invalid and requires it to be called out in Remark, or
  - it has a value but that value failed its own format validation
    (ExtractedField.validation.format_valid is False - see
    services/validation/field_validation.py for the per-field rules), or
  - the operator explicitly flagged it via ExtractedField.marked_invalid -
    independent of the first two: a field can hold a perfectly
    well-formatted, non-N.A value (e.g. a real-looking but WRONG email,
    "pranay123@gmial.com") that only a human reviewer can recognize as
    wrong. This flag is how that gets into Remark without destroying the
    value itself - see api/documents.py's update_fields, which is the only
    place that sets it, and ExtractedField's own docstring for the bug this
    replaced (the old UI wrote the literal text "N.A" over the value to
    piggyback on the first rule above, permanently losing it).

Deliberately NOT included: fields with no value at all because OCR/
extraction simply failed to find anything. The spec's "blank/space in the
image" case means the SOURCE document itself is blank there, which is not
something this pipeline can currently distinguish from "the value is
present on the page but we failed to read it" - collapsing the two would
mislabel real extraction gaps as confirmed-blank fields, which is exactly
the kind of guessed/hallucinated conclusion this project has consistently
avoided elsewhere (see FieldStatus.needs_review). Those fields keep
surfacing as "not found" for a human to check against the source, rather
than being silently written into Remark as if their blankness were
confirmed.
"""
from __future__ import annotations

from app.models.document import ExtractedField
from app.services.extraction import patterns as pat
from app.services.extraction.fields import OCR_FIELD_ORDER

# The label Remark uses to name an invalid field is its own, separate thing
# from FieldSpec.label (fields.py) - that one drives the UI form's on-screen
# label and the Excel export header, both of which should stay the nicer,
# human-readable spacing ("City 1", "Ph No 1"). Remark text instead has to
# match the legacy system's OWN naming convention, since Remark is meant to
# reproduce that system's exact output. Derived by directly scanning every
# real Remarks sentence across all 17 November-dump users' DataM.xml (2,176
# remarks, 1,952 non-"N.A" naming >=1 field) and majority-voting the exact
# spelling/casing/punctuation used per field - real data entry is
# inconsistent (typos, stray spaces), so only fields whose real convention
# clearly differs from FieldSpec.label are overridden below; every other
# field's real-data majority already matches FieldSpec.label and is left
# alone. Two entries look like typos but are in fact the dominant, repeated
# real spelling (not a one-off mistake) - "Allergiesd" (451 occurrences vs.
# 7 for "Allergies") and "Alcohlic" (57 vs. 8 for "Alcoholic") - the same
# kind of baked-in legacy-form quirk this project already preserves
# elsewhere (e.g. dump_xml_export.py's own "Allergiesd" XML tag).
REMARK_FIELD_LABELS: dict[str, str] = {
    "record_no": "Record No",
    "customer_name": "Customer Name",
    "email_address": "Email Address",
    "res_address": "Res Address",
    "city_1": "City_1",
    "state_1": "State_1",
    "zip_1": "Zip_1",
    "phno_1": "PhNo_1",
    "country_1": "Country_1",
    "sex_1": "Sex_1",
    "d_birth": "D_Birth",
    "height": "Height",
    "weight": "Weight",
    "blood_group": "Blood Group",
    "billing_name": "Billing Name",
    "shipper_name": "Shipper Name",
    "city_2": "City_2",
    "state_2": "State_2",
    "zip_2": "Zip_2",
    "country_2": "Country_2",
    "phno_2": "PHNo_2",
    "alcoholic": "Alcohlic",
    "smoker": "Smoker",
    "past_surg": "Past Surg",
    "diabetic": "Diabetic",
    "allergies": "Allergiesd",
    "policy_no": "Policy No",
    "d_b_life_assure": "D_B_Life_Assure",
    "p_inst": "P_Inst",
    "name_p_holder": "Name_P_Holder",
    "stm_name": "STM Name",
    "stm_code": "STM_Code",
    "dob": "DOB",
    "sex_2": "Sex_2",
    "card_name": "CARD Name",
    "medicine": "Medicine",
    "dosage": "Dosage",
    "tablets": "Tablets",
    "pill_rate": "Pill Rate",
    "cost": "Cost",
    "shipping_cost": "Shipping Cost",
    "total_amount": "Total Amount",
}


def _is_explicit_na(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().upper().rstrip(".") in {v.rstrip(".") for v in pat.EXPLICIT_NA_VALUES}


def generate_remark(fields: dict[str, ExtractedField]) -> str:
    invalid_labels: list[str] = []
    for name in OCR_FIELD_ORDER:
        field = fields.get(name)
        if field is None:
            continue
        # Each of the three conditions is independently sufficient - `or`,
        # not `elif`, so a field already caught by one (e.g. a genuine N.A
        # value) still only ever adds its label once, but an operator
        # marking a WELL-FORMATTED value invalid isn't silently skipped
        # just because neither of the other two happened to fire.
        if (
            _is_explicit_na(field.value)
            or (field.value and field.validation.format_valid is False)
            or field.marked_invalid
        ):
            invalid_labels.append(REMARK_FIELD_LABELS.get(name, field.label))

    if not invalid_labels:
        return "N.A"
    if len(invalid_labels) == 1:
        return f"{invalid_labels[0]} is invalid"
    if len(invalid_labels) == 2:
        return f"{invalid_labels[0]} and {invalid_labels[1]} are invalid"
    # No Oxford comma - verified against a golden legacy reference showing
    # "A, B, C and D are invalid" (no comma before "and") for a 5-field case.
    return f"{', '.join(invalid_labels[:-1])} and {invalid_labels[-1]} are invalid"
