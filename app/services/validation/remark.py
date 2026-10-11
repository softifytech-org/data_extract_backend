"""Derives the legacy form's Remark field per BN_WorkFlow.pdf rule 43
("Remarks Rule (Most Important)"): if every field is valid, Remark = "N.A";
otherwise Remark names the invalid field(s) ("X is invalid" / "X and Y are
invalid" / "X, Y, and Z are invalid").

A field counts as invalid here when either:
  - its value is an explicit N.A marker in the source ("N.A", "NA", ...) -
    the spec's "Common Rule (N.A Handling)" treats a confirmed-blank source
    field as invalid and requires it to be called out in Remark, or
  - it has a value but that value failed its own format validation
    (ExtractedField.validation.format_valid is False - see
    services/validation/field_validation.py for the per-field rules).

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
        if _is_explicit_na(field.value):
            invalid_labels.append(field.label)
        elif field.value and field.validation.format_valid is False:
            invalid_labels.append(field.label)

    if not invalid_labels:
        return "N.A"
    if len(invalid_labels) == 1:
        return f"{invalid_labels[0]} is invalid"
    if len(invalid_labels) == 2:
        return f"{invalid_labels[0]} and {invalid_labels[1]} are invalid"
    # No Oxford comma - verified against a golden legacy reference showing
    # "A, B, C and D are invalid" (no comma before "and") for a 5-field case.
    return f"{', '.join(invalid_labels[:-1])} and {invalid_labels[-1]} are invalid"
