"""Orchestrates the full validation pass: format checks + cross-field rules
+ confidence scoring, turning AggregatedField data into the final
ExtractedField objects stored on the document record.
"""
from __future__ import annotations

from app.models.document import ExtractedField, FieldSource, FieldValidation
from app.services.extraction.aggregator import AggregatedField
from app.services.extraction.fields import FIELD_SPEC_BY_NAME
from app.services.validation.confidence import compute_confidence, field_status_for_score
from app.services.validation.consistency import run_consistency_checks
from app.services.validation.field_validation import validate_field_format


def build_extracted_fields(
    aggregated: dict[str, AggregatedField], *, confidence_high_threshold: float
) -> tuple[dict[str, ExtractedField], list[str], float]:
    values = {name: agg.value for name, agg in aggregated.items()}
    consistency = run_consistency_checks(values)

    fields: dict[str, ExtractedField] = {}
    for name, agg in aggregated.items():
        spec = FIELD_SPEC_BY_NAME[name]
        format_valid = validate_field_format(name, agg.value)
        business_valid = consistency.field_validity.get(name)

        score = compute_confidence(
            ocr_confidence=agg.ocr_confidence,
            agreement_ratio=agg.agreement_ratio,
            format_valid=format_valid,
            business_rule_valid=business_valid,
        )
        status = field_status_for_score(
            score, confidence_high_threshold, has_value=bool(agg.value), field_name=name
        )

        notes: list[str] = []
        if not agg.value:
            notes.append("Not found in document - please enter manually")
        else:
            if format_valid is False:
                notes.append(f"Value does not match the expected {spec.data_type.value} format")
            if agg.passes_total > 0 and agg.agreement_ratio < 1.0:
                notes.append(f"OCR passes agreed {agg.passes_agreeing}/{agg.passes_total}")
            if business_valid is False:
                notes.append("Inconsistent with a related field - see document warnings")
            if agg.note:
                notes.append(agg.note)

        fields[name] = ExtractedField(
            field_name=name,
            label=spec.label,
            value=agg.value,
            original_ocr_value=agg.raw_ocr_text,
            data_type=spec.data_type,
            confidence=score,
            status=status,
            source=FieldSource.OCR,
            validation=FieldValidation(
                format_valid=format_valid,
                ocr_agreement=(agg.agreement_ratio >= 0.999) if agg.value else None,
                ocr_agreement_ratio=agg.agreement_ratio if agg.value else None,
                business_rule_valid=business_valid,
                notes=notes,
            ),
            bounding_box=agg.bounding_box,
        )

    overall_confidence = (
        round(sum(f.confidence for f in fields.values()) / len(fields), 1) if fields else 0.0
    )
    return fields, consistency.warnings, overall_confidence
