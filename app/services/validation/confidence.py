"""Confidence scoring: combines every independent validation signal into one
transparent 0-100 score per field.

This score is an internal risk/trust score, not a statistical probability of
correctness - it should never be presented to a user as "95% chance this is
right." It exists purely to decide which fields can be shown as verified and
which must be routed to human review.

Weighting rationale (documented here rather than left implicit):
- OCR confidence (40%): Tesseract's own signal for how clearly the glyphs
  were recognized. Necessary but explicitly NOT sufficient on its own -
  see the module docstring in services/validation/field_validation.py.
- OCR agreement (30%): whether independent preprocessing/OCR passes produced
  the same value. Weighted second-highest because agreement across genuinely
  different image variants is hard to get "by accident" - it's a strong
  independent signal that low-level pixel noise isn't fooling us all four times.
- Format validity (20%): does the value match what this field type expects.
  Catches structurally wrong values even when OCR itself was "confident."
- Business-rule / cross-field validity (10%): only applicable to a few
  fields (quantity/unit_price/total_amount, the two dates). Lower weight
  because it's unavailable for most fields, not because it matters less
  where it does apply - a business-rule failure is still surfaced as its
  own explicit warning regardless of the numeric score.

Any component that doesn't apply (None) is excluded from both the numerator
and the weight denominator, so its absence never silently drags the score
down or up.
"""
from __future__ import annotations

from app.models.document import FieldStatus

WEIGHTS = {
    "ocr_confidence": 0.40,
    "ocr_agreement": 0.30,
    "format_valid": 0.20,
    "business_rule_valid": 0.10,
}


def compute_confidence(
    *,
    ocr_confidence: float,
    agreement_ratio: float,
    format_valid: bool | None,
    business_rule_valid: bool | None,
) -> float:
    components: dict[str, float] = {
        "ocr_confidence": max(0.0, min(100.0, ocr_confidence)),
        "ocr_agreement": max(0.0, min(1.0, agreement_ratio)) * 100.0,
    }
    if format_valid is not None:
        components["format_valid"] = 100.0 if format_valid else 0.0
    if business_rule_valid is not None:
        components["business_rule_valid"] = 100.0 if business_rule_valid else 0.0

    total_weight = sum(WEIGHTS[k] for k in components)
    if total_weight == 0:
        return 0.0
    score = sum(components[k] * WEIGHTS[k] for k in components) / total_weight
    return round(score, 1)


def confidence_tier(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def field_status_for_score(
    score: float, high_threshold: float, *, has_value: bool, field_name: str | None = None
) -> FieldStatus:
    if not has_value:
        return FieldStatus.NEEDS_REVIEW
    # A field with a calibrated, measured-reliable threshold (see
    # services/validation/field_reliability.py) can auto-accept at a lower
    # confidence score than the global default - but only once real
    # held-out benchmark data has earned it that lower bar, never by default.
    effective_threshold = high_threshold
    if field_name is not None:
        from app.services.validation.field_reliability import threshold_for_field

        effective_threshold = threshold_for_field(field_name, high_threshold)
    return FieldStatus.VERIFIED if score >= effective_threshold else FieldStatus.NEEDS_REVIEW
