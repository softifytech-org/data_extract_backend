"""Combines per-pass extraction candidates into one value per field.

This is where "Validation B - Multiple OCR Agreement" (see services/validation)
gets its raw material: for every field we know exactly how many preprocessing
passes agreed on the same value out of how many were run.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.document import BoundingBox
from app.services.extraction import patterns as pat
from app.services.extraction.engine import FieldCandidate
from app.services.extraction.fields import OCR_FIELD_ORDER

_CURRENCY_FIELDS = {"p_inst", "pill_rate", "cost", "shipping_cost", "total_amount"}
_FLAG_FIELDS = ("alcoholic", "smoker", "past_surg", "diabetic", "allergies")
_NA_SPELLINGS = {v.rstrip(".") for v in pat.EXPLICIT_NA_VALUES}


def _choose_complete_flag_pass(
    pass_candidates: dict[str, dict[str, FieldCandidate]],
) -> str | None:
    """The 5 health flags print as one coherent run, so they're chosen as a
    unit from the single pass that read the COMPLETE run (all 5 matched),
    not voted per-field - per-field voting let fragments from passes that
    only caught part of the run (e.g. 'NA' from a misaligned run) outvote
    the one pass that read all five correctly. Returns None when no pass
    produced the complete run, in which case per-field voting still applies."""
    best: str | None = None
    best_score = -1.0
    for variant, field_map in pass_candidates.items():
        cands = [field_map.get(f) for f in _FLAG_FIELDS]
        if not all(c is not None and c.matched and c.value for c in cands):
            continue
        score = sum(c.confidence for c in cands)
        if score > best_score:
            best, best_score = variant, score
    return best


def _looks_like_confusable_currency_typo(value: str) -> bool:
    """True for a $-prefixed value containing a letter in place of a digit
    (e.g. "$2.o0.00") rather than a clean number. Used only to decide which
    candidate GROUP to trust more, never to alter a value's text."""
    return value.startswith("$") and any(ch.isalpha() for ch in value[1:])


def _vote_group_key(value: str) -> str:
    """Groups textually-different spellings of the same semantic answer for
    VOTING purposes only ("NA" and "N.A" both mean explicit not-applicable).
    Without this, 3 passes agreeing on the same *meaning* but split across
    2 spellings ("NA" x2, "N.A" x1) can lose a vote to a single noisier
    pass reading something else entirely ("NO" x1) just because neither
    spelling variant alone had the plurality - verified as the exact cause
    of Alcoholic returning the wrong flag despite 3 of the passes agreeing.
    The actual stored value always comes from a real candidate's literal
    text (see winner selection below), never this normalized key."""
    stripped = value.strip().upper().rstrip(".")
    return "N.A" if stripped in _NA_SPELLINGS else value


@dataclass
class AggregatedField:
    field_name: str
    value: str | None  # best-interpreted value (e.g. fuzzy-corrected against a closed vocabulary)
    raw_ocr_text: str | None  # exactly what OCR read, uncorrected - for traceability/review
    ocr_confidence: float  # 0-100, avg confidence of the passes that produced the winning value
    agreement_ratio: float  # winning_pass_count / total_passes_run
    passes_agreeing: int
    passes_total: int
    bounding_box: BoundingBox | None
    per_pass_values: dict[str, str | None]
    note: str | None = None


def aggregate_passes(
    pass_candidates: dict[str, dict[str, FieldCandidate]],
) -> dict[str, AggregatedField]:
    """pass_candidates: {variant_name: {field_name: FieldCandidate}}"""
    total_passes = len(pass_candidates) or 1
    aggregated: dict[str, AggregatedField] = {}
    flag_pass = _choose_complete_flag_pass(pass_candidates)

    for field_name in OCR_FIELD_ORDER:
        per_pass_values: dict[str, str | None] = {}
        matched: list[tuple[str, FieldCandidate]] = []
        for variant, field_map in pass_candidates.items():
            candidate = field_map.get(field_name)
            per_pass_values[variant] = candidate.value if candidate else None
            if candidate and candidate.matched and candidate.value:
                matched.append((variant, candidate))

        if field_name in _FLAG_FIELDS and flag_pass is not None:
            chosen = pass_candidates[flag_pass][field_name]
            aggregated[field_name] = AggregatedField(
                field_name=field_name,
                value=chosen.value,
                raw_ocr_text=chosen.raw_ocr_text,
                ocr_confidence=chosen.confidence,
                agreement_ratio=sum(1 for v, _ in matched if per_pass_values.get(v) == chosen.value) / total_passes,
                passes_agreeing=sum(1 for v, _ in matched if per_pass_values.get(v) == chosen.value),
                passes_total=total_passes,
                bounding_box=chosen.bounding_box,
                per_pass_values=per_pass_values,
                note=chosen.note,
            )
            continue

        if not matched:
            aggregated[field_name] = AggregatedField(
                field_name=field_name,
                value=None,
                raw_ocr_text=None,
                ocr_confidence=0.0,
                agreement_ratio=0.0,
                passes_agreeing=0,
                passes_total=total_passes,
                bounding_box=None,
                per_pass_values=per_pass_values,
            )
            continue

        # Group candidates by value, then pick the group with the highest
        # TOTAL confidence - not the group with the most votes. Pure vote
        # counting would let three low-confidence Tesseract passes always
        # beat one high-confidence PaddleOCR pass just by outnumbering it,
        # even when that one pass is clearly the more trustworthy reading.
        # Summing confidence still rewards genuine multi-pass agreement
        # (more supporting passes -> higher sum, all else equal) while
        # letting a single much-more-confident engine win when it should.
        groups: dict[str, list[FieldCandidate]] = {}
        for _, candidate in matched:
            groups.setdefault(_vote_group_key(candidate.value), []).append(candidate)
        winning_key, supporting = max(
            groups.items(), key=lambda item: sum(c.confidence for c in item[1])
        )
        # The stored value is the winning GROUP's own highest-confidence
        # candidate's literal text, not the normalized vote key - grouping
        # "NA"/"N.A" together for counting purposes never changes what
        # actually gets saved.
        winning_value = max(supporting, key=lambda c: c.confidence).value

        # Source-fidelity override for currency fields: if ANY pass read a
        # digit-confusable letter where another pass read a clean digit
        # (e.g. one pass says "$2.o0.00", another says "$2.00"), that
        # disagreement is itself evidence the printed character is
        # non-standard - a preprocessing pass (denoise/threshold) smoothing
        # a stylized character into looking more digit-like is a more
        # likely explanation than the "clean" pass being uniquely correct.
        # Preserving the anomaly beats voting it away, per explicit
        # instruction: OCR must reproduce exactly what's in the image, not
        # the most statistically popular guess at what it "should" be.
        if field_name in _CURRENCY_FIELDS and not _looks_like_confusable_currency_typo(winning_value):
            confusable_values = [v for v in groups if _looks_like_confusable_currency_typo(v)]
            if confusable_values:
                winning_value = max(
                    confusable_values, key=lambda v: sum(c.confidence for c in groups[v])
                )
                supporting = groups[winning_value]

        winning_count = len(supporting)
        ocr_confidence = sum(c.confidence for c in supporting) / len(supporting)
        best_bbox_candidate = max(supporting, key=lambda c: c.confidence)

        aggregated[field_name] = AggregatedField(
            field_name=field_name,
            value=winning_value,
            raw_ocr_text=best_bbox_candidate.raw_ocr_text,
            ocr_confidence=ocr_confidence,
            agreement_ratio=winning_count / total_passes,
            passes_agreeing=winning_count,
            passes_total=total_passes,
            bounding_box=best_bbox_candidate.bounding_box,
            per_pass_values=per_pass_values,
            note=best_bbox_candidate.note,
        )

    return aggregated
