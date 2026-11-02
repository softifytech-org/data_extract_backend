"""Whole-page Azure processing path - the production default.

Mirrors the exact pipeline validated all session in the scratchpad
confirmation rounds: ONE Azure call per page, record boundaries and every
field sliced from that single word list (record_splitter.py), never
re-analyzed per record crop. See ocr/azure_engine.py's module docstring
for why this differs from the Tesseract ensemble's per-crop, multi-variant
design.
"""
from __future__ import annotations

import numpy as np

from app.models.document import BoundingBox, RecordEntry
from app.services.extraction.aggregator import AggregatedField
from app.services.extraction.engine import RuleBasedExtractionEngine
from app.services.extraction.record_splitter import (
    choose_reference_boundaries,
    slice_words_by_boundaries,
)
from app.services.ocr.azure_engine import AzureDocumentIntelligenceEngine
from app.services.ocr.base import OCRPassResult
from app.services.processing import _add_derived_fields
from app.services.validation.pipeline import build_extracted_fields

# Azure's coordinates are precise (no deskew jitter the way Tesseract's
# preprocessing variants have), so a tight margin is correct here - a
# wider one (needed for Tesseract) was measured to pull several pixels of
# the NEXT record's own anchor row into the current record's slice.
_AZURE_BOUNDARY_MARGIN = 2


def process_page_azure(
    image: np.ndarray,
    *,
    engine: AzureDocumentIntelligenceEngine,
    extraction_engine: RuleBasedExtractionEngine,
    page_index: int,
    starting_record_index: int,
    confidence_high_threshold: float,
    original_filename: str,
) -> list[RecordEntry]:
    words = engine.recognize_words(image)
    if not words:
        return []
    boundaries = choose_reference_boundaries({"azure": words}, boundary_margin=_AZURE_BOUNDARY_MARGIN)
    if not boundaries:
        return []
    slices = slice_words_by_boundaries(words, boundaries)

    image_height, image_width = image.shape[0], image.shape[1]

    entries: list[RecordEntry] = []
    for i, sl in enumerate(slices):
        if not sl:
            continue
        boundary = boundaries[i]
        # Full page width, clipped y-range - matches how record_splitter
        # itself treats a boundary (a horizontal band across the whole
        # row), and is exactly the range slice_words_by_boundaries just
        # used to decide which words belong to this record. y_end can be
        # +inf for the last record on the page; clip it to the real image
        # height rather than storing an unusable value.
        region = BoundingBox(
            page=page_index,
            x=0,
            y=max(0, round(boundary.y_start)),
            width=image_width,
            height=max(1, round(min(boundary.y_end, image_height) - max(0, boundary.y_start))),
        )
        pass_result = OCRPassResult(
            variant="azure", page=page_index, ocr_config="azure-prebuilt-layout", words=sl
        )
        candidates = extraction_engine.extract(pass_result)
        # Single real OCR pass (no multi-pass ensemble for Azure) adapted
        # into the SAME AggregatedField shape build_extracted_fields
        # expects everywhere else, so format validation / confidence /
        # status / the review tiers all run exactly as they do for the
        # Tesseract path - agreement_ratio is architecturally 1.0 here
        # (one pass trivially "agrees" with itself; see the always-review
        # tier in field_reliability.py, which exists specifically because
        # this is not a real multi-pass agreement signal).
        aggregated = {
            name: AggregatedField(
                field_name=name,
                value=c.value,
                raw_ocr_text=c.raw_ocr_text,
                ocr_confidence=c.confidence,
                agreement_ratio=1.0,
                passes_agreeing=1 if c.matched else 0,
                passes_total=1,
                bounding_box=c.bounding_box,
                per_pass_values={"azure": c.value},
                note=c.note,
            )
            for name, c in candidates.items()
        }
        fields, warnings, overall_confidence = build_extracted_fields(
            aggregated, confidence_high_threshold=confidence_high_threshold
        )
        _add_derived_fields(fields, original_filename=original_filename)
        entries.append(
            RecordEntry(
                record_index=starting_record_index + i,
                page=page_index,
                region=region,
                fields=fields,
                cross_field_warnings=warnings,
                overall_confidence=overall_confidence,
            )
        )
    return entries
