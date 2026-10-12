"""Top-level pipeline orchestrator.

Wires together, in order: page loading (image or PDF) -> quick boundary
detection -> per-record CROPPING -> preprocessing -> multi-pass OCR (per
crop) -> extraction -> aggregation -> validation -> confidence scoring.
Nothing in here does any of those jobs itself - it only sequences calls
into the dedicated service modules, so each stage stays independently
testable and replaceable (see module docstrings under services/*).

Crop-first, not OCR-then-sort: earlier versions ran OCR once on the whole
dense page and then sorted the resulting words into per-record buckets by
Y position. That has two real costs on a page this dense - a record's
text competes for detection/recognition attention with 13 others in one
pass, and a word landing a few pixels on the wrong side of a shared
boundary (common when a preprocessing variant is rotated) silently drops
out of its correct record. Cropping each record out of the ORIGINAL image
first and re-running the full preprocessing+OCR ensemble on just that
small crop fixes both: more effective resolution per record, and no
cross-record boundary ambiguity since each crop is processed independently
from its own pixels. See services/preprocessing/record_cropping.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.models.document import DataType, DocumentRecord, DocumentStatus, ExtractedField, FieldStatus, RecordEntry
from app.models.document import utcnow
from app.services.errors import EmptyOcrResultError, InvalidImageError
from app.services.extraction.aggregator import aggregate_passes
from app.services.extraction.engine import RuleBasedExtractionEngine
from app.services.extraction.fields import FIELD_SPEC_BY_NAME
from app.services.extraction.record_splitter import choose_reference_boundaries
from app.services.ocr.base import OCREngine
from app.services.ocr.pipeline import run_multi_pass_ocr
from app.services.preprocessing.pdf_processing import pdf_to_images
from app.services.preprocessing.record_cropping import crop_record
from app.services.storage.json_storage import JsonDocumentRepository
from app.services.validation.pipeline import build_extracted_fields
from app.services.validation.remark import generate_remark

# Record crops are typically only 60-150px tall - much smaller than the
# 2x used for a full page - so they need more aggressive magnification to
# reach a legible effective character height for OCR.
RECORD_CROP_UPSCALE_FACTOR = 4.0

logger = get_logger(__name__)


def _load_pages(file_path: Path, file_type: str, poppler_path: str) -> list[np.ndarray]:
    if file_type == "pdf":
        return pdf_to_images(file_path, poppler_path)
    image = cv2.imread(str(file_path))
    if image is None:
        raise InvalidImageError("The uploaded file could not be read as an image")
    return [image]


def _add_derived_fields(fields: dict[str, ExtractedField], *, original_filename: str) -> None:
    """image_name and remark are never OCR-extracted - image_name comes from
    the upload itself, remark is computed from which OTHER fields the
    source document explicitly marked N/A. See services/validation/remark.py."""
    image_spec = FIELD_SPEC_BY_NAME["image_name"]
    fields["image_name"] = ExtractedField(
        field_name="image_name",
        label=image_spec.label,
        value=original_filename,
        original_ocr_value=None,
        data_type=DataType.TEXT,
        confidence=100.0,
        status=FieldStatus.VERIFIED,
        bounding_box=None,
    )
    remark_spec = FIELD_SPEC_BY_NAME["remark"]
    remark_value = generate_remark(fields)
    # Verified against 250 held-out records: remark values marked VERIFIED
    # are wrong 97% of the time - it's derived from all 43 other fields'
    # own validation status rather than read from the image, so it has no
    # independent confidence signal of its own and was never a candidate
    # for auto-acceptance in the first place (see ALWAYS_REVIEW_FIELDS).
    # Hardcoding VERIFIED here bypassed that entirely; route it through the
    # same review-tier decision everything else uses instead.
    from app.core.config import get_settings
    from app.services.validation.field_reliability import ALWAYS_REVIEW_FIELDS

    remark_status = FieldStatus.VERIFIED
    if get_settings().review_tiers_enabled and "remark" in ALWAYS_REVIEW_FIELDS:
        remark_status = FieldStatus.NEEDS_REVIEW
    fields["remark"] = ExtractedField(
        field_name="remark",
        label=remark_spec.label,
        value=remark_value or None,
        original_ocr_value=None,
        data_type=DataType.TEXT,
        confidence=100.0,
        status=remark_status,
        bounding_box=None,
    )


def _find_record_boundaries(image: np.ndarray, ocr_engine: OCREngine, *, page_index: int):
    """Quick whole-page pass whose ONLY job is locating where each record
    starts/ends (see record_splitter.choose_reference_boundaries) - not
    used for field extraction, which happens per-crop below."""
    pass_results = run_multi_pass_ocr(image, ocr_engine, page=page_index)
    words_by_variant = {pr.variant: pr.words for pr in pass_results}
    return choose_reference_boundaries(words_by_variant)


def _process_record_crop(
    crop: np.ndarray,
    *,
    y_offset: int,
    page_index: int,
    record_index: int,
    ocr_engine: OCREngine,
    paddle_engine: OCREngine | None,
    extraction_engine: RuleBasedExtractionEngine,
    confidence_high_threshold: float,
    paddle_escalation_threshold: float,
    original_filename: str,
) -> RecordEntry:
    """Runs the full preprocessing + multi-pass OCR ensemble on ONE
    record's cropped image, then extracts/aggregates/validates it - the
    exact same per-record pipeline as before, just fed from a crop-local
    OCR pass instead of a slice of whole-page words."""
    pass_results = run_multi_pass_ocr(
        crop, ocr_engine, page=page_index, upscale_factor=RECORD_CROP_UPSCALE_FACTOR
    )

    pass_candidates: dict[str, dict] = {}
    for pass_result in pass_results:
        for word in pass_result.words:
            word.y += y_offset  # crop-local -> full-page coordinate space
        pass_candidates[pass_result.variant] = extraction_engine.extract(pass_result)

    needs_paddle = paddle_engine is not None
    if needs_paddle:
        # Adaptive escalation: on this CPU-only setup, PaddleOCR costs far
        # more than the entire 4-pass Tesseract ensemble combined. Only pay
        # for it when Tesseract alone ISN'T already confident - a quick
        # preliminary aggregation decides this before the expensive call,
        # so a record the Tesseract ensemble already agrees on doesn't wait
        # on PaddleOCR at all. Accuracy is unaffected where it matters: a
        # genuinely uncertain record still gets the full ensemble.
        preliminary = aggregate_passes(pass_candidates)
        _, _, preliminary_confidence = build_extracted_fields(
            preliminary, confidence_high_threshold=confidence_high_threshold
        )
        needs_paddle = preliminary_confidence < paddle_escalation_threshold

    if needs_paddle:
        # Now feasible speed-wise: PaddleOCR on a small crop is a fraction
        # of the cost of running it on the whole dense page.
        paddle_result = paddle_engine.recognize(crop, page=page_index, variant="paddleocr")
        for word in paddle_result.words:
            word.y += y_offset
        pass_candidates["paddleocr"] = extraction_engine.extract(paddle_result)

    aggregated = aggregate_passes(pass_candidates)
    fields, warnings, overall_confidence = build_extracted_fields(
        aggregated, confidence_high_threshold=confidence_high_threshold
    )
    _add_derived_fields(fields, original_filename=original_filename)
    return RecordEntry(
        record_index=record_index,
        page=page_index,
        fields=fields,
        cross_field_warnings=warnings,
        overall_confidence=overall_confidence,
    )


def _process_page(
    image: np.ndarray,
    *,
    page_index: int,
    ocr_engine: OCREngine,
    paddle_engine: OCREngine | None,
    extraction_engine: RuleBasedExtractionEngine,
    confidence_high_threshold: float,
    paddle_escalation_threshold: float,
    starting_record_index: int,
    original_filename: str,
) -> list[RecordEntry]:
    boundaries = _find_record_boundaries(image, ocr_engine, page_index=page_index)
    if not boundaries:
        return []

    entries: list[RecordEntry] = []
    for i, boundary in enumerate(boundaries):
        crop, y_offset = crop_record(image, boundary.y_start, boundary.y_end)
        entries.append(
            _process_record_crop(
                crop,
                y_offset=y_offset,
                page_index=page_index,
                record_index=starting_record_index + i,
                ocr_engine=ocr_engine,
                paddle_engine=paddle_engine,
                extraction_engine=extraction_engine,
                confidence_high_threshold=confidence_high_threshold,
                paddle_escalation_threshold=paddle_escalation_threshold,
                original_filename=original_filename,
            )
        )
    return entries


def _flag_duplicate_record_numbers(entries: list[RecordEntry]) -> None:
    """Document-level safety net added after a real bug: a crop-boundary
    error once caused two different records to both read the same Record
    No, silently corrupting one of them. Record No is supposed to uniquely
    identify a record, so two records sharing a value is itself evidence
    something is wrong - this doesn't fix the cause, it makes sure that
    exact failure mode can never again pass through silently. Downgrades
    the record_no field's own status on every record sharing a duplicate
    value, and adds a warning explaining why."""
    by_value: dict[str, list[RecordEntry]] = {}
    for entry in entries:
        field = entry.fields.get("record_no")
        if field and field.value:
            by_value.setdefault(field.value, []).append(entry)

    for value, dupes in by_value.items():
        if len(dupes) < 2:
            continue
        all_indices = [e.record_index for e in dupes]
        for entry in dupes:
            others = ", ".join(str(i) for i in all_indices if i != entry.record_index)
            field = entry.fields["record_no"]
            field.status = FieldStatus.NEEDS_REVIEW
            field.confidence = min(field.confidence, 40.0)
            field.validation.notes.append(
                f"Record No '{value}' is duplicated across records {', '.join(str(i) for i in all_indices)} - "
                "at least one is almost certainly wrong"
            )
            entry.cross_field_warnings.append(f"Record No '{value}' also appears on record(s) {others}")


def process_document(
    document_id: str,
    *,
    file_path: Path,
    file_type: str,
    settings: Settings,
    repo: JsonDocumentRepository,
    ocr_engine: OCREngine,
    paddle_engine: OCREngine | None = None,
) -> DocumentRecord:
    record = repo.get(document_id)
    record.status = DocumentStatus.PROCESSING
    record.touch()
    repo.save(record)

    try:
        images = _load_pages(file_path, file_type, settings.poppler_path)
        record.source_file.pages = len(images)

        extraction_engine = RuleBasedExtractionEngine()
        all_entries: list[RecordEntry] = []
        # Azure gets its own whole-page pipeline (one API call per page,
        # not per record crop or per preprocessing variant) - see
        # services/azure_processing.py's module docstring for why this is
        # a genuinely different, separately-validated path rather than a
        # drop-in engine swap into the Tesseract ensemble below.
        from app.services.ocr.azure_engine import AzureDocumentIntelligenceEngine

        if isinstance(ocr_engine, AzureDocumentIntelligenceEngine):
            from app.services.azure_processing import process_page_azure

            for page_index, image in enumerate(images, start=1):
                page_entries = process_page_azure(
                    image,
                    engine=ocr_engine,
                    extraction_engine=extraction_engine,
                    page_index=page_index,
                    starting_record_index=len(all_entries) + 1,
                    confidence_high_threshold=settings.confidence_high_threshold,
                    original_filename=record.source_file.original_filename,
                )
                all_entries.extend(page_entries)
        else:
            for page_index, image in enumerate(images, start=1):
                page_entries = _process_page(
                    image,
                    page_index=page_index,
                    ocr_engine=ocr_engine,
                    paddle_engine=paddle_engine,
                    extraction_engine=extraction_engine,
                    confidence_high_threshold=settings.confidence_high_threshold,
                    paddle_escalation_threshold=settings.confidence_medium_threshold,
                    starting_record_index=len(all_entries) + 1,
                    original_filename=record.source_file.original_filename,
                )
                all_entries.extend(page_entries)

        if not all_entries:
            raise EmptyOcrResultError(
                "No record identifiers could be found in this document"
            )

        _flag_duplicate_record_numbers(all_entries)

        record.records = all_entries
        record.record_count = len(all_entries)
        record.overall_confidence = round(
            sum(r.overall_confidence or 0.0 for r in all_entries) / len(all_entries), 1
        )
        needs_review = any(
            f.status == FieldStatus.NEEDS_REVIEW
            for r in all_entries
            for f in r.fields.values()
        )
        record.status = DocumentStatus.NEEDS_REVIEW if needs_review else DocumentStatus.PROCESSED
        record.audit.processed_at = utcnow()
        record.error_message = None
        record.touch()
        repo.save(record)

        review_count = sum(
            1 for r in all_entries for f in r.fields.values() if f.status == FieldStatus.NEEDS_REVIEW
        )
        total_fields = sum(len(r.fields) for r in all_entries)
        logger.info(
            "Document %s processed: %d record(s), %d fields extracted, %d need review",
            document_id, len(all_entries), total_fields, review_count,
        )
        return record

    except Exception:
        record.status = DocumentStatus.FAILED
        record.error_message = "Processing failed. The document may be unreadable or corrupted."
        record.touch()
        repo.save(record)
        logger.exception("Document %s processing failed", document_id)
        raise
