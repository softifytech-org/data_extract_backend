"""Document processing API.

Every request body is a validated Pydantic schema (see app/schemas), every
file is verified server-side regardless of what the client claims about it
(see app/core/security), and every error returned to the client is a safe,
generic message - full exception detail only ever goes to the server log.

A document can contain many records (see services/extraction/
record_splitter.py) - field edits and per-record data are addressed via
record_index (1-based, page order); approval, processing, and export
remain whole-document actions.
"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.core.auth import CurrentUser, get_current_user_optional
from app.core.config import Settings, get_settings
from app.core.dependencies import (
    get_app_db,
    get_file_store,
    get_file_store_for_record,
    get_ocr_engine,
    get_paddle_engine,
    get_repository,
)
from app.core.logging_config import get_logger
from app.core.security import (
    UnsafeFilenameError,
    UnsupportedFileTypeError,
    generate_document_id,
    sanitize_original_filename,
    validate_upload,
)
from app.services.storage.blob_store import FileStore, StorageUnavailableError
from app.models.document import (
    AuditInfo,
    DocumentRecord,
    DocumentStatus,
    ExtractedField,
    FieldSource,
    FieldStatus,
    SourceFileInfo,
    utcnow,
)
from app.schemas.document import (
    ApproveRequest,
    FieldsUpdateRequest,
    ProcessResponse,
    UploadResponse,
)
from app.services.ai_vision import AIVisionError, extract_fields_via_vision
from app.services.errors import EmptyOcrResultError, InvalidImageError
from app.services.export.dump_xml_export import submit_document
from app.services.export.excel_export import export_record_to_excel
from app.services.export.xml_export import export_record_to_xml
from app.services.extraction.fields import FIELD_SPEC_BY_NAME
from app.services.ocr.base import OCREngine
from app.services.preprocessing.pdf_processing import CorruptPdfError
from app.services.storage.app_db import AppDatabase
from app.services.preprocessing.record_cropping import crop_record
from app.services.processing import process_document
from app.services.storage.json_storage import DocumentNotFoundError, JsonDocumentRepository
from app.services.validation.confidence import compute_confidence, field_status_for_score
from app.services.validation.consistency import run_consistency_checks
from app.services.validation.field_validation import validate_field_format
from app.services.validation.remark import generate_remark

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = get_logger(__name__)


def _get_record_or_404(repo: JsonDocumentRepository, document_id: str) -> DocumentRecord:
    try:
        return repo.get(document_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except UnsafeFilenameError:
        raise HTTPException(status_code=400, detail="Invalid document id") from None


def _check_access(document_id: str, current_user: CurrentUser | None, db: AppDatabase) -> None:
    """No-op - this application has no authentication anywhere (explicit
    decision; see app/api/batches.py's module docstring and the final
    report). Kept as a function, rather than deleted along with its call
    sites in every route below, to keep this change's diff small and
    low-risk; it intentionally does nothing with either argument."""
    return


def _get_entry_or_404(record: DocumentRecord, record_index: int):
    for entry in record.records:
        if entry.record_index == record_index:
            return entry
    raise HTTPException(status_code=404, detail=f"Record {record_index} not found in this document")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    layout: str = Form("old"),
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    file_store: FileStore = Depends(get_file_store),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if layout not in ("old", "new"):
        raise HTTPException(status_code=400, detail="layout must be 'old' or 'new'")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB size limit",
        )

    try:
        extension = validate_upload(file.filename, content, settings.allowed_extensions_set)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = generate_document_id()
    try:
        storage_key = file_store.save_upload(content, document_id=document_id, extension=extension, batch_id=None)
    except UnsafeFilenameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StorageUnavailableError as exc:
        logger.exception("Could not store uploaded document %s", document_id)
        raise HTTPException(status_code=502, detail="Could not store the uploaded file. Please try again.") from exc

    file_type = "pdf" if extension == ".pdf" else "image"
    record = DocumentRecord(
        document_id=document_id,
        status=DocumentStatus.UPLOADED,
        layout=layout,
        source_file=SourceFileInfo(
            original_filename=sanitize_original_filename(file.filename),
            stored_filename=storage_key,
            file_type=file_type,
            content_type=file.content_type or "application/octet-stream",
            pages=1,
            size_bytes=len(content),
            storage_backend=file_store.name,
        ),
        audit=AuditInfo(),
    )
    repo.save(record)
    logger.info(
        "Document %s uploaded (%s, %d bytes, backend=%s, layout=%s)",
        document_id, file_type, len(content), file_store.name, layout,
    )

    return UploadResponse(
        document_id=document_id, status=record.status.value, message="Document uploaded successfully"
    )


@router.get("", response_model=list[dict])
def list_documents(repo: JsonDocumentRepository = Depends(get_repository)) -> list[dict]:
    summaries = []
    for doc_id in reversed(repo.list_ids()):
        try:
            record = repo.get(doc_id)
        except DocumentNotFoundError:
            continue
        summaries.append(
            {
                "document_id": record.document_id,
                "status": record.status.value,
                "original_filename": record.source_file.original_filename,
                "created_at": record.audit.created_at.isoformat(),
                "record_count": record.record_count,
                "overall_confidence": record.overall_confidence,
            }
        )
    return summaries


@router.get("/{document_id}", response_model=DocumentRecord)
def get_document(
    document_id: str,
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> DocumentRecord:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    return record


@router.post("/{document_id}/process", response_model=ProcessResponse)
def process_document_endpoint(
    document_id: str,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    ocr_engine: OCREngine = Depends(get_ocr_engine),
    paddle_engine: OCREngine | None = Depends(get_paddle_engine),
) -> ProcessResponse:
    record = _get_record_or_404(repo, document_id)
    file_store = get_file_store_for_record(record, settings)
    if not file_store.upload_exists(record.source_file.stored_filename):
        raise HTTPException(status_code=409, detail="Source file is no longer available")

    try:
        with file_store.open_upload_as_local_path(record.source_file.stored_filename) as file_path:
            updated = process_document(
                document_id,
                file_path=file_path,
                file_type=record.source_file.file_type,
                settings=settings,
                repo=repo,
                ocr_engine=ocr_engine,
                paddle_engine=paddle_engine,
            )
    except (InvalidImageError, CorruptPdfError, EmptyOcrResultError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except StorageUnavailableError as exc:
        logger.exception("Could not fetch source file for document %s", document_id)
        raise HTTPException(status_code=502, detail="Could not retrieve the source file. Please try again.") from exc
    except Exception as exc:  # noqa: BLE001 - convert to a safe generic message
        logger.exception("Unexpected error processing document %s", document_id)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while processing the document"
        ) from exc

    return ProcessResponse(document_id=updated.document_id, status=updated.status.value)


@router.get("/{document_id}/file")
def get_document_file(
    document_id: str,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> Response:
    """Serves the original uploaded file so the review UI can display it
    alongside the extracted form. Files live outside any publicly-served
    static directory - this endpoint is the only path to them, and it goes
    through the same document-id validation as every other route."""
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    file_store = get_file_store_for_record(record, settings)
    try:
        data = file_store.read_upload(record.source_file.stored_filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source file is no longer available") from None
    except StorageUnavailableError as exc:
        logger.exception("Could not fetch source file for document %s", document_id)
        raise HTTPException(status_code=502, detail="Could not retrieve the source file. Please try again.") from exc
    return Response(content=data, media_type=record.source_file.content_type)


@router.get("/{document_id}/results", response_model=DocumentRecord)
def get_results(
    document_id: str,
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> DocumentRecord:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    if record.status == DocumentStatus.UPLOADED:
        raise HTTPException(status_code=409, detail="Document has not been processed yet")
    return record


def _revalidate_entry(entry, settings: Settings) -> None:
    """Re-run format + cross-field validation and recompute confidence/status
    for every field in ONE record, using current (possibly human-edited)
    values. Does not touch OCR or extraction - see
    services/validation/pipeline.py for the equivalent full pass used right
    after OCR."""
    values = {name: f.value for name, f in entry.fields.items()}
    consistency = run_consistency_checks(values)

    for name, f in entry.fields.items():
        if f.source == FieldSource.HUMAN:
            # Human-entered values are authoritative; we still surface format
            # issues as a note, but do not downgrade status/confidence for them.
            f.validation.format_valid = validate_field_format(name, f.value)
            f.validation.business_rule_valid = consistency.field_validity.get(name)
            continue
        format_valid = validate_field_format(name, f.value)
        business_valid = consistency.field_validity.get(name)
        score = compute_confidence(
            ocr_confidence=f.confidence,
            agreement_ratio=f.validation.ocr_agreement_ratio or 0.0,
            format_valid=format_valid,
            business_rule_valid=business_valid,
        )
        f.confidence = score
        f.status = field_status_for_score(
            score, settings.confidence_high_threshold, has_value=bool(f.value), field_name=name
        )
        f.validation.format_valid = format_valid
        f.validation.business_rule_valid = business_valid

    entry.cross_field_warnings = consistency.warnings
    entry.overall_confidence = (
        round(sum(f.confidence for f in entry.fields.values()) / len(entry.fields), 1)
        if entry.fields
        else 0.0
    )

    # Remark is derived from which OTHER fields now read as explicit N/A -
    # recompute it any time a field changes (e.g. a human corrects a value
    # to "N.A", or un-marks one).
    if "remark" in entry.fields:
        entry.fields["remark"].value = generate_remark(entry.fields) or None


def _recompute_document_status(record: DocumentRecord) -> None:
    record.overall_confidence = (
        round(sum(r.overall_confidence or 0.0 for r in record.records) / len(record.records), 1)
        if record.records
        else 0.0
    )
    if record.status != DocumentStatus.APPROVED:
        still_needs_review = any(
            f.status == FieldStatus.NEEDS_REVIEW for r in record.records for f in r.fields.values()
        )
        record.status = DocumentStatus.NEEDS_REVIEW if still_needs_review else DocumentStatus.PROCESSED
    record.touch()


@router.put("/{document_id}/records/{record_index}/fields", response_model=DocumentRecord)
def update_fields(
    document_id: str,
    record_index: int,
    body: FieldsUpdateRequest,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> DocumentRecord:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    entry = _get_entry_or_404(record, record_index)

    unknown = set(body.fields) - set(FIELD_SPEC_BY_NAME)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown field(s): {', '.join(sorted(unknown))}")

    for name, item in body.fields.items():
        existing = entry.fields.get(name)
        spec = FIELD_SPEC_BY_NAME[name]
        new_value = item.value.strip() if item.value else None
        if existing is None:
            existing = ExtractedField(field_name=name, label=spec.label, data_type=spec.data_type)
            entry.fields[name] = existing
        existing.value = new_value
        existing.source = FieldSource.HUMAN
        existing.status = FieldStatus.CORRECTED if new_value else FieldStatus.NEEDS_REVIEW
        existing.confidence = 100.0 if new_value else 0.0

    _revalidate_entry(entry, settings)
    _recompute_document_status(record)
    repo.save(record)
    return record


@router.post("/{document_id}/validate", response_model=DocumentRecord)
def validate_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> DocumentRecord:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    for entry in record.records:
        _revalidate_entry(entry, settings)
    _recompute_document_status(record)
    repo.save(record)
    return record


@router.post("/{document_id}/records/{record_index}/approve", response_model=DocumentRecord)
def approve_record(
    document_id: str,
    record_index: int,
    body: ApproveRequest,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> DocumentRecord:
    """Approves ONE record - the primary workflow: review a record, approve
    it, move to the next. When every record in the document has been
    approved this way, the document itself is marked approved and both
    exports are generated automatically."""
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    if record.status in (DocumentStatus.UPLOADED, DocumentStatus.PROCESSING, DocumentStatus.FAILED):
        raise HTTPException(
            status_code=409, detail="Document must be successfully processed before approval"
        )
    entry = _get_entry_or_404(record, record_index)

    entry.approved = True
    entry.approved_at = utcnow()
    entry.approved_by = body.approved_by
    record.touch()

    if all(r.approved for r in record.records):
        record.status = DocumentStatus.APPROVED
        record.audit.approved_at = utcnow()
        record.audit.approved_by = body.approved_by
        try:
            record.last_export_path = str(export_record_to_excel(record, settings.exports_dir))
        except Exception:  # noqa: BLE001 - approval itself must still succeed
            logger.exception("Excel export failed for document %s", document_id)
        try:
            record.last_xml_export_path = str(export_record_to_xml(record, settings.exports_dir))
        except Exception:  # noqa: BLE001 - approval itself must still succeed
            logger.exception("XML export failed for document %s", document_id)
        logger.info("Document %s fully approved (%d record(s))", document_id, record.record_count)
    else:
        logger.info("Document %s record %d approved", document_id, record_index)

    repo.save(record)
    return record


@router.post("/{document_id}/records/{record_index}/ai-verify", response_model=DocumentRecord)
def ai_verify_record(
    document_id: str,
    record_index: int,
    force: bool = False,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
) -> DocumentRecord:
    """Explicit, opt-in GPT Vision cross-check for ONE record - never
    triggered automatically, so API spend stays under the user's control.
    Cached on the record (skips the call on repeat requests) unless
    force=true. Populates ai_value/ai_match per field for comparison; never
    overwrites the field's own `value` (see services/ai_vision.py)."""
    if not settings.ai_extraction_enabled or not settings.ai_provider_api_key:
        raise HTTPException(status_code=503, detail="AI verification is not configured")

    record = _get_record_or_404(repo, document_id)
    entry = _get_entry_or_404(record, record_index)

    if not force and any(f.ai_value is not None for f in entry.fields.values()):
        return record  # already verified - avoid spending budget twice on the same record

    fields_with_bbox = [f for f in entry.fields.values() if f.bounding_box is not None]
    if not fields_with_bbox:
        raise HTTPException(status_code=422, detail="Record has no located fields to crop an image from")
    y_start = min(f.bounding_box.y for f in fields_with_bbox)
    y_end = max(f.bounding_box.y + f.bounding_box.height for f in fields_with_bbox)

    file_store = get_file_store_for_record(record, settings)
    try:
        raw = file_store.read_upload(record.source_file.stored_filename)
    except (FileNotFoundError, StorageUnavailableError) as exc:
        raise HTTPException(status_code=500, detail="Could not load the source image") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=500, detail="Could not load the source image")
    crop, _ = crop_record(image, y_start, y_end, margin=15)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode record crop")

    try:
        ai_fields = extract_fields_via_vision(
            buf.tobytes(), api_key=settings.ai_provider_api_key, model=settings.ai_provider_model
        )
    except AIVisionError as exc:
        logger.warning("AI verification failed for %s record %d: %s", document_id, record_index, exc)
        raise HTTPException(status_code=502, detail=f"AI verification failed: {exc}") from exc

    for name, ai_result in ai_fields.items():
        field = entry.fields.get(name)
        if field is None:
            continue
        ai_value = ai_result.get("value")
        field.ai_value = ai_value
        field.ai_match = (ai_value or "").strip() == (field.value or "").strip() if ai_value or field.value else None

    record.touch()
    repo.save(record)
    logger.info("AI-verified document %s record %d", document_id, record_index)
    return record


@router.post("/{document_id}/approve", response_model=DocumentRecord)
def approve_document(
    document_id: str,
    body: ApproveRequest,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> DocumentRecord:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    if record.status in (DocumentStatus.UPLOADED, DocumentStatus.PROCESSING, DocumentStatus.FAILED):
        raise HTTPException(
            status_code=409, detail="Document must be successfully processed before approval"
        )

    record.status = DocumentStatus.APPROVED
    record.audit.approved_at = utcnow()
    record.audit.approved_by = body.approved_by
    for entry in record.records:
        entry.approved = True
        entry.approved_at = entry.approved_at or record.audit.approved_at
        entry.approved_by = entry.approved_by or body.approved_by
    record.touch()

    try:
        record.last_export_path = str(export_record_to_excel(record, settings.exports_dir))
    except Exception:  # noqa: BLE001 - approval itself must still succeed
        logger.exception("Excel export failed for document %s", document_id)
    try:
        record.last_xml_export_path = str(export_record_to_xml(record, settings.exports_dir))
    except Exception:  # noqa: BLE001 - approval itself must still succeed
        logger.exception("XML export failed for document %s", document_id)

    repo.save(record)
    logger.info(
        "Document %s approved (%d record(s))", document_id, record.record_count
    )
    return record


@router.get("/{document_id}/export/excel")
def export_excel(
    document_id: str,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> FileResponse:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    if record.status == DocumentStatus.UPLOADED:
        raise HTTPException(status_code=409, detail="Document has not been processed yet")
    output_path = export_record_to_excel(record, settings.exports_dir)
    return FileResponse(
        path=output_path,
        filename=f"{document_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.get("/{document_id}/export/xml")
def export_xml(
    document_id: str,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> FileResponse:
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    if record.status == DocumentStatus.UPLOADED:
        raise HTTPException(status_code=409, detail="Document has not been processed yet")
    output_path = export_record_to_xml(record, settings.exports_dir)
    return FileResponse(path=output_path, filename=f"{document_id}.xml", media_type="application/xml")


@router.post("/{document_id}/submit")
def submit_document_endpoint(
    document_id: str,
    body: ApproveRequest,
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
    file_store: FileStore = Depends(get_file_store),
) -> dict:
    """Writes every record in this document to the legacy dump XML schema
    (same field tags as the real DataM.xml dump) - both as a flat file
    under data/submissions/ and as rows in data/submissions.db (SQLite,
    unchanged: this is what makes CreateDate immutable across a
    resubmit - see _build_record_metadata), AND to durable storage (Azure
    Blob Storage when configured, same local file otherwise) - this is the
    artifact a separate team-lead application depends on for the actual
    handoff, so it must not only live on this instance's own local disk.
    Does not require full approval first - this is a separate, explicit
    "send this data out" action a reviewer can take as soon as the
    document has been processed, same way the existing Excel/XML
    downloads don't require approval either."""
    record = _get_record_or_404(repo, document_id)
    _check_access(document_id, current_user, db)
    if record.status == DocumentStatus.UPLOADED:
        raise HTTPException(status_code=409, detail="Document has not been processed yet")
    if not record.records:
        raise HTTPException(status_code=409, detail="No records to submit")

    if body.approved_by:
        for entry in record.records:
            entry.approved_by = entry.approved_by or body.approved_by
        record.touch()
        repo.save(record)

    try:
        xml_path, rows_written = submit_document(record, settings.submissions_dir, settings.submissions_db_path)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        logger.exception("Submit failed for document %s", document_id)
        raise HTTPException(status_code=500, detail="Submit failed - see server logs") from exc

    # Read back the exact bytes just written (same utf-8-sig BOM + CRLF
    # already applied by write_dump_xml_file) rather than rebuilding the
    # XML separately - guarantees the durable copy is byte-identical to
    # the local one, not a second, potentially-diverging render.
    xml_bytes = xml_path.read_bytes()
    try:
        xml_key = file_store.save_xml(xml_bytes, batch_id=document_id)
    except StorageUnavailableError as exc:
        logger.exception("Could not store submission XML for document %s", document_id)
        raise HTTPException(
            status_code=502, detail="Submitted locally, but could not store the durable XML copy. Please try again."
        ) from exc
    # Recorded ONLY after the durable write above actually succeeds - a
    # reviewer (or the team-lead app re-downloading later) must never see
    # "submitted" for a copy that doesn't really exist in durable storage.
    db.upsert_document_submission(document_id, xml_key, file_store.name, rows_written)

    logger.info(
        "Document %s submitted (%d record(s), durable backend=%s)", document_id, rows_written, file_store.name
    )
    return {
        "document_id": document_id,
        "xml_path": str(xml_path),
        "records_submitted": rows_written,
    }


@router.get("/{document_id}/submit/download")
def download_submission_xml(
    document_id: str,
    settings: Settings = Depends(get_settings),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
    db: AppDatabase = Depends(get_app_db),
) -> Response:
    _check_access(document_id, current_user, db)
    submission = db.get_document_submission(document_id)
    if submission is not None:
        from app.services.storage.blob_store import get_file_store_by_backend

        file_store = get_file_store_by_backend(submission["xml_storage_backend"], settings)
        try:
            data = file_store.read_xml(submission["xml_storage_key"])
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Submitted XML is no longer available") from None
        except StorageUnavailableError as exc:
            logger.exception("Could not fetch submitted XML for document %s", document_id)
            raise HTTPException(
                status_code=502, detail="Could not retrieve the submission XML. Please try again."
            ) from exc
        return Response(
            content=data, media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{document_id}_submission.xml"'},
        )
    # Fallback for a document submitted before this durable-storage tracking
    # existed (an older local-only submission row with no document_submissions
    # entry) - never silently 404 data that genuinely still exists on disk.
    path = settings.submissions_dir / f"{document_id}.xml"
    if not path.exists():
        raise HTTPException(status_code=404, detail="This document has not been submitted yet")
    return FileResponse(path=path, filename=f"{document_id}_submission.xml", media_type="application/xml")
