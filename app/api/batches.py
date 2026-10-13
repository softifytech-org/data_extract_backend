"""Batch upload and status API.

A batch is 1-40 images uploaded together. Each image becomes its own
DocumentRecord via the SAME storage/validation path single uploads already
use (app/api/documents.py) - this module adds batch grouping and
asynchronous queued processing on top, it does not duplicate the
per-image OCR/review logic.

DELIBERATE, EXPLICIT DECISION: this application has NO authentication
anywhere (no accounts, no shared key, no gate of any kind) - every
endpoint here is fully public. `owner_user_id` still exists on the
`batches` row only because the database schema requires a value there
(a NOT NULL foreign key) - it is always the same fixed PUBLIC_OWNER_ID
constant below, not a real per-caller identity, and nothing here reads
it to make an access decision. See the final report for the resulting
risks (unauthenticated uploads, OCR cost exposure, no isolation between
callers - anyone with the URL can read, submit, or download ANY batch,
not just ones they created).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile

from app.core.config import Settings, get_settings
from app.core.dependencies import get_app_db, get_file_store, get_repository
from app.core.security import (
    UnsafeFilenameError,
    UnsupportedFileTypeError,
    generate_document_id,
    sanitize_original_filename,
    validate_upload,
)
from app.core.logging_config import get_logger
from app.models.document import AuditInfo, DocumentRecord, DocumentStatus, SourceFileInfo
from app.schemas.batch import (
    BatchDetail,
    BatchImageStatus,
    BatchSubmissionStatus,
    BatchSubmitResponse,
    BatchSummary,
    BatchUploadResponse,
)
from app.services.export.dump_xml_export import submit_batch_records
from app.services.storage.app_db import AppDatabase
from app.services.storage.blob_store import FileStore, StorageUnavailableError
from app.services.storage.json_storage import JsonDocumentRepository

router = APIRouter(prefix="/api/batches", tags=["batches"])
logger = get_logger(__name__)

MIN_BATCH_IMAGES = 1
MAX_BATCH_IMAGES = 40

# Fixed placeholder "owner" for every batch - see the module docstring.
# Not a real identity; exists only to satisfy the existing NOT NULL
# foreign-key column. ensure_public_user() (called once at startup, see
# app/main.py) creates this row so the FK is always satisfiable.
PUBLIC_OWNER_ID = "USER-PUBLIC"


def ensure_public_user(db: AppDatabase) -> None:
    db.ensure_user(PUBLIC_OWNER_ID, "public-no-auth", "unused")


def _get_batch_or_404(db: AppDatabase, batch_id: str) -> dict:
    batch = db.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


@router.post("", response_model=BatchUploadResponse)
async def create_batch(
    files: list[UploadFile],
    layout: str = Form("old"),
    settings: Settings = Depends(get_settings),
    repo: JsonDocumentRepository = Depends(get_repository),
    db: AppDatabase = Depends(get_app_db),
    file_store: FileStore = Depends(get_file_store),
) -> BatchUploadResponse:
    # One layout per batch (not per-image): the backend CAN process each
    # image independently, but asking a reviewer to tag 1-40 individual
    # files correctly before upload is error-prone and unnecessary here -
    # every real batch observed so far is one scanned source, one layout.
    # See app/services/extraction/engine.py's RuleBasedExtractionEngine
    # docstring for exactly what this selects.
    if layout not in ("old", "new"):
        raise HTTPException(status_code=400, detail="layout must be 'old' or 'new'")
    if len(files) < MIN_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail="A batch must contain at least 1 image")
    if len(files) > MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"A batch may contain at most {MAX_BATCH_IMAGES} images (got {len(files)})"
        )

    # Validate every file BEFORE writing or registering any of them - an
    # all-or-nothing batch create, so a single bad file can't leave a
    # partially-created batch behind.
    validated: list[tuple[UploadFile, bytes, str]] = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="One or more files has no filename")
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"'{file.filename}' is empty")
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"'{file.filename}' exceeds the {settings.max_upload_size_mb}MB size limit",
            )
        try:
            extension = validate_upload(file.filename, content, settings.allowed_extensions_set)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=400, detail=f"'{file.filename}': {exc}") from exc
        validated.append((file, content, extension))

    batch_id = db.create_batch(PUBLIC_OWNER_ID, len(validated))

    # All-or-nothing beyond validation too: if a file genuinely can't be
    # stored (e.g. Blob Storage unreachable after retries) partway through,
    # the batch/job rows and this request's own already-written files/
    # blobs are rolled back rather than left as a partially-created batch.
    created_document_ids: list[str] = []
    stored_keys: list[str] = []
    try:
        for position, (file, content, extension) in enumerate(validated):
            document_id = generate_document_id()
            try:
                storage_key = file_store.save_upload(
                    content, document_id=document_id, extension=extension, batch_id=batch_id
                )
            except UnsafeFilenameError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            created_document_ids.append(document_id)
            stored_keys.append(storage_key)

            file_type = "pdf" if extension == ".pdf" else "image"
            original_filename = sanitize_original_filename(file.filename)
            record = DocumentRecord(
                document_id=document_id,
                status=DocumentStatus.UPLOADED,
                layout=layout,
                source_file=SourceFileInfo(
                    original_filename=original_filename,
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
            db.add_batch_image(batch_id, position, document_id, original_filename)

            db.enqueue_job(
                job_type="process_image",
                payload={
                    "document_id": document_id,
                    "batch_id": batch_id,
                    "position": position,
                    "storage_key": storage_key,
                    "storage_backend": file_store.name,
                    "file_type": file_type,
                },
                idempotency_key=f"process_image:{document_id}",
            )
    except StorageUnavailableError as exc:
        logger.exception("Batch %s could not be fully stored - rolling back", batch_id)
        for document_id, storage_key in zip(created_document_ids, stored_keys):
            db.delete_job_by_idempotency_key(f"process_image:{document_id}")
            file_store.delete_upload(storage_key)
            try:
                repo.delete(document_id)
            except Exception:  # noqa: BLE001 - best-effort cleanup, never masks the real 502 below
                logger.warning("Could not clean up orphaned record %s after rollback", document_id)
        db.delete_batch(batch_id)
        raise HTTPException(
            status_code=502, detail="Could not store uploaded files. Please try again."
        ) from exc

    db.recompute_batch_status(batch_id)
    logger.info("Batch %s created (%d images, backend=%s)", batch_id, len(validated), file_store.name)
    return BatchUploadResponse(batch_id=batch_id, image_count=len(validated), status="queued")


@router.get("", response_model=list[BatchSummary])
def list_batches(db: AppDatabase = Depends(get_app_db)) -> list[BatchSummary]:
    """No per-caller scoping - returns every batch ever created, by
    anyone (see the module docstring: there is no identity to scope by)."""
    rows = db.list_batches(PUBLIC_OWNER_ID)
    return [
        BatchSummary(
            id=r["id"], status=r["status"], image_count=r["image_count"],
            created_at=r["created_at"], updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.get("/{batch_id}", response_model=BatchDetail)
def get_batch(batch_id: str, db: AppDatabase = Depends(get_app_db)) -> BatchDetail:
    batch = _get_batch_or_404(db, batch_id)
    images = db.list_batch_images(batch_id)
    return BatchDetail(
        id=batch["id"], status=batch["status"], image_count=batch["image_count"],
        created_at=batch["created_at"], updated_at=batch["updated_at"],
        images=[
            BatchImageStatus(
                position=i["position"], document_id=i["document_id"],
                original_filename=i["original_filename"], status=i["status"], error=i["error"],
            )
            for i in images
        ],
    )


def _batch_submission_blockers(images: list[dict], repo: JsonDocumentRepository) -> list[str]:
    """Requirement: the backend must refuse to submit a batch while any
    required record remains unsaved, unapproved, or invalid. "Invalid"
    here means a document that never produced usable records at all
    (still processing, or failed outright) - those block submission
    rather than being silently skipped, so a reviewer always knows
    exactly why a batch isn't ready instead of getting a partial export."""
    blockers: list[str] = []
    for img in images:
        label = f"position {img['position'] + 1} ({img['original_filename']})"
        status = img["status"]
        if status in ("queued", "processing"):
            blockers.append(f"{label} is still processing")
            continue
        if status == "failed":
            blockers.append(f"{label} failed processing: {img['error'] or 'unknown error'}")
            continue
        record = repo.get(img["document_id"])
        unapproved = [e.record_index for e in record.records if not e.approved]
        if unapproved:
            blockers.append(f"{label} has unapproved record(s): {unapproved}")
    return blockers


@router.post("/{batch_id}/submit", response_model=BatchSubmitResponse)
def submit_batch(
    batch_id: str,
    db: AppDatabase = Depends(get_app_db),
    repo: JsonDocumentRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    file_store: FileStore = Depends(get_file_store),
) -> BatchSubmitResponse:
    """Generates ONE combined XML export covering every approved record in
    this batch and uploads it to Blob Storage (or local disk, if Azure
    Blob Storage is not configured) - see submit_batch_records. Blocked
    entirely (400, with the exact list of blockers) while any image is
    still processing, failed, or has an unapproved record. Safe to call
    again after that: re-submitting re-validates readiness from scratch
    and, once allowed, overwrites the same batch's single export with
    the current approved data rather than creating a duplicate."""
    _get_batch_or_404(db, batch_id)
    images = db.list_batch_images(batch_id)
    if not images:
        raise HTTPException(status_code=400, detail="Batch has no images")

    blockers = _batch_submission_blockers(images, repo)
    if blockers:
        raise HTTPException(
            status_code=400,
            detail={"message": "Batch is not ready to submit", "blockers": blockers},
        )

    records = [repo.get(img["document_id"]) for img in images]
    try:
        combined_xml, total_records = submit_batch_records(
            records, settings.submissions_dir, settings.submissions_db_path
        )
        xml_key = file_store.save_xml(combined_xml.encode("utf-8-sig"), batch_id=batch_id)
    except StorageUnavailableError as exc:
        logger.exception("Could not store combined XML for batch %s", batch_id)
        raise HTTPException(
            status_code=502, detail="Could not store the submission XML. Please try again."
        ) from exc

    db.upsert_batch_submission(batch_id, xml_key, file_store.name, total_records)
    logger.info("Batch %s submitted (%d record(s), backend=%s)", batch_id, total_records, file_store.name)
    return BatchSubmitResponse(batch_id=batch_id, records_submitted=total_records, status="submitted")


@router.get("/{batch_id}/submission", response_model=BatchSubmissionStatus)
def get_batch_submission_status(batch_id: str, db: AppDatabase = Depends(get_app_db)) -> BatchSubmissionStatus:
    _get_batch_or_404(db, batch_id)
    submission = db.get_batch_submission(batch_id)
    if submission is None:
        return BatchSubmissionStatus(batch_id=batch_id, submitted=False)
    return BatchSubmissionStatus(
        batch_id=batch_id, submitted=True,
        record_count=submission["record_count"], submitted_at=submission["submitted_at"],
    )


@router.get("/{batch_id}/submission/download")
def download_batch_submission(
    batch_id: str,
    db: AppDatabase = Depends(get_app_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    _get_batch_or_404(db, batch_id)
    submission = db.get_batch_submission(batch_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="This batch has not been submitted yet")

    from app.services.storage.blob_store import get_file_store_by_backend

    file_store = get_file_store_by_backend(submission["xml_storage_backend"], settings)
    try:
        data = file_store.read_xml(submission["xml_storage_key"])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Submitted XML is no longer available") from None
    except StorageUnavailableError as exc:
        logger.exception("Could not fetch submitted XML for batch %s", batch_id)
        raise HTTPException(status_code=502, detail="Could not retrieve the submission XML. Please try again.") from exc

    return Response(
        content=data, media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{batch_id}.xml"'},
    )
