"""Batch API request/response schemas. No auth schemas here - this
application has no accounts (see app/api/batches.py's module docstring)."""
from __future__ import annotations

from pydantic import BaseModel


class BatchImageStatus(BaseModel):
    position: int
    document_id: str
    original_filename: str
    status: str
    error: str | None = None
    # Populated only once the document has produced records (status
    # "completed"/"needs_review") - null before that, not 0, so the
    # frontend can tell "not processed yet" apart from "processed with
    # zero records". Lets the batch screen show real review progress
    # ("12 of 25 images reviewed") without a separate round trip per image.
    record_count: int | None = None
    all_records_approved: bool | None = None


class BatchClearResponse(BaseModel):
    batch_id: str
    images_cleared: int
    errors: list[str] = []


class BatchSummary(BaseModel):
    id: str
    status: str
    image_count: int
    created_at: str
    updated_at: str
    # Computed, never stored directly (except `archived`, an explicit
    # flag) - see app/api/batches.py's _review_status for the exact
    # derivation. Additive fields: existing frontend code that only reads
    # the fields above keeps working unchanged.
    review_status: str = "in_progress"
    archived: bool = False
    # Set once by confirm_batch_completion, never cleared or pushed back
    # out by a repeat confirmation - see that endpoint's own docstring.
    # This is what the 24h retention countdown is measured from.
    completed_confirmed_at: str | None = None


class BatchDetail(BatchSummary):
    images: list[BatchImageStatus]
    images_completed: int = 0
    records_completed: int = 0
    records_total: int = 0


class BatchArchiveResponse(BaseModel):
    batch_id: str
    archived: bool


class BatchCompletionConfirmedResponse(BaseModel):
    batch_id: str
    completed_confirmed_at: str
    cleanup_scheduled_at: str


class BatchExportVersion(BaseModel):
    version: int
    record_count: int
    exported_at: str


class DeviceActivityRequest(BaseModel):
    batch_id: str | None = None
    document_id: str | None = None
    record_index: int | None = None


class DeviceResumeResponse(BaseModel):
    resumable: bool
    batch_id: str | None = None
    document_id: str | None = None
    record_index: int | None = None
    batch_status: str | None = None
    review_status: str | None = None
    image_count: int = 0
    images_completed: int = 0
    records_total: int = 0
    records_completed: int = 0
    last_saved_at: str | None = None


class BatchUploadResponse(BaseModel):
    batch_id: str
    image_count: int
    status: str


class BatchSubmitResponse(BaseModel):
    batch_id: str
    records_submitted: int
    status: str


class BatchSubmissionStatus(BaseModel):
    batch_id: str
    submitted: bool
    record_count: int | None = None
    submitted_at: str | None = None
