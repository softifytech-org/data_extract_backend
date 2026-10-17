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


class BatchDetail(BatchSummary):
    images: list[BatchImageStatus]


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
