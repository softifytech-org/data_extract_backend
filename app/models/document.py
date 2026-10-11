"""Canonical domain model for a processed document record.

This is the single source of truth for the document shape used by storage,
extraction, validation, and the API layer. It is deliberately structured so
it maps onto a normalized PostgreSQL schema without redesign later:

    documents                  (1 row per uploaded document/page-set)
        document_id PK
        status
        source_file_*  (flattened columns)
        record_count, overall_confidence
        created_at / updated_at / processed_at / approved_at / approved_by

    document_records           (many rows per document, FK -> documents)
        document_id FK, record_index (1-based, page order), page
        overall_confidence

    document_fields            (many rows per record, FK -> document_records)
        document_id FK, record_index FK, field_name
        value, original_ocr_value, confidence, status, source, data_type
        validation_* (flattened columns)
        bbox_page / bbox_x / bbox_y / bbox_width / bbox_height

    document_warnings          (many rows per record, FK -> document_records)
        document_id FK, record_index FK, message

For the prototype all three "tables" are simply nested inside one JSON file
per document (see services/storage/json_storage.py); the nesting mirrors the
FK relationship so a future migration is a mechanical unnest, not a redesign.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    FAILED = "failed"


class FieldStatus(StrEnum):
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    CORRECTED = "corrected"  # human edited an OCR value
    REJECTED = "rejected"    # human cleared a value OCR proposed


class FieldSource(StrEnum):
    OCR = "ocr"
    HUMAN = "human"


class DataType(StrEnum):
    TEXT = "text"
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    CURRENCY = "currency"
    NUMBER = "number"
    ENUM = "enum"
    ID_CODE = "id_code"
    ADDRESS = "address"


class BoundingBox(BaseModel):
    page: int = 1
    x: int
    y: int
    width: int
    height: int


class FieldValidation(BaseModel):
    """Outcome of each independent validation technique for one field.

    None means "not applicable / not evaluated" - it must never be conflated
    with False. See services/validation for how each is produced.
    """

    format_valid: bool | None = None
    ocr_agreement: bool | None = None
    ocr_agreement_ratio: float | None = None
    business_rule_valid: bool | None = None
    notes: list[str] = Field(default_factory=list)


class ExtractedField(BaseModel):
    field_name: str
    label: str
    value: str | None = None
    original_ocr_value: str | None = None
    data_type: DataType
    confidence: float = 0.0  # 0-100 internal risk score, see services/validation/confidence.py
    status: FieldStatus = FieldStatus.NEEDS_REVIEW
    source: FieldSource = FieldSource.OCR
    validation: FieldValidation = Field(default_factory=FieldValidation)
    bounding_box: BoundingBox | None = None
    # Optional GPT Vision cross-check (services/ai_vision.py) - populated
    # only when a human explicitly triggers AI verification for this record.
    # Read-only comparison data: never overwrites `value`.
    ai_value: str | None = None
    ai_match: bool | None = None


class SourceFileInfo(BaseModel):
    original_filename: str
    stored_filename: str
    file_type: str  # "image" | "pdf"
    content_type: str
    pages: int = 1
    size_bytes: int = 0


class AuditInfo(BaseModel):
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    processed_at: datetime | None = None
    approved_at: datetime | None = None
    approved_by: str | None = None
    created_by: str | None = None


class RecordEntry(BaseModel):
    """One user/record within a document. A document can contain many of
    these (see services/extraction/record_splitter.py) - record_index is
    1-based and reflects top-to-bottom page order, which is also the order
    this list is stored and exported in."""

    record_index: int
    page: int = 1
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    cross_field_warnings: list[str] = Field(default_factory=list)
    overall_confidence: float | None = None
    approved: bool = False
    approved_at: datetime | None = None
    approved_by: str | None = None


class DocumentRecord(BaseModel):
    document_id: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    source_file: SourceFileInfo
    records: list[RecordEntry] = Field(default_factory=list)
    record_count: int = 0
    overall_confidence: float | None = None  # average across all records
    error_message: str | None = None
    last_export_path: str | None = None
    last_xml_export_path: str | None = None
    audit: AuditInfo = Field(default_factory=AuditInfo)

    def touch(self) -> None:
        self.audit.updated_at = utcnow()
