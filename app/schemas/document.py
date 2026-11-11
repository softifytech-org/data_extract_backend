"""API request/response schemas. Nothing in app/api ever accepts a raw dict -
every request body is validated against one of these before touching any
service code.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str


class ProcessResponse(BaseModel):
    document_id: str
    status: str


class FieldUpdateItem(BaseModel):
    value: str | None = Field(default=None, max_length=500)
    # Optional and independent of `value` - omit entirely to leave the
    # field's current marked_invalid state untouched (e.g. a plain value
    # edit), or include it explicitly to set/clear the invalid flag without
    # touching `value` at all (the Mark Invalid toggle). Distinguishing
    # "omitted" from "explicitly false" matters here (see
    # api/documents.py's update_fields, which checks model_fields_set
    # rather than this field's own value), so this has no default baked
    # into request handling beyond "not provided."
    marked_invalid: bool | None = Field(default=None)


class FieldsUpdateRequest(BaseModel):
    fields: dict[str, FieldUpdateItem]


class ApproveRequest(BaseModel):
    approved_by: str | None = Field(default=None, max_length=120)


class ErrorResponse(BaseModel):
    message: str


class ExportResponse(BaseModel):
    document_id: str
    export_path: str
