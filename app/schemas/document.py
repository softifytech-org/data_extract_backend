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


class FieldsUpdateRequest(BaseModel):
    fields: dict[str, FieldUpdateItem]


class ApproveRequest(BaseModel):
    approved_by: str | None = Field(default=None, max_length=120)


class ErrorResponse(BaseModel):
    message: str


class ExportResponse(BaseModel):
    document_id: str
    export_path: str
