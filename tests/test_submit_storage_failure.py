"""Requirement: the backend must never report a submission as successful
when the durable Blob-Storage copy of the XML failed to write - both
app/api/documents.py's submit_document_endpoint and app/api/batches.py's
submit_batch catch StorageUnavailableError and return 502 *before*
recording a submission row (app_db.upsert_document_submission /
upsert_batch_submission). These tests force that failure via a FileStore
whose save_xml always raises, and assert both the HTTP-level failure and
that no submission row was ever recorded - proving the "recorded only
after the durable write succeeds" ordering actually holds, not just that
the code reads that way.
"""
from __future__ import annotations

import pytest

from app.core.dependencies import get_file_store
from app.main import app
from app.models.document import (
    DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus, RecordEntry,
)
from app.services.storage.blob_store import LocalFileStore, StorageUnavailableError

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


class _SaveXmlAlwaysFails(LocalFileStore):
    """A real LocalFileStore for everything except save_xml, so upload/
    processing still work normally and only the submission's durable XML
    write fails - simulating a real Azure Blob Storage outage without
    needing real Azure credentials."""

    def save_xml(self, data: bytes, *, batch_id: str) -> str:
        raise StorageUnavailableError("simulated Blob Storage outage")


def _make_processed_document(repo, db, document_id: str) -> None:
    record = repo.get(document_id)
    record.status = DocumentStatus.PROCESSED
    record.records = [
        RecordEntry(
            record_index=1, page=1,
            fields={
                "record_no": ExtractedField(
                    field_name="record_no", label="Record No", value="241001",
                    data_type=DataType.TEXT, confidence=99.0, status=FieldStatus.VERIFIED, source=FieldSource.OCR,
                ),
            },
            approved=True,
        )
    ]
    record.record_count = 1
    repo.save(record)


def test_document_submit_fails_loudly_when_durable_storage_write_fails(client, test_settings, repo, app_db):
    resp = client.post("/api/batches", files=[("files", ("img.jpg", TINY_JPEG, "image/jpeg"))])
    assert resp.status_code == 200, resp.text
    batch = client.get(f"/api/batches/{resp.json()['batch_id']}").json()
    document_id = batch["images"][0]["document_id"]
    _make_processed_document(repo, app_db, document_id)

    app.dependency_overrides[get_file_store] = lambda: _SaveXmlAlwaysFails(test_settings)
    try:
        resp = client.post(f"/api/documents/{document_id}/submit", json={})
    finally:
        app.dependency_overrides[get_file_store] = lambda: LocalFileStore(test_settings)

    assert resp.status_code == 502, resp.text
    assert app_db.get_document_submission(document_id) is None


def test_batch_submit_fails_loudly_when_durable_storage_write_fails(client, test_settings, repo, app_db):
    resp = client.post("/api/batches", files=[("files", ("img.jpg", TINY_JPEG, "image/jpeg"))])
    assert resp.status_code == 200, resp.text
    batch_id = resp.json()["batch_id"]
    batch = client.get(f"/api/batches/{batch_id}").json()
    document_id = batch["images"][0]["document_id"]
    _make_processed_document(repo, app_db, document_id)
    app_db.update_batch_image_status(batch_id, 0, "completed")
    app_db.recompute_batch_status(batch_id)

    app.dependency_overrides[get_file_store] = lambda: _SaveXmlAlwaysFails(test_settings)
    try:
        resp = client.post(f"/api/batches/{batch_id}/submit")
    finally:
        app.dependency_overrides[get_file_store] = lambda: LocalFileStore(test_settings)

    assert resp.status_code == 502, resp.text
    assert app_db.get_batch_submission(batch_id) is None
