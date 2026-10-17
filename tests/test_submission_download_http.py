"""HTTP-level regression test for the submission XML download endpoints -
calls the real routes through the TestClient (not a direct function call),
and asserts on the raw response bytes and headers. This exists because an
isolated unit test of build_dump_xml() can't catch an intermediary (CDN,
reverse proxy, corporate/ISP cache) serving a stale cached copy of a
download URL - only the actual HTTP response's headers can prove this
app itself never gives a cache permission to do that.
"""
from __future__ import annotations

from app.models.document import (
    DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus, RecordEntry,
)

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _make_processed_document(repo, document_id: str) -> None:
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


def test_document_submission_download_has_no_self_closing_update_date(client, repo):
    resp = client.post("/api/batches", files=[("files", ("img.jpg", TINY_JPEG, "image/jpeg"))])
    document_id = client.get(f"/api/batches/{resp.json()['batch_id']}").json()["images"][0]["document_id"]
    _make_processed_document(repo, document_id)

    assert client.post(f"/api/documents/{document_id}/submit", json={}).status_code == 200

    download = client.get(f"/api/documents/{document_id}/submit/download")
    assert download.status_code == 200
    text = download.content.decode("utf-8-sig")
    assert "<UpdateDate></UpdateDate>" in text
    assert "<UpdateDate/>" not in text
    assert "<UpdateDate />" not in text


def test_document_submission_download_is_never_cached(client, repo):
    """The response must tell every intermediary (CDN/proxy/browser) not
    to cache this URL - its content changes on every resubmit."""
    resp = client.post("/api/batches", files=[("files", ("img.jpg", TINY_JPEG, "image/jpeg"))])
    document_id = client.get(f"/api/batches/{resp.json()['batch_id']}").json()["images"][0]["document_id"]
    _make_processed_document(repo, document_id)
    client.post(f"/api/documents/{document_id}/submit", json={})

    download = client.get(f"/api/documents/{document_id}/submit/download")
    assert download.headers.get("cache-control") == "no-store"


def test_batch_submission_download_is_never_cached(client, repo, app_db):
    resp = client.post("/api/batches", files=[("files", ("img.jpg", TINY_JPEG, "image/jpeg"))])
    batch_id = resp.json()["batch_id"]
    document_id = client.get(f"/api/batches/{batch_id}").json()["images"][0]["document_id"]
    _make_processed_document(repo, document_id)
    app_db.update_batch_image_status(batch_id, 0, "completed")
    app_db.recompute_batch_status(batch_id)

    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200
    download = client.get(f"/api/batches/{batch_id}/submission/download")
    assert download.headers.get("cache-control") == "no-store"
