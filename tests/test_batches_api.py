"""Batch API tests for the open-access model - this application has no
accounts, no shared key, no auth of any kind (explicit decision, see
app/api/batches.py's module docstring). There is nothing left to test
for "who can see this batch" - everyone can, by design. What's still
tested: upload validation/limits, all-or-nothing rollback on a bad file,
and that every endpoint actually works end to end with no credentials
presented at all."""
from __future__ import annotations

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100
TINY_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def _files(n: int, content: bytes = TINY_JPEG, ctype: str = "image/jpeg"):
    return [("files", (f"img{i}.jpg", content, ctype)) for i in range(n)]


def test_batch_rejects_zero_files(client):
    resp = client.post("/api/batches", files=[])
    assert resp.status_code in (400, 422)


def test_batch_accepts_one_image_with_no_credentials(client):
    resp = client.post("/api/batches", files=_files(1))
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_count"] == 1
    assert body["status"] == "queued"


def test_batch_accepts_exactly_40_images(client):
    resp = client.post("/api/batches", files=_files(40))
    assert resp.status_code == 200
    assert resp.json()["image_count"] == 40


def test_batch_rejects_41_images(client):
    resp = client.post("/api/batches", files=_files(41))
    assert resp.status_code == 400
    assert "40" in resp.json()["detail"]


def test_batch_rejects_bad_file_content(client):
    bad = [("files", ("notreally.jpg", b"this is not an image", "image/jpeg"))]
    resp = client.post("/api/batches", files=bad)
    assert resp.status_code == 400


def test_batch_rejects_extension_content_mismatch(client):
    # real PNG bytes with a .jpg extension/claimed content-type
    mismatched = [("files", ("fake.jpg", TINY_PNG, "image/jpeg"))]
    resp = client.post("/api/batches", files=mismatched)
    assert resp.status_code == 400


def test_batch_all_or_nothing_on_validation_failure(client, app_db):
    """One bad file among several good ones must not create a partial batch."""
    before = len(client.get("/api/batches").json())
    files = _files(3) + [("files", ("bad.jpg", b"not an image", "image/jpeg"))]
    resp = client.post("/api/batches", files=files)
    assert resp.status_code == 400
    after = len(client.get("/api/batches").json())
    assert after == before


def test_batch_reachable_with_no_authorization_header_at_all(client):
    """The core of the open-access decision: every batch endpoint must
    work with zero credentials presented - no header, no key, nothing."""
    create = client.post("/api/batches", files=_files(1))
    assert create.status_code == 200
    batch_id = create.json()["batch_id"]

    assert client.get("/api/batches").status_code == 200
    assert client.get(f"/api/batches/{batch_id}").status_code == 200


def test_batch_created_by_one_caller_is_visible_to_another_with_no_shared_secret(client):
    """Not a bug - the explicit, requested behavior: there is no identity
    to isolate by, so a batch is visible to literally any caller."""
    create = client.post("/api/batches", files=_files(1))
    batch_id = create.json()["batch_id"]

    # A second, completely independent request (no header, no cookie, no
    # state shared with the first one beyond the batch_id itself) can
    # still read it - this is intentional, not a leak of something that
    # should have been private, since nothing in this app is private.
    resp = client.get(f"/api/batches/{batch_id}")
    assert resp.status_code == 200


def test_list_batches_returns_every_batch_not_scoped_to_a_caller(client):
    client.post("/api/batches", files=_files(1))
    client.post("/api/batches", files=_files(1))
    listing = client.get("/api/batches").json()
    assert len(listing) == 2


def test_batch_enqueues_one_job_per_image(client, app_db):
    client.post("/api/batches", files=_files(3))
    assert app_db.queue_depth() == {"pending": 3}


# --- GET /api/batches/{id}: per-image record_count/all_records_approved ---

def test_batch_detail_reports_null_progress_before_processing(client):
    """An image still queued has no records yet - must be null, not 0/false,
    so the frontend can tell "not reviewable yet" apart from "reviewed with
    zero records"."""
    resp = client.post("/api/batches", files=_files(1))
    batch_id = resp.json()["batch_id"]
    detail = client.get(f"/api/batches/{batch_id}").json()
    img = detail["images"][0]
    assert img["record_count"] is None
    assert img["all_records_approved"] is None


def test_batch_detail_reports_record_count_and_approval_once_processed(client, repo, app_db):
    from app.models.document import (
        DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus, RecordEntry,
    )

    resp = client.post("/api/batches", files=_files(1))
    batch_id = resp.json()["batch_id"]
    document_id = client.get(f"/api/batches/{batch_id}").json()["images"][0]["document_id"]

    record = repo.get(document_id)
    record.status = DocumentStatus.PROCESSED
    record.records = [
        RecordEntry(
            record_index=1, page=1,
            fields={"record_no": ExtractedField(
                field_name="record_no", label="Record No", value="241001",
                data_type=DataType.TEXT, confidence=99.0, status=FieldStatus.VERIFIED, source=FieldSource.OCR,
            )},
            approved=False,
        ),
        RecordEntry(record_index=2, page=1, fields={}, approved=True),
    ]
    repo.save(record)
    app_db.update_batch_image_status(batch_id, 0, "completed")
    app_db.recompute_batch_status(batch_id)

    detail = client.get(f"/api/batches/{batch_id}").json()
    img = detail["images"][0]
    assert img["record_count"] == 2
    assert img["all_records_approved"] is False  # one of the two records is still unapproved

    record.records[0].approved = True
    repo.save(record)
    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["images"][0]["all_records_approved"] is True
