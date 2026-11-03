"""Tests for the batch lifecycle additions: archive/unarchive, permanent
export-version history (never overwritten by a later resubmit), and
device-aware resume state. No credentials anywhere, same open-access
model as every other batch test (see test_batches_api.py)."""
from __future__ import annotations

from app.models.document import (
    DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus, RecordEntry,
)

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _create_batch(client, n=1):
    resp = client.post("/api/batches", files=[("files", (f"img{i}.jpg", TINY_JPEG, "image/jpeg")) for i in range(n)])
    assert resp.status_code == 200, resp.text
    return resp.json()["batch_id"]


def _mark_image_completed_and_approved(repo, db, batch_id, position, *, customer_name="Test Customer"):
    images = db.list_batch_images(batch_id)
    img = next(i for i in images if i["position"] == position)
    document_id = img["document_id"]

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
                "customer_name": ExtractedField(
                    field_name="customer_name", label="Customer Name", value=customer_name,
                    data_type=DataType.NAME, confidence=99.0, status=FieldStatus.VERIFIED, source=FieldSource.OCR,
                ),
            },
            approved=True,
        )
    ]
    record.record_count = 1
    repo.save(record)
    db.update_batch_image_status(batch_id, position, "completed")
    db.recompute_batch_status(batch_id)
    return document_id


# --- archive / unarchive ----------------------------------------------------

def test_archive_hides_batch_from_default_list_but_keeps_its_data(client, app_db):
    batch_id = _create_batch(client)

    resp = client.post(f"/api/batches/{batch_id}/archive")
    assert resp.status_code == 200
    assert resp.json() == {"batch_id": batch_id, "archived": True}

    active_ids = [b["id"] for b in client.get("/api/batches").json()]
    assert batch_id not in active_ids

    archived_ids = [b["id"] for b in client.get("/api/batches?archived=true").json()]
    assert batch_id in archived_ids

    # Data itself must still be fully intact and readable.
    detail = client.get(f"/api/batches/{batch_id}")
    assert detail.status_code == 200
    assert detail.json()["archived"] is True
    assert detail.json()["review_status"] == "archived"


def test_unarchive_restores_batch_to_active_list(client):
    batch_id = _create_batch(client)
    client.post(f"/api/batches/{batch_id}/archive")

    resp = client.post(f"/api/batches/{batch_id}/unarchive")
    assert resp.status_code == 200
    assert resp.json()["archived"] is False

    active_ids = [b["id"] for b in client.get("/api/batches").json()]
    assert batch_id in active_ids


def test_archiving_one_batch_never_touches_another(client):
    batch_a = _create_batch(client)
    batch_b = _create_batch(client)
    client.post(f"/api/batches/{batch_a}/archive")

    active_ids = [b["id"] for b in client.get("/api/batches").json()]
    assert batch_b in active_ids
    assert batch_a not in active_ids


# --- review_status computation ----------------------------------------------

def test_review_status_ready_for_export_when_fully_approved(client, repo, app_db):
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["review_status"] == "ready_for_export"
    assert detail["images_completed"] == 1
    assert detail["records_completed"] == 1
    assert detail["records_total"] == 1


def test_review_status_exported_after_submit(client, repo, app_db):
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    submit = client.post(f"/api/batches/{batch_id}/submit")
    assert submit.status_code == 200, submit.text

    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["review_status"] == "exported"


# --- export version history --------------------------------------------------

def test_first_submit_creates_export_version_1(client, repo, app_db):
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0, customer_name="First Name")
    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200

    history = client.get(f"/api/batches/{batch_id}/exports").json()
    assert len(history) == 1
    assert history[0]["version"] == 1


def test_resubmitting_adds_a_new_version_without_losing_the_old_one(client, repo, app_db):
    batch_id = _create_batch(client)
    document_id = _mark_image_completed_and_approved(repo, app_db, batch_id, 0, customer_name="Original Name")
    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200

    # Correct the record, then resubmit - simulates "reopen an exported
    # batch for an authorized correction" from the spec.
    record = repo.get(document_id)
    record.records[0].fields["customer_name"].value = "Corrected Name"
    repo.save(record)
    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200

    history = client.get(f"/api/batches/{batch_id}/exports").json()
    assert sorted(h["version"] for h in history) == [1, 2]

    v1_xml = client.get(f"/api/batches/{batch_id}/exports/1/download").text
    v2_xml = client.get(f"/api/batches/{batch_id}/exports/2/download").text
    assert "Original Name" in v1_xml
    assert "Original Name" not in v2_xml
    assert "Corrected Name" in v2_xml

    # The existing "latest" endpoint must still work exactly as before,
    # and reflect the newest export.
    latest_xml = client.get(f"/api/batches/{batch_id}/submission/download").text
    assert "Corrected Name" in latest_xml


def test_downloading_a_nonexistent_export_version_is_404(client, repo, app_db):
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200

    resp = client.get(f"/api/batches/{batch_id}/exports/99/download")
    assert resp.status_code == 404


# --- device-aware resume ------------------------------------------------------

def test_resume_is_not_resumable_for_an_unknown_device(client):
    resp = client.get("/api/devices/some-device-never-seen/resume")
    assert resp.status_code == 200
    assert resp.json()["resumable"] is False


def test_activity_then_resume_returns_last_position(client):
    batch_id = _create_batch(client)
    device_id = "device-abc-123"

    put_resp = client.put(
        f"/api/devices/{device_id}/activity",
        json={"batch_id": batch_id, "document_id": "DOC-000000000001", "record_index": 2},
    )
    assert put_resp.status_code == 204

    resume = client.get(f"/api/devices/{device_id}/resume").json()
    assert resume["resumable"] is True
    assert resume["batch_id"] == batch_id
    assert resume["document_id"] == "DOC-000000000001"
    assert resume["record_index"] == 2
    assert resume["last_saved_at"] is not None


def test_resume_reports_false_once_batch_is_exported(client, repo, app_db):
    """Once a batch is fully exported there is nothing left to "resume" -
    it belongs under Completed Batches, not the Welcome-back banner."""
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    device_id = "device-xyz"
    client.put(f"/api/devices/{device_id}/activity", json={"batch_id": batch_id})
    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200

    resume = client.get(f"/api/devices/{device_id}/resume").json()
    assert resume["resumable"] is False


def test_resume_reports_false_once_batch_is_archived(client):
    batch_id = _create_batch(client)
    device_id = "device-archived-case"
    client.put(f"/api/devices/{device_id}/activity", json={"batch_id": batch_id})
    client.post(f"/api/batches/{batch_id}/archive")

    resume = client.get(f"/api/devices/{device_id}/resume").json()
    assert resume["resumable"] is False


def test_two_devices_track_independent_positions(client):
    batch_a = _create_batch(client)
    batch_b = _create_batch(client)
    client.put("/api/devices/device-1/activity", json={"batch_id": batch_a})
    client.put("/api/devices/device-2/activity", json={"batch_id": batch_b})

    assert client.get("/api/devices/device-1/resume").json()["batch_id"] == batch_a
    assert client.get("/api/devices/device-2/resume").json()["batch_id"] == batch_b
