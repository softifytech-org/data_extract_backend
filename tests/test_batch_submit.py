"""Tests for the batch-level submit workflow: the readiness gate
(requirement: refuse submission while any image is unprocessed, failed,
or has an unapproved record), the combined XML this produces, and
idempotent/safe-to-retry resubmission. No credentials are used anywhere
- this application has no authentication (explicit decision, see
app/api/batches.py's module docstring), so there is no "other caller
can't submit this batch" case to test; everyone can, by design.

The `client` fixture's worker is never started (see conftest.py), so
images stay "queued" after a real batch upload - tests that need a
"completed, approved" image drive that state directly through the same
repo/app_db fixtures the real worker would use, exactly like
test_batches_api.py already does for queued-state assertions.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from app.models.document import (
    AuditInfo, DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus,
    RecordEntry, SourceFileInfo,
)

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _create_batch(client, n=1):
    resp = client.post("/api/batches", files=[("files", (f"img{i}.jpg", TINY_JPEG, "image/jpeg")) for i in range(n)])
    assert resp.status_code == 200, resp.text
    return resp.json()["batch_id"]


def _mark_image_completed_and_approved(repo, db, batch_id, position, *, approved=True):
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
                    field_name="customer_name", label="Customer Name", value="Test Customer",
                    data_type=DataType.NAME, confidence=99.0, status=FieldStatus.VERIFIED, source=FieldSource.OCR,
                ),
            },
            approved=approved,
        )
    ]
    record.record_count = 1
    repo.save(record)
    db.update_batch_image_status(batch_id, position, "completed")
    db.recompute_batch_status(batch_id)
    return document_id


def test_submit_blocked_while_images_still_queued(client):
    batch_id = _create_batch(client, n=1)

    resp = client.post(f"/api/batches/{batch_id}/submit")
    assert resp.status_code == 400
    assert "processing" in resp.json()["detail"]["blockers"][0]


def test_submit_blocked_when_record_unapproved(client, repo, app_db):
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0, approved=False)

    resp = client.post(f"/api/batches/{batch_id}/submit")
    assert resp.status_code == 400
    assert "unapproved" in resp.json()["detail"]["blockers"][0]


def test_submit_blocked_when_image_failed(client, app_db):
    batch_id = _create_batch(client, n=1)
    app_db.update_batch_image_status(batch_id, 0, "failed", "OCR engine error")
    app_db.recompute_batch_status(batch_id)

    resp = client.post(f"/api/batches/{batch_id}/submit")
    assert resp.status_code == 400
    assert "failed" in resp.json()["detail"]["blockers"][0]


def test_submit_succeeds_when_fully_approved(client, repo, app_db):
    batch_id = _create_batch(client, n=2)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 1)

    resp = client.post(f"/api/batches/{batch_id}/submit")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["records_submitted"] == 2
    assert body["status"] == "submitted"


def test_submitted_xml_is_well_formed_and_uses_na_convention(client, repo, app_db):
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    client.post(f"/api/batches/{batch_id}/submit")
    download = client.get(f"/api/batches/{batch_id}/submission/download")
    assert download.status_code == 200

    text = download.content.decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])
    datam_elements = root.findall("DataM")
    assert len(datam_elements) == 1
    assert datam_elements[0].findtext("CustomerName") == "Test Customer"
    # res_address was never set on this synthetic record - must round-trip
    # to the documented "N.A" placeholder, never a fabricated value or an
    # empty tag.
    assert datam_elements[0].findtext("ResAddress") == "N.A"


def test_resubmit_is_idempotent_not_duplicated(client, repo, app_db):
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    first = client.post(f"/api/batches/{batch_id}/submit")
    second = client.post(f"/api/batches/{batch_id}/submit")
    assert first.status_code == 200 and second.status_code == 200

    download1 = client.get(f"/api/batches/{batch_id}/submission/download")
    download2 = client.get(f"/api/batches/{batch_id}/submission/download")
    assert download1.content == download2.content  # no duplicate/inconsistent export

    status = client.get(f"/api/batches/{batch_id}/submission").json()
    assert status["submitted"] is True
    assert status["record_count"] == 1


def test_submission_status_before_submit(client):
    batch_id = _create_batch(client, n=1)

    status = client.get(f"/api/batches/{batch_id}/submission").json()
    assert status["submitted"] is False

    download = client.get(f"/api/batches/{batch_id}/submission/download")
    assert download.status_code == 404


def test_submit_works_with_no_credentials_presented(client, repo, app_db):
    """The core of the open-access decision: submitting and downloading
    require no Authorization header, no key, nothing."""
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200
    assert client.get(f"/api/batches/{batch_id}/submission").status_code == 200
    assert client.get(f"/api/batches/{batch_id}/submission/download").status_code == 200


def test_edit_persists_and_is_reflected_in_submission(client, repo, app_db):
    """Saved field corrections must survive into the generated XML -
    exercises the explicit-save-then-submit requirement end to end."""
    batch_id = _create_batch(client, n=1)
    document_id = _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    update = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"customer_name": {"value": "Corrected Name"}}},
    )
    assert update.status_code == 200, update.text
    assert update.json()["records"][0]["fields"]["customer_name"]["value"] == "Corrected Name"

    client.post(f"/api/batches/{batch_id}/submit")
    download = client.get(f"/api/batches/{batch_id}/submission/download")
    root = ET.fromstring(download.content.decode("utf-8-sig").split("?>", 1)[1])
    assert root.find("DataM").findtext("CustomerName") == "Corrected Name"


def test_combined_batch_xml_has_one_record_per_image_in_order_no_duplicates(client, repo, app_db):
    """The one-root-element requirement: a 3-image batch must produce a
    SINGLE <doc> containing exactly 3 <DataM> records, each carrying its
    own image's distinct data and ImageName in upload-position order -
    not missing, not duplicated, and not mixed up between images."""
    batch_id = _create_batch(client, n=3)
    document_ids = []
    for position in range(3):
        document_id = _mark_image_completed_and_approved(repo, app_db, batch_id, position)
        record = repo.get(document_id)
        record.records[0].fields["record_no"].value = f"24100{position}"
        record.records[0].fields["customer_name"].value = f"Customer {position}"
        repo.save(record)
        document_ids.append(document_id)

    resp = client.post(f"/api/batches/{batch_id}/submit")
    assert resp.status_code == 200, resp.text
    assert resp.json()["records_submitted"] == 3

    download = client.get(f"/api/batches/{batch_id}/submission/download")
    text = download.content.decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])

    # One combined root, not three separate files concatenated.
    assert root.tag == "doc"
    datam_elements = root.findall("DataM")
    assert len(datam_elements) == 3

    for position, datam in enumerate(datam_elements):
        assert datam.findtext("ImageName") == f"img{position}.jpg"
        assert datam.findtext("RecordNo") == f"24100{position}"
        assert datam.findtext("CustomerName") == f"Customer {position}"

    # No record's data leaked into another record's position.
    customer_names = [d.findtext("CustomerName") for d in datam_elements]
    assert len(set(customer_names)) == 3
    image_names = [d.findtext("ImageName") for d in datam_elements]
    assert len(set(image_names)) == 3


def test_separate_batches_stay_isolated_in_their_own_combined_xml(client, repo, app_db):
    """Submitting one batch must never pull in, or be confused with,
    another batch's records - each batch's combined XML contains only
    its own images."""
    batch_a = _create_batch(client, n=2)
    batch_b = _create_batch(client, n=1)
    for position in range(2):
        _mark_image_completed_and_approved(repo, app_db, batch_a, position)
    _mark_image_completed_and_approved(repo, app_db, batch_b, 0)

    client.post(f"/api/batches/{batch_a}/submit")
    client.post(f"/api/batches/{batch_b}/submit")

    text_a = client.get(f"/api/batches/{batch_a}/submission/download").content.decode("utf-8-sig")
    text_b = client.get(f"/api/batches/{batch_b}/submission/download").content.decode("utf-8-sig")
    root_a = ET.fromstring(text_a.split("?>", 1)[1])
    root_b = ET.fromstring(text_b.split("?>", 1)[1])

    assert len(root_a.findall("DataM")) == 2
    assert len(root_b.findall("DataM")) == 1


# --- DELETE /api/batches/{id}: clearing a completed batch ---

def test_clear_batch_blocked_before_submission(client, repo, app_db):
    """Must never clear a batch that hasn't been submitted/downloaded yet -
    that would destroy in-progress review work."""
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    resp = client.delete(f"/api/batches/{batch_id}")
    assert resp.status_code == 400
    assert "submit" in resp.json()["detail"].lower()


def test_clear_batch_removes_data_after_submission(client, repo, app_db):
    batch_id = _create_batch(client, n=2)
    document_id_0 = _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    document_id_1 = _mark_image_completed_and_approved(repo, app_db, batch_id, 1)
    assert client.post(f"/api/batches/{batch_id}/submit").status_code == 200

    resp = client.delete(f"/api/batches/{batch_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["images_cleared"] == 2
    assert body["errors"] == []

    # The batch itself (and its tracking rows) are gone...
    assert client.get(f"/api/batches/{batch_id}").status_code == 404
    # ...and so are the per-document records.
    assert not repo.exists(document_id_0)
    assert not repo.exists(document_id_1)


def test_clear_batch_never_touches_a_different_batch(client, repo, app_db):
    """Clearing one batch must never remove another batch's data, even
    one created around the same time with overlapping document indices."""
    batch_a = _create_batch(client, n=1)
    batch_b = _create_batch(client, n=1)
    doc_a = _mark_image_completed_and_approved(repo, app_db, batch_a, 0)
    doc_b = _mark_image_completed_and_approved(repo, app_db, batch_b, 0)
    client.post(f"/api/batches/{batch_a}/submit")
    client.post(f"/api/batches/{batch_b}/submit")

    resp = client.delete(f"/api/batches/{batch_a}")
    assert resp.status_code == 200

    assert client.get(f"/api/batches/{batch_a}").status_code == 404
    assert not repo.exists(doc_a)
    # batch_b is completely untouched.
    assert client.get(f"/api/batches/{batch_b}").status_code == 200
    assert repo.exists(doc_b)


def test_clear_batch_reports_partial_failure_instead_of_claiming_success(client, repo, app_db, test_settings):
    """A document whose record file is already missing must be reported as
    an error, not silently counted as cleared - "claim success only when
    it actually happened" also applies to cleanup, not just submission."""
    batch_id = _create_batch(client, n=1)
    document_id = _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    client.post(f"/api/batches/{batch_id}/submit")

    # Simulate the record file already being gone before cleanup runs.
    repo.delete(document_id)

    resp = client.delete(f"/api/batches/{batch_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["images_cleared"] == 0
    assert len(body["errors"]) == 1


# --- DELETE /api/batches/{id}/images/{position}: removing one image ---

def test_remove_image_from_unsubmitted_batch(client, repo, app_db):
    batch_id = _create_batch(client, n=3)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    doc1 = _mark_image_completed_and_approved(repo, app_db, batch_id, 1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 2)

    resp = client.delete(f"/api/batches/{batch_id}/images/1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["image_count"] == 2
    remaining_positions = [i["position"] for i in body["images"]]
    assert 1 not in remaining_positions
    assert len(body["images"]) == 2
    assert not repo.exists(doc1)


def test_remove_image_blocked_after_submission(client, repo, app_db):
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    client.post(f"/api/batches/{batch_id}/submit")

    resp = client.delete(f"/api/batches/{batch_id}/images/0")
    assert resp.status_code == 400
    assert "submitted" in resp.json()["detail"].lower()


def test_remove_unknown_image_position_404s(client, repo, app_db):
    batch_id = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)

    resp = client.delete(f"/api/batches/{batch_id}/images/99")
    assert resp.status_code == 404


def test_remove_image_does_not_affect_other_batches(client, repo, app_db):
    batch_a = _create_batch(client, n=1)
    batch_b = _create_batch(client, n=1)
    _mark_image_completed_and_approved(repo, app_db, batch_a, 0)
    doc_b = _mark_image_completed_and_approved(repo, app_db, batch_b, 0)

    client.delete(f"/api/batches/{batch_a}/images/0")

    detail_b = client.get(f"/api/batches/{batch_b}").json()
    assert detail_b["image_count"] == 1
    assert repo.exists(doc_b)
