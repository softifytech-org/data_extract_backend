"""Regression tests for ExtractedField.marked_invalid.

Real, confirmed bug: the review UI's "Mark Invalid" button used to express
"this field is wrong" by overwriting the field's own value with the literal
text "N.A" - permanently destroying whatever the operator had typed or OCR
had read (e.g. a wrong-but-readable email, "pranay123@gmial.com", became
unrecoverable the moment it was flagged). marked_invalid is a separate
property precisely so marking a field invalid never touches its value -
see ExtractedField's own docstring (app/models/document.py) and
generate_remark's (services/validation/remark.py).
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _create_batch(client, n=1):
    resp = client.post("/api/batches", files=[("files", (f"img{i}.jpg", TINY_JPEG, "image/jpeg")) for i in range(n)])
    assert resp.status_code == 200, resp.text
    return resp.json()["batch_id"]


def _mark_image_completed_and_approved(repo, db, batch_id, position, *, approved=True):
    from app.models.document import DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus, RecordEntry

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
                "email_address": ExtractedField(
                    field_name="email_address", label="Email Address", value="pranay123@gmial.com",
                    data_type=DataType.EMAIL, confidence=92.0, status=FieldStatus.VERIFIED, source=FieldSource.OCR,
                ),
                "remark": ExtractedField(
                    field_name="remark", label="Remark", value="N.A",
                    data_type=DataType.TEXT, confidence=99.0, status=FieldStatus.VERIFIED, source=FieldSource.OCR,
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


def _setup(client, repo, db):
    batch_id = _create_batch(client, n=1)
    document_id = _mark_image_completed_and_approved(repo, db, batch_id, 0)
    return batch_id, document_id


def test_marking_invalid_preserves_the_value(client, repo, app_db):
    """The headline bug report: clicking Invalid on a populated field must
    not clear or replace it, and the field must read as invalid afterward."""
    _, document_id = _setup(client, repo, app_db)

    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    assert resp.status_code == 200, resp.text
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["value"] == "pranay123@gmial.com"
    assert field["marked_invalid"] is True


def test_unmarking_invalid_also_preserves_the_value(client, repo, app_db):
    """Un-marking must not restore/guess a value either - it was never
    touched in the first place, so there's nothing to restore."""
    _, document_id = _setup(client, repo, app_db)
    client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": False}}},
    )
    assert resp.status_code == 200, resp.text
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["value"] == "pranay123@gmial.com"
    assert field["marked_invalid"] is False


def test_editing_value_does_not_clear_the_invalid_flag(client, repo, app_db):
    """Marking invalid must not prevent normal editing, and a separate
    value edit must not silently clear a flag the operator didn't touch."""
    _, document_id = _setup(client, repo, app_db)
    client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"value": "pranay123@gmail.com"}}},
    )
    assert resp.status_code == 200, resp.text
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["value"] == "pranay123@gmail.com"
    assert field["marked_invalid"] is True  # untouched by the value-only request


def test_intentional_clear_while_marked_invalid_is_saved(client, repo, app_db):
    """If the operator deliberately empties a marked-invalid field, that
    deletion must be saved like any other edit - the invalid flag must not
    fight it or restore the old value."""
    _, document_id = _setup(client, repo, app_db)
    client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"value": ""}}},
    )
    assert resp.status_code == 200, resp.text
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["value"] is None
    assert field["marked_invalid"] is True


def test_remark_names_the_field_when_marked_invalid_with_a_valid_value(client, repo, app_db):
    """A well-formatted, non-N.A value that an operator still knows is
    wrong must show up in Remark - this is the whole reason the flag
    exists (format validation alone would never catch a wrong-but-valid
    email like a typo'd domain)."""
    _, document_id = _setup(client, repo, app_db)
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    assert resp.status_code == 200, resp.text
    remark = resp.json()["records"][0]["fields"]["remark"]["value"]
    assert remark is not None and "Email Address" in remark and "invalid" in remark


def test_remark_does_not_duplicate_on_repeated_toggles(client, repo, app_db):
    """Toggling invalid on, off, and on again must never accumulate
    duplicate mentions of the same field in Remark - it is recomputed
    fresh from current field state every time, not appended to."""
    _, document_id = _setup(client, repo, app_db)
    for marked in (True, False, True):
        resp = client.put(
            f"/api/documents/{document_id}/records/1/fields",
            json={"fields": {"email_address": {"marked_invalid": marked}}},
        )
    remark = resp.json()["records"][0]["fields"]["remark"]["value"]
    assert remark.count("Email Address") == 1


def test_genuine_na_value_is_not_confused_with_marked_invalid(client, repo, app_db):
    """A field whose REAL, correctly-extracted value is the explicit N.A
    marker must still be reported invalid in Remark via the pre-existing
    rule, entirely independent of the new flag (which stays False)."""
    _, document_id = _setup(client, repo, app_db)
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"value": "N.A"}}},
    )
    assert resp.status_code == 200, resp.text
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["value"] == "N.A"
    assert field["marked_invalid"] is False
    remark = resp.json()["records"][0]["fields"]["remark"]["value"]
    assert "Email Address" in remark and "invalid" in remark


def test_multiple_invalid_fields_all_named_in_remark(client, repo, app_db):
    _, document_id = _setup(client, repo, app_db)
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={
            "fields": {
                "email_address": {"marked_invalid": True},
                "customer_name": {"marked_invalid": True},
            }
        },
    )
    assert resp.status_code == 200, resp.text
    remark = resp.json()["records"][0]["fields"]["remark"]["value"]
    assert "Email Address" in remark and "Customer Name" in remark


def test_marked_invalid_persists_across_reload(client, repo, app_db):
    """Simulates refresh/reopen: GET must return the same flag and value
    a later request set, proving it round-trips through storage, not just
    the single response that set it."""
    _, document_id = _setup(client, repo, app_db)
    client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    resp = client.get(f"/api/documents/{document_id}")
    assert resp.status_code == 200, resp.text
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["value"] == "pranay123@gmial.com"
    assert field["marked_invalid"] is True


def test_invalid_flag_value_exported_as_is_in_xml(client, repo, app_db):
    """The exported XML must carry the real (possibly wrong) value, not
    "N.A" - Remark is where the invalid flag shows up, not the field
    itself."""
    batch_id, document_id = _setup(client, repo, app_db)
    client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    client.post(f"/api/batches/{batch_id}/submit")
    download = client.get(f"/api/batches/{batch_id}/submission/download")
    assert download.status_code == 200, download.text
    root = ET.fromstring(download.content.decode("utf-8-sig").split("?>", 1)[1])
    datam = root.find("DataM")
    assert datam.findtext("EmailAddress") == "pranay123@gmial.com"
    assert "Email Address" in (datam.findtext("Remarks") or "")


def test_marking_invalid_alone_does_not_touch_source_or_status(client, repo, app_db):
    """Toggling the flag is not a value edit - it must not flip an
    OCR-sourced field to human-edited, or mark it "corrected" the way an
    actual value edit does (status is otherwise recomputed from confidence
    on every save regardless of what changed - see _revalidate_entry -
    so "corrected" specifically, not any particular status, is the
    tell-tale sign the value-edit code path wrongly fired here)."""
    _, document_id = _setup(client, repo, app_db)
    resp = client.put(
        f"/api/documents/{document_id}/records/1/fields",
        json={"fields": {"email_address": {"marked_invalid": True}}},
    )
    field = resp.json()["records"][0]["fields"]["email_address"]
    assert field["source"] == "ocr"
    assert field["status"] != "corrected"
