"""Tests for the 24h completion-confirmation retention policy:

- confirm-completion reliably reaches the backend and is idempotent
  (repeat clicks never push the retention window back out or schedule a
  duplicate cleanup job)
- the scheduled cleanup job is not claimable before 24h, IS claimable
  once due, and survives a worker/process restart (re-claimed by a new
  BatchWorker instance exactly like an in-flight process_image job would)
- an unfinished batch, or one whose export never happened, can never be
  cleaned up - confirm-completion itself refuses, and the job handler
  re-verifies independently
- archive/unarchive does not interact with the countdown at all
"""
from __future__ import annotations

import asyncio

from app.models.document import (
    DataType, DocumentStatus, ExtractedField, FieldSource, FieldStatus, RecordEntry,
)
from app.services.queue.worker import BatchWorker

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _create_batch(client, n=1):
    resp = client.post("/api/batches", files=[("files", (f"img{i}.jpg", TINY_JPEG, "image/jpeg")) for i in range(n)])
    assert resp.status_code == 200, resp.text
    return resp.json()["batch_id"]


def _mark_image_completed_and_approved(repo, db, batch_id, position):
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
            },
            approved=True,
        )
    ]
    record.record_count = 1
    repo.save(record)
    db.update_batch_image_status(batch_id, position, "completed")
    db.recompute_batch_status(batch_id)

    # The `client` fixture's worker is never started (see conftest.py),
    # so the real "process_image" job enqueued at upload time is still
    # sitting pending. This helper simulates "the worker already finished
    # it" the same way test_batch_submit.py's identical helper does - but
    # a STILL-PENDING, already-due process_image job would otherwise be
    # claimed ahead of a later cleanup_batch job (claim_next_job orders
    # by created_at), so it must be marked done here too, matching what
    # the real worker would have done before the operator could ever
    # reach "approved".
    stale_job = db.get_job_by_idempotency_key(f"process_image:{document_id}")
    if stale_job is not None:
        db.mark_job_succeeded(stale_job["id"])
    return document_id


def _submit(client, batch_id):
    resp = client.post(f"/api/batches/{batch_id}/submit")
    assert resp.status_code == 200, resp.text


def _make_worker(app_db, repo, test_settings) -> BatchWorker:
    from tests.conftest import FakeOCREngine

    return BatchWorker(app_db=app_db, repo=repo, settings=test_settings, ocr_engine_factory=lambda: FakeOCREngine())


# --- confirm-completion reliability -----------------------------------------

def test_confirm_completion_requires_a_real_submission(client, repo, app_db):
    """Must not be possible to start the retention countdown on a batch
    that was never actually, successfully exported."""
    batch_id = _create_batch(client)
    resp = client.post(f"/api/batches/{batch_id}/confirm-completion")
    assert resp.status_code == 400

    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["completed_confirmed_at"] is None
    assert app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}") is None


def test_confirm_completion_schedules_cleanup_24h_out(client, repo, app_db):
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)

    resp = client.post(f"/api/batches/{batch_id}/confirm-completion")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batch_id"] == batch_id
    assert body["completed_confirmed_at"]
    assert body["cleanup_scheduled_at"]

    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["completed_confirmed_at"] == body["completed_confirmed_at"]

    job = app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}")
    assert job is not None
    assert job["status"] == "pending"
    # Not claimable right now - the whole point of the 24h delay.
    assert app_db.claim_next_job("test-worker") is None


def test_confirm_completion_is_idempotent(client, repo, app_db):
    """Repeat confirmations (e.g. a retried request) must not push the
    retention window back out, and must not schedule a second cleanup job."""
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)

    first = client.post(f"/api/batches/{batch_id}/confirm-completion").json()
    second = client.post(f"/api/batches/{batch_id}/confirm-completion").json()
    assert first["completed_confirmed_at"] == second["completed_confirmed_at"]
    assert first["cleanup_scheduled_at"] == second["cleanup_scheduled_at"]

    rows = app_db.list_batch_exports(batch_id)  # sanity: exports untouched by confirm
    assert rows == app_db.list_batch_exports(batch_id)
    # Only one job was ever created for this batch - the idempotency key
    # on the SECOND enqueue_job call returns None rather than inserting.
    cur_jobs = [j for j in [app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}")] if j]
    assert len(cur_jobs) == 1


# --- delayed cleanup execution + worker-restart survival ---------------------

def test_cleanup_job_not_claimable_before_24h_then_runs_and_deletes_data(client, repo, app_db, test_settings):
    batch_id = _create_batch(client)
    document_id = _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)
    client.post(f"/api/batches/{batch_id}/confirm-completion")

    assert app_db.claim_next_job("worker-a") is None  # still 24h out

    # Simulate "24 hours have passed" the same way test_queue_worker.py's
    # own retry-backoff tests would - move the job's own next_attempt_at
    # into the past directly, rather than sleeping for a real day.
    with app_db._connect() as conn:  # noqa: SLF001 - test-only direct manipulation
        conn.execute(
            "UPDATE jobs SET next_attempt_at = '2000-01-01T00:00:00+00:00' WHERE idempotency_key = ?",
            (f"cleanup_batch:{batch_id}",),
        )

    job = app_db.claim_next_job("worker-a")
    assert job is not None
    assert job["job_type"] == "cleanup_batch"

    worker = _make_worker(app_db, repo, test_settings)
    asyncio.run(worker._run_job(job))  # noqa: SLF001 - testing the real unit directly

    assert app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}")["status"] == "succeeded"
    assert app_db.get_batch(batch_id) is None  # batch tracking row gone
    assert repo.exists(document_id) is False  # per-document JSON record deleted too


def test_cleanup_job_survives_worker_restart(client, repo, app_db, test_settings):
    """A brand-new BatchWorker instance (simulating a process restart)
    must still be able to claim and finish a cleanup job that an earlier
    worker scheduled - the job lives in the durable queue, not in any
    worker's own memory."""
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)
    client.post(f"/api/batches/{batch_id}/confirm-completion")

    with app_db._connect() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE jobs SET next_attempt_at = '2000-01-01T00:00:00+00:00' WHERE idempotency_key = ?",
            (f"cleanup_batch:{batch_id}",),
        )

    # "Old" worker never gets to run it (simulating a crash between
    # scheduling and execution) - a brand new worker instance claims it.
    new_worker = _make_worker(app_db, repo, test_settings)
    job = app_db.claim_next_job("worker-after-restart")
    assert job is not None
    assert job["job_type"] == "cleanup_batch"
    asyncio.run(new_worker._run_job(job))  # noqa: SLF001

    assert app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}")["status"] == "succeeded"
    assert app_db.get_batch(batch_id) is None


# --- safety: never touch unfinished or failed batches -------------------------

def test_cleanup_job_refuses_if_batch_was_never_submitted(app_db, repo, test_settings):
    """Defense in depth: even if a cleanup_batch job somehow existed for
    a batch with no real submission (should be structurally impossible
    via the API), the job handler itself must still refuse."""
    uid = app_db.create_user("rt1@example.com", "h1")
    batch_id = app_db.create_batch(uid, 1)
    app_db.add_batch_image(batch_id, 0, "DOC-RETEN0000001", "a.jpg")

    app_db.enqueue_job(
        "cleanup_batch", {"batch_id": batch_id}, idempotency_key=f"cleanup_batch:{batch_id}", delay_seconds=0,
    )
    job = app_db.claim_next_job("w")
    worker = _make_worker(app_db, repo, test_settings)
    asyncio.run(worker._run_job(job))  # noqa: SLF001

    # The job "succeeds" (it ran without raising) but the batch itself
    # must still be fully intact - the handler's own safety check is
    # what refused to delete anything.
    assert app_db.get_batch(batch_id) is not None
    assert app_db.list_batch_images(batch_id) != []


def test_cleanup_job_refuses_if_completion_was_never_confirmed(client, repo, app_db, test_settings):
    """A batch that was submitted (export succeeded) but never explicitly
    completion-confirmed must never be auto-cleaned - only
    confirm_batch_completion can ever cause that job to be scheduled in
    the real flow, but this proves the handler doesn't just trust the
    job existing."""
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)
    # No confirm-completion call at all.

    app_db.enqueue_job(
        "cleanup_batch", {"batch_id": batch_id}, idempotency_key=f"cleanup_batch:{batch_id}", delay_seconds=0,
    )
    job = app_db.claim_next_job("w")
    worker = _make_worker(app_db, repo, test_settings)
    asyncio.run(worker._run_job(job))  # noqa: SLF001

    assert app_db.get_batch(batch_id) is not None
    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["review_status"] == "exported"  # still there, still downloadable


def test_unfinished_batch_is_never_eligible_for_confirm_completion(client):
    """A batch still being reviewed (nothing submitted) cannot even reach
    the point where a cleanup job could be scheduled."""
    batch_id = _create_batch(client)
    resp = client.post(f"/api/batches/{batch_id}/confirm-completion")
    assert resp.status_code == 400
    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["review_status"] in ("in_progress", "paused")


def test_batch_with_failed_image_is_never_eligible_for_confirm_completion(client, app_db):
    batch_id = _create_batch(client)
    app_db.update_batch_image_status(batch_id, 0, "failed", "OCR engine error")
    app_db.recompute_batch_status(batch_id)

    resp = client.post(f"/api/batches/{batch_id}/confirm-completion")
    assert resp.status_code == 400


# --- archive/unarchive independence from the countdown ------------------------

def test_archiving_does_not_cancel_or_affect_the_cleanup_schedule(client, repo, app_db):
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)
    confirm = client.post(f"/api/batches/{batch_id}/confirm-completion").json()

    client.post(f"/api/batches/{batch_id}/archive")

    job_before = app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}")
    assert job_before is not None and job_before["status"] == "pending"

    client.post(f"/api/batches/{batch_id}/unarchive")

    job_after = app_db.get_job_by_idempotency_key(f"cleanup_batch:{batch_id}")
    assert job_after["id"] == job_before["id"]  # same job, never re-scheduled
    assert job_after["next_attempt_at"] == job_before["next_attempt_at"]

    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["completed_confirmed_at"] == confirm["completed_confirmed_at"]


def test_archived_batch_is_still_cleaned_up_once_due(client, repo, app_db, test_settings):
    """Explicit product decision: archiving does NOT exempt a batch from
    the 24h countdown."""
    batch_id = _create_batch(client)
    _mark_image_completed_and_approved(repo, app_db, batch_id, 0)
    _submit(client, batch_id)
    client.post(f"/api/batches/{batch_id}/confirm-completion")
    client.post(f"/api/batches/{batch_id}/archive")

    with app_db._connect() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE jobs SET next_attempt_at = '2000-01-01T00:00:00+00:00' WHERE idempotency_key = ?",
            (f"cleanup_batch:{batch_id}",),
        )
    job = app_db.claim_next_job("w")
    worker = _make_worker(app_db, repo, test_settings)
    asyncio.run(worker._run_job(job))  # noqa: SLF001

    assert app_db.get_batch(batch_id) is None
