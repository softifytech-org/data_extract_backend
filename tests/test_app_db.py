"""Direct tests of the durable queue/user/batch persistence layer -
codifies what was manually verified via curl during development (idempotent
enqueue, claim/lock, bounded retry with backoff, dead-letter) as repeatable
automated tests."""
from __future__ import annotations

import time


def test_user_create_and_lookup(app_db):
    uid = app_db.create_user("a@example.com", "hash1")
    found = app_db.get_user_by_api_key_hash("hash1")
    assert found["id"] == uid
    assert found["email"] == "a@example.com"
    assert app_db.get_user_by_api_key_hash("nonexistent-hash") is None


def test_batch_and_images(app_db):
    uid = app_db.create_user("a@example.com", "hash1")
    batch_id = app_db.create_batch(uid, 2)
    app_db.add_batch_image(batch_id, 0, "DOC-AAAAAAAAAAAA", "a.jpg")
    app_db.add_batch_image(batch_id, 1, "DOC-BBBBBBBBBBBB", "b.jpg")

    images = app_db.list_batch_images(batch_id)
    assert [i["document_id"] for i in images] == ["DOC-AAAAAAAAAAAA", "DOC-BBBBBBBBBBBB"]
    assert all(i["status"] == "queued" for i in images)

    assert app_db.get_batch_id_for_document("DOC-AAAAAAAAAAAA") == batch_id
    assert app_db.get_batch_id_for_document("DOC-NOT-IN-ANY-BATCH") is None


def test_recompute_batch_status_transitions(app_db):
    uid = app_db.create_user("a@example.com", "hash1")
    batch_id = app_db.create_batch(uid, 2)
    app_db.add_batch_image(batch_id, 0, "DOC-A", "a.jpg")
    app_db.add_batch_image(batch_id, 1, "DOC-B", "b.jpg")

    app_db.recompute_batch_status(batch_id)
    assert app_db.get_batch(batch_id)["status"] == "queued"

    app_db.update_batch_image_status(batch_id, 0, "processing")
    app_db.recompute_batch_status(batch_id)
    assert app_db.get_batch(batch_id)["status"] == "processing"

    app_db.update_batch_image_status(batch_id, 0, "completed")
    app_db.update_batch_image_status(batch_id, 1, "needs_review")
    app_db.recompute_batch_status(batch_id)
    assert app_db.get_batch(batch_id)["status"] == "completed"

    app_db.update_batch_image_status(batch_id, 0, "failed")
    app_db.update_batch_image_status(batch_id, 1, "failed")
    app_db.recompute_batch_status(batch_id)
    assert app_db.get_batch(batch_id)["status"] == "failed"


def test_enqueue_is_idempotent(app_db):
    job_id = app_db.enqueue_job("process_image", {"x": 1}, idempotency_key="process:DOC-A")
    duplicate = app_db.enqueue_job("process_image", {"x": 1}, idempotency_key="process:DOC-A")
    assert job_id is not None
    assert duplicate is None
    assert app_db.queue_depth() == {"pending": 1}


def test_claim_locks_and_increments_attempts(app_db):
    app_db.enqueue_job("process_image", {"x": 1}, idempotency_key="k1")
    job = app_db.claim_next_job("worker-1")
    assert job is not None
    assert job["status"] == "running"
    assert job["attempts"] == 1
    # A second worker must not be able to claim the same job while it's locked.
    assert app_db.claim_next_job("worker-2") is None


def test_job_success_path(app_db):
    app_db.enqueue_job("process_image", {}, idempotency_key="k1")
    job = app_db.claim_next_job("worker-1")
    app_db.mark_job_succeeded(job["id"])
    assert app_db.queue_depth() == {"succeeded": 1}


def test_job_retries_then_dead_letters(app_db):
    app_db.enqueue_job("process_image", {}, idempotency_key="k1", max_attempts=2)

    job = app_db.claim_next_job("worker-1")
    assert job["attempts"] == 1
    app_db.mark_job_failed(job["id"], "boom 1", job["attempts"], job["max_attempts"])
    assert app_db.queue_depth() == {"pending": 1}
    # Backoff means it must NOT be immediately reclaimable.
    assert app_db.claim_next_job("worker-1") is None

    # Force the backoff to have elapsed rather than sleeping in a test.
    import sqlite3

    conn = sqlite3.connect(app_db._path)  # noqa: SLF001 - test-only direct manipulation
    conn.execute("UPDATE jobs SET next_attempt_at = '2000-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()

    job2 = app_db.claim_next_job("worker-1")
    assert job2["attempts"] == 2
    app_db.mark_job_failed(job2["id"], "boom 2", job2["attempts"], job2["max_attempts"])
    assert app_db.queue_depth() == {"dead_letter": 1}


def test_concurrent_claims_never_double_claim_the_same_job(app_db):
    """20 'workers' racing to claim 5 jobs via real OS threads against the
    SAME AppDatabase/SQLite file - the BEGIN IMMEDIATE transaction in
    claim_next_job must make this safe: every job claimed exactly once,
    never twice, even under genuine concurrent access."""
    import threading

    for i in range(5):
        app_db.enqueue_job("process_image", {"i": i}, idempotency_key=f"concurrency:{i}")

    claimed_job_ids: list[str] = []
    lock = threading.Lock()

    def worker(worker_id: int) -> None:
        while True:
            job = app_db.claim_next_job(f"worker-{worker_id}")
            if job is None:
                return
            with lock:
                claimed_job_ids.append(job["id"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(claimed_job_ids) == 5
    assert len(set(claimed_job_ids)) == 5  # no job claimed twice


def test_data_persists_across_simulated_restart(test_settings):
    """Rebuilding AppDatabase/JsonDocumentRepository against the SAME
    on-disk path (simulating a process restart - the in-memory object is
    new, the files are not) must still see everything written before."""
    from app.services.storage.app_db import AppDatabase
    from app.services.storage.json_storage import JsonDocumentRepository

    db1 = AppDatabase(test_settings.app_db_path)
    uid = db1.create_user("restart@example.com", "hash1")
    batch_id = db1.create_batch(uid, 1)
    db1.add_batch_image(batch_id, 0, "DOC-333333333333", "a.jpg")

    repo1 = JsonDocumentRepository(test_settings.records_dir)
    from app.models.document import AuditInfo, DocumentRecord, DocumentStatus, SourceFileInfo

    record = DocumentRecord(
        document_id="DOC-333333333333", status=DocumentStatus.UPLOADED,
        source_file=SourceFileInfo(
            original_filename="a.jpg", stored_filename="DOC-333333333333.jpg",
            file_type="image", content_type="image/jpeg",
        ),
        audit=AuditInfo(),
    )
    repo1.save(record)

    # Fresh instances, same paths - this is the "restart".
    db2 = AppDatabase(test_settings.app_db_path)
    repo2 = JsonDocumentRepository(test_settings.records_dir)

    assert db2.get_batch(batch_id)["image_count"] == 1
    assert db2.list_batch_images(batch_id)[0]["document_id"] == "DOC-333333333333"
    assert db2.get_user_by_email("restart@example.com")["id"] == uid
    assert repo2.get("DOC-333333333333").source_file.original_filename == "a.jpg"


def test_stale_lock_is_reclaimable(app_db):
    """Simulates a worker crash: a job stuck 'running' past the staleness
    window must become claimable again, not lost forever."""
    import sqlite3

    app_db.enqueue_job("process_image", {}, idempotency_key="k1")
    job = app_db.claim_next_job("worker-1")
    assert job is not None

    conn = sqlite3.connect(app_db._path)  # noqa: SLF001
    conn.execute("UPDATE jobs SET locked_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (job["id"],))
    conn.commit()
    conn.close()

    reclaimed = app_db.claim_next_job("worker-2")
    assert reclaimed is not None
    assert reclaimed["id"] == job["id"]
