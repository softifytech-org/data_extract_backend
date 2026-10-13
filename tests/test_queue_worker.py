"""Tests the BatchWorker's actual job-processing logic directly (not
through HTTP/TestClient - see conftest.py's `client` fixture docstring for
why the lifespan-started worker can't be reached that way). Uses
asyncio.run() directly rather than adding a pytest-asyncio dependency this
project doesn't otherwise need."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from app.services.queue.worker import BatchWorker


def _make_worker(app_db, repo, test_settings, ocr_engine) -> BatchWorker:
    return BatchWorker(
        app_db=app_db, repo=repo, settings=test_settings, ocr_engine_factory=lambda: ocr_engine,
    )


def _upload_fake_image(test_settings, repo, app_db, batch_id, position, document_id):
    from app.models.document import AuditInfo, DocumentRecord, DocumentStatus, SourceFileInfo

    src = Path(r"D:\data extraction\data\ground_truth\user_25\USER - 25\Vim_17_Live_Data-User 12\User 12 Live Data\09101.jpeg")
    dest = test_settings.uploads_dir / f"{document_id}.jpeg"
    shutil.copy(src, dest)
    record = DocumentRecord(
        document_id=document_id, status=DocumentStatus.UPLOADED,
        source_file=SourceFileInfo(
            original_filename="09101.jpeg", stored_filename=dest.name,
            file_type="image", content_type="image/jpeg",
        ),
        audit=AuditInfo(),
    )
    repo.save(record)
    app_db.add_batch_image(batch_id, position, document_id, "09101.jpeg")
    app_db.enqueue_job(
        "process_image",
        {"document_id": document_id, "batch_id": batch_id, "position": position,
         "file_path": str(dest), "file_type": "image"},
        idempotency_key=f"process_image:{document_id}",
    )
    return dest


def test_worker_processes_job_successfully(app_db, repo, test_settings):
    from tests.conftest import FakeOCREngine

    uid = app_db.create_user("w@example.com", "h1")
    batch_id = app_db.create_batch(uid, 1)
    _upload_fake_image(test_settings, repo, app_db, batch_id, 0, "DOC-F00000000001")

    worker = _make_worker(app_db, repo, test_settings, FakeOCREngine())
    job = app_db.claim_next_job("test-worker")
    asyncio.run(worker._run_job(job))  # noqa: SLF001 - testing the real unit directly

    assert app_db.queue_depth() == {"succeeded": 1}
    images = app_db.list_batch_images(batch_id)
    assert images[0]["status"] in ("completed", "needs_review")
    assert app_db.get_batch(batch_id)["status"] in ("completed",)

    # The document itself must be genuinely processed, not just marked done.
    record = repo.get("DOC-F00000000001")
    assert record.record_count > 0


def _upload_fake_image_via_file_store(test_settings, repo, app_db, batch_id, position, document_id):
    """Same as _upload_fake_image above, but through the FileStore
    abstraction with the current "storage_key"/"storage_backend" job
    payload shape (app/api/batches.py) - _upload_fake_image above
    deliberately keeps using the pre-blob-storage "file_path" shape so
    that backward-compatibility path stays covered too."""
    from app.models.document import AuditInfo, DocumentRecord, DocumentStatus, SourceFileInfo
    from app.services.storage.blob_store import LocalFileStore

    src = Path(r"D:\data extraction\data\ground_truth\user_25\USER - 25\Vim_17_Live_Data-User 12\User 12 Live Data\09101.jpeg")
    store = LocalFileStore(test_settings)
    storage_key = store.save_upload(src.read_bytes(), document_id=document_id, extension=".jpeg", batch_id=batch_id)

    record = DocumentRecord(
        document_id=document_id, status=DocumentStatus.UPLOADED,
        source_file=SourceFileInfo(
            original_filename="09101.jpeg", stored_filename=storage_key,
            file_type="image", content_type="image/jpeg", storage_backend="local",
        ),
        audit=AuditInfo(),
    )
    repo.save(record)
    app_db.add_batch_image(batch_id, position, document_id, "09101.jpeg")
    app_db.enqueue_job(
        "process_image",
        {"document_id": document_id, "batch_id": batch_id, "position": position,
         "storage_key": storage_key, "storage_backend": "local", "file_type": "image"},
        idempotency_key=f"process_image:{document_id}",
    )


def test_worker_processes_job_with_storage_key_payload(app_db, repo, test_settings):
    """The current (post-Blob-Storage-integration) job payload shape -
    resolved through FileStore.open_upload_as_local_path rather than a
    bare file_path - must process correctly end to end."""
    from tests.conftest import FakeOCREngine

    uid = app_db.create_user("w4@example.com", "h4")
    batch_id = app_db.create_batch(uid, 1)
    _upload_fake_image_via_file_store(test_settings, repo, app_db, batch_id, 0, "DOC-F00000000004")

    worker = _make_worker(app_db, repo, test_settings, FakeOCREngine())
    job = app_db.claim_next_job("test-worker")
    asyncio.run(worker._run_job(job))  # noqa: SLF001

    assert app_db.queue_depth() == {"succeeded": 1}
    record = repo.get("DOC-F00000000004")
    assert record.record_count > 0


def test_worker_job_failure_updates_batch_image_status(app_db, repo, test_settings):
    uid = app_db.create_user("w2@example.com", "h2")
    batch_id = app_db.create_batch(uid, 1)
    document_id = "DOC-F00000000002"
    app_db.add_batch_image(batch_id, 0, document_id, "missing.jpeg")
    app_db.enqueue_job(
        "process_image",
        {"document_id": document_id, "batch_id": batch_id, "position": 0,
         "file_path": str(test_settings.uploads_dir / "does-not-exist.jpeg"), "file_type": "image"},
        idempotency_key=f"process_image:{document_id}",
        max_attempts=1,
    )

    class FailingOCR:
        single_pass = True

        def recognize_words(self, image):
            raise RuntimeError("should not be reached - file load fails first")

    worker = _make_worker(app_db, repo, test_settings, FailingOCR())
    job = app_db.claim_next_job("test-worker")
    asyncio.run(worker._run_job(job))  # noqa: SLF001

    assert app_db.queue_depth() == {"dead_letter": 1}
    images = app_db.list_batch_images(batch_id)
    assert images[0]["status"] == "failed"
    assert images[0]["error"]
    assert app_db.get_batch(batch_id)["status"] == "failed"


def test_worker_does_not_duplicate_on_retry(app_db, repo, test_settings):
    """A job claimed, then re-enqueued for the same document_id
    (idempotency_key collision) must not create a second job."""
    uid = app_db.create_user("w3@example.com", "h3")
    batch_id = app_db.create_batch(uid, 1)
    document_id = "DOC-F00000000003"
    dest = _upload_fake_image(test_settings, repo, app_db, batch_id, 0, document_id)

    dup = app_db.enqueue_job(
        "process_image",
        {"document_id": document_id, "batch_id": batch_id, "position": 0,
         "file_path": str(dest), "file_type": "image"},
        idempotency_key=f"process_image:{document_id}",
    )
    assert dup is None
    assert app_db.queue_depth() == {"pending": 1}
