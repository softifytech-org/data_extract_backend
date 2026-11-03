"""Shared batch working-data cleanup - the actual deletion logic behind
both the manual DELETE /api/batches/{id} endpoint (app/api/batches.py's
clear_batch) and the automatic 24h-retention cleanup job (services/queue/
worker.py). One implementation, two triggers, so they can never silently
diverge in what "cleared" means.

Deletes: each image's uploaded file, its per-document JSON record (the
correction/review data), and the batch's own tracking rows (batch_images,
batch_submissions, batch_exports, batches - see AppDatabase.delete_batch).
Does NOT delete the already-exported XML blob itself in storage - that is
the one durable deliverable a downstream system may still depend on; only
the DATABASE's ability to list/re-download it goes away with the tracking
rows, matching the exact same limitation delete_batch already documents
for upsert_batch_submission.
"""
from __future__ import annotations

from app.services.storage.app_db import AppDatabase
from app.services.storage.blob_store import FileStore
from app.services.storage.json_storage import JsonDocumentRepository


def clear_batch_data(
    batch_id: str, db: AppDatabase, repo: JsonDocumentRepository, file_store: FileStore
) -> tuple[int, list[str]]:
    """Best-effort per-image cleanup: one image's upload or record delete
    failing does not abort the rest - every failure is collected and
    returned in `errors` rather than silently claiming full success. The
    batch's own DB rows are only removed once every image has at least
    been attempted. Returns (images_cleared, errors)."""
    images = db.list_batch_images(batch_id)
    errors: list[str] = []
    cleared = 0
    for img in images:
        document_id = img["document_id"]
        label = f"position {img['position'] + 1} ({img['original_filename']})"
        try:
            record = repo.get(document_id)
            file_store.delete_upload(record.source_file.stored_filename)
        except Exception as exc:  # noqa: BLE001 - best-effort; report, don't abort the rest
            errors.append(f"{label}: could not delete uploaded image - {exc}")
            continue
        try:
            repo.delete(document_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: could not delete stored record - {exc}")
            continue
        cleared += 1

    db.delete_batch(batch_id)
    return cleared, errors
