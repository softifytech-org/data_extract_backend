"""Device-aware workspace resume.

No login exists in this application (see app/api/batches.py's module
docstring), so there is no real "user" to resume work for - only a
browser installation. The frontend generates a random device_id once and
keeps it in localStorage (see frontend/src/services/device.ts); every
other field this module tracks (last batch/image/record, timestamps)
lives in the backend database, which is what actually survives a
cleared browser cache, a closed laptop, or a totally different machine.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.batches import _batch_progress, _review_status
from app.core.dependencies import get_app_db, get_repository
from app.schemas.batch import DeviceActivityRequest, DeviceResumeResponse
from app.services.storage.app_db import AppDatabase
from app.services.storage.json_storage import JsonDocumentRepository

router = APIRouter(prefix="/api/devices", tags=["devices"])

# A batch in one of these review_status values still has something an
# operator could come back to finish. "exported" and "archived" are
# deliberately excluded - once a batch is exported the operator has
# already finished it (it shows up under Completed Batches instead, see
# GET /api/batches?archived=... and the existing review_status/EXPORTED
# value), and an archived batch was explicitly put away.
_RESUMABLE_REVIEW_STATUSES = {"in_progress", "paused", "ready_for_export"}


@router.put("/{device_id}/activity", status_code=204, response_model=None)
def report_activity(
    device_id: str, body: DeviceActivityRequest, db: AppDatabase = Depends(get_app_db)
) -> None:
    """Called (debounced) by the frontend whenever the operator opens a
    different image/record or navigates within a batch - see Phase 2's
    "Save Workspace Position". Intentionally fire-and-forget from the
    frontend's perspective: a dropped/delayed call here only means a
    slightly stale resume point, never lost correction data (corrections
    themselves are saved durably by the existing per-record fields
    endpoint, not by this one)."""
    db.save_device_session(device_id, body.batch_id, body.document_id, body.record_index)


@router.get("/{device_id}/resume", response_model=DeviceResumeResponse)
def get_resume_state(
    device_id: str,
    db: AppDatabase = Depends(get_app_db),
    repo: JsonDocumentRepository = Depends(get_repository),
) -> DeviceResumeResponse:
    """Returns this device's last-known workspace position, if it still
    points at something resumable. Never auto-redirects anything by
    itself - the frontend decides whether to show the "Welcome back"
    banner and only jumps there if the operator explicitly clicks Resume
    (see App.tsx's own history on why automatic resume was tried before
    and reverted)."""
    session = db.get_device_session(device_id)
    if session is None or not session["batch_id"]:
        return DeviceResumeResponse(resumable=False)

    batch = db.get_batch(session["batch_id"])
    # The batch may be gone entirely (clear_batch already ran) or simply
    # not resumable right now (exported/archived) - either way, nothing
    # to offer back.
    if batch is None:
        return DeviceResumeResponse(resumable=False)

    images = db.list_batch_images(batch["id"])
    status = _review_status(batch, images, db, repo)
    if status not in _RESUMABLE_REVIEW_STATUSES:
        return DeviceResumeResponse(resumable=False)

    images_completed, records_total, records_completed = _batch_progress(images, repo)
    return DeviceResumeResponse(
        resumable=True,
        batch_id=batch["id"], document_id=session["document_id"], record_index=session["record_index"],
        batch_status=batch["status"], review_status=status,
        image_count=batch["image_count"], images_completed=images_completed,
        records_total=records_total, records_completed=records_completed,
        last_saved_at=session["updated_at"],
    )
