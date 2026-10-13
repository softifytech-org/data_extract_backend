"""The background worker loop that drains the durable job queue
(services/storage/app_db.py). Runs as an asyncio background task started
at FastAPI startup (see app/main.py) - a real worker, not a thread the
request handler fires and forgets about: it survives for the life of the
process, polls the SAME durable SQLite queue every request uses, and any
job it doesn't finish (crash, restart, deploy) is picked back up by
whichever worker claims it next (see AppDatabase.claim_next_job's stale-
lock recovery).

Only one job type exists today: "process_image", which wraps the existing,
already-validated process_document() pipeline unchanged - this worker adds
durability/retry/concurrency-control around that function, it does not
reimplement OCR processing.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.models.document import DocumentStatus
from app.services.ocr.base import OCREngine
from app.services.processing import process_document
from app.services.storage.app_db import AppDatabase
from app.services.storage.blob_store import get_file_store_by_backend
from app.services.storage.json_storage import JsonDocumentRepository

logger = get_logger(__name__)

# How many "process_image" jobs may run at once in this process. Azure's
# resource is S0 Standard (see the Azure readiness doc for the rate-limit
# history that made this necessary) - bounding local concurrency is the
# other half of "don't send everything to Azure OCR simultaneously"
# alongside Azure's own per-resource limit.
_DEFAULT_CONCURRENCY = 3
_POLL_INTERVAL_SECONDS = 1.0
_IDLE_POLL_INTERVAL_SECONDS = 3.0


class BatchWorker:
    def __init__(
        self,
        app_db: AppDatabase,
        repo: JsonDocumentRepository,
        settings: Settings,
        ocr_engine_factory,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        self._db = app_db
        self._repo = repo
        self._settings = settings
        self._ocr_engine_factory = ocr_engine_factory
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._concurrency = concurrency
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._run_lane(i)) for i in range(self._concurrency)]
        logger.info("BatchWorker %s started with %d lane(s)", self._worker_id, self._concurrency)

    async def stop(self) -> None:
        """Graceful shutdown: stop claiming new jobs, let in-flight ones
        finish rather than abandoning them mid-write."""
        self._stop.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("BatchWorker %s stopped", self._worker_id)

    async def _run_lane(self, lane: int) -> None:
        lane_id = f"{self._worker_id}:{lane}"
        while not self._stop.is_set():
            job = await asyncio.to_thread(self._db.claim_next_job, lane_id)
            if job is None:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=_IDLE_POLL_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._run_job(job)
            await asyncio.sleep(0)  # yield between jobs, don't starve other lanes

    async def _run_job(self, job: dict) -> None:
        try:
            if job["job_type"] == "process_image":
                await asyncio.to_thread(self._process_image_job, job["payload"])
            else:
                raise ValueError(f"Unknown job type: {job['job_type']}")
            self._db.mark_job_succeeded(job["id"])
        except Exception as exc:  # noqa: BLE001 - a job failure must never kill the worker loop
            logger.exception("Job %s failed (attempt %d/%d)", job["id"], job["attempts"], job["max_attempts"])
            self._db.mark_job_failed(job["id"], str(exc), job["attempts"], job["max_attempts"])
            payload = job["payload"]
            batch_id = payload.get("batch_id")
            position = payload.get("position")
            if batch_id is not None and position is not None and job["attempts"] >= job["max_attempts"]:
                self._db.update_batch_image_status(batch_id, position, "failed", str(exc)[:500])
                self._db.recompute_batch_status(batch_id)

    def _process_image_job(self, payload: dict) -> None:
        document_id = payload["document_id"]
        batch_id = payload.get("batch_id")
        position = payload.get("position")
        file_type = payload["file_type"]

        if batch_id is not None and position is not None:
            self._db.update_batch_image_status(batch_id, position, "processing")
            self._db.recompute_batch_status(batch_id)

        ocr_engine: OCREngine = self._ocr_engine_factory()

        # "storage_key"/"storage_backend" is the current payload shape
        # (see app/api/batches.py), resolved through the FileStore
        # abstraction so this works unchanged whether the upload lives on
        # local disk or in Azure Blob Storage. "file_path" is kept as a
        # fallback for any job enqueued before this existed (a plain local
        # path) so an in-flight job from before this change still
        # completes normally rather than failing on restart.
        if "storage_key" in payload:
            file_store = get_file_store_by_backend(payload.get("storage_backend", "local"), self._settings)
            with file_store.open_upload_as_local_path(payload["storage_key"]) as file_path:
                record = process_document(
                    document_id,
                    file_path=file_path,
                    file_type=file_type,
                    settings=self._settings,
                    repo=self._repo,
                    ocr_engine=ocr_engine,
                    paddle_engine=None,
                )
        else:
            record = process_document(
                document_id,
                file_path=Path(payload["file_path"]),
                file_type=file_type,
                settings=self._settings,
                repo=self._repo,
                ocr_engine=ocr_engine,
                paddle_engine=None,
            )

        if batch_id is not None and position is not None:
            final_status = "needs_review" if record.status == DocumentStatus.NEEDS_REVIEW else "completed"
            self._db.update_batch_image_status(batch_id, position, final_status)
            self._db.recompute_batch_status(batch_id)
