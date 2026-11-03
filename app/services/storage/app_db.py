"""The application database: users, batches, batch-image membership, and the
durable job queue. Separate from JsonDocumentRepository (which still owns
the per-document OCR/review data - see that module's own docstring) because
these are genuinely relational, multi-row concerns the JSON-file-per-document
model was never meant to express.

SQLite, not a new framework: this project already uses sqlite3 directly for
submissions.db (see services/export/dump_xml_export.py) with no ORM
anywhere, so this follows the same established pattern rather than
introducing SQLAlchemy/Alembic for one more table set.

Durability note: this file IS what satisfies every "durable" requirement
locally - a job survives a process crash because it's a committed SQLite
row, not an in-memory queue. It is explicitly NOT Azure Storage Queue /
Blob Storage, which this environment has no credentials for and is
forbidden from provisioning (see the Azure readiness doc). The job-queue
functions below are written as the seam where an Azure-backed
implementation would plug in later with the same call signatures.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    api_key_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (owner_user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS batch_images (
    batch_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (batch_id, position),
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);
CREATE INDEX IF NOT EXISTS idx_batch_images_document_id ON batch_images(document_id);

CREATE TABLE IF NOT EXISTS batch_submissions (
    batch_id TEXT PRIMARY KEY,
    xml_storage_key TEXT NOT NULL,
    xml_storage_backend TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    submitted_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);

-- Same purpose as batch_submissions above, for a SINGLE document's own
-- submit (app/api/documents.py's submit_document_endpoint) - tracks where
-- that document's submitted XML actually landed (Blob Storage when
-- configured, local disk otherwise) so a later re-download reads back
-- from the SAME durable location instead of assuming local disk. No FK to
-- batches: a document can be submitted this way whether or not it's part
-- of a batch.
CREATE TABLE IF NOT EXISTS document_submissions (
    document_id TEXT PRIMARY KEY,
    xml_storage_key TEXT NOT NULL,
    xml_storage_backend TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    submitted_at TEXT NOT NULL
);

-- Device-aware resume: one row per browser installation (device_id is
-- generated client-side once and kept in localStorage - see
-- frontend/src/services/device.ts). Tracks where that device last was so
-- an operator who closes Chrome, restarts their laptop, or clears
-- browser storage can still recover their place from the backend, which
-- is the actual source of truth - localStorage only carries the
-- device_id itself, never the workspace state.
CREATE TABLE IF NOT EXISTS device_sessions (
    device_id TEXT PRIMARY KEY,
    batch_id TEXT,
    document_id TEXT,
    record_index INTEGER,
    updated_at TEXT NOT NULL
);

-- Append-only export history: batch_submissions (above) stays a single
-- "latest" row per batch for the existing download-XML endpoint; this
-- table instead gains one new row every time a batch is submitted, so an
-- earlier export version is never lost when the batch is corrected and
-- re-submitted later.
CREATE TABLE IF NOT EXISTS batch_exports (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    xml_storage_key TEXT NOT NULL,
    xml_storage_backend TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    exported_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);
CREATE INDEX IF NOT EXISTS idx_batch_exports_batch_id ON batch_exports(batch_id);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    job_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    next_attempt_at TEXT NOT NULL,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    locked_by TEXT,
    locked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_next_attempt ON jobs(status, next_attempt_at);
"""

# A job whose lock is older than this is assumed to belong to a crashed
# worker and becomes reclaimable - this is the worker-restart recovery
# mechanism: no separate heartbeat process, just a timestamp comparison on
# the next claim attempt.
_STALE_LOCK_SECONDS = 300


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"


class AppDatabase:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_batches_columns(conn)

    def _migrate_batches_columns(self, conn: sqlite3.Connection) -> None:
        """Additive migration for columns added after this table's original
        CREATE TABLE - IF NOT EXISTS only guards table creation, not new
        columns on a `batches` row that already exists on disk from before
        this feature. Safe to run on every startup: PRAGMA table_info is
        checked first, so an already-migrated database does nothing here."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(batches)")}
        if "archived" not in existing:
            conn.execute("ALTER TABLE batches ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        if "last_worked_at" not in existing:
            conn.execute("ALTER TABLE batches ADD COLUMN last_worked_at TEXT")
        if "completed_confirmed_at" not in existing:
            conn.execute("ALTER TABLE batches ADD COLUMN completed_confirmed_at TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- users -----------------------------------------------------------

    def create_user(self, email: str, api_key_hash: str) -> str:
        user_id = new_id("USER")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, api_key_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email, api_key_hash, _now_iso()),
            )
        return user_id

    def ensure_user(self, user_id: str, email: str, api_key_hash: str) -> None:
        """Idempotent upsert for a user row with a FIXED, caller-chosen id -
        unlike create_user (random id, used for normal per-user rows), this
        exists for the single shared-identity row the company-access-key
        model creates at startup (see core/auth.py's ensure_company_user).
        Safe to call every startup: if the configured access code changes,
        the stored hash is updated to match on the next call, with no
        migration needed."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, api_key_hash, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET api_key_hash = excluded.api_key_hash",
                (user_id, email, api_key_hash, _now_iso()),
            )

    def get_user_by_api_key_hash(self, api_key_hash: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, created_at FROM users WHERE api_key_hash = ?", (api_key_hash,)
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, api_key_hash, created_at FROM users WHERE email = ?", (email,)
            ).fetchone()
        return dict(row) if row else None

    # --- batches -----------------------------------------------------------

    def create_batch(self, owner_user_id: str, image_count: int) -> str:
        batch_id = new_id("BATCH")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batches (id, owner_user_id, status, image_count, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, ?, ?)",
                (batch_id, owner_user_id, image_count, now, now),
            )
        return batch_id

    def add_batch_image(self, batch_id: str, position: int, document_id: str, original_filename: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batch_images (batch_id, position, document_id, original_filename, status) "
                "VALUES (?, ?, ?, ?, 'queued')",
                (batch_id, position, document_id, original_filename),
            )

    def get_batch(self, batch_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        return dict(row) if row else None

    def list_batches(self, owner_user_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM batches WHERE owner_user_id = ? ORDER BY created_at DESC", (owner_user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_batch_id_for_document(self, document_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT batch_id FROM batch_images WHERE document_id = ?", (document_id,)
            ).fetchone()
        return row["batch_id"] if row else None

    def list_batch_images(self, batch_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM batch_images WHERE batch_id = ? ORDER BY position", (batch_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_batch_image_status(self, batch_id: str, position: int, status: str, error: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE batch_images SET status = ?, error = ? WHERE batch_id = ? AND position = ?",
                (status, error, batch_id, position),
            )

    def remove_batch_image(self, batch_id: str, position: int) -> None:
        """Removes one image from a batch that hasn't been submitted yet
        (see app/api/batches.py's remove_batch_image endpoint, which
        refuses to call this once a submission exists). Deliberately does
        NOT renumber the remaining images' `position` values - a gap in
        position numbers is harmless (list_batch_images still returns them
        correctly ordered; the frontend numbers rows by their position in
        that list, not by the raw position value), and renumbering would
        touch every row after the removed one for no real benefit.
        image_count is decremented to match what's actually left, and the
        batch's own status is left for the caller to recompute (mirrors
        update_batch_image_status, which does the same)."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM batch_images WHERE batch_id = ? AND position = ?", (batch_id, position)
            )
            conn.execute(
                "UPDATE batches SET image_count = image_count - 1, updated_at = ? WHERE id = ?",
                (_now_iso(), batch_id),
            )

    def delete_batch(self, batch_id: str) -> None:
        """Two legitimate callers (see app/api/batches.py): rolling back a
        batch create that failed partway through storing its files, and
        clear_batch's explicit, user-confirmed cleanup of a batch that has
        already been submitted and downloaded. Neither leaves this as a
        general "delete any batch anytime" operation - clear_batch itself
        refuses to run unless a real submission already exists."""
        with self._connect() as conn:
            conn.execute("DELETE FROM batch_images WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM batch_submissions WHERE batch_id = ?", (batch_id,))
            # Same tracking-row-only cleanup as batch_submissions just
            # above: the actual exported XML files in storage are never
            # touched here, only these rows' ability to list/re-download
            # them by batch_id (which clear_batch already accepts losing
            # for the single "latest" submission, for the same reason).
            conn.execute("DELETE FROM batch_exports WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))

    def upsert_batch_submission(
        self, batch_id: str, xml_storage_key: str, xml_storage_backend: str, record_count: int
    ) -> None:
        """Resubmitting a batch REPLACES its one row here (and overwrites
        the one blob/file at that same key) - never creates a second,
        inconsistent export. See submit_batch in app/api/batches.py."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batch_submissions (batch_id, xml_storage_key, xml_storage_backend, "
                "record_count, submitted_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(batch_id) DO UPDATE SET xml_storage_key = excluded.xml_storage_key, "
                "xml_storage_backend = excluded.xml_storage_backend, record_count = excluded.record_count, "
                "submitted_at = excluded.submitted_at",
                (batch_id, xml_storage_key, xml_storage_backend, record_count, _now_iso()),
            )

    def get_batch_submission(self, batch_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_submissions WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_document_submission(
        self, document_id: str, xml_storage_key: str, xml_storage_backend: str, record_count: int
    ) -> None:
        """Same semantics as upsert_batch_submission above: resubmitting a
        document REPLACES its one row here (and overwrites the one
        blob/file at that same key), never creating a second, inconsistent
        export. See submit_document_endpoint in app/api/documents.py."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO document_submissions (document_id, xml_storage_key, xml_storage_backend, "
                "record_count, submitted_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(document_id) DO UPDATE SET xml_storage_key = excluded.xml_storage_key, "
                "xml_storage_backend = excluded.xml_storage_backend, record_count = excluded.record_count, "
                "submitted_at = excluded.submitted_at",
                (document_id, xml_storage_key, xml_storage_backend, record_count, _now_iso()),
            )

    def get_document_submission(self, document_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM document_submissions WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def recompute_batch_status(self, batch_id: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status FROM batch_images WHERE batch_id = ?", (batch_id,)
            ).fetchall()
            statuses = [r["status"] for r in rows]
            if not statuses:
                overall = "queued"
            elif all(s in ("completed", "needs_review", "failed") for s in statuses):
                overall = "failed" if all(s == "failed" for s in statuses) else "completed"
            elif any(s == "processing" for s in statuses):
                overall = "processing"
            else:
                overall = "queued"
            conn.execute(
                "UPDATE batches SET status = ?, updated_at = ? WHERE id = ?",
                (overall, _now_iso(), batch_id),
            )

    # --- batch lifecycle (archive, export history) --------------------------

    def archive_batch(self, batch_id: str) -> None:
        """Hides a batch from the active workspace only - never touches its
        images, records, corrections, or export history (see clear_batch,
        the actually-destructive operation, for contrast)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE batches SET archived = 1, updated_at = ? WHERE id = ?", (_now_iso(), batch_id)
            )

    def unarchive_batch(self, batch_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE batches SET archived = 0, updated_at = ? WHERE id = ?", (_now_iso(), batch_id)
            )

    def touch_batch_activity(self, batch_id: str) -> None:
        """Records that SOME device actively worked this batch just now -
        the signal the IN_PROGRESS/PAUSED distinction is computed from
        (see api/batches.py's review_status logic): recently touched means
        someone has it open right now, not touched in a while means the
        work is real but paused, not abandoned."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE batches SET last_worked_at = ? WHERE id = ?", (_now_iso(), batch_id)
            )

    def confirm_batch_completion(self, batch_id: str) -> str:
        """Records the operator's EXPLICIT "I downloaded the XML, I'm done
        with this batch" confirmation - distinct from, and later than,
        batch_submissions' own submitted_at (export happening at all does
        not by itself mean the operator actually got the file). This is
        the timestamp the 24h retention window (see api/batches.py's
        confirm_batch_completion endpoint and services/batch_cleanup.py)
        counts from. Idempotent: a repeat confirmation keeps the FIRST
        timestamp rather than pushing the retention window back out, so
        clicking the button twice can never silently extend how long a
        completed batch's data sticks around."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT completed_confirmed_at FROM batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if existing and existing["completed_confirmed_at"]:
                return existing["completed_confirmed_at"]
            now = _now_iso()
            conn.execute(
                "UPDATE batches SET completed_confirmed_at = ?, updated_at = ? WHERE id = ?",
                (now, now, batch_id),
            )
        return now

    def next_export_version(self, batch_id: str) -> int:
        """Computed BEFORE the XML is written (see submit_batch): the
        version-specific storage key (f"{batch_id}-v{N}.xml") needs this
        number up front, so it can't be assigned as a side effect of
        inserting the history row after the fact the way batch_id/job_id
        generation works elsewhere in this file."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) as v FROM batch_exports WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        return row["v"] + 1

    def record_batch_export(
        self, batch_id: str, version: int, xml_storage_key: str, xml_storage_backend: str, record_count: int
    ) -> dict:
        """Appends one new, permanent export-history row (never replaced or
        deleted) alongside the existing upsert_batch_submission "latest"
        pointer - called from the same submit_batch request, so every
        export is both instantly downloadable (via batch_submissions) AND
        permanently recorded (via this table), satisfying "never overwrite
        a previous XML export" without changing the existing download
        endpoint's behavior at all."""
        export_id = new_id("EXPORT")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batch_exports (id, batch_id, version, xml_storage_key, xml_storage_backend, "
                "record_count, exported_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (export_id, batch_id, version, xml_storage_key, xml_storage_backend, record_count, now),
            )
        return {"id": export_id, "batch_id": batch_id, "version": version, "record_count": record_count, "exported_at": now}

    def list_batch_exports(self, batch_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM batch_exports WHERE batch_id = ? ORDER BY version DESC", (batch_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- device-aware resume --------------------------------------------------

    def save_device_session(
        self, device_id: str, batch_id: str | None, document_id: str | None, record_index: int | None
    ) -> None:
        """Upserts where this device last was - called (debounced) from the
        frontend on batch/image/record navigation. The backend row is the
        actual source of truth for recovery; localStorage on the client
        only ever needs to remember `device_id` itself (see
        frontend/src/services/device.ts), so a cleared browser cache loses
        nothing recoverable."""
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO device_sessions (device_id, batch_id, document_id, record_index, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(device_id) DO UPDATE SET batch_id = excluded.batch_id, "
                "document_id = excluded.document_id, record_index = excluded.record_index, "
                "updated_at = excluded.updated_at",
                (device_id, batch_id, document_id, record_index, now),
            )
        if batch_id:
            self.touch_batch_activity(batch_id)

    def get_device_session(self, device_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM device_sessions WHERE device_id = ?", (device_id,)
            ).fetchone()
        return dict(row) if row else None

    # --- job queue -----------------------------------------------------------
    # A durable, idempotent, retrying, crash-recoverable local queue. See
    # the module docstring for why this is local SQLite rather than Azure
    # Storage Queue, and how it's meant to be swapped out later.

    def enqueue_job(
        self, job_type: str, payload: dict, idempotency_key: str, max_attempts: int = 3, delay_seconds: float = 0
    ) -> str | None:
        """Returns the new job id, or None if a job with this idempotency_key
        already exists (duplicate-submission protection - e.g. a retried
        upload request can't create a second processing job for the same
        image). `delay_seconds` (0 by default, preserving every existing
        caller's "run as soon as a worker is free" behavior) schedules the
        job's first attempt into the future instead - see
        services/batch_cleanup.py's 24h-retention cleanup job, the first
        caller that actually needs this. claim_next_job's own
        `next_attempt_at <= now` check is what makes the delay real: a
        delayed job simply isn't claimable yet, the same mechanism that
        already spaces out process_image's own retry backoff."""
        job_id = new_id("JOB")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        next_attempt_at = (now_dt + timedelta(seconds=delay_seconds)).isoformat() if delay_seconds else now
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO jobs (id, idempotency_key, job_type, payload, status, attempts, "
                    "max_attempts, next_attempt_at, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)",
                    (job_id, idempotency_key, job_type, json.dumps(payload), max_attempts, next_attempt_at, now, now),
                )
            except sqlite3.IntegrityError:
                return None
        return job_id

    def claim_next_job(self, worker_id: str) -> dict | None:
        """Atomically claims one due job for this worker, including jobs
        whose previous owner's lock has gone stale (crash recovery) -
        nothing is lost if a worker process dies mid-job, it just becomes
        claimable again after _STALE_LOCK_SECONDS."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        stale_before = (now - timedelta(seconds=_STALE_LOCK_SECONDS)).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'pending' AND next_attempt_at <= ? "
                "ORDER BY created_at LIMIT 1",
                (now_iso,),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE status = 'running' AND locked_at <= ? "
                    "ORDER BY created_at LIMIT 1",
                    (stale_before,),
                ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', locked_by = ?, locked_at = ?, updated_at = ?, "
                "attempts = attempts + 1 WHERE id = ?",
                (worker_id, now_iso, now_iso, row["id"]),
            )
            job = dict(row)
            job["payload"] = json.loads(job["payload"])
            # Reflect the UPDATE just issued above - `row` was fetched
            # before it ran, so these fields would otherwise still show
            # the job's pre-claim values (e.g. status 'pending' instead of
            # 'running') even though the database itself is now correct.
            job["status"] = "running"
            job["attempts"] += 1
            job["locked_by"] = worker_id
            job["locked_at"] = now_iso
            job["updated_at"] = now_iso
            return job

    def mark_job_succeeded(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'succeeded', updated_at = ? WHERE id = ?", (_now_iso(), job_id)
            )

    def mark_job_failed(self, job_id: str, error: str, attempts: int, max_attempts: int) -> None:
        now = datetime.now(timezone.utc)
        if attempts >= max_attempts:
            status = "dead_letter"
            next_attempt_at = now.isoformat()
        else:
            status = "pending"
            # Exponential backoff: 2^attempts seconds, capped at 5 minutes -
            # spreads out retries after a transient failure (e.g. an Azure
            # 429) instead of hammering it again immediately.
            backoff = min(300, 2**attempts)
            next_attempt_at = (now + timedelta(seconds=backoff)).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, last_error = ?, next_attempt_at = ?, updated_at = ? WHERE id = ?",
                (status, error[:2000], next_attempt_at, now.isoformat(), job_id),
            )

    def delete_job_by_idempotency_key(self, idempotency_key: str) -> None:
        """Used only to roll back a batch create that failed partway
        through (see create_batch in app/api/batches.py) - a normal job,
        once enqueued, is never deleted, only ever retried to success or
        dead_letter."""
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE idempotency_key = ?", (idempotency_key,))

    def get_job_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return dict(row) if row else None

    def queue_depth(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) as n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}
