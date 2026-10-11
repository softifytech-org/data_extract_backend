"""JSON-file-backed document storage.

This is the prototype persistence layer. It exposes the same narrow
interface (get / save / list / delete) a future PostgreSQL-backed
repository would expose, so callers (the API layer) never need to change
when that migration happens - only this file gets replaced by one that
runs SQL instead of reading/writing JSON.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from app.core.security import UnsafeFilenameError
from app.models.document import DocumentRecord

_ID_PATTERN_MSG = "Invalid document id"


class DocumentNotFoundError(KeyError):
    pass


class JsonDocumentRepository:
    def __init__(self, records_dir: Path) -> None:
        self._dir = records_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, document_id: str) -> Path:
        # document_id is always server-generated (DOC-<12 hex chars>), but we
        # re-validate here too since this method is the last line of defense
        # against path traversal before touching the filesystem.
        import re

        if not re.fullmatch(r"DOC-[0-9A-F]{12}", document_id):
            raise UnsafeFilenameError(_ID_PATTERN_MSG)
        return self._dir / f"{document_id}.json"

    def save(self, record: DocumentRecord) -> None:
        path = self._path_for(record.document_id)
        payload = record.model_dump(mode="json")
        # Atomic write: write to a temp file in the same directory, then replace.
        # Prevents a half-written JSON file if the process is interrupted mid-write.
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, prefix=".tmp-", suffix=".json")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            # Windows can transiently deny a rename right after a file is
            # created (antivirus/Defender briefly holding a scan handle on
            # it) - retry a few times with a short backoff before giving up,
            # rather than failing the whole save over a momentary lock.
            for attempt in range(5):
                try:
                    Path(tmp_name).replace(path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.2 * (attempt + 1))
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    def get(self, document_id: str) -> DocumentRecord:
        path = self._path_for(document_id)
        if not path.exists():
            raise DocumentNotFoundError(document_id)
        # Same transient-Windows-lock retry as save() (see there) - a read
        # landing in the brief window where Defender/an indexer has the
        # file open shouldn't surface as "document broken" to the user.
        for attempt in range(5):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
        return DocumentRecord.model_validate(data)

    def exists(self, document_id: str) -> bool:
        return self._path_for(document_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("DOC-*.json"))

    def delete(self, document_id: str) -> None:
        path = self._path_for(document_id)
        path.unlink(missing_ok=True)
