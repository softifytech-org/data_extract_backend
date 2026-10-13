"""Durable file storage abstraction for uploaded images and generated
submission XML.

Two implementations behind one interface:
  - LocalFileStore: the project's original behavior (data/uploads,
    data/submissions on local disk). Used whenever Azure Blob Storage is
    not configured - local dev and the automated test suite.
  - AzureBlobFileStore: uploads/XML live in Azure Blob Storage (two
    private containers), for real durability across restarts/redeploys -
    local disk on a host like Render's free tier is demonstrably wiped on
    restart (confirmed directly earlier in this project), which is what
    this replaces for any installation that configures it.

Every existing DocumentRecord on disk before this was added has no
`storage_backend` field; SourceFileInfo defaults it to "local" exactly so
old records keep resolving through LocalFileStore forever, regardless of
whether Azure Blob Storage is configured for new uploads - see
get_file_store_for_record in core/dependencies.py. This is what makes
switching backends non-destructive to data that already exists.

Blob names are scoped by batch (`<batch_id>/<document_id><ext>`, or
`legacy/<document_id><ext>` for documents uploaded outside the batch
flow) - document_id is already server-generated and globally unique (see
core/security.py), so this isn't needed for collision-avoidance, but it
keeps a batch's files grouped together and means guessing one user's
object name requires already knowing their batch_id too, not just a
single unguessable id.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterator

from app.core.logging_config import get_logger
from app.core.security import _EXTENSION_TO_KIND, UnsafeFilenameError, safe_storage_path

if TYPE_CHECKING:
    from app.core.config import Settings

logger = get_logger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 1.0


class StorageUnavailableError(RuntimeError):
    """Raised when a storage operation fails after all retries - a safe,
    generic error for the API layer to turn into a 502/503, never letting
    a raw Azure SDK exception (which could include connection details)
    reach the client."""


def _with_retries(op_name: str, fn, *, no_retry: tuple[type[Exception], ...] = ()):
    """Retries only transient-looking failures (network blips, timeouts,
    429s). `no_retry` exceptions (e.g. ResourceNotFoundError - a blob that
    genuinely doesn't exist will never start existing just because we
    waited) are re-raised immediately, unwrapped, on the first attempt -
    retrying those would only add latency with zero chance of success."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except no_retry:
            raise
        except Exception as exc:  # noqa: BLE001 - retried uniformly, re-raised safely below
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.warning("%s failed (attempt %d/%d), retrying in %.1fs: %s",
                                op_name, attempt + 1, _RETRY_ATTEMPTS, delay, exc)
                time.sleep(delay)
    logger.error("%s failed after %d attempts: %s", op_name, _RETRY_ATTEMPTS, last_exc)
    raise StorageUnavailableError(f"{op_name} failed after {_RETRY_ATTEMPTS} attempts") from last_exc


def _batch_scope(batch_id: str | None) -> str:
    return batch_id if batch_id else "legacy"


def _upload_blob_name(document_id: str, extension: str, batch_id: str | None) -> str:
    if extension not in _EXTENSION_TO_KIND:
        raise UnsafeFilenameError("Invalid extension")
    return f"{_batch_scope(batch_id)}/{document_id}{extension}"


class FileStore(ABC):
    name: str

    @abstractmethod
    def save_upload(self, data: bytes, *, document_id: str, extension: str, batch_id: str | None) -> str:
        """Persists an uploaded file's bytes. Returns a storage key opaque
        to the caller - store it as-is (SourceFileInfo.stored_filename) and
        pass it back unchanged to the other methods here."""

    @abstractmethod
    def read_upload(self, storage_key: str) -> bytes:
        """Raises FileNotFoundError if the key doesn't resolve to a file."""

    @abstractmethod
    @contextmanager
    def open_upload_as_local_path(self, storage_key: str) -> Iterator[Path]:
        """Yields a real local filesystem Path to the upload's bytes, for
        the OCR pipeline's cv2/pdf2image calls, which need a real path.
        For a local-disk backend this is the existing file, zero copy; for
        a remote backend this downloads to a temp file that is deleted
        when the context exits. Raises FileNotFoundError up front if the
        key doesn't resolve."""

    @abstractmethod
    def upload_exists(self, storage_key: str) -> bool:
        ...

    @abstractmethod
    def delete_upload(self, storage_key: str) -> None:
        """Best-effort cleanup (e.g. rolling back a batch create that
        failed partway through). Never raises if the key is already
        gone."""

    @abstractmethod
    def save_xml(self, data: bytes, *, batch_id: str) -> str:
        """Persists one batch's combined submission XML. Returns a storage
        key. Always overwrites any existing XML for the same batch_id -
        resubmitting a batch replaces its export with the current one,
        never creates a second, inconsistent copy (see
        app/api/batches.py's submit_batch)."""

    @abstractmethod
    def read_xml(self, storage_key: str) -> bytes:
        """Raises FileNotFoundError if the key doesn't resolve to a file."""


class LocalFileStore(FileStore):
    name = "local"

    def __init__(self, settings: "Settings") -> None:
        self._uploads_dir = settings.uploads_dir
        self._submissions_dir = settings.submissions_dir

    def save_upload(self, data: bytes, *, document_id: str, extension: str, batch_id: str | None) -> str:
        path = safe_storage_path(self._uploads_dir, document_id, extension)
        path.write_bytes(data)
        return path.name

    def read_upload(self, storage_key: str) -> bytes:
        path = self._uploads_dir / storage_key
        if not path.is_file():
            raise FileNotFoundError(storage_key)
        return path.read_bytes()

    @contextmanager
    def open_upload_as_local_path(self, storage_key: str) -> Iterator[Path]:
        path = self._uploads_dir / storage_key
        if not path.is_file():
            raise FileNotFoundError(storage_key)
        yield path

    def upload_exists(self, storage_key: str) -> bool:
        return (self._uploads_dir / storage_key).is_file()

    def delete_upload(self, storage_key: str) -> None:
        (self._uploads_dir / storage_key).unlink(missing_ok=True)

    def save_xml(self, data: bytes, *, batch_id: str) -> str:
        self._submissions_dir.mkdir(parents=True, exist_ok=True)
        name = f"{batch_id}.xml"
        (self._submissions_dir / name).write_bytes(data)
        return name

    def read_xml(self, storage_key: str) -> bytes:
        path = self._submissions_dir / storage_key
        if not path.is_file():
            raise FileNotFoundError(storage_key)
        return path.read_bytes()


class AzureBlobFileStore(FileStore):
    name = "azure"

    def __init__(self, settings: "Settings") -> None:
        from azure.storage.blob import BlobServiceClient

        if not settings.azure_storage_configured:
            raise StorageUnavailableError(
                "Azure Blob Storage is not fully configured (AZURE_STORAGE_CONNECTION_STRING, "
                "AZURE_STORAGE_ACCOUNT_NAME, AZURE_UPLOAD_CONTAINER, AZURE_XML_CONTAINER)"
            )
        self._client = BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
        self._upload_container = settings.azure_upload_container
        self._xml_container = settings.azure_xml_container

    def _blob(self, container: str, blob_name: str):
        return self._client.get_blob_client(container=container, blob=blob_name)

    def save_upload(self, data: bytes, *, document_id: str, extension: str, batch_id: str | None) -> str:
        blob_name = _upload_blob_name(document_id, extension, batch_id)
        _with_retries(
            f"Blob upload {blob_name}",
            lambda: self._blob(self._upload_container, blob_name).upload_blob(data, overwrite=True),
        )
        return blob_name

    def read_upload(self, storage_key: str) -> bytes:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return _with_retries(
                f"Blob download {storage_key}",
                lambda: self._blob(self._upload_container, storage_key).download_blob().readall(),
                no_retry=(ResourceNotFoundError,),
            )
        except ResourceNotFoundError:
            raise FileNotFoundError(storage_key) from None

    @contextmanager
    def open_upload_as_local_path(self, storage_key: str) -> Iterator[Path]:
        data = self.read_upload(storage_key)  # raises FileNotFoundError up front if missing
        suffix = Path(storage_key).suffix
        tmp = NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.close()
            yield Path(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def upload_exists(self, storage_key: str) -> bool:
        return self._blob(self._upload_container, storage_key).exists()

    def delete_upload(self, storage_key: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self._blob(self._upload_container, storage_key).delete_blob()
        except ResourceNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - best-effort cleanup only, never raises
            logger.warning("Could not delete orphaned blob %s during rollback", storage_key)

    def save_xml(self, data: bytes, *, batch_id: str) -> str:
        blob_name = f"{batch_id}.xml"
        _with_retries(
            f"Blob XML upload {blob_name}",
            lambda: self._blob(self._xml_container, blob_name).upload_blob(data, overwrite=True),
        )
        return blob_name

    def read_xml(self, storage_key: str) -> bytes:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return _with_retries(
                f"Blob XML download {storage_key}",
                lambda: self._blob(self._xml_container, storage_key).download_blob().readall(),
                no_retry=(ResourceNotFoundError,),
            )
        except ResourceNotFoundError:
            raise FileNotFoundError(storage_key) from None


def get_write_file_store(settings: "Settings") -> FileStore:
    """The backend NEW uploads/XML go to - Azure when fully configured,
    local disk otherwise. Never a silent partial fallback: Settings itself
    refuses to start if Azure storage is only partially configured (see
    get_settings in core/config.py)."""
    return AzureBlobFileStore(settings) if settings.azure_storage_configured else LocalFileStore(settings)


def get_file_store_by_backend(backend: str, settings: "Settings") -> FileStore:
    """The backend a SPECIFIC, already-stored record was written with
    (SourceFileInfo.storage_backend) - always resolves old data correctly
    even after the live write backend above has since changed."""
    if backend == "azure":
        return AzureBlobFileStore(settings)
    return LocalFileStore(settings)
