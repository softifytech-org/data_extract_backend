"""FastAPI dependency providers. Centralized so the API layer never
constructs services itself - it only asks for them."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.ocr.base import OCREngine

# Tesseract was the original local-OCR engine for this project; retired in
# favor of Azure Document Intelligence (see services/ocr/azure_engine.py
# and its module docstring) once Azure was measured to be both more
# accurate and not dependent on a system binary being installed on the
# deployment host. Left importable (not deleted) for anyone who needs to
# compare against it again, but get_ocr_engine() below no longer
# constructs it by default.
# from app.services.ocr.tesseract_engine import TesseractEngine
from app.models.document import DocumentRecord
from app.services.storage.app_db import AppDatabase
from app.services.storage.blob_store import FileStore, get_file_store_by_backend, get_write_file_store
from app.services.storage.json_storage import JsonDocumentRepository


class AzureCredentialsMissingError(RuntimeError):
    pass


@lru_cache
def get_repository() -> JsonDocumentRepository:
    settings = get_settings()
    return JsonDocumentRepository(settings.records_dir)


@lru_cache
def get_app_db() -> AppDatabase:
    settings = get_settings()
    return AppDatabase(settings.app_db_path)


@lru_cache
def get_ocr_engine() -> OCREngine:
    settings: Settings = get_settings()
    if not settings.documentintelligence_endpoint or not settings.documentintelligence_api_key:
        raise AzureCredentialsMissingError(
            "DOCUMENTINTELLIGENCE_ENDPOINT and DOCUMENTINTELLIGENCE_API_KEY must be set "
            "(as environment variables in production, or in .env locally) - Azure "
            "Document Intelligence is the only active OCR engine."
        )
    from app.services.ocr.azure_engine import AzureDocumentIntelligenceEngine

    return AzureDocumentIntelligenceEngine(
        settings.documentintelligence_endpoint, settings.documentintelligence_api_key
    )


@lru_cache
def get_file_store() -> FileStore:
    """The backend NEW uploads/XML are written to (Azure Blob Storage if
    fully configured, local disk otherwise - see blob_store.py). Used
    directly as a route dependency (Depends(get_file_store)), so test
    overrides work the normal FastAPI way."""
    return get_write_file_store(get_settings())


def get_file_store_for_record(record: DocumentRecord, settings: Settings) -> FileStore:
    """The backend a SPECIFIC document was actually stored with - may
    differ from get_file_store() above if the live configuration changed
    after that document was uploaded (see SourceFileInfo.storage_backend
    and blob_store.py's module docstring).

    Deliberately NOT decorated with @lru_cache and NOT calling
    get_settings() internally: this is called as a plain function from
    inside route bodies (it needs `record`, only available after the
    route already fetched it), so dependency_overrides can never
    intercept it directly the way it intercepts get_file_store() above -
    the caller must pass its OWN already-injected `settings` (from its
    own Depends(get_settings) parameter) so a test override on
    get_settings still takes effect. See get_app_db/get_repository's
    bare-function-call pitfall in conftest.py's client fixture docstring
    for the exact same failure mode this avoids."""
    return get_file_store_by_backend(record.source_file.storage_backend, settings)


@lru_cache
def get_paddle_engine() -> OCREngine | None:
    """Lazily constructed and cached - PaddleOCR's own model load takes
    ~50s, so this must happen once per process, not once per request."""
    settings: Settings = get_settings()
    if not settings.paddleocr_enabled:
        return None
    from app.services.ocr.paddle_engine import PaddleOCREngine

    return PaddleOCREngine()
