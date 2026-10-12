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
from app.services.storage.json_storage import JsonDocumentRepository


class AzureCredentialsMissingError(RuntimeError):
    pass


@lru_cache
def get_repository() -> JsonDocumentRepository:
    settings = get_settings()
    return JsonDocumentRepository(settings.records_dir)


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
def get_paddle_engine() -> OCREngine | None:
    """Lazily constructed and cached - PaddleOCR's own model load takes
    ~50s, so this must happen once per process, not once per request."""
    settings: Settings = get_settings()
    if not settings.paddleocr_enabled:
        return None
    from app.services.ocr.paddle_engine import PaddleOCREngine

    return PaddleOCREngine()
