"""FastAPI dependency providers. Centralized so the API layer never
constructs services itself - it only asks for them."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.ocr.base import OCREngine
from app.services.ocr.tesseract_engine import TesseractEngine
from app.services.storage.json_storage import JsonDocumentRepository


@lru_cache
def get_repository() -> JsonDocumentRepository:
    settings = get_settings()
    return JsonDocumentRepository(settings.records_dir)


@lru_cache
def get_ocr_engine() -> OCREngine:
    settings: Settings = get_settings()
    return TesseractEngine(settings.tesseract_cmd, settings.tessdata_prefix)


@lru_cache
def get_paddle_engine() -> OCREngine | None:
    """Lazily constructed and cached - PaddleOCR's own model load takes
    ~50s, so this must happen once per process, not once per request."""
    settings: Settings = get_settings()
    if not settings.paddleocr_enabled:
        return None
    from app.services.ocr.paddle_engine import PaddleOCREngine

    return PaddleOCREngine()
