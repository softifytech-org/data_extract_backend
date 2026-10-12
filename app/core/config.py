"""Central application configuration.

All environment-dependent values (tool paths, limits, thresholds) live here so
nothing else in the codebase hard-codes a path or a secret. Backed by a .env
file at the project root; see .env.example for the full list of variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure Document Intelligence - the production OCR source (see
    # services/ocr/azure_engine.py). Required in any real deployment;
    # get_ocr_engine() raises a clear startup error if these are missing
    # rather than silently falling back to a lower-accuracy local engine.
    documentintelligence_endpoint: str = ""
    documentintelligence_api_key: str = ""

    # External tools. Tesseract is for the retired local-OCR path only
    # (services/ocr/tesseract_engine.py), kept for reference and not used
    # by default - see get_ocr_engine() in core/dependencies.py. Not
    # required in production and safe to leave unset. Poppler is still
    # needed regardless of OCR engine whenever a PDF (not image) is
    # uploaded (services/preprocessing/pdf_processing.py converts pages to
    # images before either engine ever sees them) - a Linux deployment
    # needs the `poppler-utils` system package and this path updated
    # accordingly (e.g. "/usr/bin" on most distros).
    tesseract_cmd: str = r"D:\Tools\Tesseract-OCR\tesseract.exe"
    tessdata_prefix: str = r"D:\Tools\Tesseract-OCR\tessdata"
    poppler_path: str = r"D:\Tools\Poppler\Library\bin"

    # Storage
    data_dir: Path = _PROJECT_ROOT / "data"

    # Upload limits
    max_upload_size_mb: int = 15
    allowed_extensions: str = ".jpg,.jpeg,.png,.pdf"

    # Confidence thresholds (0-100 internal risk score, NOT a probability)
    confidence_high_threshold: float = 95.0
    confidence_medium_threshold: float = 80.0

    # Retention
    processing_artifact_retention_days: int = 7

    # CORS - comma-separated list of exact allowed origins, e.g.
    # "http://localhost:5173,https://my-app.vercel.app". A Vercel preview
    # deployment gets its own unique *.vercel.app subdomain per build, which
    # a fixed list can't anticipate, so any https://*.vercel.app subdomain
    # is additionally allowed via allow_origin_regex in main.py.
    frontend_origin: str = "http://localhost:5173"

    # Optional GPT Vision verification layer (services/ai_vision.py) -
    # explicit per-record trigger only, never auto-run across a document.
    # Disabled by default; the API key is never sent to the frontend.
    ai_extraction_enabled: bool = False
    ai_provider_api_key: str = ""
    ai_provider_model: str = "gpt-4o-mini"

    # PaddleOCR: a genuine second OCR engine, measured to be more accurate
    # than Tesseract on some fields of this document family (see
    # services/ocr/paddle_engine.py) but ~100x slower on this CPU-only
    # machine (~350s/page vs ~1-2s for the full 4-pass Tesseract ensemble).
    # Off by default so that cost is opt-in, not silently imposed.
    paddleocr_enabled: bool = False

    # Per-field review tiers (services/validation/field_reliability.py):
    # routes fields with a measured false-acceptance rate above the agreed
    # safety limit to mandatory review regardless of confidence score, and
    # lets fields proven safe at a lower score (via held-out measurement,
    # never a guess) auto-accept sooner. On by default since both lists are
    # populated only from measured benchmark data; set false to fall back
    # to the single global confidence_high_threshold for every field, e.g.
    # to isolate whether a regression is tier-related during debugging.
    review_tiers_enabled: bool = True

    @property
    def frontend_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",") if ext.strip()}

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processing_dir(self) -> Path:
        return self.data_dir / "processing"

    @property
    def records_dir(self) -> Path:
        return self.data_dir / "records"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def submissions_dir(self) -> Path:
        return self.data_dir / "submissions"

    @property
    def submissions_db_path(self) -> Path:
        return self.data_dir / "submissions.db"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for directory in (
        settings.uploads_dir,
        settings.processing_dir,
        settings.records_dir,
        settings.exports_dir,
        settings.submissions_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
