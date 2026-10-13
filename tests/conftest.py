"""Shared pytest fixtures.

Every fixture here is isolated to a pytest tmp_path - nothing touches the
real data/ directory, real Azure, or the real app.db. The fake OCR engine
below exists so the batch/queue/API tests can run fast, offline, and
deterministically without spending real Azure quota on every test run;
actual OCR accuracy is covered separately by the dev/held-out ground-truth
comparisons documented in the project history, not by this suite.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.batches import ensure_public_user
from app.core.config import Settings
from app.core.dependencies import get_app_db, get_file_store, get_ocr_engine, get_paddle_engine, get_repository
from app.main import app
from app.services.ocr.base import OCREngine, OCRPassResult, OCRWord
from app.services.storage.app_db import AppDatabase
from app.services.storage.blob_store import LocalFileStore
from app.services.storage.json_storage import JsonDocumentRepository


class FakeOCREngine(OCREngine):
    """Returns one trivial, always-the-same record's worth of words for any
    image - enough to exercise the full upload -> queue -> process ->
    review pipeline end to end without a real Azure call."""

    single_pass = True

    def recognize_words(self, image: np.ndarray) -> list[OCRWord]:
        words = [
            ("241001", 47, 50, 40, 13),
            ("Test", 96, 50, 30, 13),
            ("Customer", 130, 50, 50, 13),
            ("US", 300, 50, 16, 11),
            ("MALE", 400, 50, 30, 11),
        ]
        return [
            OCRWord(text=t, confidence=95.0, x=x, y=y, width=w, height=h, page=1, line_num=0, word_num=i)
            for i, (t, x, y, w, h) in enumerate(words)
        ]

    def recognize(self, image: np.ndarray, *, page: int, variant: str, config: str = "") -> OCRPassResult:
        words = self.recognize_words(image)
        for w in words:
            w.page = page
        return OCRPassResult(variant=variant, page=page, ocr_config=config or "fake", words=words)


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        documentintelligence_endpoint="https://fake.example.com",
        documentintelligence_api_key="fake-key",
        _env_file=None,  # don't read the real project .env during tests
    )
    # get_settings() normally creates these as a side effect (see
    # app/core/config.py) - constructing Settings directly here, to point
    # it at an isolated tmp_path instead of the real data dir, bypasses
    # that, so it's redone explicitly.
    for directory in (
        settings.uploads_dir, settings.processing_dir, settings.records_dir,
        settings.exports_dir, settings.submissions_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def app_db(test_settings) -> AppDatabase:
    db = AppDatabase(test_settings.app_db_path)
    ensure_public_user(db)  # the fixed FK target create_batch's owner_user_id needs - see batches.py
    return db


@pytest.fixture
def repo(test_settings) -> JsonDocumentRepository:
    return JsonDocumentRepository(test_settings.records_dir)


@pytest.fixture
def client(test_settings, app_db, repo):
    """Deliberately NOT used as a `with TestClient(app) as c:` context
    manager - that form triggers app/main.py's lifespan, which starts the
    real background worker against the REAL app_db/repo/OCR engine (it
    calls those dependency providers directly, not through FastAPI's
    Depends() injection, so dependency_overrides below can't reach it).
    Plain instantiation exercises routes without starting that worker, so
    these tests check HTTP-level validation/auth/ownership/enqueue
    behavior; worker/queue processing itself is tested directly in
    test_queue_worker.py against isolated fixtures, not through here."""
    from app.core.config import get_settings

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_app_db] = lambda: app_db
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_ocr_engine] = lambda: FakeOCREngine()
    app.dependency_overrides[get_paddle_engine] = lambda: None
    # Without this override, get_file_store() would call the REAL
    # get_settings() internally (a bare function call, not resolved
    # through Depends(), so the override two lines up can't reach it -
    # the exact same pitfall documented on the `client` fixture itself)
    # and, now that real Azure Blob Storage credentials are configured in
    # the project's own .env, tests would genuinely upload to production
    # Blob Storage. Overriding get_file_store directly keeps every test
    # on local disk under tmp_path, regardless of what's live-configured.
    app.dependency_overrides[get_file_store] = lambda: LocalFileStore(test_settings)
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()
