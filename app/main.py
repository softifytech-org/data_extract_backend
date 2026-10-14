"""FastAPI application entrypoint."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.batches import ensure_public_user
from app.api.batches import router as batches_router
from app.api.documents import router as documents_router
from app.core.config import get_settings
from app.core.dependencies import get_app_db, get_repository
from app.core.logging_config import configure_logging, get_logger
from app.services.queue.worker import BatchWorker

configure_logging()
logger = get_logger(__name__)

settings = get_settings()
_worker: BatchWorker | None = None

# Harmless build-version marker - not a secret, just the commit this
# running process was built from (see Dockerfile's GIT_SHA build-arg and
# .github/workflows/publish-image.yml, which passes the real commit SHA
# on every build). "unknown" means either a local dev run or an image
# built without that workflow - a real, useful signal either way, not an
# error to hide.
_GIT_SHA = os.environ.get("GIT_SHA", "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker
    # Lazy import so a missing/misconfigured Azure credential doesn't block
    # the whole app from starting - it only matters once a job actually
    # needs to run (see BatchWorker._process_image_job), at which point a
    # failure there is a normal, retried job failure, not a crash.
    from app.core.dependencies import get_ocr_engine

    ensure_public_user(get_app_db())
    _worker = BatchWorker(
        app_db=get_app_db(), repo=get_repository(), settings=settings, ocr_engine_factory=get_ocr_engine,
    )
    _worker.start()
    logger.info("Startup complete - batch worker running")
    yield
    if _worker is not None:
        await _worker.stop()
    logger.info("Shutdown complete")


app = FastAPI(
    title="OCR-Assisted Data Entry API",
    description=(
        "Uploads documents, runs OCR + rule-based extraction + multi-technique "
        "validation, and serves the results to the review UI for human approval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins_list,
    # Vercel gives every preview build its own unique subdomain, and Azure
    # Static Web Apps does the same for its own default hostname (both the
    # production slot and every per-PR "preview environment" get a fresh
    # *.azurestaticapps.net subdomain) - neither can be enumerated in a
    # fixed origin list ahead of time. A real custom domain, once attached
    # to either host, still needs to be added to FRONTEND_ORIGIN explicitly.
    allow_origin_regex=r"https://.*\.(vercel\.app|azurestaticapps\.net)",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last line of defense: never let a raw traceback reach the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred"})


app.include_router(documents_router)
app.include_router(batches_router)


@app.get("/api/health")
def health() -> dict:
    """Liveness: is the process up at all. Never touches a dependency -
    this must stay fast and always-200 if the process is running, so an
    orchestrator restarting on health-check failure can't mistake a slow
    database for a dead process."""
    return {"status": "ok", "build": _GIT_SHA}


@app.get("/api/ready")
def ready() -> JSONResponse:
    """Readiness: can this instance actually serve requests right now -
    checked against its real dependencies, not just "the process started".
    A load balancer/orchestrator should stop routing traffic here (but not
    necessarily restart the process) when this returns non-200."""
    checks: dict[str, bool] = {}
    try:
        db = get_app_db()
        db.queue_depth()
        checks["database"] = True
    except Exception:  # noqa: BLE001
        checks["database"] = False
    checks["azure_configured"] = bool(
        settings.documentintelligence_endpoint and settings.documentintelligence_api_key
    )
    # Informational only - does not gate `healthy`. Local disk is a
    # legitimate, intentional mode for local dev/test (see
    # services/storage/blob_store.py); only a PARTIAL Azure Storage
    # configuration is ever treated as an error, and that is refused at
    # startup (get_settings), not surfaced as a readiness failure here.
    checks["azure_storage_configured"] = settings.azure_storage_configured
    checks["worker_running"] = _worker is not None
    healthy = checks["database"] and checks["azure_configured"] and checks["worker_running"]
    return JSONResponse(status_code=200 if healthy else 503, content={"ready": healthy, "checks": checks})
