"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title="OCR-Assisted Data Entry API",
    description=(
        "Uploads documents, runs OCR + rule-based extraction + multi-technique "
        "validation, and serves the results to the review UI for human approval."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
