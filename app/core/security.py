"""File and path security helpers.

Nothing downstream should ever trust a client-supplied filename, extension,
or Content-Type header directly. Everything here exists to make sure of that:
- document IDs are generated server-side, never client-supplied
- on-disk filenames are derived from the server-side ID, never from the
  original filename, which eliminates path traversal by construction
- file type is verified from the actual byte content (magic numbers), not
  the extension or the browser-supplied MIME type
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

# Magic-byte signatures for the file types this system accepts.
# Checked against the real file content - never trust extension/MIME alone.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "pdf": (b"%PDF-",),
}

_EXTENSION_TO_KIND = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".pdf": "pdf",
}


class UnsafeFilenameError(ValueError):
    pass


class UnsupportedFileTypeError(ValueError):
    pass


def generate_document_id() -> str:
    """Server-side, unguessable document identifier. Never derived from client input."""
    return f"DOC-{uuid.uuid4().hex[:12].upper()}"


# Only characters that are either control codes or genuinely illegal in a
# Windows/Unix filename (a real uploaded file could never have had one of
# these in its name to begin with, so replacing them loses no real
# information). Everything else - parentheses, brackets, "&", "'", "+",
# "#", "@", "%", "~", etc. - is meaningful, naturally-occurring filename
# content (e.g. a duplicate-upload suffix like "scan (1).jpg") and must be
# preserved exactly, since this value becomes <ImageName> in the legacy
# XML dump and is the handoff key the team-lead app matches against.
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


def sanitize_original_filename(filename: str) -> str:
    """Produce a display-safe copy of the client filename for audit/logging only.

    This value is NEVER used to build a filesystem path - see safe_storage_path().
    """
    name = Path(filename).name  # strips any directory components / traversal segments
    name = _FORBIDDEN_FILENAME_CHARS.sub("_", name)
    return name[:255] if name else "unnamed"


def detect_file_kind(content: bytes) -> str | None:
    """Identify the real file type from its content bytes. Returns None if unrecognized."""
    for kind, signatures in _SIGNATURES.items():
        if any(content.startswith(sig) for sig in signatures):
            return kind
    return None


def validate_upload(filename: str, content: bytes, allowed_extensions: set[str]) -> str:
    """Validate an uploaded file end-to-end. Returns the verified file extension.

    Raises UnsafeFilenameError / UnsupportedFileTypeError on any mismatch between
    what the client claims (extension) and what the content actually is.
    """
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise UnsupportedFileTypeError(f"Extension '{ext}' is not permitted")

    kind = detect_file_kind(content)
    if kind is None:
        raise UnsupportedFileTypeError("File content does not match any supported format")

    expected_kind = _EXTENSION_TO_KIND.get(ext)
    if kind != expected_kind:
        raise UnsupportedFileTypeError(
            f"File content ({kind}) does not match its extension ({ext})"
        )

    return ext


def safe_storage_path(base_dir: Path, document_id: str, extension: str) -> Path:
    """Build the on-disk path for a document from its server-generated ID only.

    Because document_id is always server-generated (see generate_document_id) and
    extension is validated against an allow-list, this path can never escape base_dir.
    """
    if not re.fullmatch(r"DOC-[0-9A-F]{12}", document_id):
        raise UnsafeFilenameError("Invalid document id")
    if extension not in _EXTENSION_TO_KIND:
        raise UnsafeFilenameError("Invalid extension")
    base_dir = base_dir.resolve()
    candidate = (base_dir / f"{document_id}{extension}").resolve()
    if base_dir not in candidate.parents and candidate.parent != base_dir:
        raise UnsafeFilenameError("Resolved path escapes storage directory")
    return candidate
