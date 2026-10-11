"""Domain-level processing errors.

Caught at the API boundary and translated into safe, generic HTTP error
messages - see app/api/documents.py. Never let a raw exception/stack trace
reach the client.
"""
from __future__ import annotations


class ProcessingError(Exception):
    """Base class for anything that can go wrong turning a file into a record."""


class InvalidImageError(ProcessingError):
    pass


class EmptyOcrResultError(ProcessingError):
    pass
