"""Logging setup.

Rule: log events and counts, never field values. A log line should read like
"Document DOC-123 processed, 3 fields need review", never "email=foo@bar.com".
Call sites are responsible for honoring this - see docstring notes on each
service function - but centralizing the logger config here keeps format
consistent and makes it easy to redirect output (e.g. to a file) later.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (avoid duplicate handlers under reload)
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
