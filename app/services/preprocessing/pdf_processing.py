"""PDF -> per-page image conversion, via Poppler (pdf2image).

Every page is converted independently and handed downstream exactly like an
uploaded image would be, so the rest of the pipeline (preprocessing, OCR,
extraction) never needs to know whether the original upload was a PDF.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError


class CorruptPdfError(ValueError):
    pass


def pdf_to_images(pdf_path: Path, poppler_path: str, dpi: int = 300) -> list[np.ndarray]:
    """Convert every page of a PDF to a BGR numpy array (OpenCV format)."""
    try:
        pil_pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise CorruptPdfError("The PDF file is corrupted or unreadable") from exc

    if not pil_pages:
        raise CorruptPdfError("The PDF contains no readable pages")

    images: list[np.ndarray] = []
    for page in pil_pages:
        rgb = np.array(page.convert("RGB"))
        images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return images
