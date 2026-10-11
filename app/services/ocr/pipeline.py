"""Ties preprocessing + OCR together: one image in, multiple OCR passes out.

Kept separate from both preprocessing and the OCR engine so each stays
independently testable, and so the extraction layer only ever depends on
this one function rather than reaching into preprocessing internals.
"""
from __future__ import annotations

import numpy as np

from app.services.ocr.base import OCREngine, OCRPassResult
from app.services.preprocessing.image_preprocessing import generate_variants, variant_scale

# Running every variant through OCR is the most thorough option and, for a
# single-page document processed synchronously, still fast enough for the
# prototype. If pages start taking too long, trim this list first.
VARIANTS_FOR_OCR = (
    "grayscale",
    "grayscale_threshold",
    "upscaled_denoised",
    "deskewed_threshold",
)


def _normalize_to_original_scale(result: OCRPassResult, scale: float) -> OCRPassResult:
    """Bounding boxes must line up with the ORIGINAL uploaded image, since
    that's what the review UI displays and overlays highlights on - a
    variant that resized the image (e.g. upscale_and_denoise) produces word
    coordinates in its own, larger pixel space otherwise."""
    if scale == 1.0:
        return result
    for word in result.words:
        word.x = round(word.x / scale)
        word.y = round(word.y / scale)
        word.width = round(word.width / scale)
        word.height = round(word.height / scale)
    return result


def run_multi_pass_ocr(
    image: np.ndarray, engine: OCREngine, *, page: int, upscale_factor: float = 2.0
) -> list[OCRPassResult]:
    """upscale_factor controls how aggressively the "upscaled_denoised"
    variant magnifies `image` before OCR. Small record crops need a bigger
    factor than a full page does to reach a legible effective character
    height - see services/processing.py for how this is chosen per call."""
    variants = generate_variants(image, upscale_factor=upscale_factor)
    results: list[OCRPassResult] = []
    for name in VARIANTS_FOR_OCR:
        variant_image = variants[name]
        result = engine.recognize(variant_image, page=page, variant=name)
        result = _normalize_to_original_scale(result, variant_scale(name, upscale_factor))
        results.append(result)
    return results
