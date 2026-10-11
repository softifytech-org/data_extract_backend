"""Image preprocessing: turns one input image into several candidate variants.

No single preprocessing technique works for every document, so instead of
picking one, we generate a small set of variants and let the OCR layer run
against each of them independently (see services/ocr). Their agreement (or
disagreement) later becomes one of the validation signals for each field.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_binary_otsu(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def upscale_and_denoise(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    gray = to_grayscale(image)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(resized, h=10)
    return denoised


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate rotation angle (degrees) of the dominant text using the
    minimum-area bounding rectangle of foreground pixels."""
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns an angle in [-90, 0); normalize to a small rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Ignore implausible skew estimates on dense text (avoid over-rotating)
    if abs(angle) > 15:
        return 0.0
    return angle


def deskew_and_threshold(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    angle = _estimate_skew_angle(gray)
    if abs(angle) > 0.1:
        (h, w) = gray.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(
            gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    return to_binary_otsu(gray)


def variant_scale(name: str, upscale_factor: float) -> float:
    """Pixel scale of a named variant relative to whatever image was passed
    into generate_variants(). Callers normalize OCR word coordinates back to
    that input image's own coordinate space by dividing by this - see
    services/ocr/pipeline.py. Only the upscaled variant actually resizes;
    everything else stays 1:1. upscale_factor is a parameter (not a fixed
    constant) because record crops use much more aggressive upscaling than
    a full page does - small crops need more magnification to reach a
    legible effective character height."""
    return upscale_factor if name == "upscaled_denoised" else 1.0


def generate_variants(image: np.ndarray, *, upscale_factor: float = 2.0) -> dict[str, np.ndarray]:
    """Return the named preprocessing variants used for multi-pass OCR.

    Keys are stable identifiers referenced elsewhere (OCR pass results carry
    the variant name so validation can report which passes agreed).
    """
    gray = to_grayscale(image)
    return {
        "original": image,
        "grayscale": gray,
        "grayscale_threshold": to_binary_otsu(gray),
        "upscaled_denoised": upscale_and_denoise(image, scale=upscale_factor),
        "deskewed_threshold": deskew_and_threshold(image),
    }
