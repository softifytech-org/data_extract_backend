"""Crops one record's region out of a full page image.

This is the core of the crop-first pipeline: instead of running OCR once
on a dense multi-record page and trying to sort the resulting words into
records afterward, each record is cropped out of the ORIGINAL image and
OCR'd independently, on its own small image. A record gets far more
effective resolution and attention from the OCR engine this way than it
would competing for space with 13 other records in one dense page - the
same principle behind why a passport-photo crop reads better than the
same face found inside a group photo.
"""
from __future__ import annotations

import numpy as np


def crop_record(image: np.ndarray, y_start: float, y_end: float, *, margin: int = 15) -> tuple[np.ndarray, int]:
    """Crops the full width of `image` from y_start to y_end, expanded by
    `margin` pixels on each side (clamped to the image bounds) so a record
    positioned slightly off from the detected boundary isn't clipped.

    Returns (crop, actual_y_offset) - actual_y_offset is how far down the
    ORIGINAL image the crop's row 0 actually starts, which callers need to
    translate the crop's own OCR word coordinates back into full-image
    space (see services/processing.py).
    """
    height = image.shape[0]
    y0 = max(0, int(y_start) - margin)
    y1 = min(height, int(y_end) + margin) if y_end != float("inf") else height
    y1 = max(y1, y0 + 1)  # never produce an empty crop
    return image[y0:y1, :], y0
