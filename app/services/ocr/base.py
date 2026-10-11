"""OCR abstraction layer.

Everything downstream (extraction, validation) works against OCRWord /
OCRPassResult, never against a raw Tesseract call. That makes it possible to
swap Tesseract for a different engine later (see TesseractEngine below for
the only class that would need replacing) without touching extraction logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

import numpy as np


class OCRWord(BaseModel):
    """One recognized word with its position and confidence.

    This is the normalized internal representation referenced throughout the
    pipeline - bounding boxes are never discarded, since the extraction layer
    relies on them to figure out which text belongs to which field.
    """

    text: str
    confidence: float  # 0-100, Tesseract's native scale ("-1" is normalized to 0)
    x: int
    y: int
    width: int
    height: int
    page: int
    line_num: int
    word_num: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class OCRPassResult(BaseModel):
    """The output of running OCR once, against one preprocessing variant."""

    variant: str
    page: int
    ocr_config: str
    words: list[OCRWord]

    @property
    def full_text(self) -> str:
        lines: dict[int, list[OCRWord]] = {}
        for w in self.words:
            lines.setdefault(w.line_num, []).append(w)
        ordered_lines = []
        for line_num in sorted(lines):
            line_words = sorted(lines[line_num], key=lambda w: w.x)
            ordered_lines.append(" ".join(w.text for w in line_words))
        return "\n".join(ordered_lines)


class OCREngine(ABC):
    """Interface every OCR engine implementation must satisfy."""

    @abstractmethod
    def recognize(
        self, image: np.ndarray, *, page: int, variant: str, config: str = ""
    ) -> OCRPassResult:
        raise NotImplementedError
