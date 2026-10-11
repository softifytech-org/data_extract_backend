"""Tesseract implementation of the OCR engine interface."""
from __future__ import annotations

import os

import numpy as np
import pytesseract
from pytesseract import Output

from app.services.ocr.base import OCREngine, OCRPassResult, OCRWord

DEFAULT_CONFIG = "--oem 3 --psm 6"


def _assign_visual_line_ids(rows: list[tuple[int, int]]) -> list[int]:
    """Compute a page-global visual line id per word from (top, height) pairs.

    Tesseract's own `line_num` is only unique *within* a block/paragraph, not
    across the page - on dense, unusually-shaped text it can (and does, in
    testing) group visually-separate lines under colliding line_num values,
    scrambling reading order if trusted directly. Clustering by actual y
    position is what the rest of the pipeline relies on for correct
    left-to-right/top-to-bottom reconstruction.
    """
    if not rows:
        return []
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    line_ids = [0] * len(rows)
    current_id = 0
    ref_top, ref_height = rows[order[0]]
    line_ids[order[0]] = current_id
    for idx in order[1:]:
        top, height = rows[idx]
        tolerance = max(ref_height, height) * 0.6
        if top - ref_top > tolerance:
            current_id += 1
            ref_top, ref_height = top, height
        else:
            ref_height = max(ref_height, height)
        line_ids[idx] = current_id
    return line_ids


class TesseractEngine(OCREngine):
    def __init__(self, tesseract_cmd: str, tessdata_prefix: str | None = None) -> None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        if tessdata_prefix:
            # Set via environment rather than the --tessdata-dir CLI flag:
            # pytesseract splits its config string on whitespace without
            # shell-aware quoting, so a quoted path is not safe to pass there.
            os.environ["TESSDATA_PREFIX"] = tessdata_prefix

    def recognize(
        self, image: np.ndarray, *, page: int, variant: str, config: str = ""
    ) -> OCRPassResult:
        config = config or DEFAULT_CONFIG
        raw = pytesseract.image_to_data(image, output_type=Output.DICT, config=config, lang="eng")

        n = len(raw["text"])
        kept_indices = [i for i in range(n) if raw["text"][i].strip()]
        visual_lines = _assign_visual_line_ids(
            [(int(raw["top"][i]), int(raw["height"][i])) for i in kept_indices]
        )

        words: list[OCRWord] = []
        for pos, i in enumerate(kept_indices):
            text = raw["text"][i].strip()
            try:
                confidence = float(raw["conf"][i])
            except (ValueError, TypeError):
                confidence = 0.0
            confidence = max(confidence, 0.0)  # Tesseract uses -1 for "no confidence"
            words.append(
                OCRWord(
                    text=text,
                    confidence=confidence,
                    x=int(raw["left"][i]),
                    y=int(raw["top"][i]),
                    width=int(raw["width"][i]),
                    height=int(raw["height"][i]),
                    page=page,
                    line_num=visual_lines[pos],
                    word_num=int(raw["word_num"][i]),
                )
            )
        return OCRPassResult(variant=variant, page=page, ocr_config=config, words=words)
