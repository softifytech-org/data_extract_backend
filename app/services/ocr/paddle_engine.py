"""PaddleOCR implementation of the OCR engine interface.

Measured against Tesseract on the actual sample document (not assumed):
PaddleOCR read several fields (record ID, email) correctly on the very
first pass where all 4 Tesseract preprocessing passes misread them -
genuinely higher per-field accuracy on this document family in some
cases. The tradeoff is processing time: ~350 seconds per page on this
CPU-only machine, versus ~1-2 seconds for the full 4-pass Tesseract
ensemble combined - roughly two orders of magnitude slower. That's why
this engine is wired in but OFF by default (see core/config.py
`paddleocr_enabled`) - enabling it adds several minutes to processing a
multi-record page, which is a real cost a user should opt into knowingly
rather than have silently imposed.

PaddleOCR recognizes whole LINES of text, not individual words the way
Tesseract does. To fit the same OCRWord (word-level) structure the rest
of the pipeline expects, each recognized line is split on whitespace and
its bounding box divided proportionally across the resulting words by
character count - an approximation, not a re-detection, so per-word
boxes from this engine are less precise than Tesseract's but still
usable for click-to-highlight and for text/regex extraction (which only
cares about the reconstructed text, not exact box precision).
"""
from __future__ import annotations

import os

import numpy as np

from app.services.ocr.base import OCREngine, OCRPassResult, OCRWord

# Must be set before paddle is imported anywhere in the process - a known
# oneDNN/PIR-executor incompatibility on this Windows/CPU build otherwise
# raises "NotImplementedError: ConvertPirAttribute2RuntimeAttribute" on the
# very first inference call. Disabling mkldnn avoids the affected code path.
os.environ.setdefault("FLAGS_use_mkldnn", "0")


def _polygon_to_bbox(poly) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0 = min(xs), min(ys)
    return int(x0), int(y0), int(max(xs) - x0), int(max(ys) - y0)


class PaddleOCREngine(OCREngine):
    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # deferred: heavy import, only pay the cost if enabled

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
        )

    def recognize(
        self, image: np.ndarray, *, page: int, variant: str, config: str = ""
    ) -> OCRPassResult:
        results = self._ocr.predict(image)
        words: list[OCRWord] = []
        line_num = 0
        for res in results:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            boxes = res.get("rec_boxes")
            polys = res.get("rec_polys") or res.get("dt_polys")

            for i, text in enumerate(texts):
                text = (text or "").strip()
                if not text:
                    continue
                if boxes is not None:
                    x0, y0, x1, y1 = (int(v) for v in boxes[i])
                    line_x, line_y, line_w, line_h = x0, y0, x1 - x0, y1 - y0
                elif polys is not None:
                    line_x, line_y, line_w, line_h = _polygon_to_bbox(polys[i])
                else:
                    continue
                score = float(scores[i]) * 100.0 if i < len(scores) else 0.0

                tokens = text.split()
                total_chars = sum(len(t) for t in tokens) or 1
                cursor = 0.0
                for tok in tokens:
                    frac = len(tok) / total_chars
                    tok_w = max(1, int(line_w * frac))
                    tok_x = line_x + int(cursor)
                    words.append(
                        OCRWord(
                            text=tok,
                            confidence=score,
                            x=tok_x,
                            y=line_y,
                            width=tok_w,
                            height=max(1, line_h),
                            page=page,
                            line_num=line_num,
                            word_num=0,
                        )
                    )
                    cursor += line_w * frac
                line_num += 1

        return OCRPassResult(variant=variant, page=page, ocr_config=config, words=words)
