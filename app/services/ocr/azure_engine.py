"""Azure Document Intelligence OCR engine - the production OCR source.

This ports the exact logic validated over the project's whole Azure
evaluation effort (see the scratchpad round-by-round confirmations): one
`prebuilt-layout` call per PAGE (not per record crop, and not once per
preprocessing variant like the Tesseract ensemble below) - Azure's own
recognition is accurate enough on a single pass that re-running it per
variant or per crop would only multiply API cost for no measured benefit.
Record boundaries and per-field extraction both work from this one
page-level word list, sliced by services/extraction/record_splitter.py -
see services/azure_processing.py for that whole-page pipeline.

Azure's own line segmentation is NOT used - verified against real ground-
truth data that trusting it breaks same-line-based extraction rules (e.g.
the 5 flag values, city-before-state adjacency) that were tuned against a
coarser y-tolerance line clustering. Line membership is re-derived with
the SAME clustering function the Tesseract adapter uses
(_assign_visual_line_ids), so every extraction rule sees consistent line
grouping regardless of which OCR engine produced the words.
"""
from __future__ import annotations

import cv2
import numpy as np
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from app.services.ocr.base import OCREngine, OCRPassResult, OCRWord
from app.services.ocr.tesseract_engine import _assign_visual_line_ids


class AzureDocumentIntelligenceEngine(OCREngine):
    # Marks this engine as single-pass to callers that branch on it (see
    # services/azure_processing.py) - unlike Tesseract's 4-variant
    # ensemble, there is only ever one real analysis per page.
    single_pass = True

    def __init__(self, endpoint: str, api_key: str) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )

    def recognize_words(self, image: np.ndarray) -> list[OCRWord]:
        """The real entry point used in production - one Azure call on a
        whole page image, returning words in the project's own OCRWord
        shape with page-global, re-clustered line numbers."""
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode image for Azure Document Intelligence")
        poller = self._client.begin_analyze_document(
            "prebuilt-layout", body=buf.tobytes(), content_type="image/jpeg"
        )
        result = poller.result()
        if not result.pages:
            return []
        page = result.pages[0]
        raw_words = page.words or []

        boxes = []
        for w in raw_words:
            xs = w.polygon[0::2]
            ys = w.polygon[1::2]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))

        line_ids = _assign_visual_line_ids([(int(y0), int(y1 - y0)) for _, y0, _, y1 in boxes])

        words: list[OCRWord] = []
        for word_idx, (w, (x0, y0, x1, y1)) in enumerate(zip(raw_words, boxes)):
            words.append(
                OCRWord(
                    text=w.content,
                    confidence=round((w.confidence or 0.0) * 100, 1),
                    x=int(x0),
                    y=int(y0),
                    width=max(1, int(x1 - x0)),
                    height=max(1, int(y1 - y0)),
                    page=1,
                    line_num=line_ids[word_idx],
                    word_num=word_idx,
                )
            )
        return words

    def recognize(
        self, image: np.ndarray, *, page: int, variant: str, config: str = ""
    ) -> OCRPassResult:
        """Satisfies the shared OCREngine interface (e.g. for any code
        still calling run_multi_pass_ocr generically) by wrapping the same
        single real call - NOT used by the production per-page path in
        azure_processing.py, which calls recognize_words() directly to
        avoid the Tesseract-era multi-variant loop re-analyzing the same
        page 4 times."""
        words = self.recognize_words(image)
        for w in words:
            w.page = page
        return OCRPassResult(variant=variant, page=page, ocr_config=config or "azure-prebuilt-layout", words=words)
