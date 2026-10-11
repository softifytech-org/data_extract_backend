"""Splits a densely multi-record page into per-record word clusters.

Real documents in this system contain many stacked records on one page,
each anchored by a distinctive record-ID token (e.g. MI296T23). No single
preprocessing variant reliably reads every one of 10-40+ such anchors on a
dense page - each variant tends to miss a different subset. So instead of
trusting one variant's anchor list, every variant's anchors are pooled
and merged by vertical proximity: anchors found within a few pixels of
each other (across different variants, already normalized to the original
image's coordinate space - see services/ocr/pipeline.py) are treated as
the same physical record start. This is multi-pass agreement applied to
*boundary detection*, the same principle used for field values.

Each resulting cluster is handed to the existing single-record field
extractors unchanged - RuleBasedExtractionEngine has no idea whether it's
looking at a 1-record or 40-record document.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.extraction import patterns as pat
from app.services.ocr.base import OCRWord


@dataclass
class RecordBoundary:
    record_id_hint: str  # best-effort ID text read at this boundary; may be wrong/blank
    y_start: float
    y_end: float


def _find_anchor_words(words: list[OCRWord]) -> list[tuple[str, OCRWord]]:
    """Every word that looks like the SHAPE of a record ID, sorted top to
    bottom. Two distinct shapes are recognized (see patterns.py):
    - the alphanumeric "MI296T23" shape (patterns.RECORD_ID_ANCHOR,
      deliberately tolerant of badly-misread leading letters - this only
      needs to locate where each record starts, not read the ID correctly).
    - a plain 5-6 digit number (patterns.RECORD_ID_NUMERIC), used by a
      different document family in this same schema. Bare digits are too
      easy to confuse with a ZIP/phone/other numeric field if matched
      anywhere, so this shape is only accepted when the token is also the
      LEFTMOST word on its line - genuine record numbers lead the line,
      other numeric fields don't.
    """
    by_line: dict[tuple[int, int], list[OCRWord]] = {}
    for w in words:
        by_line.setdefault((w.page, w.line_num), []).append(w)
    leftmost_ids = set()
    for line_words in by_line.values():
        leftmost_ids.add(id(min(line_words, key=lambda w: w.x)))

    found: list[tuple[str, OCRWord]] = []
    for w in words:
        text = w.text.strip()
        m = pat.RECORD_ID_ANCHOR.search(text)
        if m:
            found.append((m.group(0).upper(), w))
            continue
        if id(w) in leftmost_ids and pat.RECORD_ID_NUMERIC.fullmatch(text):
            found.append((text, w))
    found.sort(key=lambda item: item[1].y)
    return found


# Anchors within this many pixels (in original-image coordinates) are
# assumed to be different OCR passes' readings of the same physical
# record-start line, not two separate records. Comfortably smaller than
# the vertical gap between genuinely different records on this document
# family (tens of pixels), comfortably larger than cross-pass jitter on
# the same line (a few pixels).
_Y_MERGE_TOLERANCE = 15


def choose_reference_boundaries(
    words_by_variant: dict[str, list[OCRWord]], boundary_margin: int = 12
) -> list[RecordBoundary]:
    """Pool every variant's detected record anchors, merge ones that are
    vertically close together (same record, different pass), and turn the
    result into one Y-range per record in top-to-bottom document order.

    boundary_margin defaults to 12px, sized for Tesseract's deskew-rotation
    jitter (see below) - verified against real ground-truth data that this
    is too generous for an OCR source with tight, precise row spacing (~19px
    between rows in Azure's output): it was eating over half the gap to the
    NEXT record and pulling that record's first row into this one's slice
    (confirmed directly: record 241011's computed y_end reached to y=150,
    six pixels past where record 241012's own row actually starts at
    y=144), corrupting state/city/name extraction with contamination from
    the wrong record. Callers with precise, low-jitter word coordinates
    should pass a smaller value."""
    all_anchors: list[tuple[str, OCRWord]] = []
    for words in words_by_variant.values():
        all_anchors.extend(_find_anchor_words(words))
    if not all_anchors:
        return []
    all_anchors.sort(key=lambda item: item[1].y)

    clusters: list[list[tuple[str, OCRWord]]] = []
    for anchor in all_anchors:
        if clusters and anchor[1].y - clusters[-1][-1][1].y <= _Y_MERGE_TOLERANCE:
            clusters[-1].append(anchor)
        else:
            clusters.append([anchor])

    starts: list[tuple[str, float]] = []
    for cluster in clusters:
        y = min(w.y for _, w in cluster)
        # Prefer a reading that at least matches one ID shape cleanly
        # (alphanumeric or plain-numeric); otherwise keep whatever the
        # first pass saw.
        clean_hits = [
            text
            for text, w in cluster
            if pat.RECORD_ID.fullmatch(text) or pat.RECORD_ID_NUMERIC.fullmatch(text)
        ]
        hint = clean_hits[0] if clean_hits else cluster[0][0]
        starts.append((hint, y))

    boundaries: list[RecordBoundary] = []
    # Deskew rotation shifts a variant's text vertically by a position-
    # dependent amount (not a uniform offset), so a word right at a shared
    # boundary can land a few pixels on the wrong side of it in some
    # variants even though the reference boundaries themselves are fine.
    # A small overlap margin lets that word still be picked up by its
    # correct record's cluster instead of silently vanishing from that
    # variant's contribution entirely.
    for i, (hint, y) in enumerate(starts):
        y_start = y - 6  # small tolerance so the anchor word itself is included
        y_end = starts[i + 1][1] - 6 + boundary_margin if i + 1 < len(starts) else float("inf")
        boundaries.append(RecordBoundary(hint, y_start, y_end))
    return boundaries


def slice_words_by_boundaries(
    words: list[OCRWord], boundaries: list[RecordBoundary]
) -> list[list[OCRWord]]:
    """Partition one variant's word list into per-record clusters using the
    shared reference boundaries computed above."""
    return [[w for w in words if b.y_start <= w.y < b.y_end] for b in boundaries]
