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
    bottom. Three distinct shapes are recognized (see patterns.py):
    - the alphanumeric "MI296T23" shape (patterns.RECORD_ID_ANCHOR,
      deliberately tolerant of badly-misread leading letters - this only
      needs to locate where each record starts, not read the ID correctly).
    - a plain 5-8 digit number (patterns.RECORD_ID_NUMERIC), used by other
      document families in this same schema. Bare digits are too easy to
      confuse with a ZIP/phone/other numeric field if matched anywhere, so
      this shape is only accepted when the token is also the LEFTMOST word
      on its line - genuine record numbers lead the line, other numeric
      fields don't. For exactly 6 digits (the original, most-verified
      shape) that is the only requirement, unchanged from before. For any
      other length (5, 7 or 8 digits - newer document families), a bare
      digit run is shape-identical to a wrapped ZIP/other numeric field
      that happens to land leftmost on its own line, so two additional,
      content-based checks are required: a name-shaped word immediately
      after it on the same line, AND an email address somewhere later on
      that same line - the structural signature of a genuine record start,
      not just its digit count. See patterns.looks_like_name_start/
      patterns.line_has_email for the evidence behind each check (shared
      with extraction/engine.py's _find_numeric_record_ids, which needs
      the identical disambiguation for the same reason).
    - a 2-character confusable prefix glued to 6 digits (patterns.
      RECORD_ID_PREFIXED_NUMERIC, e.g. "S8360099"), used by a third
      document family. Same leftmost-of-line safeguard as the plain-numeric
      shape, for the same reason. Checked BEFORE the plain-numeric shape so
      an 8-digit token that also fits this more specific shape (e.g.
      "58360441") is never suppressed by the plain-numeric path's extra
      requirements.

    A real document uses exactly ONE of these three shapes - never a mix
    (confirmed against 125 real documents: 81 cached eval files plus the
    November dataset's 44-image baseline sample, zero exceptions among
    GENUINE anchors). So once the alphanumeric shape is confidently
    present on a page, the plain-numeric and prefixed-numeric branches
    are skipped entirely for that page - a bare digit run elsewhere is
    then necessarily noise that happens to coincidentally satisfy the
    numeric shape's own leftmost/companion checks, not a second,
    different record family sharing the page. Real, confirmed bug this
    fixes (not hypothetical): a phone-number fragment, "811880", sitting
    in the MIDDLE of an existing "MI296T3952" record's own body text
    (between its DOB and currency fields) became its own phantom extra
    "record" this way - confirmed in 4 of 5 real November-dataset images
    that showed more anchor-shaped tokens found than records actually
    produced.

    "Confidently present" requires 2+ alphanumeric matches, not just
    one - RECORD_ID_ANCHOR itself has rare false positives against
    ordinary text that happens to fit its confusable-character shape
    (confirmed real: "BALDORIOTY", a place name, and "ob90@aol.com", an
    email fragment, both in real cached numeric-family documents). A
    single stray match there isn't enough evidence to suppress that
    file's real numeric anchors entirely - a genuine alphanumeric-family
    page has one such anchor per record, far more than 2, so this
    threshold costs nothing on real alnum pages while no longer wiping
    out an entire numeric-family page over one coincidental false match.
    """
    by_line: dict[tuple[int, int], list[OCRWord]] = {}
    for w in words:
        by_line.setdefault((w.page, w.line_num), []).append(w)
    leftmost_ids = set()
    for line_words in by_line.values():
        leftmost_ids.add(id(min(line_words, key=lambda w: w.x)))

    alnum_match_count = sum(1 for w in words if pat.RECORD_ID_ANCHOR.search(w.text.strip()))
    has_alnum_anchor = alnum_match_count >= 2

    found: list[tuple[str, OCRWord]] = []
    for w in words:
        text = w.text.strip()
        m = pat.RECORD_ID_ANCHOR.search(text)
        # Real, confirmed bug (November dataset, USER 7 69N/1649.jpeg
        # record 29622519): the repeated, synthetic username-style name
        # "milo1123" (printed 3x per this record's own duplicate-name
        # rule) was misread as "miloll23" on its third occurrence - l/1
        # and O/0 are already in RECORD_ID_ANCHOR's own confusable
        # class, and that misread happens to satisfy the anchor shape.
        # Unlike the numeric/prefixed-numeric branches below, this
        # branch had no "leftmost word on its line" requirement at all,
        # even though Record No is schema-guaranteed to lead its own
        # line exactly like those other two shapes - the false match sat
        # mid-line (a name field's 3rd repeat), far from the left
        # margin, and was accepted anyway, splitting one real record
        # into a corrupted pair (the false anchor's own record_no came
        # back blank, and its neighbor absorbed/lost fields across the
        # bogus boundary). Same safeguard, same reason, now applied here
        # too - a genuine anchor is never the exception to this rule.
        if m and id(w) in leftmost_ids:
            found.append((m.group(0).upper(), w))
            continue
        if has_alnum_anchor:
            continue
        if id(w) in leftmost_ids and pat.RECORD_ID_PREFIXED_NUMERIC.fullmatch(text):
            found.append((text.upper(), w))
            continue
        if id(w) in leftmost_ids and pat.RECORD_ID_NUMERIC.fullmatch(text):
            if len(text) == 6:
                found.append((text, w))
                continue
            # Y-proximity, not the OCR engine's own line_num grouping - see
            # _ROW_Y_TOLERANCE above for why.
            row_words = sorted(
                (o for o in words if o.page == w.page and abs(o.y - w.y) <= _ROW_Y_TOLERANCE),
                key=lambda x: x.x,
            )
            after = [lw for lw in row_words if lw.x > w.x]
            if after and pat.looks_like_name_start(after[0].text) and pat.line_has_email(row_words):
                found.append((text, w))
            continue
    found.sort(key=lambda item: item[1].y)
    return found


# Anchors within this many pixels (in original-image coordinates) are
# assumed to be different OCR passes' readings of the same physical
# record-start line, not two separate records. Comfortably smaller than
# the vertical gap between genuinely different records on this document
# family (tens of pixels), comfortably larger than cross-pass jitter on
# the same line (a few pixels).
_Y_MERGE_TOLERANCE = 15

# How close two words' Y-coordinates must be (in original-image pixels)
# to be treated as sitting on the same PRINTED row, for the name+email
# companion check below - deliberately NOT the same thing as the OCR
# engine's own line_num grouping. Real-world evidence (reviewer-reported:
# a whole record silently missing from the output, its text instead
# absorbed into the PRECEDING record) traced to this: on some scans the
# OCR engine assigns two different line_num values to words that are
# visually on the same printed row (e.g. a wide row where the left half -
# the ID and name - gets one line_num and the right half - the email -
# gets another, a few pixels lower). The old by-line_num-only lookup then
# never saw the email at all, so the companion check failed and the
# anchor was silently dropped, same as if the record didn't exist. Set to
# under half of Azure's own ~19px row spacing (see choose_reference_
# boundaries' boundary_margin comment) - wide enough to recombine a
# fragmented row, narrow enough that it can never reach into a genuinely
# different, NEXT printed row and borrow that row's name/email instead.
_ROW_Y_TOLERANCE = 10


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
            if pat.RECORD_ID.fullmatch(text)
            or pat.RECORD_ID_NUMERIC.fullmatch(text)
            or pat.RECORD_ID_PREFIXED_NUMERIC.fullmatch(text)
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
