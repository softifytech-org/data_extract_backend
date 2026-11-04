"""Structured data extraction: turns one OCR pass into field candidates.

This answers "what value belongs to each field?" - nothing here judges
whether that value is *correct* (that's services/validation). Kept behind
the ExtractionEngine interface so a future AI-assisted or hybrid engine can
be swapped in (see AIExtractionEngine / HybridExtractionEngine below) without
touching OCR or validation.

The legacy form this reproduces has several field categories that repeat
twice (or more) per record - two addresses, two genders, two dates, four
currency amounts, up to four name slots. The source document usually only
contains ONE clearly-structured occurrence of some of these (e.g. one real
proper name, not four) - see _extract_names() for the documented, honest
best-effort assumption used when the source genuinely doesn't distinguish
them. Fields with no reliable anchor at all (allergies, P_Inst, STM Name)
are left unmatched rather than guessed - see the module docstring in
services/validation/field_validation.py for why that's the correct behavior.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models.document import BoundingBox
from app.services.extraction import patterns as pat
from app.services.extraction.fields import OCR_FIELD_ORDER
from app.services.extraction.parser import (
    average_confidence,
    bounding_box_for_words,
    build_indexed_text,
    closest_enum_match,
    normalize_confusable_digits,
    normalize_currency,
    normalize_date_long,
    normalize_date_numeric,
    words_in_span,
)
from app.services.extraction.record_splitter import _ROW_Y_TOLERANCE
from app.services.ocr.base import OCRPassResult, OCRWord

NOISE_TOKENS = {
    "NO", "YES", "N.A", "N.A.", "NA", "US", "USA", "MALE", "FEMALE", "MG",
}

# Loose shape for the punctuation-heavy alphanumeric codes this document
# family uses for policy/STM-style reference numbers (e.g.
# "B.alX_,0lacM-40010", "ZtAO,9vB.891504"). Two sub-shapes confirmed against
# real ground truth: most have a punctuation separator - including a leading
# hyphen (e.g. "FR-DLI09_i14482", "Pi03-Ur_2048"), which the separator class
# was missing even though the trailing continuation class already allowed
# hyphens, so these were silently never matched at all. "#" and "@" are two
# more separator characters this same code family genuinely uses (confirmed
# real, November dump USER 13/1.jpeg: "BaX#_0,1acM-36812", and separately
# "Gyt@09_OPYu_t22911") - without them in the class, the regex can't even
# begin a match at the code's own first character ("#"/"@" aren't letters,
# digits, or in the old separator set), so the ENTIRE code was unmatched
# whenever one of these sat before the first real separator. Policy No is also
# sometimes a clean, punctuation-free mixed-case alphanumeric run (e.g.
# "ZtA09vB8643", "N12aXp1990") that no separator-based pattern can match -
# the second alternative requires upper+lower+digit all present in an 8-14
# char token to recognize that shape without accidentally matching ordinary
# capitalized words (which never mix in digits) or pure numbers/codes (which
# never mix in letters). Verified against real ground-truth data that this
# still wrongly matched real names with a trailing number, a documented real
# shape in this schema (e.g. "Cocaine71", "Pomona Cn4524") - real codes mix
# digits in THROUGHOUT (a digit is followed by another letter somewhere,
# e.g. "...09v..."), while a name+trailing-number only ever has digits at
# the very end with nothing after them. Requiring a digit-then-letter
# transition keeps the real code shapes matching while excluding that name
# shape.
HASH_CODE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]{0,4}[.,_#@-][A-Za-z0-9_.,#-]{4,20}\b"
    r"|\b(?=[A-Za-z0-9]{8,14}\b)(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*\d[A-Za-z])"
    r"[A-Za-z][A-Za-z0-9]{7,13}\b"
)


@dataclass
class FieldCandidate:
    field_name: str
    value: str | None
    raw_ocr_text: str | None
    confidence: float  # OCR-only confidence for this candidate, 0-100
    bounding_box: BoundingBox | None
    matched: bool
    note: str | None = None  # e.g. "assumed same as customer name"


def _empty_candidate(field_name: str) -> FieldCandidate:
    return FieldCandidate(field_name, None, None, 0.0, None, matched=False)


def _smart_join(words: list[OCRWord]) -> str:
    """Join word texts in reading order for a free-text field (address,
    city, shipper/STM name), omitting the space between two adjacent words
    when their pixel gap is small enough that they're almost certainly OCR
    fragments of one token, not two genuinely separate words the source
    ever printed with a real space between them.

    This is a defensive floor, NOT a confirmed fix for a specific bug.
    Observed real cases ("David"/"Lewis" -> "Dav id Lewis", "Drivi" ->
    "Dri vi", "GARLAND" -> "GARLA ND") show this happens, but the exact
    gap for a genuine split was never captured: 12 live Azure calls
    targeting a known-affected image/word all came back clean (Azure's own
    OCR word segmentation appears deterministic per the overwhelming
    majority of calls; whatever produced the stored split records was not
    reproducible on demand). What IS directly measured is a safe floor: a
    captured clean case of two genuinely DIFFERENT adjacent words ("David"
    next to "Lewis") sat 4px apart. A 2px cutoff is comfortably under that
    confirmed-real gap, so this cannot merge two legitimate separate words
    - verified against the full 9-image evaluation set with zero
    regressions - even though it isn't proven to catch every split case
    (a genuine split with a gap of, say, 3px would pass through unfixed).
    """
    if not words:
        return ""
    ordered = sorted(words, key=lambda w: (w.line_num, w.x))
    parts = [ordered[0].text]
    for prev, w in zip(ordered, ordered[1:]):
        # A line change always implies at least the space a wrap in the
        # original layout would have had - two words from different
        # printed lines were never glued with zero gap the way two same-
        # line OCR fragments sometimes are. Without this, a value that
        # legitimately wraps across lines (e.g. Shipper Name's second word
        # starting the next printed line) silently lost its space -
        # confirmed real (November dump, USER 06/1758.jpeg record 412942):
        # "Pnorthripbarrygaw" (line 2) + "Dansullivan" (line 3) joined as
        # "PnorthripbarrygawDansullivan", ground truth "Pnorthripbarrygaw
        # Dansullivan". The x-gap check below is meaningless across lines
        # anyway (prev.right and w.x belong to unrelated horizontal scales).
        if w.line_num != prev.line_num or w.x - prev.right >= 2:
            parts.append(" ")
        parts.append(w.text)
    return "".join(parts)


class ExtractionEngine(ABC):
    @abstractmethod
    def extract(self, ocr_result: OCRPassResult) -> dict[str, FieldCandidate]:
        raise NotImplementedError


class RuleBasedExtractionEngine(ExtractionEngine):
    """Regex + positional-layout extraction. No network calls, fully local.

    This is the default and only engine wired into the prototype. It works
    directly off OCRWord bounding boxes so every candidate can be traced
    back to a region of the source image (used by the review UI to
    highlight the matching area when a field is selected).

    There is deliberately ONE engine here, not two separate "old-layout"
    and "new-layout" classes: the two document families share the same
    43-field schema and the same positional extraction strategy almost
    entirely - only a bounded set of fallback heuristics (added while
    investigating the dense wrapped-paragraph layout) are specific to it.
    `layout` ("old" or "new", selected by the caller - see
    app/services/processing.py, threaded from DocumentRecord.layout) gates
    exactly those new-layout-only fallback tiers, each marked below with
    "NEW-LAYOUT ONLY". With layout="old" (the default), every one of those
    tiers is skipped entirely, so behavior is byte-for-byte identical to
    the code as it stood before that investigation - this is what makes
    the held-out regression benchmark (90.67%/88.64%) a true guarantee for
    old-layout documents, not just an currently-measured coincidence.
    """

    def __init__(self, layout: str = "old") -> None:
        self.layout = layout

    @staticmethod
    def _earliest_record_id_match(text: str, pos: int = 0) -> tuple[int, int, str, str] | None:
        """Finds the record-ID-shaped match starting EARLIEST in `text` from
        `pos` onward, trying the strict and fuzzy patterns and taking
        whichever begins first - NOT "prefer strict wherever it appears".
        Verified bug: a record's own ID can fail strict matching (OCR reads
        "MI296T3l" - a lowercase L glued to the next word - instead of a
        clean "MI296T31"), while a DIFFERENT, LATER record's cleanly-printed
        ID still matches strict. Unconditionally preferring strict then
        anchors record-boundary windowing AND record_no on the wrong
        record's ID instead of this record's own, earlier, fuzzy-shaped one.
        Returns (start, end, digits_only_value, raw_matched_text)."""
        candidates: list[tuple[int, int, str, str]] = []
        strict = pat.RECORD_ID.search(text, pos)
        if strict:
            digits = "".join(ch for ch in strict.group(0) if ch.isdigit())
            candidates.append((strict.start(), strict.end(), digits, strict.group(0)))
        fuzzy = pat.RECORD_ID_FUZZY.search(text, pos)
        if fuzzy:
            normalized = (
                fuzzy.group(1)
                + normalize_confusable_digits(fuzzy.group(2))
                + fuzzy.group(3)
                + normalize_confusable_digits(fuzzy.group(4))
            )
            digits = "".join(ch for ch in normalized if ch.isdigit())
            candidates.append((fuzzy.start(), fuzzy.end(), digits, fuzzy.group(0)))
        if not candidates:
            return None
        return min(candidates, key=lambda c: c[0])

    @staticmethod
    def _find_numeric_record_ids(words: list[OCRWord], spans) -> list[tuple[int, str, str]]:
        """For the plain-numeric Record No document family (see
        patterns.RECORD_ID_NUMERIC) - there's no alphanumeric shape to
        search for in running text, so this instead looks at the LEFTMOST
        word on each OCR line and accepts it only if it's a bare 5-6 digit
        token (same leftmost-word safeguard as record_splitter.py, so a
        ZIP/phone fragment elsewhere on a line is never mistaken for a
        Record No).

        Also recognizes a third family's prefixed-numeric shape (patterns.
        RECORD_ID_PREFIXED_NUMERIC, e.g. "S8360099") under the same
        leftmost-word safeguard. Boundary detection (record_splitter.py)
        already used this shape to split records correctly, but that alone
        left record_no blank - this is the separate field-population path,
        which needs the same shape recognized here too. The stored digits
        value is the trailing 6 digits only (the prefix is a confusable
        OCR artifact, not part of the real Record No - verified against
        DataM (2).xml: RecordNo "360099" matches the trailing digits of the
        anchor actually printed on that record's image, not the full
        8-character token), while the raw text keeps the full matched token
        so window/consumption logic still spans the whole anchor.

        Returns (text_start_pos, digits, raw_text) sorted by document
        position.

        RECORD_ID_NUMERIC covers 5-8 digits, not just 6 (see its own
        comment in patterns.py for why). For exactly 6 digits that's the
        only requirement, unchanged. For any other length, a bare digit
        run is shape-identical to a wrapped ZIP/other numeric field that
        happens to land leftmost on its own line, so the same content-
        based checks record_splitter.py's boundary detection uses
        (patterns.looks_like_name_start / patterns.line_has_email) are
        required here too - this path populates the record_no FIELD
        value, and would otherwise be just as liable to mistake a wrapped
        field for a Record No as boundary detection originally was.
        RECORD_ID_PREFIXED_NUMERIC is checked BEFORE RECORD_ID_NUMERIC so
        an 8-digit token that also fits that more specific shape (e.g.
        "58360441") is never shadowed by the plain-numeric path.

        The name/email companion lookup below uses Y-proximity, not the
        OCR engine's own line_num grouping - see record_splitter.py's
        _ROW_Y_TOLERANCE for why (same real failure mode, same fix,
        needed identically in both places per this function's own
        docstring above).

        Also mirrors record_splitter.py's has_alnum_anchor guard, for the
        identical reason: a real document uses exactly ONE of the three
        shapes, never a mix, so once this record's OWN alphanumeric ID is
        genuinely present in its word slice, a bare digit run elsewhere in
        that same slice is noise, not a competing Record No - without this
        guard it could silently override the correct alphanumeric
        record_no with a false numeric match (the caller prefers whatever
        this function returns over the alphanumeric path).

        Unlike record_splitter.py's page-wide version, this only needs
        (and only uses) a single genuine alphanumeric match - one record's
        own slice normally contains exactly one real ID, not many, so a
        2+ count threshold would almost never fire here. Instead this
        requires the match to be LEFTMOST on its own line, the same
        signal the numeric/prefixed branches below already trust: a real
        record ID is always its line's leading token. Confirmed against
        the same two real false positives that motivated the page-wide
        guard (RECORD_ID_ANCHOR matching "BALDORIOTY", a place name, and
        "ob90@aol.com", an email fragment, in real numeric-family
        documents) - neither is leftmost on its own line, so this guard
        does not fire for them, while a genuine record's own ID always
        is."""
        by_line: dict[int, list[OCRWord]] = {}
        for w in words:
            by_line.setdefault(w.line_num, []).append(w)
        leftmost_ids = {id(min(ws, key=lambda w: w.x)): True for ws in by_line.values()}

        if any(
            id(w) in leftmost_ids and pat.RECORD_ID_ANCHOR.search(w.text.strip()) for w in words
        ):
            return []

        hits: list[tuple[int, str, str]] = []
        for s, e, w in spans:
            text = w.text.strip()
            if id(w) not in leftmost_ids:
                continue
            if pat.RECORD_ID_PREFIXED_NUMERIC.fullmatch(text):
                hits.append((s, text[-6:], text))
            elif pat.RECORD_ID_NUMERIC.fullmatch(text):
                if len(text) == 6:
                    hits.append((s, text, text))
                    continue
                row_words = sorted(
                    (o for o in words if abs(o.y - w.y) <= _ROW_Y_TOLERANCE), key=lambda x: x.x
                )
                after = [lw for lw in row_words if lw.x > w.x]
                if after and pat.looks_like_name_start(after[0].text) and pat.line_has_email(row_words):
                    hits.append((s, text, text))
        hits.sort(key=lambda h: h[0])
        return hits

    # "IA" (Iowa), "NE" (Nebraska) and "IN" (Indiana) are also extremely
    # common real-word endings - verified against real ground-truth data
    # three times now: "IDALIA" (a real city) was split into "IDAL" + "IA",
    # "PHENTERMINE" (a real, documented drug name) into "PHENTERMI" + "NE"
    # every time it appeared, and "Sutherlin" (a real surname in this
    # schema, confirmed as ONE clean Azure token before our own code split
    # it) into "Sutherl" + "IN". Excluded specifically (not a broader
    # guessed list - the other state codes have no measured evidence of the
    # same problem, and the original motivating case for this split,
    # "MissoulaOR", relies on "OR" still being eligible).
    _RISKY_STATE_SUFFIXES = {"IA", "NE", "IN"}

    @staticmethod
    def _split_glued_state_suffixes(words: list[OCRWord]) -> list[OCRWord]:
        result: list[OCRWord] = []
        for w in words:
            if w.text.isalpha() and len(w.text) >= 5:
                suffix = w.text[-2:].upper()
                prefix = w.text[:-2]
                # A genuinely glued state code is conventionally printed in
                # caps even within an otherwise mixed-case token (the
                # motivating case, "MissoulaOR", has its own "OR" already
                # uppercase in the real OCR text) - requiring the RAW
                # suffix to already be uppercase (not just uppercase AFTER
                # normalizing for comparison) rules out the far more common
                # case of an ordinary mixed-case word/surname that merely
                # ENDS in a letter pair matching a state code (verified
                # directly against USER01: dozens of real false positives
                # this way - "Ssubherwal"/AL, "Meade"/DE, "Drivi"/VI,
                # "Oneal"/AL, "Prior"/OR, "David"/ID, "Johnny"/NY - none of
                # which have their last two characters capitalized in the
                # source). An ALL-CAPS word's own suffix is trivially
                # "already uppercase" too, so this doesn't touch the
                # existing IA/NE/IN exclusions (those need the explicit
                # list regardless) - it only narrows the far larger
                # mixed-case false-positive surface uncovered here.
                if (
                    suffix in pat.US_STATE_ABBREVIATIONS
                    and suffix not in RuleBasedExtractionEngine._RISKY_STATE_SUFFIXES
                    and len(prefix) >= 3
                    and prefix[0].isupper()
                    and w.text[-2:].isupper()
                ):
                    split_ratio = len(prefix) / len(w.text)
                    prefix_width = max(1, round(w.width * split_ratio))
                    result.append(w.model_copy(update={"text": prefix, "width": prefix_width}))
                    result.append(
                        w.model_copy(
                            update={
                                "text": w.text[-2:],
                                "x": w.x + prefix_width,
                                "width": max(1, w.width - prefix_width),
                            }
                        )
                    )
                    continue
            result.append(w)
        return result

    # A date is sometimes glued with no space directly onto the FOLLOWING
    # field's own leading digits (e.g. OCR word "01/22/1959170" - a date
    # immediately followed by Height's "170", verified directly against a
    # real record). _extract_dates's match only covers the date portion
    # syntactically, but words_in_span returns the WHOLE containing OCR
    # word when claiming it (the word's span overlaps the match even
    # though the match itself ends earlier) - consuming the trailing
    # digits along with the date and silently removing them from the
    # pool height/weight (and, once that number is gone, tablets) search
    # for. Same split-a-glued-word technique already proven for
    # _split_glued_state_suffixes, scoped narrowly to a complete
    # MM/DD/YYYY date immediately followed by 1-3 extra digits with no
    # separator - a plain date (ending in just the 4-digit year, no more)
    # never matches, so this can't touch the overwhelming majority of
    # ordinary date words.
    # Same glue, a second observed shape: a date immediately followed by a
    # "$"-prefixed dollar amount with no separator (e.g. "01/22/1959$250.00"
    # - verified directly in the same real record). Checked as an
    # alternative to the bare-digits case above, not merged into one
    # pattern, since the trailing part's own shape differs (currency
    # punctuation allowed, digit count unbounded).
    _DATE_PLUS_TRAILING_DIGITS = re.compile(
        r"\A(\d{1,2}/\d{1,2}/\d{4})(\d{1,3}|\$[\d.,]+)\Z"
    )

    @staticmethod
    def _split_glued_trailing_digits_after_date(words: list[OCRWord]) -> list[OCRWord]:
        result: list[OCRWord] = []
        for w in words:
            m = RuleBasedExtractionEngine._DATE_PLUS_TRAILING_DIGITS.match(w.text)
            if m:
                date_part, digit_part = m.group(1), m.group(2)
                split_ratio = len(date_part) / len(w.text)
                date_width = max(1, round(w.width * split_ratio))
                result.append(w.model_copy(update={"text": date_part, "width": date_width}))
                result.append(
                    w.model_copy(
                        update={
                            "text": digit_part,
                            "x": w.x + date_width,
                            "width": max(1, w.width - date_width),
                        }
                    )
                )
                continue
            result.append(w)
        return result

    # A name repeated across Customer Name/Billing Name/Name_P_Holder's 3
    # printed occurrences (see _find_name_run) sometimes carries an extra
    # trailing word that is ALSO printed 2-3 times (one synthetic
    # "username-style" token per occurrence, e.g. "jed563"), but with NO
    # space before whatever field happens to follow it on that particular
    # occurrence's own row (Email Address / Shipper Name / STM Name -
    # whichever the layout prints next). The OCR engine's own word
    # segmentation then reads the trailing word and the following field as
    # ONE glued token (e.g. "jed563moukie08@gmail.com"), so _find_name_run's
    # exact-repeat check never sees a 3rd, trailing word at all - only the
    # clean 2-word prefix forms a run, and the genuinely-repeated trailing
    # word silently corrupts whatever field follows it instead. Confirmed
    # real (November dump, USER 06/1758.jpeg record 412942): extraction read
    # customer_name "Merlin Landon" (GT "Merlin Landon jed563"),
    # email_address "jed563moukie08@gmail.com" (GT "moukie08@gmail.com"),
    # shipper_name "jed563Pnorthripbarrygaw..." (GT "Pnorthripbarrygaw..."),
    # stm_name "jed563Kenklee Skizard" (GT "Kenklee Skizard") - the exact
    # same "jed563" prefix stolen three separate times by three otherwise
    # unrelated fields.
    #
    # The glued prefix's own text can't be read off a single occurrence -
    # there's no delimiter inside "jed563moukie08@gmail.com" marking where
    # it ends - but when the SAME 2-4 word name run recurs with a DIFFERENT
    # glued successor on 2+ of its occurrences, the successors' longest
    # common prefix IS that boundary: "jed563moukie08@gmail.com",
    # "jed563Pnorthripbarrygaw" and "jed563Kenklee" share nothing but
    # "jed563" itself, since what follows it (an email address, a shipper
    # name, an STM name) genuinely differs. Same repetition-as-evidence
    # principle _find_name_run itself already relies on for the 2-word base
    # name, applied one level deeper - not a guess at what the prefix means,
    # just recognizing recurring boilerplate glued onto different content.
    @staticmethod
    def _split_glued_repeated_name_suffix(words: list[OCRWord]) -> list[OCRWord]:
        word_like_re = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")

        def is_word_like(w: OCRWord) -> bool:
            return (
                len(w.text) >= 2
                and w.text.upper() not in NOISE_TOKENS
                and bool(word_like_re.fullmatch(w.text))
            )

        ordered = sorted(words, key=lambda w: (w.line_num, w.x))
        runs: list[list[OCRWord]] = []
        current: list[OCRWord] = []
        prev_line = None
        for w in ordered:
            same_line = prev_line is None or w.line_num == prev_line
            close_enough = (
                not current or w.x - current[-1].right <= max(w.height, current[-1].height) * 1.0
            )
            if is_word_like(w) and same_line and close_enough:
                current.append(w)
            else:
                current = [w] if is_word_like(w) else []
            if 2 <= len(current) <= 4:
                runs.append(list(current))
            prev_line = w.line_num

        counts: dict[tuple[str, ...], int] = {}
        for run in runs:
            key = tuple(x.text.upper() for x in run)
            counts[key] = counts.get(key, 0) + 1
        repeated_keys = {k for k, c in counts.items() if c >= 2}
        if not repeated_keys:
            return words
        # Longest repeated phrase wins if more than one distinct phrase
        # repeats - same tie-break _find_name_run itself uses.
        best_key = max(repeated_keys, key=len)
        occurrences = [run for run in runs if tuple(x.text.upper() for x in run) == best_key]

        # The glued successor is NOT reliably distinguishable from an
        # ordinary next word by its own shape or gap alone - verified
        # directly against record 412942's raw OCR words: "Landon" sits a
        # perfectly normal 7px (same scale as the run's own internal word
        # gaps) from the next word on EVERY one of its 3 occurrences, and
        # the glued word is itself pure alphanumeric (word-like) whenever
        # the field it happened to fuse with has no punctuation of its own
        # (a name/STM Name) - only an email's "@" makes some occurrences
        # visibly non-word-like. So every occurrence's immediate next word
        # on the same line is taken as a successor candidate regardless of
        # shape or gap width; the LCP-length floor below (not gap or shape)
        # is what rejects occurrences where the next word is genuinely
        # unrelated.
        run_word_ids = {id(w) for run in occurrences for w in run}
        successors: list[OCRWord] = []
        for run in occurrences:
            run_end = run[-1]
            same_line_after = [
                w
                for w in ordered
                if w.line_num == run_end.line_num and w.x > run_end.x and id(w) not in run_word_ids
            ]
            if not same_line_after:
                continue
            successors.append(min(same_line_after, key=lambda w: w.x))

        if len(successors) < 2:
            return words

        texts = [s.text for s in successors]
        lcp = texts[0]
        for t in texts[1:]:
            i = 0
            while i < len(lcp) and i < len(t) and lcp[i] == t[i]:
                i += 1
            lcp = lcp[:i]
            if not lcp:
                break
        # A short/coincidental shared prefix isn't strong enough evidence (a
        # single shared leading digit, say), and it must leave a real
        # remainder in EVERY successor - otherwise there's nothing to split.
        if len(lcp) < 4 or any(len(t) <= len(lcp) for t in texts):
            return words

        split_ids = {id(s) for s in successors}
        result: list[OCRWord] = []
        for w in words:
            if id(w) in split_ids:
                split_ratio = len(lcp) / len(w.text)
                prefix_width = max(1, round(w.width * split_ratio))
                result.append(w.model_copy(update={"text": lcp, "width": prefix_width}))
                result.append(
                    w.model_copy(
                        update={
                            "text": w.text[len(lcp):],
                            "x": w.x + prefix_width,
                            "width": max(1, w.width - prefix_width),
                        }
                    )
                )
            else:
                result.append(w)
        return result

    def extract(self, ocr_result: OCRPassResult) -> dict[str, FieldCandidate]:
        words = ocr_result.words
        text, spans = build_indexed_text(words)

        # Restrict extraction to this record's own text window. A crop
        # always keeps a small margin (see services/preprocessing/
        # record_cropping.py), and on a dense multi-record page that margin
        # can catch a sliver of the previous or next record's line - verified
        # directly against the source image: a record's Res Address and
        # Shipper Name were found to be text that actually belongs to the
        # FOLLOWING record, not this one. The per-field record_no_pos guard
        # added for dates only protected dates; this generalizes it to every
        # field by dropping any word before this record's own Record No, or
        # at/after a second Record No match (= the next record starting).
        numeric_ids = self._find_numeric_record_ids(words, spans)
        if numeric_ids:
            # Plain-numeric Record No document family (patterns.
            # RECORD_ID_NUMERIC) - window between this record's own leftmost
            # numeric ID and the next one, same purpose as the alphanumeric
            # path below, just a different ID shape.
            window_start = numeric_ids[0][0]
            window_end = numeric_ids[1][0] if len(numeric_ids) > 1 else len(text)
            windowed = [w for s, e, w in spans if s >= window_start and e <= window_end]
            if windowed:
                words = windowed
        else:
            first_id = self._earliest_record_id_match(text)
            if first_id:
                window_start, first_end, _, _ = first_id
                next_id = self._earliest_record_id_match(text, first_end)
                window_end = next_id[0] if next_id else len(text)
                windowed = [w for s, e, w in spans if s >= window_start and e <= window_end]
                if windowed:
                    words = windowed

        # Tried disabling this entirely (a full ground-truth scan showed
        # EVERY state abbreviation collides with common real name endings,
        # e.g. "OR" with "Metaphor" 127 times) - the dev set showed a big
        # gain (stm_name/city_2/state_2 all up), but a FRESH held-out batch
        # showed the opposite: city_1/state_2/res_address all dropped, and
        # the metric that matters most (records >=40/44) went DOWN
        # (9.6%->9.2%). That means some held-out records DO have a genuine
        # glued city+state token after all - the dev-set gain didn't
        # generalize. Reverted to keep the step enabled, with the three
        # individually-confirmed exclusions (IA, NE, IN) from direct
        # mismatch tracing, per "retain changes only when evidence
        # supports them, measured on held-out data, not the dev set alone."
        words = self._split_glued_state_suffixes(words)
        words = self._split_glued_repeated_name_suffix(words)
        # NEW-LAYOUT ONLY: a date glued with no space to the following
        # field's own digits/dollar amount (e.g. "01/22/1959170") is a
        # dense wrapped-paragraph-layout artifact - no instance of it was
        # ever found or needed on old-layout documents.
        if self.layout == "new":
            words = self._split_glued_trailing_digits_after_date(words)
        text, spans = build_indexed_text(words)

        consumed: set[int] = set()  # ids() of words already claimed by a field

        candidates: dict[str, FieldCandidate] = {}

        def claim(
            field_name: str,
            match_words: list[OCRWord],
            value: str | None,
            raw: str,
            note: str | None = None,
            confidence_override: float | None = None,
        ) -> None:
            for w in match_words:
                consumed.add(id(w))
            candidates[field_name] = FieldCandidate(
                field_name=field_name,
                value=value,
                raw_ocr_text=raw,
                confidence=confidence_override if confidence_override is not None else average_confidence(match_words),
                bounding_box=bounding_box_for_words(match_words),
                matched=value is not None,
                note=note,
            )

        # --- record_no (strict, then fuzzy fallback for O/0/@ confusion) ---
        # The printed anchor is alphanumeric ("MI296T23"), but BN_Workflow.pdf's
        # own rule (and the verified golden reference: "MI296T23" -> "29623")
        # says Record No is the 5-6 DIGIT number, letters stripped. The full
        # alphanumeric shape is still what's matched/consumed here (it's the
        # reliable anchor record-boundary detection and every other field's
        # position is measured against) - only the STORED value is reduced
        # to digits-only.
        record_no_pos: int | None = None
        numeric_ids = self._find_numeric_record_ids(words, spans)
        if numeric_ids:
            start, digits, raw = numeric_ids[0]
            record_no_pos = start
            mw = words_in_span(spans, start, start + len(raw))
            claim("record_no", mw, digits, raw)
        else:
            m = self._earliest_record_id_match(text)
            if m:
                start, end, digits, raw = m
                record_no_pos = start
                mw = words_in_span(spans, start, end)
                claim("record_no", mw, digits, raw)

        # --- email ---
        m = pat.EMAIL.search(text)
        email_end_key: tuple[int, int] | None = None
        if m:
            mw = words_in_span(spans, m.start(), m.end())
            claim("email_address", mw, m.group(0), m.group(0))
            if mw:
                email_end_key = max((w.line_num, w.x) for w in mw)

        # --- dates x2: D_Birth (first occurrence), DOB (second) ---
        # Extracted early (before city/address/etc.) so their words are
        # marked consumed before anything else can grab them - a spelled-out
        # date like "Monday, September 7, 1996" sitting on the same OCR
        # line as a city/state pair would otherwise let "September" get
        # vacuumed into city_1 by _extract_city's same-line heuristic.
        date_positions = self._extract_dates(text, spans, consumed, candidates, record_no_pos)

        def suggestion_note(suggestion: str | None) -> str | None:
            return f"Closest known value: {suggestion}" if suggestion else None

        # --- gender x2 (Sex_1 early, Sex_2 late) ---
        # A looser-cutoff fallback for the case where OCR garbles one
        # occurrence badly (e.g. "FAM?LE" for "FEMALE") was tried and
        # measured directly against the November-dump sample: it recovered
        # that case but cost far more elsewhere, matching unrelated words
        # (a real regression found this way: "MUSCLE" - from the city
        # "Muscle Shoals" - fuzzy-matched "MALE" at a 0.6 cutoff) and,
        # even after tightening to 0.65, still net-regressed Sex_1 badly
        # (280 correct vs 442 before, on the same sample) by pulling a
        # spurious earlier "match" into the Sex_1 slot and bumping the
        # real one down to Sex_2. Reverted - no cutoff tried came close to
        # paying for itself; the plain 0.8-cutoff match below remains the
        # evidence-backed behavior.
        genders = self._match_all_single_token_enum(words, consumed, pat.GENDER_VALUES)
        if len(genders) == 1:
            # Position-aware disambiguation using DOB's own already-
            # extracted position (date extraction runs immediately above)
            # instead of a text-similarity fallback - requested
            # specifically after the fuzzy-cutoff attempt above regressed
            # Sex_1 badly. Schema order places Sex_1 well before DOB and
            # Sex_2 well after it, so a single match's position relative
            # to DOB reliably says which slot it really is, without
            # guessing at what the OTHER slot's value might be - that
            # slot is correctly left blank either way, never filled with
            # a lower-confidence guess the way the reverted approach did.
            only_w, only_value, only_sugg = genders[0]
            dob_candidate = candidates.get("dob")
            if (
                dob_candidate is not None
                and dob_candidate.bounding_box is not None
                and only_w.y > dob_candidate.bounding_box.y
            ):
                claim("sex_2", [only_w], only_value, only_value, note=suggestion_note(only_sugg))
            else:
                claim("sex_1", [only_w], only_value, only_value, note=suggestion_note(only_sugg))
            genders = []
        if genders:
            w, value, sugg = genders[0]
            claim("sex_1", [w], value, value, note=suggestion_note(sugg))
        if len(genders) > 1:
            # Verified against real ground-truth data: a third gender-shaped
            # token sometimes appears between Sex_1 and the real Sex_2
            # (same "extra occurrence in the middle" pattern already
            # confirmed for dates) - taking the LAST match rather than the
            # second keeps this robust to it, and is identical to the
            # second match whenever exactly 2 are found (the common case).
            w, value, sugg = genders[-1]
            claim("sex_2", [w], value, value, note=suggestion_note(sugg))

        # --- country x2 ---
        countries = self._match_all_single_token_enum(words, consumed, pat.COUNTRY_VALUES)
        # Verified against real ground-truth data (17 occurrences across
        # the full dump) and record 241054 (dev set) directly: "US" is a
        # common American highway-naming convention ("US Hwy 19", "US
        # Route 50") that can appear embedded in Res Address, genuinely
        # earlier in reading order than the real Country_1 slot. Its
        # matched VALUE is harmless (it spells "US" either way, so
        # Country_1/Country_2 still read correctly), but claiming that
        # WORD consumes it, breaking Res Address's own contiguous-run
        # walk. Same technique already proven for the state/city "CO
        # SPGS" collision: a genuine Country slot is never immediately
        # followed by a highway-designation word on the same line.
        _HWY_WORDS = {"HWY", "HIGHWAY", "ROUTE", "RTE"}
        if countries:
            by_line: dict[int, list] = {}
            for w in sorted(words, key=lambda w: (w.line_num, w.x)):
                by_line.setdefault(w.line_num, []).append(w)
            filtered_countries = []
            for w, v, s in countries:
                same_line = by_line.get(w.line_num, [])
                earlier = [o for o in same_line if o.x < w.x]
                later = [o for o in same_line if o.x > w.x]
                if later and later[0].text.upper() in _HWY_WORDS and later[0].x - w.right <= max(w.height, later[0].height) * 4:
                    continue
                # A genuine printed Country value in this schema is always
                # ALL CAPS ("US"/"USA" - the same all-caps convention
                # already established for city names elsewhere in this
                # file); a LOWERCASE match sitting between two digit-shaped
                # tokens on both sides is instead a coincidental mid-address
                # token (e.g. a stray "us" inside a street address), not a
                # genuine Country slot - a real address routinely has
                # digits flanking ordinary text on both sides, but Country
                # itself always sits next to text (email/address before,
                # phone/sex after), never digits on both sides. Requiring
                # the lowercase condition too (not digit-flanking alone)
                # matters: digit-flanking alone is common and legitimate
                # for a genuine Country_2 (Zip_2 before, PhNo_2 after) -
                # measured a real regression trying that (country_2
                # 92.6%->79.8% on the November-dump sample, genuine "US"
                # values silently dropped across many records). Confirmed
                # real motivating case (November dump, USER 13/2.jpeg
                # record 94370045): a lowercase "us" between ZIP-shaped
                # "21149" and house number "52" was claimed as Country_1/
                # consumed, which then made Res Address's own contiguous-
                # run walk see a sudden wide gap spanning the now-invisible
                # consumed word and silently cut the address down to just
                # "21149".
                if not w.text.isupper():
                    prev_digit = bool(earlier) and earlier[-1].text.isdigit() and w.x - earlier[-1].right <= max(w.height, earlier[-1].height) * 4
                    next_digit = bool(later) and later[0].text.isdigit() and later[0].x - w.right <= max(w.height, later[0].height) * 4
                    if prev_digit and next_digit:
                        continue
                filtered_countries.append((w, v, s))
            countries = filtered_countries
        if countries:
            w, value, sugg = countries[0]
            claim("country_1", [w], value, value, note=suggestion_note(sugg))
        if len(countries) > 1:
            w, value, sugg = countries[1]
            claim("country_2", [w], value, value, note=suggestion_note(sugg))

        # --- state x2 ---
        states = self._match_all_single_token_enum(words, consumed, pat.US_STATE_ABBREVIATIONS)
        states = self._filter_real_state_matches(states, words, consumed, self.layout)
        state_1_word = states[0][0] if states else None
        state_2_word = states[1][0] if len(states) > 1 else None
        if states:
            w, value, sugg = states[0]
            claim("state_1", [w], value, value, note=suggestion_note(sugg))
        if len(states) > 1:
            w, value, sugg = states[1]
            claim("state_2", [w], value, value, note=suggestion_note(sugg))

        # --- zip x2: nearest 5-digit token to each state ---
        zip1 = self._extract_zip(text, spans, consumed, state_1_word)
        if zip1:
            mw, value, raw = zip1
            claim("zip_1", mw, value, raw)
        zip2 = self._extract_zip(text, spans, consumed, state_2_word)
        if zip2:
            mw, value, raw = zip2
            claim("zip_2", mw, value, raw)

        # --- phone x2: punctuated grammar first, then bare 10-digit ---
        phones = self._extract_phones(text, spans, consumed)
        if len(phones) >= 2:
            mw, value, raw = phones[0]
            claim("phno_1", mw, value, raw)
            mw, value, raw = phones[1]
            claim("phno_2", mw, value, raw)
        elif len(phones) == 1:
            mw, value, raw = phones[0]
            slot = self._assign_lone_phone_slot(mw[0], state_2_word)
            claim(slot, mw, value, raw)

        # --- city x2: word(s) immediately before each matched state token ---
        # Extracted BEFORE Res Address on purpose: Res Address is then
        # recovered by POSITION (everything between Email and City_1), which
        # needs City_1's own position already known.
        city1_start_key: tuple[int, int] | None = None
        if state_1_word is not None:
            city1 = self._extract_city(words, consumed, state_1_word)
            if not city1:
                city1 = self._extract_city_fallback(words, consumed, state_1_word)
            # NEW-LAYOUT ONLY (all 4 fallback tiers below): SAME-LINE
            # relaxed-case tried before any PREV-LINE fallback - a correct
            # city on state's own line, even in an uncommon case (Title
            # Case/lowercase with no recognized place-name suffix), must
            # win over a same-shaped-but-wrong-line ALL-CAPS guess.
            # Verified directly, e.g. "...LN New\nPhiladelphia FL..." -
            # trying prev-line ALL-CAPS before this wrongly grabbed the
            # leftover street-suffix word "LN" instead of same-line
            # "Philadelphia" (which has no recognized suffix, so the
            # Title-Case fallback above also misses it).
            #
            # No longer NEW-LAYOUT ONLY: these 4 tiers only ever run when
            # BOTH strict tiers above already returned None (the `if not
            # city1` gate), so by construction they can never override an
            # already-correct strict match - the old "old layout never
            # changes" boundary was a scope choice from when old-layout
            # never needed this, not a proven-unsafe result. Real,
            # confirmed need found since (November dataset, multiple
            # images, layout="old"): genuine mixed-case city names
            # ("midlothian", "Modesto") and state-wrapped-to-the-previous-
            # line cases were silently left unfound, which (per Res
            # Address's own comment below) also took Res Address down
            # with them. Verified safe against the full 81-file real
            # old-layout eval corpus: zero already-correct city_1/city_2
            # values changed, only previously-None ones filled in.
            if not city1:
                city1 = self._extract_city(words, consumed, state_1_word, require_upper=False)
            if not city1:
                city1 = self._extract_city(words, consumed, state_1_word, line_offset=-1)
            if not city1:
                city1 = self._extract_city(
                    words, consumed, state_1_word, line_offset=-1, require_upper=False
                )
            if city1:
                mw, value = city1
                claim("city_1", mw, value, value)
                if mw:
                    city1_start_key = min((w.line_num, w.x) for w in mw)
        city2_start_key: tuple[int, int] | None = None
        city2_mw: list[OCRWord] = []
        if state_2_word is not None:
            city2 = self._extract_city(words, consumed, state_2_word)
            if not city2:
                city2 = self._extract_city_fallback(words, consumed, state_2_word)
            # No longer NEW-LAYOUT ONLY - see the matching City_1 block above for why.
            if not city2:
                city2 = self._extract_city(words, consumed, state_2_word, require_upper=False)
            if not city2:
                city2 = self._extract_city(words, consumed, state_2_word, line_offset=-1)
            if not city2:
                city2 = self._extract_city(
                    words, consumed, state_2_word, line_offset=-1, require_upper=False
                )
            if city2:
                mw, value = city2
                claim("city_2", mw, value, value)
                city2_mw = mw
                if mw:
                    city2_start_key = min((w.line_num, w.x) for w in mw)

        # --- res address (only one street-address field in this schema) ---
        # Prefer POSITION over shape: Res Address has no reliable shape of
        # its own in this document family (the "number + street suffix"
        # regex missed real addresses like "1134 monroi Circli", where OCR
        # read the suffix word wrong) - verified against real ground-truth
        # data that this was the field with the single worst accuracy (0%).
        # Email and City_1 are both independently reliable anchors, so
        # whatever sits between them in reading order IS Res Address,
        # regardless of what it looks like.
        addr_result = None
        if email_end_key is not None and city1_start_key is not None:
            addr_result = self._extract_gap(words, consumed, email_end_key, city1_start_key)
        if not addr_result:
            addr_result = self._extract_street_address(text, spans, consumed)
        if addr_result:
            mw, value = addr_result[:2]
            raw = addr_result[2] if len(addr_result) > 2 else value
            claim("res_address", mw, value, raw)

        # --- height / weight: two standalone small integers, in order ---
        hw = self._extract_height_weight(words, consumed)
        weight_word: OCRWord | None = None
        if hw:
            (hw_word, hw_value), *rest = hw
            claim("height", [hw_word], hw_value, hw_word.text)
            if rest:
                weight_word, v2 = rest[0]
                claim("weight", [weight_word], v2, weight_word.text)

        # --- blood group: real blood-type shape first, else an explicit
        # N.A immediately after Weight on the same line (its position on
        # this form). Must run - and consume its N.A token - BEFORE the
        # flags block below: Blood Group's own N.A sits earlier in reading
        # order than the 5 flag tokens, so if flags ran first it would
        # wrongly steal this N.A as if it were Alcoholic, corrupting both
        # fields at once (verified: this was exactly why Alcoholic was
        # coming back "N.A" instead of the flags' own real first answer).
        # Deterministic corrections run BEFORE the fuzzy match below, not
        # after: a word like "A+?" already clears the fuzzy cutoff (0.6)
        # on its own, so gating these on "fuzzy match failed" would never
        # actually run for the dominant real case - the fuzzy path would
        # claim it first and store the noisy raw text with just a
        # suggestion note (same as the "0"->"O" case before this fix was
        # reordered).
        blood: tuple[OCRWord, str, str | None] | None = None
        for w in words:
            if id(w) in consumed:
                continue
            # Verified against real ground-truth data: Azure sometimes
            # prints the blood-type letter "O" as digit "0" (a classic OCR
            # confusable) - e.g. raw token "0+" where the real value must
            # be "O+" (no blood type can contain a literal zero; this is a
            # closed 8-value vocabulary, so there's no ambiguity about what
            # "0+"/"0-" actually means).
            corrected = w.text.replace("0", "O")
            if corrected in pat.BLOOD_GROUP_VALUES:
                blood = (w, corrected, None)
                break
            # A trailing OCR noise character glued onto an otherwise clean,
            # unambiguous blood type (e.g. "A+?", "A+7", "AB++") is the
            # dominant invalid-value shape for this field in the November
            # dump (confirmed: ~90% of invalid blood_group values fit this
            # exact pattern). Unlike Medicine's "XANA,X" (explicitly
            # verified against a golden legacy reference to NOT be
            # auto-corrected, since that shape can genuinely mean a
            # different, distinct garbled reading), a blood type has only
            # 8 possible values total with no real ambiguity once a clean
            # prefix match is found - stripping 1-2 trailing characters to
            # recover one is the same deterministic-substitution precedent
            # as the "0"->"O" case just above, not a new kind of guess.
            for strip in (1, 2):
                prefix = w.text[:-strip] if strip < len(w.text) else ""
                if len(prefix) >= 2 and prefix.upper() in pat.BLOOD_GROUP_VALUES:
                    blood = (w, prefix.upper(), None)
                    break
            if blood:
                break
        if not blood:
            # The default fuzzy cutoff (0.6) catches everything else this
            # schema's real OCR noise produces that isn't one of the two
            # deterministic shapes above.
            blood = self._match_single_token_enum(words, consumed, pat.BLOOD_GROUP_VALUES, cutoff=0.6)
        flag_run = self._find_flag_run(words, consumed, self.layout)
        if blood:
            w, value, sugg = blood
            claim("blood_group", [w], value, value, note=suggestion_note(sugg))
        elif flag_run:
            # Reading-order rule: Blood Group is the explicit N.A printed
            # just before the 5-flag run (schema order: Blood Group comes
            # before Alcoholic). Anchoring to the flags run instead of to
            # Weight, whose own detection is unreliable across OCR passes.
            run_ids = {id(w) for w, _ in flag_run}
            run_start = (flag_run[0][0].line_num, flag_run[0][0].x)
            na_before = [
                w
                for w in words
                if id(w) not in consumed
                and id(w) not in run_ids
                and (w.line_num, w.x) < run_start
                and closest_enum_match(w.text, pat.EXPLICIT_NA_VALUES, cutoff=0.8)
            ]
            if na_before:
                w = max(na_before, key=lambda w: (w.line_num, w.x))
                claim("blood_group", [w], "N.A", w.text)

        # PhNo_2 is also immediately before the flags run in schema order
        # (City_2, State_2, Zip_2, Country_2, PhNo_2, Alcoholic, ...) -
        # verified against real ground-truth data that it's very often
        # explicitly "N.A" rather than a real second phone number, which
        # _extract_phones alone has no way to recognize (it only looks for
        # phone-shaped digit patterns). Same reading-order fallback as
        # Blood Group: the last explicit-N.A word before the flags run.
        if len(phones) <= 1 and flag_run:
            run_ids = {id(w) for w, _ in flag_run}
            run_start = (flag_run[0][0].line_num, flag_run[0][0].x)
            na_before_phone = [
                w
                for w in words
                if id(w) not in consumed
                and id(w) not in run_ids
                and (w.line_num, w.x) < run_start
                and closest_enum_match(w.text, pat.EXPLICIT_NA_VALUES, cutoff=0.8)
            ]
            if na_before_phone:
                w = max(na_before_phone, key=lambda w: (w.line_num, w.x))
                claim("phno_2", [w], "N.A", w.text)

        # --- YES/NO/N.A flags x5: Alcoholic, Smoker, Past Surg, Diabetic, Allergies ---
        # Best-effort positional assumption (documented, not guaranteed): the
        # legacy form asks these five questions in this fixed order, and the
        # source record answers them as a run of standalone YES/NO/N.A
        # tokens in the same order. There is no per-flag label in the OCR
        # text to confirm which answer belongs to which question.
        flags = self._extract_flags(words, consumed, self.layout)
        flag_fields = ("alcoholic", "smoker", "past_surg", "diabetic", "allergies")
        # A pass that found the COMPLETE run of all 5 flags together is much
        # stronger evidence than an isolated single match from a noisier
        # pass (which may well be reading a stray, unrelated YES/NO/N.A
        # token) - boosted so a complete-run pass can win cross-pass
        # aggregation against fragments, the same principle already used
        # for repeated name detection.
        flags_conf = None
        if len(flags) == 5:
            avg = sum(average_confidence([w]) for w, _ in flags) / 5
            flags_conf = min(100.0, avg * 1.3)
        for field_name, flag in zip(flag_fields, flags):
            w, value = flag
            claim(
                field_name,
                [w],
                value,
                w.text,
                note="Positional guess: order of YES/NO/N.A tokens, not labeled in source",
                confidence_override=flags_conf,
            )

        # --- card name (payment card type vocabulary) ---
        # Window widened 3->4: CARD_TYPES itself has 4-word entries
        # ("AMERICAN EXPRESS GREEN CARD", "AMERICAN EXPRESS PLATINUM
        # CARD") that a 3-word cap could never reach - confirmed the
        # dominant card_name failure in the November-dump sample
        # (31 of 60 invalid instances): the search (largest window first)
        # was truncating both to "American Express Green"/"... Platinum",
        # one word short, instead of ever trying the full 4-word match.
        card_match = self._match_multiword_enum(words, consumed, pat.CARD_TYPES, max_window=4)
        if card_match:
            mw, value, sugg = card_match
            # Verified against real ground-truth data: the following
            # Medicine value is sometimes glued directly onto Card Name
            # with no space in the source ("American ExpressXANAX" instead
            # of "American Express"). Drug names are a small closed
            # vocabulary (same technique already used for day-names glued
            # onto hash codes) - stripping a trailing match is safe and
            # unambiguous.
            # Verified against records 241030/241036 (dev set): PhNo_2's
            # own explicit "N.A" sometimes glues directly onto the FRONT
            # of Card Name with no real gap ("N.A American Express",
            # "N,.A American Express" with OCR's stray comma) - the
            # 3-word fuzzy window still clears the match cutoff against
            # "AMERICAN EXPRESS" since it's a near-total substring, so the
            # raw joined text (including the glued prefix) gets stored
            # as-is. A loose N.A-shaped prefix (tolerating the OCR comma
            # noise already seen elsewhere, e.g. field_validation.py's own
            # handling of "N,.A") is a small, closed shape - stripping it
            # is as safe and unambiguous as the existing trailing-strip
            # cases below.
            value = re.sub(r"^N[.,]+A\.?,?\s+", "", value, flags=re.IGNORECASE)
            upper = value.upper()
            stripped_drug = None
            for drug in pat.DRUG_NAMES:
                if upper.endswith(drug) and len(value) > len(drug):
                    value = value[: -len(drug)].rstrip()
                    stripped_drug = drug
                    break
            claim("card_name", mw, value, value, note=suggestion_note(sugg))
            # Verified against record 241021 (dev set): the stripped drug
            # name ("XANAX" in "American ExpressXANAX") was being
            # discarded entirely - the whole glued word is consumed by
            # card_name's own claim above, so medicine's own later,
            # separate search (which needs its own standalone word) could
            # never find it. We already know exactly which drug it is
            # here, so claim medicine with it directly instead of losing
            # the information.
            if stripped_drug:
                claim("medicine", mw, stripped_drug, stripped_drug)

        # --- medicine (drug name vocabulary) ---
        drug_match = self._match_single_token_enum(words, consumed, pat.DRUG_NAMES)
        if drug_match:
            w, value, sugg = drug_match
            claim("medicine", [w], value, value, note=suggestion_note(sugg))
            # Card Name has no vocabulary match above when the form prints
            # an explicit "N.A" in that slot instead of a real card type
            # (verified real: USER 13/2.jpeg record 94370048, row
            # "...FEMALE N.A XANAX N.A" - Sex_2, Card Name, Medicine,
            # Dosage in schema order) - same "explicit N.A, no vocabulary
            # shape of its own" situation already handled for Blood Group
            # via its own na_before anchor, applied here with Medicine's
            # own just-found, reliably-anchored position as the right
            # boundary instead (Card Name directly precedes Medicine in
            # schema order). Only claimed when nothing already filled
            # card_name (a real card-type match always wins) and the N.A
            # sits on the SAME line, immediately before Medicine, with a
            # tight gap - the same adjacency requirement used throughout
            # this file to avoid grabbing an unrelated earlier N.A.
            if "card_name" not in candidates:
                same_line = sorted(
                    (o for o in words if id(o) not in consumed and o.line_num == w.line_num and o.x < w.x),
                    key=lambda o: o.x,
                )
                if same_line:
                    na_word = same_line[-1]
                    gap = w.x - na_word.right
                    if gap <= max(w.height, na_word.height) * 4 and closest_enum_match(
                        na_word.text, pat.EXPLICIT_NA_VALUES, cutoff=0.8
                    ):
                        claim("card_name", [na_word], na_word.text, na_word.text)

        # --- dosage (strict, then fuzzy fallback for O/0/@ confusion) ---
        dosage_pos: int | None = None
        m = pat.DOSAGE.search(text)
        if m:
            dosage_pos = m.start()
            mw = words_in_span(spans, m.start(), m.end())
            claim("dosage", mw, m.group(0).upper().replace("  ", " "), m.group(0))
        else:
            m = pat.DOSAGE_FUZZY.search(text)
            if m:
                dosage_pos = m.start()
                mw = words_in_span(spans, m.start(), m.end())
                value = f"{normalize_confusable_digits(m.group(1))} MG"
                claim("dosage", mw, value, m.group(0))

        # --- currency: D_B_Life_Assure (before medicine block) + 4 after ---
        # Prefer anchoring on the LAST date's position (== DOB, see
        # _extract_dates): dates are found consistently across OCR passes,
        # whereas the "MG" dosage unit is frequently dropped entirely by OCR
        # noise (see _extract_currency_fields docstring) and so is a much
        # less stable anchor on its own.
        currency_anchor = date_positions[-1] if len(date_positions) > 1 else dosage_pos
        # Real, confirmed bug (November dataset, 752.jpeg record 29610274):
        # when only 2 distinct dates are found, date_positions[-1] is
        # D_B_Life_Assure's OWN date (not DOB - DOB was genuinely blank on
        # this record), which prints on the SAME line as, and just BEFORE,
        # P_Inst's own dollar value. Anchoring there put P_Inst's "$250.00"
        # into the tail group instead of head, shifting Pill Rate/Cost/
        # Shipping Cost/Total Amount each one slot out of place too (the
        # same cascade already fixed elsewhere for the 5-flag run). Taking
        # the later of the two available signals is strictly safer than
        # either alone: dosage_pos, when actually found, sits right at the
        # Medicine/Dosage block - always structurally after P_Inst - so it
        # can only correct an anchor that's too early, never push a correct
        # one later than the real boundary.
        if dosage_pos is not None and currency_anchor is not None:
            currency_anchor = max(currency_anchor, dosage_pos)
        elif dosage_pos is not None:
            currency_anchor = dosage_pos
        self._extract_currency_fields(text, spans, consumed, candidates, currency_anchor)

        # --- tablets: standalone integer near dosage ---
        qty_result = self._extract_quantity(words, consumed, candidates.get("dosage"))
        if qty_result:
            w, value = qty_result
            claim("tablets", [w], value, w.text)

        # --- policy_no / stm_code: hash-shaped codes, first then second ---
        hashes = self._extract_hash_codes(text, spans, consumed, self.layout)
        # Policy No always prints AFTER the 5-flag run in this schema
        # (Alcoholic..Allergiesd, PloicyNo, ...) - a hash-shaped match
        # positioned BEFORE it can only be a false positive from an
        # unrelated glued OCR word (confirmed real, DataM (4).xml record
        # 29610265: the email "realty4u@msn.comRr1" glues a stray code-
        # shaped suffix onto itself, which HASH_CODE's own shape rules
        # can't tell apart from a real code). Left unfiltered, that
        # earlier false match took the policy_no slot (hashes[0]) and
        # bumped the REAL policy_no into stm_code's slot instead - the
        # same off-by-one cascade already seen elsewhere in this file,
        # just from a different root cause. Verified against 8 of 9
        # traced real cases from the November dump; the reading-order
        # anchor itself (flag_run's own end position) reuses the same
        # precedent already trusted for Blood Group/PhNo_2 just above.
        if flag_run:
            flags_end = (flag_run[-1][0].line_num, flag_run[-1][0].x)
            hashes = [
                h for h in hashes
                if not h[0] or min((w.line_num, w.x) for w in h[0]) > flags_end
            ]
        stm_code_start_key: tuple[int, int] | None = None
        policy_no_start_key: tuple[int, int] | None = None
        # NOTE: tried routing a lone hash candidate to stm_code (instead
        # of the default policy_no) whenever it matched the common
        # "aC"-family shape (01aCm/01aCp/OlaCn etc) that most real
        # STM_Code values share - measured against DataM (4).xml: fixed
        # several genuine lone-stm_code cases (e.g. record 29610264,
        # where Policy No's own value was invisible to HASH_CODE due to
        # a non-ASCII OCR substitution) but caused a larger number of new
        # regressions, because a real, confirmed minority of Policy No
        # values ALSO use this exact shape family (e.g. GT PolicyNo
        # "OlaCo-13462", "XX_OlaCp_21848", "B.alX,_OlaCm-8839" on records
        # 29610340/29610306/29610348) - net effect was flat-to-negative
        # on the full ground-truth comparison. Reverted; no reliable
        # content-only signal was found to distinguish the two fields
        # when only one candidate exists.
        if hashes:
            mw, value = hashes[0]
            claim("policy_no", mw, value, value)
            if mw:
                policy_no_start_key = min((w.line_num, w.x) for w in mw)
        if len(hashes) > 1:
            mw, value = hashes[1]
            claim("stm_code", mw, value, value)
            if mw:
                stm_code_start_key = min((w.line_num, w.x) for w in mw)

        # --- names: see _extract_names for the documented duplication logic ---
        state_2_key = (state_2_word.line_num, state_2_word.x) if state_2_word is not None else None
        self._extract_names(
            words,
            consumed,
            candidates,
            claim,
            state_2_key,
            city2_start_key,
            stm_code_start_key,
            policy_no_start_key,
            city2_mw,
        )

        for field_name in OCR_FIELD_ORDER:
            candidates.setdefault(field_name, _empty_candidate(field_name))
        return candidates

    # -- individual field heuristics -------------------------------------------------

    @staticmethod
    def _match_single_token_enum(
        words: list[OCRWord], consumed: set[int], vocabulary: set[str], cutoff: float = 0.8
    ) -> tuple[OCRWord, str, str | None] | None:
        """Returns (word, raw_text, suggested_correction). raw_text is
        exactly what OCR read - the fuzzy vocabulary match is only ever a
        suggestion, never silently substituted as the value (verified
        against a golden legacy reference: "XANA,X" must stay "XANA,X" and
        be flagged invalid, never silently "corrected" to "XANAX")."""
        for w in words:
            if id(w) in consumed:
                continue
            # No character-class prefilter here on purpose: vocabularies in
            # this schema include meaningful punctuation (blood groups like
            # "A-"/"O+"), so any pre-strip risks rejecting a valid token
            # before it's even compared. closest_enum_match's own cutoff is
            # what decides relevance; this is just a cheap length guard.
            if len(w.text) < 2:
                continue
            match = closest_enum_match(w.text, vocabulary, cutoff=cutoff)
            if match:
                suggestion = match if match.upper() != w.text.upper() else None
                return w, w.text, suggestion
        return None

    @staticmethod
    def _match_all_single_token_enum(
        words: list[OCRWord], consumed: set[int], vocabulary: set[str], cutoff: float = 0.8
    ) -> list[tuple[OCRWord, str, str | None]]:
        """Like _match_single_token_enum but returns every match, in reading
        order. Used for fields that appear twice per record (Sex_1/Sex_2,
        Country_1/Country_2, ...).

        Does NOT consume matches itself - callers only claim(), and
        therefore only consume, the 1-2 entries they actually use (e.g.
        genders[0]/genders[-1]). Verified against record 241054 (dev set):
        this used to consume every match found, including ones never
        claimed by anything - a third "US" embedded in "US Hwy 19" (a
        common American highway-naming convention, not country data at
        all) was silently swallowed even though only the first two "US"
        occurrences became Country_1/Country_2, breaking the res_address
        gap-walk's contiguous run by removing a word from its middle.

        Verified against real ground-truth data that fuzzy matching (cutoff
        0.8) is unsafe for very short vocabularies like the 2-letter US state
        codes: difflib's ratio for "NEW" vs "NE" is exactly 0.8, so the word
        "NEW" (from a city like "NORTH NEW HYDE PARK") was being accepted as
        the state "NE" - one bad OCR-tolerant word fuzzy-matching an entirely
        unrelated short token. For any vocabulary whose longest entry is 3
        characters or less, require an exact (case-insensitive) match on a
        token of the same length instead of falling back to fuzzy matching.
        """
        short_vocab = max(len(v) for v in vocabulary) <= 3
        vocab_lengths = {len(v) for v in vocabulary}
        found: list[tuple[OCRWord, str, str | None]] = []
        # Verified while fixing the over-consumption issue above: this
        # never sorted its input, unlike _match_multiword_enum elsewhere
        # in this file, which explicitly does. Callers rely on "first
        # match" / "second match" meaning first/second BY POSITION
        # (Sex_1 before Sex_2, Country_1 before Country_2, ...) - without
        # this, whatever raw order `words` happens to already be in
        # decides it instead, which is not guaranteed to be reading
        # order.
        for w in sorted(words, key=lambda w: (w.page, w.line_num, w.x)):
            if id(w) in consumed:
                continue
            if len(w.text) < 2:
                continue
            if short_vocab:
                if len(w.text) not in vocab_lengths:
                    continue
                match = w.text.upper() if w.text.upper() in vocabulary else None
            else:
                match = closest_enum_match(w.text, vocabulary, cutoff=cutoff)
            if match:
                suggestion = match if match.upper() != w.text.upper() else None
                found.append((w, w.text, suggestion))
        return found

    @staticmethod
    def _filter_real_state_matches(
        matches: list[tuple[OCRWord, str, str | None]],
        words: list[OCRWord],
        consumed: set[int],
        layout: str = "old",
    ) -> list[tuple[OCRWord, str, str | None]]:
        """A 2-letter state code match can be a false positive when it's
        actually one word of a multi-word city name that happens to coincide
        with a real abbreviation - verified against real ground-truth data:
        "CO SPGS" is literally how this document spells "Colorado Springs",
        and "CO" alone exact-matches Colorado, duplicating State_1 into
        State_2 and leaving City_2 wrong. A genuine state slot is always
        immediately followed by a ZIP code on the same line (or is the last
        word on its line); a state-shaped word immediately followed by
        something else is far more likely sitting inside a city phrase.
        Rejected matches are un-consumed so they remain available to city
        extraction instead of being silently dropped.
        """
        by_line: dict[int, list[OCRWord]] = {}
        for w in words:
            by_line.setdefault(w.line_num, []).append(w)

        def _looks_like_zip(text: str) -> bool:
            # Tolerant of OCR-garbled characters (e.g. "7200?11" for what's
            # almost certainly a real zip with one misread digit) rather
            # than requiring pat.ZIP_CODE's exact clean fullmatch -
            # verified directly (originally new-layout data: a real state
            # match ("VA") was being rejected and un-consumed here,
            # promoting the WRONG state from a different line to State_1,
            # purely because its own real zip had a garbled character and
            # so never fullmatched ZIP_CODE). Confirmed the identical
            # failure in old-layout November-dump data too (e.g. a genuine
            # "CA" rejected over a badly garbled zip, which then let a
            # same-shaped word from the record's SECOND address block get
            # mistaken for State_1, leaving City_2/State_2/Zip_2 all blank)
            # - no longer gated to new layout, since the underlying cause
            # (OCR sometimes garbles a zip) isn't layout-specific.
            if pat.ZIP_CODE.fullmatch(text):
                return True
            if not (3 <= len(text) <= 10):
                return False
            digit_count = sum(ch.isdigit() for ch in text)
            return digit_count >= 3 and digit_count >= len(text) - 1

        kept: list[tuple[OCRWord, str, str | None]] = []
        for w, value, suggestion in matches:
            same_line = sorted(by_line.get(w.line_num, []), key=lambda x: x.x)
            later = [o for o in same_line if o.x > w.x]
            # A real state slot is just as validly followed by an explicit
            # "N.A"/"N/A" placeholder as by an actual zip - verified
            # directly (e.g. "...Linvillet ARKANSAS FL N.A US..."): zip_2
            # itself is blank/not-applicable for this record, and the
            # genuine state match "FL" was being rejected here purely
            # because nothing zip-shaped followed it, when the real
            # explanation is just that there's no real zip to find at all.
            # Not layout-specific - confirmed the same N.A-as-zip pattern
            # in old-layout data too.
            na_ok = bool(later) and later[0].text.upper() in ("N.A", "N/A")
            # "Last word on its line" is only weak evidence on its own - a
            # street-suffix word that collides with a real state code (here
            # "CT"/Court vs Connecticut, the same kind of 2-letter
            # coincidence already confirmed for "CO"/Colorado above) is
            # just as often simply the last word of a WRAPPED address line
            # that ends before reaching the real City/State/Zip block on
            # the next line, not a genuine trailing state. Confirmed real
            # (November dump, USER 13/1.jpeg record 94370035): "...APT
            # L-104 Ct" ends the address line with nothing after it, and
            # the real state ("PA") sits on the FOLLOWING line - without
            # this, "Ct" was kept as State_1, bumping the real State_1 into
            # State_2 and losing the real State_2 entirely.
            is_suffix_word = w.text.upper() in (pat.STREET_SUFFIX_WORDS | pat.STREET_SUFFIX_WORDS_OCR_VARIANTS)
            if (not later and not is_suffix_word) or (
                later
                and (na_ok or _looks_like_zip(later[0].text))
                and later[0].x - w.right <= max(w.height, later[0].height) * 4
            ):
                kept.append((w, value, suggestion))
            else:
                consumed.discard(id(w))
        return kept

    @staticmethod
    def _match_multiword_enum(
        words: list[OCRWord], consumed: set[int], vocabulary: set[str], max_window: int
    ) -> tuple[list[OCRWord], str, str | None] | None:
        available = [w for w in words if id(w) not in consumed]
        available.sort(key=lambda w: (w.page, w.line_num, w.x))
        for window in range(max_window, 0, -1):
            for i in range(len(available) - window + 1):
                chunk = available[i : i + window]
                if len({(w.page, w.line_num) for w in chunk}) > 1:
                    continue
                joined = " ".join(w.text for w in chunk)
                match = closest_enum_match(joined, vocabulary, cutoff=0.82)
                if match:
                    suggestion = match if match.upper() != joined.upper() else None
                    return chunk, joined, suggestion
        return None

    @staticmethod
    def _extract_zip(
        text: str, spans, consumed: set[int], state_word: OCRWord | None
    ) -> tuple[list[OCRWord], str, str] | None:
        candidates_found = []
        for m in pat.ZIP_CODE.finditer(text):
            mw = words_in_span(spans, m.start(), m.end())
            mw = [w for w in mw if id(w) not in consumed]
            if not mw:
                continue
            candidates_found.append((m, mw))
        if not candidates_found:
            return None
        if state_word is not None:
            # Verified against real ground-truth data: now that short
            # (3-4 digit) zips are matchable, a plain nearest-by-Y-distance
            # sort can tie with an unrelated same-row number (e.g. a street
            # address's leading number, "140" in "140 Willow Criik Canyon
            # Rd", sits at the same y as the real zip and is the same
            # length). The real Zip always sits immediately after the State
            # token on its own line with only a small gap (the same
            # adjacency already used to validate a state match itself) -
            # prefer that candidate first, before falling back to pure
            # Y-distance for the rare case where Zip wraps to a different
            # line than its State (as City_2 itself sometimes does).
            same_line_after = [
                (m, mw)
                for m, mw in candidates_found
                if mw[0].line_num == state_word.line_num
                and mw[0].x > state_word.x
                and mw[0].x - state_word.right <= max(mw[0].height, state_word.height) * 4
            ]
            if same_line_after:
                same_line_after.sort(key=lambda item: item[1][0].x)
                m, mw = same_line_after[0]
                return mw, m.group(0), m.group(0)
            # Verified against record 241023 (dev set): State_2 ("MN") and
            # its own Zip_2 ("56662") sometimes wrap across the line
            # break, the same way City_2 itself does - Zip starts the
            # line directly below State with no same-line candidate at
            # all. Checked before the Y-distance fallback below (not
            # instead of it) since it's a more precise, position-based
            # signal than pure nearest-by-Y-distance, which previously
            # picked an unrelated same-length number from a different
            # line instead (here, a street address's leading "10333").
            # Verified against record 241101 (dev set): real Zip_2 "955"
            # is only 3 digits (a short/truncated Puerto Rico zip, a
            # confirmed real shape in this dataset) and was being
            # excluded by a 5-digit-only restriction here, leaving this
            # check empty and falling through to the unanchored Y-
            # distance fallback below, which grabbed an unrelated 5-digit
            # number instead (the res address's own house number). That
            # 5-digit restriction exists to keep the UNANCHORED fallback
            # below from confusing Height/Weight with a real zip by pure
            # proximity - this check isn't unanchored, it additionally
            # requires being the leftmost word on the specific line right
            # after State, which is precise enough to safely accept the
            # full 3-5 digit ZIP_CODE range.
            next_line = [
                (m, mw)
                for m, mw in candidates_found
                if mw[0].line_num == state_word.line_num + 1
            ]
            if next_line:
                next_line.sort(key=lambda item: item[1][0].x)
                m, mw = next_line[0]
                # Only accept if this candidate is genuinely the leftmost
                # word on its line (not just leftmost among zip
                # candidates), confirming it starts the wrapped line
                # rather than sitting mid-row as an unrelated number.
                all_words_on_line = [w for _, _, w in spans if w.line_num == mw[0].line_num]
                if all_words_on_line and min(all_words_on_line, key=lambda w: w.x).x == mw[0].x:
                    return mw, m.group(0), m.group(0)
            # No precisely-anchored candidate on State's own line - fall
            # back to nearest-by-Y-distance, but restrict it to genuine
            # 5-digit zips only. Verified against real ground-truth data
            # that allowing the newly-matchable 3-4 digit candidates into
            # this UNANCHORED distance-only comparison caused a real
            # regression: Height/Weight are also 3-digit numbers and
            # sometimes the geometrically nearest short number to a state
            # with no real zip found nearby, corrupting Height/Weight/
            # Tablets (which then shift into each other's slots) instead of
            # just leaving Zip unfound. The short-zip support only pays off
            # in the precisely-anchored same-line case above.
            candidates_found = [
                (m, mw) for m, mw in candidates_found if len(m.group(0).split("-")[0]) == 5
            ]
            # A real Zip always comes AFTER its own State in reading order,
            # never before - verified directly against a real record where
            # this unanchored Y-distance fallback picked the RES ADDRESS's
            # own house number as Zip_2 ("10722", positioned near the very
            # TOP of the record, far before State_2) simply because it was
            # the only remaining 5-digit-shaped candidate anywhere in the
            # text once the real Zip_1 had already consumed the genuine
            # zip number. Excluding anything before State_word in reading
            # order is a precise, principled constraint (not an arbitrary
            # pixel distance) that directly rules this out while leaving
            # every genuine trailing/wrapped-zip case untouched.
            candidates_found = [
                (m, mw)
                for m, mw in candidates_found
                if mw and (mw[0].line_num, mw[0].x) > (state_word.line_num, state_word.x)
            ]
            if not candidates_found:
                return None
            state_y = state_word.y
            candidates_found.sort(key=lambda item: abs((item[1][0].y if item[1] else 0) - state_y))
            m, mw = candidates_found[0]
            return mw, m.group(0), m.group(0)
        # Verified against real ground-truth data (record 241016): with no
        # state anchor at all, this returned the FIRST zip-shaped (3-5
        # digit) token found anywhere in the record's text, in plain
        # reading order - no position check of any kind. That's the res
        # address's own house number far more often than it's a real zip,
        # since the address is read before city/state/zip in this form's
        # layout. No anchor means no reliable way to find zip at all here;
        # returning nothing is safer than confidently claiming the wrong
        # field's value; also stops zip_2 from stealing a word that
        # res_address would otherwise correctly claim later.
        return None

    @staticmethod
    def _assign_lone_phone_slot(phone_word: OCRWord, state_2_word: OCRWord | None) -> str:
        """Bug fix: when exactly one phone-shaped match is found, it must
        never default to Phone_1 just because it's the only match - if
        Phone_1 genuinely isn't on the page (OCR/regex found nothing
        there) but Phone_2 is, the one real match is Phone_2, not a
        "shifted" Phone_1. FIELD_TO_DUMP_TAG's own schema order confirms
        Phone_1 always precedes Country_1 (block 1) while Phone_2 always
        follows Country_2 (block 2, which starts after State_2) - so a
        lone match at or after State_2's own position is Phone_2. Falls
        back to the original phno_1 default only when there's no State_2
        anchor to compare against (unchanged behavior for that case)."""
        is_phone_2 = (
            state_2_word is not None
            and (phone_word.line_num, phone_word.x) >= (state_2_word.line_num, state_2_word.x)
        )
        return "phno_2" if is_phone_2 else "phno_1"

    @staticmethod
    def _extract_phones(
        text: str, spans, consumed: set[int]
    ) -> list[tuple[list[OCRWord], str, str]]:
        found: list[tuple[int, list[OCRWord], str, str]] = []
        for m in pat.PHONE_PUNCTUATED.finditer(text):
            mw = [w for w in words_in_span(spans, m.start(), m.end()) if id(w) not in consumed]
            if mw:
                # PHONE_PUNCTUATED's separator class intentionally accepts
                # whitespace (needed when a variant's own OCR genuinely
                # prints a gap, e.g. "(205) 527-8209"), but when the number
                # was split across two separate OCR word objects,
                # build_indexed_text's single-space join between them lands
                # right next to a hyphen and leaks into the value - verified
                # directly: 4 separate real records all showed the same
                # "907- 722-5546" shape (space immediately after the first
                # hyphen only, never elsewhere). A hyphen-adjacent space is
                # never real phone-number content in this format, so it's
                # collapsed away; hyphens, digits and parens are untouched.
                cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
                cleaned = re.sub(r"\s*-\s*", "-", cleaned)
                found.append((m.start(), mw, cleaned, m.group(0)))
        for m in pat.PHONE_BARE.finditer(text):
            mw = [w for w in words_in_span(spans, m.start(), m.end()) if id(w) not in consumed]
            if mw and not any(id(w) in {id(x) for x in f[1]} for f in found for w in mw):
                # Store exactly what's in the image, digit-for-digit - a
                # leading "1" (US country code) is real data on the page and
                # must not be silently dropped, even though it's optional
                # for the regex to match a bare 10-digit number.
                found.append((m.start(), mw, m.group(0), m.group(0)))
        found.sort(key=lambda item: item[0])
        results = []
        seen_ids: set[int] = set()
        for _, mw, value, raw in found:
            if any(id(w) in seen_ids for w in mw):
                continue
            seen_ids.update(id(w) for w in mw)
            results.append((mw, value, raw))
        return results

    # Words from any other closed vocabulary this document uses (gender,
    # yes/no/na flags, country codes, day/month names) that _extract_city's
    # same-line heuristic could otherwise mistake for part of a city name -
    # NOISE_TOKENS already existed for this purpose elsewhere (_find_name_run,
    # _extract_street_address) but _extract_city never consulted it.
    #
    # Street-suffix words (St/Ave/Rd/...) included too - real, confirmed bug
    # found via manual verification against the actual source image
    # (November dataset, MI_New T(132).jpeg record 1): the line above
    # State_1 read "...WOOTiIN RD Gaithersburg" - the real city
    # ("Gaithersburg") is lowercase and invisible to the ALL-CAPS
    # line_offset=-1 fallback tier, but "RD" (the street suffix right
    # before it) IS all-caps and otherwise unexcluded, so it won city_1
    # outright instead of being skipped past to reach the real city after
    # it. The primary (same-line) search already stops ITS OWN walk at a
    # street suffix to avoid absorbing the address - this closes the same
    # gap for the word being picked AS the city in the first place.
    _CITY_EXCLUSION_VOCAB = (
        NOISE_TOKENS | pat.DAY_MONTH_NAMES | pat.STREET_SUFFIX_WORDS | pat.STREET_SUFFIX_WORDS_OCR_VARIANTS
    )

    @staticmethod
    def _looks_alphabetic_city_word(text: str) -> bool:
        """True for a word that's alphabetic once a stray apostrophe is
        removed - confirmed real, not hypothetical: Azure read "DENTON"
        as "DENTO'N" (one inserted apostrophe) on a real document (751.
        jpeg), and a bare .isalpha() check excluded it outright from
        EVERY city candidate pool below. Since Res Address is recovered
        by POSITION relative to City_1 (see that field's own comment
        above), losing City_1 this way silently took Res Address down
        with it too - not a city-only bug. A real US place name can
        also legitimately contain an apostrophe (e.g. "Coeur d'Alene"),
        so this isn't purely a noise accommodation either way."""
        return text.replace("'", "").replace("’", "").isalpha()

    @classmethod
    def _is_city_noise_word(cls, word: str) -> bool:
        upper = word.upper()
        if upper in cls._CITY_EXCLUSION_VOCAB:
            return True
        # Fuzzy fallback for OCR-misread vocabulary words (e.g. "Septenber"
        # for September, "FIIALE" for FEMALE) that slip past both the exact
        # set above and their own field's extraction (too garbled to be
        # confidently claimed as the real field, but still clearly not a
        # city name). Restricted to len>=4 so short real city-name
        # fragments can't spuriously fuzzy-match a short word like "MAY".
        if len(upper) < 4:
            return False
        return closest_enum_match(upper, cls._CITY_EXCLUSION_VOCAB, cutoff=0.8) is not None

    @classmethod
    def _extract_city(
        cls,
        words: list[OCRWord],
        consumed: set[int],
        state_word: OCRWord,
        line_offset: int = 0,
        require_upper: bool = True,
    ) -> tuple[list[OCRWord], str] | None:
        # Date words are excluded even though they're consumed earlier (see
        # _extract_dates call order) - and gender/flag/country vocabulary
        # words too - as a second layer of defense for cases where the
        # owning field's own extraction couldn't confidently claim a
        # garbled OCR reading, but it's still clearly not a city name.
        # Verified against real ground-truth data that city names in this
        # document family print in ALL CAPS ("BROOKSHIRE", "TAMWORTH",
        # "NORTH NEW HYDE PARK") - requiring full uppercase instead of just
        # a capitalized first letter correctly excludes an adjacent Title
        # Case Shipper Name that the state-line window also catches (e.g.
        # "Wendy Heynen BROOKSHIRE" -> "BROOKSHIRE", not the whole phrase).
        # line_offset=-1 is a LAST-RESORT fallback (see the caller) for the
        # dense wrapped-paragraph layout (USER01/USER12), which line-wraps
        # at a fixed character width, not at field/cell boundaries - the
        # city frequently ends up as the last word(s) of the PHYSICAL LINE
        # ABOVE state_word, with state starting the next wrapped line,
        # purely by coincidence of where the wrap landed (verified directly,
        # e.g. "...OLIVE HILL\nMI 79233...", ground truth city_1="OLIVE
        # HILL"). Must be tried AFTER _extract_city_fallback, not before:
        # confirmed directly that trying it first steals cases where the
        # real city is Title Case on the CORRECT same line (e.g.
        # "...LN New\nPhiladelphia FL..." - the real city "Philadelphia" is
        # same-line but Title Case, invisible to this ALL-CAPS check; tried
        # before the Title-Case fallback got a chance, this wrongly grabbed
        # the leftover ALL-CAPS street-suffix word "LN" from the line
        # above instead). The old tabular layout always has a same-line
        # ALL-CAPS city (the verified assumption this function is built
        # on), so line_offset=0 never returns None there and this is never
        # reached for it.
        #
        # require_upper=False is a SEPARATE, even-later fallback (tried by
        # the caller only after same-line ALL-CAPS, the Title-Case suffix
        # fallback, AND prev-line ALL-CAPS have all failed): this document
        # family's cities aren't always ALL CAPS ("Los Angeles", "Shawnee
        # Mission", even lowercase "laguna hills", "campbellsville" -
        # dozens of real examples). Originally added for City_1 only, on
        # the theory that City_2's adjacency to Shipper Name (the actual
        # motivating reason for the ALL-CAPS requirement) made this too
        # risky there - measured directly on the held-out set instead of
        # just reasoning about it: City_2 and Shipper Name were BOTH
        # exactly unchanged on held-out (the earlier, stricter fallbacks
        # already succeed for every real old-layout case, so this one
        # never even gets reached there), while enabling it for City_2 too
        # recovered a large number of real new-layout cases AND fixed
        # Shipper Name's own boundary in the process (Shipper Name's right
        # edge is anchored on City_2's position when found - a correctly
        # found City_2 stops Shipper Name from overrunning into it). The
        # suffix-stop walk below still protects against over-extending
        # into Res Address/Shipper Name either way.
        #
        # A candidate word immediately followed (tight gap, no real word
        # space) by a recognized street-suffix word is the TAIL of a
        # street name ("Francisca St."), not a city, even though the
        # suffix word itself never enters `same_line` below (a period-
        # bearing suffix like "ST." fails the alphabetic-only check, and
        # even a periodless one like "AVi" would still just look like an
        # ordinary all-caps word in isolation). Without this, an all-caps
        # street-name word sitting directly before the real, genuinely
        # Title-Case city on the SAME line wins the strict ALL-CAPS tier
        # outright, and the correct looser tiers never get a chance to run
        # at all (city1 is already truthy). Confirmed real (November dump,
        # USER 09/2582.jpeg record 421142): "...3 FRANCISCA ST. Hornbrook
        # NC" - "FRANCISCA" (all caps) was picked as City_1 instead of the
        # real "Hornbrook" (Title Case), which also truncated Res Address
        # (bounded by City_1's position) to "P.o.box 59 3", dropping its
        # own real tail "FRANCISCA ST.". Gap/suffix-set reuse matches the
        # contiguous walk's own street-suffix stop further below exactly -
        # same evidence, applied one step earlier.
        suffix_vocab = pat.STREET_SUFFIX_WORDS | pat.STREET_SUFFIX_WORDS_OCR_VARIANTS
        line_words = sorted(
            (
                w
                for w in words
                if id(w) not in consumed
                and w.page == state_word.page
                and w.line_num == state_word.line_num + line_offset
            ),
            key=lambda w: w.x,
        )
        # Walk backward from EVERY street-suffix word through its tightly-
        # adjacent predecessors (not just the one immediately before it) -
        # a street name can be more than one word ("FRANCISCA" here is
        # itself later split into "FRANCIS"+"CA" by the earlier glued-
        # state-suffix step, since "CA" is a valid state code; both halves
        # are equally part of the same street-name phrase and must both be
        # excluded, not just whichever one happens to sit directly next to
        # the suffix word).
        street_tail_ids: set[int] = set()
        for i, w in enumerate(line_words):
            if w.text.upper().rstrip(".") not in suffix_vocab:
                continue
            j = i - 1
            while j >= 0:
                gap = line_words[j + 1].x - line_words[j].right
                if gap > max(line_words[j].height, line_words[j + 1].height) * 0.8:
                    break
                street_tail_ids.add(id(line_words[j]))
                j -= 1

        same_line = [
            w
            for w in words
            if id(w) not in consumed
            and w.page == state_word.page
            and w.line_num == state_word.line_num + line_offset
            and (line_offset == 0 and w.x < state_word.x or line_offset != 0)
            and cls._looks_alphabetic_city_word(w.text)
            and (w.text.isupper() if require_upper else True)
            and not cls._is_city_noise_word(w.text)
            and id(w) not in street_tail_ids
        ]
        if not same_line:
            return None
        same_line.sort(key=lambda w: w.x)
        # Verified against real ground-truth data that city names are often
        # 3-4 words ("NORTH NEW HYDE PARK", "LAND O LAKES", "VISTAS DE LA
        # VEGA") - a hard "last 2 words" cap was silently truncating these to
        # their last word or two.
        #
        # "Same line" alone isn't always a tight enough window, though: some
        # OCR sources (verified with Azure Document Intelligence) put far
        # more of the record on one physical row than others do - record_no,
        # name, email, address, city AND state can all share one row - so
        # blindly taking "the last 5 qualifying words before state" can reach
        # back across a whole cell boundary and capture address/name text
        # along with the city. Real measured gaps on this document family:
        # within a city/address phrase, ~2-8px between words; between two
        # distinct cells on the same row, ~22-45px. Walking backward from the
        # word closest to the state and stopping at the first gap wider than
        # the words' own height (a resolution-independent proxy for "about
        # one character wide") keeps the run to one contiguous phrase.
        # Verified against real ground-truth data: when Res Address and
        # City share one line with no real gap at all between them (e.g.
        # "13070 PFilFLi AVi AK STEEL" - every word is ~4px apart
        # throughout), the gap check alone can't tell them apart, and the
        # walk reached straight through the street address too. A street-
        # suffix word ("AVi" = "AVE") marks exactly where the address ends
        # and the city begins - stop there rather than include it.
        upper_suffix = pat.STREET_SUFFIX_WORDS | pat.STREET_SUFFIX_WORDS_OCR_VARIANTS
        contiguous: list[OCRWord] = [same_line[-1]]
        for w in reversed(same_line[:-1]):
            gap = contiguous[0].x - w.right
            if gap > max(w.height, contiguous[0].height) * 0.8:
                break
            if w.text.upper() in upper_suffix:
                break
            contiguous.insert(0, w)
        picked = contiguous[-5:]
        # Tried: recovering a City_2 that wraps across the line break
        # itself (e.g. "NEW" on the row above State_2, "HAVEN" starting
        # State_2's own row - confirmed via records 241066/241096) by
        # prepending the last ALL-CAPS word of the row above. Reverted:
        # measured on the dev set, this regressed city_2 (-12.1pp),
        # shipper_name (-18.1pp), and res_address (-4.0pp) - the row
        # directly above State_2 is NOT a fixed, reliable column position
        # across records (it holds Shipper Name's own trailing word in
        # many records, which this then wrongly stole whenever Shipper
        # Name's real value happened to be missing/short on that row).
        # Net regression, not a net improvement - left unfixed pending a
        # more precise signal than "last word of the previous row".
        return picked, _smart_join(picked)

    # A real US place name is recognizable even without the ALL CAPS
    # signal via these extremely common endings. Used ONLY as a narrow,
    # single-word fallback below (see _extract_city_fallback) for the
    # genuine minority of city names in this document that print in
    # ordinary Title Case (e.g. "Mechanicsville") instead - never as a
    # general city/name disambiguator, since it can't distinguish a
    # multi-word Title Case city from an adjacent Title Case Shipper Name
    # with no gap between them (a real, acknowledged limitation).
    _PLACE_NAME_SUFFIXES = (
        "VILLE", "BURG", "BORO", "BOROUGH", "TON", "TOWN", "FIELD", "DALE",
        "WOOD", "LAND", "VIEW", "PARK", "SIDE", "FORD", "PORT", "CHESTER",
        "SHIRE", "WORTH", "FALLS", "SPRINGS", "HEIGHTS", "POINT", "CREEK",
    )

    @classmethod
    def _extract_city_fallback(
        cls, words: list[OCRWord], consumed: set[int], state_word: OCRWord
    ) -> tuple[list[OCRWord], str] | None:
        """Only used when the ALL-CAPS city search above finds nothing on
        this line. Conservative by design: claims at most the single word
        immediately before State whose ending matches a common US place-
        name suffix, rather than guessing how many of the preceding Title
        Case words belong to the city instead of an adjacent Shipper/STM
        Name run (there is no reliable gap to cut on in this case, so
        anything wider than one word would be a guess, not a measurement)."""
        same_line = [
            w
            for w in words
            if id(w) not in consumed
            and w.page == state_word.page
            and w.line_num == state_word.line_num
            and w.x < state_word.x
            and cls._looks_alphabetic_city_word(w.text)
            and not cls._is_city_noise_word(w.text)
        ]
        if not same_line:
            return None
        same_line.sort(key=lambda w: w.x)
        last = same_line[-1]
        if any(last.text.upper().endswith(suf) for suf in cls._PLACE_NAME_SUFFIXES):
            return [last], last.text
        return None

    @staticmethod
    def _find_flag_run(
        words: list[OCRWord], consumed: set[int], layout: str = "old"
    ) -> list[tuple[OCRWord, str]]:
        # A word already claimed by an earlier field (e.g. a phone
        # number) is normally excluded here - but when that word has a
        # flag answer glued onto its own tail with no space (see the
        # digit-suffix recovery below), the SAME OCR word genuinely
        # carries two separate pieces of information, and the earlier
        # claim only accounted for one of them. Re-admitting just these
        # words (not consumed words generally) lets that glued answer
        # still be found instead of silently vanishing because the word
        # was already spoken for.
        def _has_glued_flag_suffix(text: str) -> bool:
            return bool(re.search(r"\d(YES|N\.?A\.?|NO)$", text.replace(".", ""), re.IGNORECASE))

        ordered = sorted(
            (w for w in words if id(w) not in consumed or _has_glued_flag_suffix(w.text)),
            key=lambda w: (w.line_num, w.x),
        )
        # NEW-LAYOUT ONLY (the wraps_run allowance below): dense wrapped-paragraph layout (USER01/USER12) sometimes wraps
        # mid-run, with the flag sequence's remaining tokens continuing as
        # the very FIRST word of the next physical line (verified directly,
        # e.g. "...NO NO NO\nNO N.A ..." where ground truth has all 5 as one
        # answer run - the strict same-line assumption below silently
        # dropped the wrapped tail, costing Allergies specifically since
        # it's the LAST of the 5). Allowed to continue across a line change
        # only when the word is the LEFTMOST word on its line - i.e.
        # genuinely next in reading order with nothing before it - so an
        # unrelated flag-shaped word deep into a new, unrelated line still
        # breaks the run exactly as before.
        leftmost_ids: set[int] = set()
        by_line: dict[int, list[OCRWord]] = {}
        for w in ordered:
            by_line.setdefault(w.line_num, []).append(w)
        for line_words in by_line.values():
            leftmost_ids.add(id(min(line_words, key=lambda w: w.x)))

        best: list[tuple[OCRWord, str]] = []
        current: list[tuple[OCRWord, str]] = []
        prev_line = None
        for w in ordered:
            # Periods stripped before the vocabulary comparison - real,
            # confirmed bug (November dataset, 787.jpeg record 9): OCR
            # read "NO" as "N.O.", which closest_enum_match's raw
            # difflib ratio against "NO" (0.67) never clears the 0.85
            # cutoff - the periods alone were enough to make a real flag
            # answer invisible. Safe for "N.A" too: stripped to "NA",
            # already a distinct accepted member of YES_NO_NA_VALUES
            # itself, not a new fuzzy match.
            # Leading/trailing quote, backtick and backslash characters
            # also stripped - a confirmed recurring OCR artifact around
            # "YES" specifically (753.jpeg record 29610284: raw token
            # '"YES\' for Smoker, ratio 0.75 against "YES", just under
            # the 0.85 cutoff - independently corroborated by ground
            # truth's own OCR pass showing the identical noise shape,
            # e.g. "'YES\\", on other records). None of these characters
            # can ever be part of a real YES/NO/N.A answer, so stripping
            # them is a deterministic cleanup, not a fuzzy guess. Left
            # unrecovered, this single noisy word broke the run right
            # after its first match and silently shifted every
            # subsequent flag (Alcoholic through Allergies) onto the
            # wrong field name instead of just losing one.
            stripped = w.text.replace(".", "").strip("'\"`\\")
            match = closest_enum_match(stripped, pat.YES_NO_NA_VALUES, cutoff=0.85)
            if not match and stripped.upper() in ("NOT", "NOO"):
                # A confirmed real, recurring OCR misread of "NO" (62
                # occurrences across 19 of 44 real November-dataset
                # images) - a trailing "T" is just different enough that
                # the raw difflib ratio against "NO" (0.80) falls short
                # of the 0.85 cutoff by a narrow margin, the same kind of
                # near-miss as "N.O." above, not a coincidence. Breaking
                # the run on this one word silently cost the LAST flag
                # (Allergies) in every affected record, the same failure
                # mode as the period/glued-suffix cases - this is a
                # third, independent instance of the identical bug
                # class, not a new mechanism.
                #
                # "NOO" (period-stripped "N.OO") is a fourth, same-class
                # instance - a doubled trailing "O", the same character-
                # doubling artifact already confirmed elsewhere in this
                # November-dump family (e.g. shipping_cost's "$2.o0.00").
                # Ratio against "NO" is exactly 0.80, the identical
                # near-miss margin as "NOT" above. Confirmed real (USER
                # 13/4.jpeg record 94370059): ground truth's own Smoker
                # value is "N.OO" verbatim (the raw OCR noise, preserved
                # rather than cleaned up) - breaking the run here cost
                # BOTH the preceding Alcoholic answer (reduced to a
                # length-1 run, discarded once a longer run formed later)
                # and silently shifted Past Surg/Diabetic/Allergies each
                # one field early.
                match = "NO"
            if not match:
                # A flag answer is sometimes glued with no space onto the
                # END of the PRECEDING field's own text - confirmed real
                # (November dataset, MI_New T(132).jpeg record 1): a
                # phone number read as "880-6110YES", with the run's
                # first real flag answer ("YES") completely invisible as
                # its own token. Left unrecovered, the run comes back
                # short from the FRONT - which the caller (zip(flag_
                # fields, flags), front-aligned) then silently maps onto
                # the WRONG field names for every flag after it, not just
                # a missing last one; see that zip's own comment. Only
                # recovered when a DIGIT sits immediately before the
                # suffix - the real signature of text glued onto a
                # number (a phone/ID/currency field) - so a genuine word
                # that merely ends in "no"/"yes" (e.g. a city "Reno")
                # is never mistaken for this, since it's preceded by a
                # letter, not a digit.
                suffix_match = re.search(r"\d(YES|N\.?A\.?|NO)$", stripped, re.IGNORECASE)
                if suffix_match:
                    match = closest_enum_match(suffix_match.group(1), pat.YES_NO_NA_VALUES, cutoff=0.85)
            if not match and len(current) >= 2:
                # Already 2+ confirmed flag answers immediately before this
                # word in reading order - OCR garbles a flag token badly
                # enough to miss every recovery above often enough to
                # matter (confirmed real variants: "NOE", "NOI", "~YnO",
                # all meant to read "NO") and, left unrecovered, breaks the
                # run right before the end, silently losing the trailing
                # flags even though the row position makes this word
                # unambiguously the next answer, not unrelated text. A
                # much looser cutoff is safe ONLY in this exact context -
                # continuing an already-established run of 2+ confirmed
                # matches, immediately next in reading order - unlike
                # scanning the whole page for any similarly-shaped word
                # (tried for Sex_2 and reverted: that produced far more
                # false positives than it recovered, since nothing already
                # anchors which word SHOULD be the answer). Here the
                # schema itself guarantees this position holds a flag
                # answer; the only question is which one.
                match = closest_enum_match(stripped, pat.YES_NO_NA_VALUES, cutoff=0.6)
            same_line = prev_line is None or w.line_num == prev_line
            # Confirmed the identical wrapped-run pattern in old-layout
            # November-dump data too (not new-layout-specific after all -
            # same conclusion already reached twice elsewhere in this file
            # for the hash-code-split and garbled-zip cases): a record's
            # flag run regularly wraps mid-run onto the next physical
            # line, e.g. "...879-2312 NO YES NO\nNO NO Zc_Bqn..." - the
            # strict same-line requirement broke the run right there,
            # costing the trailing flags (Diabetic, Allergies) even
            # though the wrap is unambiguous (leftmost word, immediately
            # continuing the run).
            wraps_run = not same_line and bool(current) and id(w) in leftmost_ids
            # A token that's ENTIRELY periods (stripped to nothing) is the
            # other half of the same real bug: OCR sometimes splits one
            # garbled flag token into a bare "." plus the rest ("N.O.")
            # as two separate words. Confirmed real (787.jpeg record 9):
            # "YES YES .N.O. .N.O. . N.O." - the lone "." sits between
            # two genuine flag tokens and, as a plain non-match, used to
            # reset the run right before the real 5th flag, silently
            # losing it even after the period-stripping fix above. Skip
            # it - mid-run, same line, no real content - rather than
            # treating it as a run-ending non-match.
            is_punctuation_noise = bool(w.text) and not stripped
            if match and (same_line or wraps_run):
                current.append((w, match))
            elif is_punctuation_noise and current and same_line:
                pass
            else:
                current = [(w, match)] if match else []
            if len(current) > len(best):
                best = list(current)
            prev_line = w.line_num
        # Verified against real ground-truth data that the detected run is
        # sometimes 6 tokens long because PhNo_2's own explicit "N.A" sits
        # with no gap immediately before the true 5-flag run (PhNo_2 is the
        # field directly before Alcoholic in schema order) - truncating to
        # the FIRST 5 took that leading N.A as if it were Alcoholic's own
        # answer and silently dropped the real last flag (Allergies) off
        # the end. The trailing 5 are the real flags; whatever's ahead of
        # them belongs to an earlier field, not this one.
        return best[-5:]

    @staticmethod
    def _extract_flags(
        words: list[OCRWord], consumed: set[int], layout: str = "old"
    ) -> list[tuple[OCRWord, str]]:
        # The 5 flag answers print as one tight run of YES/NO/N.A tokens on a
        # single line (verified against the source image). Allergies is the
        # 5th slot; its strict YES/NO-only rule is applied at validation time.
        found = RuleBasedExtractionEngine._find_flag_run(words, consumed, layout)
        for w, _ in found:
            consumed.add(id(w))
        return found

    @staticmethod
    def _extract_height_weight(
        words: list[OCRWord], consumed: set[int]
    ) -> list[tuple[OCRWord, str]]:
        pool = [
            w
            for w in words
            if id(w) not in consumed
            and w.text.isdigit()
            and 2 <= len(w.text) <= 3
            and 40 <= int(w.text) <= 400
        ]
        pool.sort(key=lambda w: (w.line_num, w.x))
        # NOTE: tried restricting the 2nd match to Height's own line_num
        # (Height and Weight are on the same physical line on this form) to
        # stop a pass where Height's token was unreadable from reaching all
        # the way down to an unrelated field for its "Weight" - measured to
        # cause a net regression instead (broke a different pass that was
        # correctly finding Weight via the wider scan), so reverted. Left
        # as a known open issue rather than shipping the regression.
        found = [(w, w.text) for w in pool[:2]]
        for w, _ in found:
            consumed.add(id(w))
        return found

    @staticmethod
    def _extract_dates(
        text: str, spans, consumed: set[int], candidates: dict, record_no_pos: int | None = None
    ) -> list[int]:
        """Returns the text positions of every date found (in order) so
        callers needing a reliable positional anchor (e.g. the currency
        split - dates are found consistently across passes, unlike "MG"
        which OCR frequently drops) can use them."""
        # Stores the exact source text ("Sunday, October 10, 1999" /
        # "01/22/1998"), not an ISO-reformatted value - normalize_date_* is
        # used only to confirm the match is a real, parseable date before
        # accepting it (rejects e.g. "Sunday, Febluary 31, 1999"), never to
        # rewrite the stored value.
        found: list[tuple[int, list[OCRWord], str]] = []
        for m in pat.DATE_LONG.finditer(text):
            mw = [w for w in words_in_span(spans, m.start(), m.end()) if id(w) not in consumed]
            if not mw:
                continue
            if normalize_date_long(*m.groups()) is not None:
                raw = re.sub(r"\s+", " ", m.group(0)).strip()
                found.append((m.start(), mw, raw))
        for m in pat.DATE_NUMERIC.finditer(text):
            mw = [w for w in words_in_span(spans, m.start(), m.end()) if id(w) not in consumed]
            if not mw:
                continue
            if normalize_date_numeric(*m.groups()) is not None:
                raw = re.sub(r"\s+", " ", m.group(0)).strip()
                # Tried and reverted: stripping the space after the weekday's
                # comma ("Wednesday, 07/26/1950" -> "Wednesday,07/26/1950").
                # A first sample (09101/09102/09104/09109/09116) showed
                # ground truth with no space in every case, but reprocessing
                # the full 9-image set showed the opposite on 09107/09110/
                # 09113/09120: ground truth genuinely HAS a space there on
                # many real records. This is real variation in the source
                # images, not a join artifact - no blanket rule fits both,
                # and there's no cheap per-record signal (word-level OCR gap
                # data) available to tell them apart. Measured regression:
                # d_birth 73.1%->47.6% across all 9 images. Reverted.
                found.append((m.start(), mw, raw))

        # Recovery pass: verified against 5 dev-set examples, DOB sometimes
        # glues directly onto a following STM_Code/Policy_No hash-code with
        # no separator (e.g. "BaX_01acM_9548" + "11/23/1952" read as
        # "...954811/23/1952") - the date regex's leading \b can't match
        # mid-digit-run there, so the normal DATE_NUMERIC pass above misses
        # it entirely. Reuses the hash-code shape (not a blanket
        # unanchored date search) to find exactly where such a code would
        # overshoot into a trailing date, matching the same overshoot
        # _extract_hash_codes trims on its own independent pass later -
        # every traced example overshoots by exactly the date's 2-digit
        # day field.
        # Words recovered this way physically overlap a hash-code match
        # (the two values are glued onto the SAME OCR word) - consumption is
        # tracked per word, not per character range, so these words must
        # NOT be marked consumed here, or _extract_hash_codes (which runs
        # later and needs that same word for stm_code/policy_no) would find
        # nothing left to claim.
        glued_recovery_word_ids: set[int] = set()
        for hm in HASH_CODE.finditer(text):
            if not any(ch.isdigit() for ch in hm.group(0)):
                continue
            for overshoot in (2, 1):
                trial_end = hm.end() - overshoot
                if trial_end <= hm.start():
                    continue
                dm = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})\b", text[trial_end:])
                if dm and normalize_date_numeric(*dm.groups()) is not None:
                    mw = [w for w in words_in_span(spans, trial_end, trial_end + dm.end()) if id(w) not in consumed]
                    if mw:
                        glued_recovery_word_ids.update(id(w) for w in mw)
                        found.append((trial_end, mw, dm.group(0)))
                    break

        found.sort(key=lambda item: item[0])

        # A date whose text position comes BEFORE this record's own Record
        # No can only be a fragment of the PREVIOUS record leaking in
        # through the crop's top margin (a record's own content never
        # precedes its own Record No label) - drop it rather than let it
        # steal the D_Birth slot from the record's genuine first date.
        if record_no_pos is not None:
            found = [item for item in found if item[0] >= record_no_pos]

        # Verified by scanning the FULL real ground-truth XML directly (not
        # a sample): D_Birth, DOB, and D_B_Life_Assure are the SAME
        # underlying date, printed 2-3 times on the form (D_Birth == DOB in
        # 95.1% of all 3597 records; all three match in 93.2% of records
        # that have D_B_Life_Assure) - the exact same "duplicated field"
        # pattern already known for Customer Name/Billing Name/
        # Name_P_Holder, not three distinct, sequentially-positioned dates.
        # The previous "first/middle/last position" assumption meant
        # D_B_Life_Assure (and sometimes DOB) only got a value when 3+
        # separate date matches happened to be found in this one OCR pass -
        # missing them whenever OCR read fewer occurrences, even though the
        # real shared value was available from a different, successfully-
        # read occurrence in the same record. Groups matches by their
        # numeric date (day-name spelling/garbling ignored for grouping
        # only, never for the stored value) and assigns whichever
        # repeated-occurrence reading is most complete to all three.
        def date_key(raw: str) -> str:
            m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", raw)
            return m.group(0) if m else raw.upper()

        date_keys = [date_key(raw) for _, _, raw in found]
        key_counts: dict[str, int] = {}
        for k in date_keys:
            key_counts[k] = key_counts.get(k, 0) + 1
        most_common_key = max(key_counts, key=lambda k: key_counts[k]) if key_counts else None

        if most_common_key is not None and key_counts[most_common_key] >= 2:
            group = [i for i, k in enumerate(date_keys) if k == most_common_key]
            best_idx = max(group, key=lambda i: len(found[i][2]))
            _, best_mw, shared_value = found[best_idx]
            for i in group:
                for w in found[i][1]:
                    if id(w) not in glued_recovery_word_ids:
                        consumed.add(id(w))
            conf = average_confidence(best_mw)
            bbox = bounding_box_for_words(best_mw)
            for field_name in ("d_birth", "dob", "d_b_life_assure"):
                candidates[field_name] = FieldCandidate(
                    field_name, shared_value, shared_value, conf, bbox, True
                )
            # The 93.2% statistic above is about records where D_Birth,
            # DOB AND D_B_Life_Assure all three match - it says nothing
            # about what to do with a genuinely distinct date found
            # ALONGSIDE a repeated pair, and the loop above was silently
            # overwriting that distinct date's own slot with the shared
            # one regardless. Real, confirmed bug (November dump, USER 13/
            # 1.jpeg record 94370030): three dates are actually printed -
            # "07/28/1945" (D_Birth's row), "Wednesday, 08/13/2010"
            # (D_B_Life_Assure's row, between the other two in reading
            # order), "07/28/1945" again (DOB's row) - ground truth
            # confirms D_B_Life_Assure genuinely is "Wednesday,
            # 08/13/2010", not a third copy of the birth date, but the
            # loop above assigned the shared "07/28/1945" to all three
            # slots, discarding the real D_B_Life_Assure entirely. Only
            # acts when there's exactly ONE date outside the repeated
            # pair (an unambiguous "two match, one genuinely differs"
            # shape) - any messier split (2+ found outside, or the group
            # itself not a clean pair) is left to the existing behavior
            # rather than guess which slot the extra date(s) belong to.
            outside = [i for i in range(len(found)) if i not in group]
            if len(outside) == 1 and len(group) == 2:
                _, mw2, raw2 = found[outside[0]]
                for w in mw2:
                    if id(w) not in glued_recovery_word_ids:
                        consumed.add(id(w))
                candidates["d_b_life_assure"] = FieldCandidate(
                    "d_b_life_assure", raw2, raw2, average_confidence(mw2), bounding_box_for_words(mw2), True
                )
        else:
            # No repeat evidenced in this pass (0 or 1 date found, or
            # several genuinely distinct ones) - fall back to the prior
            # positional heuristic rather than guess at a duplication this
            # pass gives no evidence for. Order of appearance decides
            # D_Birth vs DOB; taking the two ends (not first+second) stays
            # robust to an unrelated extra date (e.g. a transaction date)
            # landing in the middle.
            if found:
                _, mw, raw_date = found[0]
                for w in mw:
                    if id(w) not in glued_recovery_word_ids:
                        consumed.add(id(w))
                candidates["d_birth"] = FieldCandidate(
                    "d_birth", raw_date, raw_date, average_confidence(mw), bounding_box_for_words(mw), True
                )
            if len(found) > 1:
                _, mw, raw_date = found[-1]
                for w in mw:
                    if id(w) not in glued_recovery_word_ids:
                        consumed.add(id(w))
                candidates["dob"] = FieldCandidate(
                    "dob", raw_date, raw_date, average_confidence(mw), bounding_box_for_words(mw), True
                )
            if len(found) > 2:
                _, mw, raw_date = found[1]
                for w in mw:
                    if id(w) not in glued_recovery_word_ids:
                        consumed.add(id(w))
                candidates["d_b_life_assure"] = FieldCandidate(
                    "d_b_life_assure", raw_date, raw_date, average_confidence(mw), bounding_box_for_words(mw), True
                )
        return [item[0] for item in found]

    @staticmethod
    def _extract_currency_fields(
        text: str, spans, consumed: set[int], candidates: dict, dosage_pos: int | None
    ) -> None:
        found = []
        for m in pat.CURRENCY.finditer(text):
            mw = [w for w in words_in_span(spans, m.start(), m.end()) if id(w) not in consumed]
            if not mw:
                continue
            # If it cleanly parses as a number, use the normalized form
            # (consistent "123.45" shape for the cross-field math checks).
            # If it doesn't (e.g. "$2.o0.00" - a real OCR-garbled amount),
            # keep the exact raw text rather than dropping the candidate -
            # validation will correctly mark it invalid; extraction must
            # not silently make garbled source data disappear.
            value = normalize_currency(m.group(0)) or m.group(0)
            found.append((m.start(), mw, value))
        if not found:
            return
        found.sort(key=lambda item: item[0])

        def make(field_name: str, item) -> None:
            _, mw, value = item
            for w in mw:
                consumed.add(id(w))
            candidates[field_name] = FieldCandidate(
                field_name, value, value, average_confidence(mw), bounding_box_for_words(mw), True
            )

        # Anchor the split on the dosage match's text position rather than
        # "the last 4 matches" - counting from the end is unstable when a
        # given OCR pass fails to read one of the amounts (a very common
        # partial failure on this document family), since the total count
        # of found amounts then varies pass-to-pass and silently shifts
        # which field gets which value. Anything before the medicine/dosage
        # block is P_Inst (the only "$"-prefixed field in that part of the
        # form per BN_WorkFlow.pdf - D_B_Life_Assure, despite sitting next
        # to it, is a date and is handled in _extract_dates instead); anything
        # at or after the dosage block is Pill Rate / Cost / Shipping Cost /
        # Total Amount, in order of appearance.
        if dosage_pos is not None:
            head = [f for f in found if f[0] < dosage_pos]
            tail = [f for f in found if f[0] >= dosage_pos]
        elif len(found) == 5:
            # Verified against real ground-truth data: when Dosage wasn't
            # matched in this pass (no anchor to split on), P_Inst's own
            # currency value was falling into the tail group instead of
            # being recognized as P_Inst, bumping Pill Rate/Cost/Shipping
            # Cost/Total Amount each one slot out of place. P_Inst
            # structurally comes first in reading order among this
            # record's currency fields, so with exactly 5 found and no
            # better anchor, the first one is the safest P_Inst guess.
            head, tail = [found[0]], found[1:]
        else:
            head, tail = [], found

        tail_fields = ["pill_rate", "cost", "shipping_cost", "total_amount"]
        # If fewer than 4 are available, keep pill_rate first and
        # total_amount last (the two most reliably identifiable), leaving
        # the middle ones unmatched rather than mis-assigned.
        #
        # Real, confirmed bug (November dump, USER 13/1.jpeg record
        # 94370029): the OLD formula `range(4 - len(tail), 4)` does NOT
        # implement that stated intent for any count between 1 and 4 - for
        # 3 found it produces positions [1,2,3], which drops PILL RATE
        # (position 0) and shifts the other three down by one, not
        # "pill_rate first". Confirmed from this record's own raw OCR text
        # ("$2.86 $505.00 $#20.00 $525.00" - Shipping Cost's own "$#20.00"
        # fails the currency regex outright because of its embedded "#",
        # so only 3 of the 4 are ever found): the old formula assigned
        # pill_rate's real value "$2.86" to Cost, Cost's real value
        # "$505.00" to Shipping Cost, and left Pill Rate None entirely -
        # three fields wrong/missing from one single garbled character.
        tail = tail[:4]
        if len(tail) == 4:
            positions = list(range(4))
        elif len(tail) == 3:
            # The only real ambiguity left is which of the two MIDDLE
            # slots (Cost, Shipping Cost) is the missing one - use the
            # actual gap between the 3 found positions (same gap-based
            # boundary reasoning already used throughout this file) rather
            # than assuming it's always the same slot: the gap spanning a
            # skipped slot runs noticeably wider than an ordinary adjacent
            # gap between two genuinely consecutive currency values.
            gap_before = tail[1][0] - tail[0][0]
            gap_after = tail[2][0] - tail[1][0]
            positions = [0, 2, 3] if gap_before > gap_after * 1.4 else [0, 1, 3]
        elif len(tail) == 2:
            positions = [0, 3]
        elif len(tail) == 1:
            positions = [3]
        else:
            positions = []
        for pos, item in zip(positions, tail):
            make(tail_fields[pos], item)

        if head:
            make("p_inst", head[-1])

    @staticmethod
    def _extract_quantity(
        words: list[OCRWord], consumed: set[int], dosage_candidate
    ) -> tuple[OCRWord, str] | None:
        pool = [
            w
            for w in words
            if id(w) not in consumed
            and w.text.isdigit()
            and 1 <= len(w.text) <= 4
            and int(w.text) <= 999
        ]
        if not pool:
            return None
        if dosage_candidate and dosage_candidate.bounding_box:
            ref_y = dosage_candidate.bounding_box.y
            pool.sort(key=lambda w: abs(w.y - ref_y))
        return pool[0], pool[0].text

    _TRAILING_GLUED_SUFFIX = re.compile(r"[.,=_#@-]{1,3}\d{2,6}")
    # Cross-word bridge: a genuinely separate OCR word that is PURELY
    # separator characters (no letters, no digits - e.g. "=-", "--", ",--")
    # directly between the code and a trailing pure-digit word (e.g.
    # "BaX_01aCm" + "=-" + "83454", three separate OCR tokens with real
    # word gaps - confirmed via record 241054, 09106.jpeg - every word
    # gap in the reconstructed text is exactly one space, see
    # parser.build_indexed_text). Deliberately narrower than an earlier
    # general-purpose cross-word continuation attempt that was reverted
    # for gluing onto unrelated words (a following "N.A", a bare
    # adjacent "10"): this only bridges when the CONNECTING word has no
    # letters or digits at all, which "N.A" and a bare number never
    # satisfy, so it cannot reproduce that regression.
    # Verified against record 241036 (dev set): "BaX_01acM, -- 5168" has
    # a dangling comma attached to the CODE's own word (no space before
    # it - the regex's own trailing \b requirement stops the base match
    # right before it, same mechanism as _TRAILING_GLUED_SUFFIX above),
    # THEN a genuine word gap, then the separator-only word, then
    # another gap, then the digit word - a 4-part shape the plain
    # single-space version above didn't anticipate. The optional leading
    # [.,=_-]{0,2} consumes that dangling same-word punctuation before
    # the cross-word part starts.
    # Real, confirmed bug (DataM (4).xml, 751.jpeg record 29610259):
    # "B.alX,_OlaCn" + "=-" (same line, real space) + "70755" (the SAME
    # trailing digit fragment, but wrapped onto the NEXT physical line as
    # its first word - build_indexed_text joins a line change with "\n",
    # not " ", so the literal-space version matched the first gap but not
    # the second, leaving the whole code unrecovered). Each gap now
    # independently accepts either separator - the same "either is fine,
    # as long as it's immediately next in reading order" tolerance
    # already established for the 5-flag run's own cross-line wrap.
    _CROSS_WORD_BRIDGE = re.compile(r"[.,=_#@-]{0,2}[ \n][.,=_#@-]{1,3}[ \n]\d{2,6}\b")

    # Unlike the ambiguous per-instance O/0 and I/l confusables elsewhere in
    # this code family (where the SAME rendered character is genuinely a
    # digit in some records and a letter in others, with no reliable way to
    # tell which - verified directly, not a guess), this one specific
    # template is a confirmed, FIXED, repeating prefix: every real
    # "[Y]+t_" -prefixed code in the ground-truth corpus has "i7" right
    # after the prefix, and Azure consistently misreads it as "17" - 6
    # independent real instances checked (policy_no values for records
    # 402424, 402434, 402437, 402441, 402442, 402444), zero counter-
    # examples. Safe to correct unconditionally for this one template only
    # (not generalized to other O/0 or I/l positions in this code family,
    # which remain genuinely ambiguous per-record).
    _KNOWN_CODE_TEMPLATE_FIXES = (
        (re.compile(r"(Y+t_)17(-Un)"), r"\g<1>i7\g<2>"),
    )
    # Tried: a third template fix for "O(l*)a" immediately before "C"/"c"
    # in this same code family, on the theory that it was always really
    # "0" + "1"*n (Azure misreading 0->O, 1->l), the same confirmed pattern
    # as the Yt_i7 fix above - measured a real net gain on the November
    # dump (stm_code 34.0%->51.6% on a 53-image sample, e.g. USER 06/
    # 1758.jpeg record 412942's real "BaX_01aCo-6577"). Reverted anyway:
    # direct ground-truth comparison (USER 07/2170.jpeg, records 417038/
    # 417039/417042/417043/417046/417047 - 6 separate records, zero
    # exceptions among them) shows the SAME "_Ol[l]aC"/"_OlacM" shape,
    # including the identical "BaX_" prefix used by USER 06's own
    # digit-convention records, genuinely means to stay LETTERS there
    # ("Ba.X_OlaCp_7845", "BaX_OlaCp_8747" - confirmed directly against
    # the source XML, not a script bug). So this character is genuinely
    # ambiguous per-batch with no reliable signal in the record itself to
    # tell which convention applies - unlike Yt_i7, which has zero known
    # counter-examples across the whole corpus - and "fix" it blindly is
    # indistinguishable from guessing, which the no-guessing rule forbids
    # even when it nets positive on one particular sample.


    @classmethod
    def _normalize_known_code_confusables(cls, value: str) -> str:
        for pattern, repl in cls._KNOWN_CODE_TEMPLATE_FIXES:
            value = pattern.sub(repl, value)
        return value

    @staticmethod
    def _extract_hash_codes(
        text: str, spans, consumed: set[int], layout: str = "old"
    ) -> list[tuple[list[OCRWord], str]]:
        found: list[tuple[int, list[OCRWord], str]] = []
        for m in HASH_CODE.finditer(text):
            # Verified against real ground-truth data: the separator-based
            # alternative also matches ordinary period-joined person names,
            # a real documented name shape in this schema (e.g.
            # "Eric.Turcotte", "H.Garland") - a period is a valid separator
            # character, and nothing else in the pattern rules out a plain
            # alphabetic surname on each side. Every confirmed real code
            # contains at least one digit somewhere; a period-joined name
            # never does - reject matches with no digit at all rather than
            # loosen the shared regex itself.
            # Deferred to after the glued/bridge extensions below (was
            # "continue" here) - real, confirmed bug (DataM (4).xml,
            # 751.jpeg record 29610259): "B.alX,_OlaCn" has no digit at
            # all in the BASE match (its own trailing digits "70755" are
            # a separate OCR word, recovered only by the cross-word-
            # bridge logic further down) - rejecting here, before that
            # recovery ever runs, discarded the whole code even though
            # the complete, digit-bearing value was fully recoverable.
            has_base_digit = any(ch.isdigit() for ch in m.group(0))
            # The second HASH_CODE alternative (no punctuation separator)
            # has no real code-shape requirement beyond "has upper, lower,
            # and a digit-letter transition somewhere in 8-14 chars" - that
            # shape is also satisfied by this dataset's own synthetic
            # customer names, which routinely embed a digit (confirmed real
            # false positives: "Faith2Gm", "Bootst3K", both 12.5% digits).
            # Checked against every distinct real STM_Code/PloicyNo value in
            # the full 25-file ground-truth corpus that has no punctuation
            # (5,452 distinct values): the lowest digit density among ALL of
            # them is 37.5% ("ZtA019vB"), clustering mostly at 44-60%
            # ("N12aXp###", "ZtA09vB###" prefix families) - a wide, clean
            # margin above the false positives. 30% sits safely in that gap
            # without rejecting any confirmed real example. Only applies to
            # this branch - the punctuated branch has its own, unrelated
            # false-positive history (period-joined names) already handled
            # by the digit check above, and ordinary names don't naturally
            # contain underscores/hyphens/commas mid-word the way these
            # codes do, so no density check is needed there.
            if not any(ch in ".,_#@-" for ch in m.group(0)):
                digit_count = sum(ch.isdigit() for ch in m.group(0))
                if digit_count / len(m.group(0)) < 0.3:
                    continue
            # A real Policy No/STM Code is always a one-off value in this
            # schema - it never repeats elsewhere in the same record. A
            # synthetic customer-name fragment that happens to satisfy
            # HASH_CODE's shape (confirmed real: "angel_5200", density
            # 40% - well above the digit-density guard above, which can't
            # catch it) typically DOES repeat, since Customer Name,
            # Billing Name, and Name_P_Holder are frequently identical
            # copies of each other per this form's own rules - "angel_5200"
            # appeared 3 separate times in one real record this was traced
            # from, wrongly consumed as the code and cascading into STM
            # Name losing its anchor entirely. Catches both branches, not
            # just the no-punctuation one, since the false positive here
            # used the punctuated branch (via its own "_").
            if text.count(m.group(0)) > 1:
                continue
            end = m.end()
            # Verified against 5 dev-set examples (e.g. "BaX_01acM_9548" +
            # "11/23/1952" glued with no separator at all, read as
            # "BaX_01acM-954811/23/1952"): when DOB's date is glued directly
            # onto this code, the code's character class has nothing to
            # distinguish "its own trailing digits" from "the date's
            # leading digits" and overshoots into the date - every traced
            # example overshoots by exactly the date's 2-digit day field.
            # Trim back 1-2 digits whenever doing so reveals a complete
            # DD/MM/YYYY date starting immediately after - the matching
            # recovery of that date into _extract_dates is what actually
            # lets DOB be found (its own regex can't match mid-digit-run).
            for overshoot in (2, 1):
                trial_end = end - overshoot
                if trial_end <= m.start():
                    continue
                if re.match(r"\d{1,2}/\d{1,2}/\d{4}\b", text[trial_end:]):
                    end = trial_end
                    break
            # Verified against 6 dev-set examples (e.g. "BaX_01acM" next to
            # ",--5168", ground truth "BaX_01acM,--5168"): the regex's own
            # character classes correctly stop at a punctuation mark they
            # can't continue through, but when that punctuation is
            # immediately followed by more digits with NO space in the
            # reconstructed text, it's still the same OCR word (a real word
            # gap always inserts a space - see parser.build_indexed_text),
            # so it's the same code, not unrelated adjacent text. This never
            # crosses an actual word boundary, unlike an earlier cross-word
            # continuation attempt that was reverted for gluing on unrelated
            # separate words (e.g. a following "N.A").
            glued = RuleBasedExtractionEngine._TRAILING_GLUED_SUFFIX.match(text, end)
            if glued:
                end = glued.end()
            else:
                bridge = RuleBasedExtractionEngine._CROSS_WORD_BRIDGE.match(text, end)
                if bridge:
                    end = bridge.end()
            start = m.start()
            # HASH_CODE's own leading \b (both branches) requires starting
            # on a letter, so when the real code genuinely has a leading
            # comma glued directly onto it with no space, the match starts
            # one character late and silently drops it. Verified directly
            # via raw Azure OCR on two independent records (09104.jpeg and
            # 09107.jpeg): the comma is not a separate token or a join
            # artifact - Azure reads it as part of the SAME word, e.g.
            # text=',BBaX_01aCm-9834' as one contiguous OCR word - so this
            # recovers it only when it's immediately adjacent with no
            # whitespace, never a comma that belongs to earlier text.
            if start > 0 and text[start - 1] == ",":
                start -= 1
            elif start > 0 and text[start - 1] == " ":
                # Originally restricted to new-layout only (that document
                # family's dense wrapped-paragraph layout, USER01/USER12) -
                # widened to both layouts after direct ground-truth
                # comparison (November dump, USER 06/1758.jpeg record
                # 412949) found the identical pattern in old-layout data:
                # "B," + "aaX_01aCm-6849" as two separate OCR words with a
                # real 3px space between them (ground truth
                # "B,aaX_01aCm-6849"), the exact same leading-fragment shape
                # already confirmed on the other document family. Same
                # mirror-image widening already applied to the TRAILING
                # bridge case below for the identical reason - this
                # fragmentation is an OCR-tokenization artifact of the hash
                # code shape itself, not something tied to one document
                # family's structural layout.
                #
                # Sometimes reads a short leading fragment of the code as
                # its OWN OCR word, with a real space before the rest -
                # verified directly against 6+ real records, e.g. "Ba,
                # XX_OlaCp_21741" (ground truth "Ba,XX_01aCp_21741") and
                # "BaX_, OlaCm-12893" (ground truth "BaX_,01aCm-12893").
                # Recovered only when gluing that single preceding word
                # directly onto this match reproduces a complete,
                # independently-valid HASH_CODE match - not a blind grab;
                # this re-uses the same shape validation already trusted
                # for the whole code, so an ordinary preceding name/address
                # word can only be swept in if it happens to itself be
                # code-shaped, which measured zero false positives on the
                # held-out regression set. A bare "," is handled as its own
                # case (not required to pass the shape check) since it's
                # the same already-verified leading-comma value as the
                # adjacent case above, just split off as a separate word
                # instead of glued with no space.
                prev_span = next((s for s in spans if s[1] == start - 1), None)
                if prev_span is not None:
                    s2, e2, w2 = prev_span
                    # Require the fragment to already contain one of this
                    # code family's separator characters itself (not just
                    # "any word that happens to glue cleanly") - a bare
                    # fullmatch check alone is too permissive: concrete
                    # false positive found swallowing the survey answer
                    # "NO" immediately before a real code ("NO" +
                    # "FR-DLI019_21918" also happens to satisfy HASH_CODE's
                    # shape). Ordinary enum/name words in this schema never
                    # contain ",._-" themselves, so this still accepts
                    # every real fragment observed ("Ba,", "BaX_,", "BaX.")
                    # while rejecting "NO". A second concrete false
                    # positive found the same way: "N.A" (this schema's
                    # ubiquitous not-applicable placeholder, appearing in
                    # dozens of records for a blank blood group/zip/etc.)
                    # contains a period and glues cleanly onto a following
                    # real code's shape too ("N.A" + "RnM-0194_4073" also
                    # fullmatches) - excluded explicitly, since it is a
                    # documented closed-vocabulary placeholder in this
                    # schema, never a real code fragment, in every
                    # instance checked.
                    if (
                        id(w2) not in consumed
                        and w2.text.upper() not in ("N.A", "N/A")
                        and (
                            w2.text == ","
                            or (
                                any(ch in ",._-" for ch in w2.text)
                                and HASH_CODE.fullmatch(w2.text + m.group(0))
                            )
                        )
                    ):
                        start = s2
            # Mirror image of the leading-fragment case - the code's own
            # trailing portion sometimes lands as a separate word after a
            # real space instead of glued (e.g. "Zc_Bqn, 3140", ground
            # truth "Zc_Bqn,_3140") - unlike the glued trailing-suffix case
            # above, _CROSS_WORD_BRIDGE doesn't span a real space.
            # Originally restricted to a bare digit run and new-layout
            # only; widened to any next word on BOTH layouts after direct
            # ground-truth comparison (DataM (4).xml) found the identical
            # pattern in old-layout data with a non-digit trailing
            # fragment too (e.g. "B.alX,_0," + "lack-37232" - ground truth
            # "B.a1X,_0,1acM-37232" - the first word alone already
            # satisfies HASH_CODE on its own, so the match stopped there
            # and silently dropped the rest of the code, repeated across 5
            # separate records). Still only extends when gluing produces a
            # complete, independently-valid HASH_CODE match - the same
            # self-validating safety net every other extension in this
            # function relies on, not a blind word grab.
            # HASH_CODE's own trailing \b means the match itself can never
            # include a trailing separator char that's immediately
            # followed by a space (non-word -> non-word is never a \b) -
            # confirmed directly: "B.alX,_0," (the complete OCR word,
            # ending in a comma) only ever matches as far as "B.alX,_0",
            # one character short, leaving that trailing comma as
            # text[end] instead of the space this check originally
            # assumed. Skipped over (not required to be a separator
            # itself - just whatever's left of THIS word) before looking
            # for the following space + word.
            bridge_end = end
            while bridge_end < len(text) and text[bridge_end] != " " and text[bridge_end] != "\n":
                bridge_end += 1
            if bridge_end < len(text) and text[bridge_end] == " ":
                next_span = next((s for s in spans if s[0] == bridge_end + 1), None)
                if next_span is not None:
                    s3, e3, w3 = next_span
                    if id(w3) not in consumed and HASH_CODE.fullmatch(text[start:bridge_end] + w3.text):
                        end = e3
            mw = [w for w in words_in_span(spans, start, end) if id(w) not in consumed]
            if mw:
                value = text[start:end]
                # Same build_indexed_text word-join artifact as the already-
                # validated phone-number fix, confirmed here too: checked
                # directly, 3 separate real records all showed a stray space
                # next to the code's own separator punctuation
                # ("BaX_01acM, -- 5168" / "BaX_01acM, -- 2741" /
                # "BaX_01acM, -- 6531" vs ground truth's "BaX_01acM,--5168"
                # etc., always with no real space in the source). Collapsed
                # the same way, generalized to every separator character this
                # code family actually uses (",", "-", "_", ".", "=").
                value = re.sub(r"\s*([,\-_.=])\s*", r"\1", value)
                # Not layout-gated: per this fix's own docstring, the
                # template it corrects is confirmed fixed regardless of
                # which document family printed it - restricting it to
                # new-layout only was never a safety requirement, just an
                # artifact of where it was first found.
                value = RuleBasedExtractionEngine._normalize_known_code_confusables(value)
                if has_base_digit or any(ch.isdigit() for ch in value):
                    found.append((m.start(), mw, value))

        # NEW-LAYOUT ONLY: a code ending in a separator character with
        # no word-char after it (e.g. "Zc_Bqn,") can never satisfy
        # HASH_CODE's own trailing \b once a real space follows - comma
        # then space is a non-word-to-non-word transition, never a word
        # boundary, so finditer finds no match there AT ALL and the
        # trailing-digit-bridge above (which only extends an
        # already-found match) never gets a chance to run. Verified
        # directly (e.g. "Zc_Bqn, 3130", ground truth "Zc_Bqn,_3130"):
        # scan adjacent word pairs directly instead, requiring the
        # rejoined pair to independently satisfy HASH_CODE.fullmatch (not
        # a blind pair grab - same shape-validation principle as the two
        # extensions above).
        if layout == "new":
            for i in range(len(spans) - 1):
                s1, e1, w1 = spans[i]
                s2, e2, w2 = spans[i + 1]
                if id(w1) in consumed or id(w2) in consumed:
                    continue
                if s2 != e1 + 1 or text[e1:s2] != " ":
                    continue
                if not w1.text or w1.text[-1] not in ",._-" or not w2.text.isdigit():
                    continue
                if not HASH_CODE.fullmatch(w1.text + w2.text):
                    continue
                value = re.sub(r"\s*([,\-_.=])\s*", r"\1", text[s1:e2])
                value = RuleBasedExtractionEngine._normalize_known_code_confusables(value)
                found.append((s1, [w1, w2], value))

        # HASH_CODE's leading \b[A-Za-z] requires a LETTER as the very
        # first character of the match - but this document family's
        # "liB_..." STM Code prefix is routinely OCR'd with BOTH leading
        # letters ("l", "i") misread as digit "1" ("11B_OaCN-1127" for
        # the real "liB_OaCN-1127"), which makes the whole word invisible
        # to the regex from the start, not just trimmed short - confirmed
        # directly via ground-truth comparison (DataM (4).xml): the
        # resulting code was missing entirely, which then also silently
        # took STM Name down with it (no stm_code_start_key to anchor
        # its own search). Restricted to this exact, confirmed 4-char
        # template prefix (never a blanket "1 means l" substitution
        # anywhere else, where a genuine digit "1" is common and
        # expected) and still only accepted when doing so reproduces a
        # complete, independently-valid HASH_CODE match.
        for s, e, w in spans:
            if id(w) in consumed or len(w.text) < 5 or w.text[:3] != "11B":
                continue
            corrected = "liB" + w.text[3:]
            if HASH_CODE.fullmatch(corrected) and any(ch.isdigit() for ch in corrected):
                found.append((s, [w], corrected))

        # A stray "#" or "'" occasionally lands inside this family's own
        # "FR-DLI09_..." Policy No prefix (confirmed real ground-truth
        # values: "F#R'-DLI09_i4259", "FR-D'LI#I09_13741") - neither
        # character is in HASH_CODE's own class, so the word fails to
        # match AT ALL, not just trimmed short, and (as traced on a real
        # record) the slot it should have filled was then wrongly taken
        # by a DIFFERENT code found elsewhere, corrupting both Policy No
        # and STM Code at once. Stripped only when doing so reproduces a
        # complete, independently-valid HASH_CODE match - same
        # self-validating guard as every other recovery here.
        for s, e, w in spans:
            if id(w) in consumed or not any(ch in "#'" for ch in w.text):
                continue
            stripped = w.text.replace("#", "").replace("'", "")
            if HASH_CODE.fullmatch(stripped) and any(ch.isdigit() for ch in stripped):
                found.append((s, [w], stripped))

        # November-dump data regularly splits a Policy No/STM Code into
        # SEVERAL separate OCR words with real spaces between them (not a
        # glued-word artifact like the cases above - confirmed directly
        # across many real records, e.g. "B.alX, _. 00laCo-5328" or
        # "ZtAO, 9vB.894359" as 2-3 separate words). Unlike the glued
        # cases, neither fragment can independently match HASH_CODE here:
        # the piece right after the space routinely starts with a digit,
        # and BOTH of HASH_CODE's alternatives require a leading letter -
        # so finditer never finds an anchor to extend from, the code goes
        # missing entirely, and a later trailing fragment (which DOES
        # start with a letter, e.g. "OlaCp_24810") gets mistaken for a
        # separate, independently-valid code and wrongly claimed as the
        # FIRST one (Policy No) instead of being part of the second
        # (STM Code) - i.e. this silently corrupts Policy No too, not
        # just STM Code.
        #
        # Recovered generally, not by hardcoding this template's exact
        # prefix spelling (would not generalize to other layouts/prefixes
        # - see this module's own policy against per-template hardcoding):
        # scans not-yet-claimed words in reading order, and for every run
        # of up to 5 adjacent words on the same line within the same
        # horizontal-gap contiguity threshold already used for city/name
        # extraction elsewhere in this file, joins their text with NO
        # space and requires the join to reproduce a complete,
        # independently-valid HASH_CODE match (same digit/density
        # safeguards as the main pass above) - exactly the same shape
        # validation already trusted for every other code in this schema,
        # so ordinary adjacent words that don't happen to join into that
        # exact shape are never swept in.
        claimed_so_far: set[int] = {id(w) for _, mw, _ in found for w in mw}
        candidates = sorted(
            ((s, w) for s, e, w in spans if id(w) not in consumed and id(w) not in claimed_so_far),
            key=lambda sw: (sw[1].line_num, sw[1].x),
        )
        i = 0
        while i < len(candidates):
            run = [candidates[i]]
            j = i + 1
            while j < len(candidates) and len(run) < 5:
                _, w = candidates[j]
                _, prev_w = run[-1]
                if w.line_num != prev_w.line_num:
                    break
                gap = w.x - prev_w.right
                # 1.25x, matching _trailing_contiguous_run's own widening
                # (see that function's comment) - confirmed via ground-
                # truth comparison this run-builder has the identical
                # too-tight-gap failure mode: "B.alX," + "_0" alone
                # already passes HASH_CODE validation, so a 1.0x cutoff
                # stopped the run right there instead of continuing to
                # the real, longer code ("B.alX,_0,1acM-37232" etc,
                # repeated truncation confirmed across 5 separate records).
                if gap > max(w.height, prev_w.height) * 1.25:
                    break
                run.append(candidates[j])
                j += 1
            advanced = False
            for length in range(len(run), 1, -1):
                sub = run[:length]
                joined = "".join(w.text for _, w in sub)
                if not HASH_CODE.fullmatch(joined):
                    continue
                if not any(ch.isdigit() for ch in joined):
                    continue
                if not any(ch in ".,_-" for ch in joined):
                    digit_count = sum(ch.isdigit() for ch in joined)
                    if digit_count / len(joined) < 0.3:
                        continue
                # A multi-word join needs real evidence it's one broken-up
                # code, not an accidental joint match of unrelated words -
                # this dataset's own synthetic name fields routinely embed
                # a digit AND an underscore (e.g. "ps_kanwar"), so digit
                # presence alone isn't enough once several words are
                # joined (confirmed false positive: "ps_kanwar" +
                # "egg0517" + "waltre" -> "ps_kanwaregg0517waltre" passes
                # every check above). Every real multi-word code observed
                # carries at least one separator character PER WORD in
                # the run (OCR tends to break these codes apart AT their
                # own separator punctuation, not in the middle of a
                # letter/digit run) - that false positive has only 1
                # separator across 3 words and is correctly rejected,
                # while every confirmed real case (e.g. "B.alX," + "_." +
                # "00laCo-3574") clears this by a wide margin.
                if length > 1:
                    sep_count = sum(1 for ch in joined if ch in ".,_-")
                    if sep_count < length:
                        continue
                mw = [w for _, w in sub]
                found.append((sub[0][0], mw, joined))
                claimed_so_far.update(id(w) for w in mw)
                i += length
                advanced = True
                break
            if not advanced:
                i += 1

        found.sort(key=lambda item: item[0])
        results = []
        for _, mw, value in found[:2]:
            for w in mw:
                consumed.add(id(w))
            # Verified against real ground-truth data: the source sometimes
            # glues the following D_B_Life_Assure date's day name directly
            # onto a code with no space (e.g. "BaX_01aC,o-2634Wednesday" ->
            # "BaX_01aC,o-2634"). Day/month names are a small closed
            # vocabulary, so stripping one off the end when present is safe
            # and unambiguous - unlike guessing at other trailing noise.
            upper = value.upper()
            for name in pat.DAY_MONTH_NAMES:
                if upper.endswith(name) and len(value) > len(name):
                    value = value[: -len(name)]
                    break
            # Verified against the real ground-truth XML directly: of 568
            # Policy No values with the "FR-DLI09_" prefix, 566 (99.6%) have
            # a lowercase "i" immediately after it - the only 2 exceptions
            # are themselves ground-truth noise (a stray "1" and a stray
            # "!"), not evidence of real template variation. This is a
            # constant code prefix, not free-form user data, and "1"/"I"/
            # "l" are classic OCR-confusable glyphs for "i" - narrow,
            # evidence-backed, and restricted to this exact template
            # position only.
            prefix = "FR-DLI09_"
            if value.upper().startswith(prefix) and len(value) > len(prefix):
                next_char = value[len(prefix)]
                if next_char in "1Il":
                    value = value[: len(prefix)] + "i" + value[len(prefix) + 1 :]
            # Verified against the dev-set ground truth: every "liB_" STM
            # Code prefix instance found so far (6/6) has a digit "0"
            # immediately after it, while OCR reads a letter "O" - the same
            # constant-prefix digit-confusable pattern as FR-DLI09_ above,
            # restricted the same way to this exact template position only.
            lib_prefix = "LIB_"
            if value.upper().startswith(lib_prefix) and len(value) > len(lib_prefix):
                next_char = value[len(lib_prefix)]
                if next_char in "oO":
                    value = value[: len(lib_prefix)] + "0" + value[len(lib_prefix) + 1 :]
            results.append((mw, value))
        return results

    @staticmethod
    def _extract_gap(
        words: list[OCRWord],
        consumed: set[int],
        start_key: tuple[int, int],
        end_key: tuple[int, int],
        max_tokens: int = 8,
    ) -> tuple[list[OCRWord], str] | None:
        """Returns the unconsumed tokens strictly between two other fields'
        already-known positions, in reading order - recovers a field by
        its POSITION in the fixed schema instead of by what it looks like.
        For a field with no reliable shape of its own (e.g. Res Address,
        STM Name), this is far more robust than guessing from a regex: two
        independently-anchored neighboring fields bound exactly where it
        must be, regardless of what text or OCR noise ends up there.

        Only the LEADING contiguous run (closest to start_key) is kept,
        stopping at the first horizontal gap wider than the words' own
        height - same mechanism proven for city/name extraction. Verified
        against real ground-truth data that end_key is often a soft,
        approximate boundary (the next field's position, not a hard field
        edge), so without this the raw window reached straight through
        Res Address into City_1/State_1/Zip_1 and beyond (e.g. "7789
        Granvilli Drivi WINSLOW NJ 8095 173 170" instead of just "7789
        Granvilli Drivi")."""
        candidates = sorted(
            (w for w in words if id(w) not in consumed and start_key < (w.line_num, w.x) < end_key),
            key=lambda w: (w.line_num, w.x),
        )
        if not candidates:
            return None
        run = [candidates[0]]
        for w in candidates[1:]:
            if w.line_num != run[-1].line_num:
                # A genuine field value can wrap onto the next physical
                # line entirely, not just stretch close to a same-line gap
                # - allowed only when `w` is the LEFTMOST word on its own
                # line (genuinely first in reading order there, not a
                # stray mid-line word this window happened to reach),
                # mirroring the same leftmost-of-line safeguard already
                # used for the 5-flag run's own cross-line continuation
                # elsewhere in this file. This is safe specifically here
                # (unlike a field with no independent right-hand anchor)
                # because `end_key` is itself another field's own
                # independently-found position - candidates are already
                # filtered to strictly before it, so crossing a line can
                # never reach past the real boundary. Confirmed real
                # (November dump, USER 13/1.jpeg record 94370029): Res
                # Address's real text wraps as "2320 Bithany Rd Avi"
                # (row 1) + "Walgrovi" (row 2, leftmost word on that row,
                # with City_1 - this call's end_key - starting right after
                # it on the SAME row), ground truth "2320 Bithany Rd Avi
                # Walgrovi".
                is_leftmost = not any(
                    o.page == w.page and o.line_num == w.line_num and o.x < w.x for o in words
                )
                if not is_leftmost:
                    break
                run.append(w)
                continue
            gap = w.x - run[-1].right
            # Widened from 0.8x to 1.0x, HERE ONLY (not the other 5 sites
            # in this file using the same 0.8x threshold - tried widening
            # all 6 together and measured a real held-out regression, -2
            # records; isolated to just this one site, held-out recovers
            # exactly while the gain here is kept): verified directly
            # against a real record ("18513 Grackli Way Rd Denair") that a
            # genuine same-phrase word gap can reach 13px against a 14px
            # word height (0.93x) - just over the old 0.8x cutoff, silently
            # dropping "Rd" from Res Address's own gap-based extraction.
            # The real field-boundary gap measured elsewhere in this file
            # is 22-45px (2x+ even this widened threshold), so this stays
            # comfortably below any confirmed real cell boundary.
            if gap > max(w.height, run[-1].height) * 1.0:
                break
            run.append(w)
        run = run[:max_tokens]
        value = _smart_join(run)
        return run, value

    @staticmethod
    def _extract_street_address(
        text: str, spans, consumed: set[int]
    ) -> tuple[list[OCRWord], str, str] | None:
        m = pat.STREET_ADDRESS.search(text)
        if m:
            mw = [w for w in words_in_span(spans, m.start(), m.end()) if id(w) not in consumed]
            if mw:
                cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
                return mw, cleaned, m.group(0)

        ordered = sorted(
            [w for _, _, w in spans if id(w) not in consumed], key=lambda w: (w.line_num, w.x)
        )
        for i, w in enumerate(ordered):
            if w.text[:1].isdigit() and len(w.text) <= 6:
                run = [w]
                # Verified against record 241064 (dev set): this had no
                # line or gap boundary at all, just "next 7 unconsumed
                # words" - with no real street address to find (this
                # record's res_address is "PO Box 605", which doesn't
                # start with a digit and so never reaches this fallback;
                # res_address proper was empty/unfound), it happily
                # walked straight across line and field boundaries into
                # height/weight/blood_group ("605 DULLES INTERNATION AL
                # AIRP 185 170 AB+"). Same contiguous-run discipline used
                # everywhere else in this file: stop at a line change or
                # a gap wider than the words' own height.
                for nxt in ordered[i + 1 : i + 8]:
                    if nxt.text.upper() in NOISE_TOKENS:
                        break
                    if nxt.line_num != run[-1].line_num:
                        break
                    if nxt.x - run[-1].right > max(nxt.height, run[-1].height) * 0.8:
                        break
                    run.append(nxt)
                if len(run) >= 3:
                    value = " ".join(x.text for x in run)
                    return run, value, value
        return None

    @staticmethod
    def _find_name_run(
        words: list[OCRWord], consumed: set[int], allow_alnum: bool = True
    ) -> tuple[list[OCRWord], str, bool] | None:
        available = [w for w in words if id(w) not in consumed]
        ordered = sorted(available, key=lambda w: (w.line_num, w.x))

        def word_like(w: OCRWord) -> bool:
            # Alphanumeric, not just alphabetic (when allow_alnum): verified
            # against real ground-truth data that real CUSTOMER names in
            # this schema sometimes embed digits (e.g. "Pomona Cn4524",
            # "Spot1 Metaphor") - a pure-alpha check made the true name run
            # invisible to this detector, falling through to a different
            # (wrong) Title-Case run instead. STM Name, by contrast, is
            # reliably a clean human name in the real data - kept pure-alpha
            # for that search (allow_alnum=False) so it isn't tempted away
            # from its own correct run by an unrelated alphanumeric one.
            t = w.text
            if len(t) < 2 or t.upper() in NOISE_TOKENS:
                return False
            if allow_alnum:
                # Verified against record 241034 (dev set): a real,
                # genuinely-repeated customer name containing a period
                # ("Eric.Satterly", the same real name shape already
                # confirmed elsewhere in this document family - e.g.
                # "Thomas.Rimmler", "H.Garland") failed isalnum() outright
                # (a period isn't alphanumeric), so EVERY occurrence of
                # that name silently never formed a word-like run at all -
                # the repeat-detection below never saw it as a candidate,
                # and a different, unrelated, non-repeating run won by
                # default instead. Matches the same allowed-character set
                # already used by the shipper_name candidate regex
                # elsewhere in this file ([A-Za-z][A-Za-z_.]+).
                return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", t))
            return t.isalpha()

        # Collect every same-line run of 2-4 consecutive word-like tokens,
        # regardless of case (a run growing from 2->3->4 words contributes
        # one entry per length, which is fine - counting below only cares
        # whether an exact sequence recurs).
        # Verified against real ground-truth data: checking only "same
        # line" let a run merge straight into unrelated adjacent text with
        # no real gap check (e.g. "Lamb103 Cmt579" - the real 2-word name,
        # repeated elsewhere for Billing Name/Name_P_Holder - absorbing a
        # directly-adjacent STM Name run into one 4-word window,
        # "Lamb103 Cmt579Hergert Kemler"). Same gap-based contiguity check
        # already proven for city/shipper/STM name extraction: stop
        # extending a run at the first horizontal gap wider than the
        # words' own height.
        runs: list[list[OCRWord]] = []
        current: list[OCRWord] = []
        prev_line = None
        for w in ordered:
            same_line = prev_line is None or w.line_num == prev_line
            # Widened from 0.8x to 1.0x, HERE ONLY (same precedent as the
            # city-extraction contiguous walk elsewhere in this file) -
            # real, confirmed need (November dataset): a genuinely
            # repeated 2-3 word customer name's own real inter-word gap
            # measured 11-12px against an 11-12px word height (ratio
            # 0.92-1.0), just over the old 0.8x cutoff, in EVERY one of
            # its 3 occurrences in the same record ("tedjustok
            # micheal246") - each occurrence then became two isolated
            # single words instead of one run, invisible to the repeat-
            # detection below entirely, and a completely unrelated
            # Title-Case fragment ("Helen Frederick", itself actually the
            # real Shipper Name) won by default instead, which then also
            # consumed the words Shipper Name's own extraction needed.
            # Verified safe against the real 81-file + 44-image corpora:
            # only ONE already-found repeated name changed, and that was
            # a genuine extension (a real 3rd repeated word, "ekristina",
            # recovered in front of an already-correct "wang mile" - the
            # same gap-measurement issue, not a new one) - zero
            # corruptions of an already-correct result.
            close_enough = (
                not current or w.x - current[-1].right <= max(w.height, current[-1].height) * 1.0
            )
            if word_like(w) and same_line and close_enough:
                current.append(w)
            else:
                current = [w] if word_like(w) else []
            if 2 <= len(current) <= 4:
                runs.append(list(current))
            prev_line = w.line_num

        # Verified against the source image: this document's actual names
        # print in lowercase and get reused across Customer Name/Billing
        # Name/Shipper Name/Name_P_Holder (e.g. "kthkt amneet" appearing 3x
        # in one record) - a Title-Case-only search skips straight past
        # them and grabs an unrelated capitalized address fragment instead
        # (e.g. "Park Avi Apt"). An exact sequence repeating 2+ times is a
        # far stronger name signal than capitalization shape alone, so it's
        # tried first; the longest repeated run wins if more than one repeats.
        # Tried folding classic OCR confusables (0/O, 1/I/L) into the repeat-
        # detection key, on the theory that the same physical name printed
        # 2-3 times sometimes differs between occurrences by exactly one
        # such character (e.g. "Dlcrouse" vs "DIcrouse"). Measured on real
        # held-out data: it did not fix the targeted case (the real
        # mismatch there has a different cause, not yet found) and
        # introduced a new one elsewhere ("Epike Dlb2" vs "Dib2" - folding
        # merged two genuinely different short alnum tokens). Reverted; a
        # net regression, not a net improvement.
        counts: dict[tuple[str, ...], int] = {}
        for run in runs:
            key = tuple(w.text.upper() for w in run)
            counts[key] = counts.get(key, 0) + 1
        repeated = [run for run in runs if counts[tuple(w.text.upper() for w in run)] >= 2]
        if repeated:
            best = max(repeated, key=len)
            return best, _smart_join(best), True

        # Fall back to the longest Title Case run - covers document
        # families/records where real names ARE capitalized normally and
        # nothing repeats (the original heuristic).
        # Verified against real ground-truth data that this was matching
        # ALL-CAPS place-name noise too (e.g. "BOULDER CREEK", "SAN ANTONIO",
        # "WASHINGTON DT") as if it were a name, because checking only the
        # first letter doesn't distinguish "Title Case" from "ALL CAPS" -
        # both have an uppercase first letter. Words of 3+ letters that are
        # entirely uppercase are excluded; the length floor keeps real
        # single/double-letter middle initials (e.g. the "D" in "Hector D
        # Villalavazo") from being excluded too.
        best_run: list[OCRWord] = []
        current = []
        prev_line = None
        for w in ordered:
            is_title = (
                word_like(w)
                and w.text[0].isupper()
                and not (len(w.text) >= 3 and w.text.isupper())
            )
            same_line = prev_line is None or w.line_num == prev_line
            close_enough = (
                not current or w.x - current[-1].right <= max(w.height, current[-1].height) * 0.8
            )
            if is_title and same_line and close_enough:
                current.append(w)
            else:
                current = [w] if is_title else []
            if len(current) > len(best_run):
                best_run = current
            prev_line = w.line_num
        if 2 <= len(best_run) <= 4:
            return best_run, _smart_join(best_run), False
        return None

    @staticmethod
    def _trailing_contiguous_run(
        candidates: list[OCRWord], max_count: int = 4, allow_line_wrap: bool = False
    ) -> list[OCRWord]:
        """Among words already filtered to "name-shaped", keep only the
        trailing contiguous phrase closest to the right-hand anchor, instead
        of everything found in a (possibly wide) search window.

        Verified against real ground-truth data that Shipper Name and STM
        Name's search windows often contain leftover noise from OTHER,
        unrelated fields earlier in the window (whatever field precedes them
        failed to fully consume its own text in this OCR pass) - e.g.
        "Erogers Lordinfamous David H Worley" where only "David H Worley" is
        the real name. The noise and the real name are reliably separated by
        a horizontal gap much wider than the gap between two words of the
        same name/phrase (same mechanism already proven for city extraction -
        see _extract_city), so walking backward from the anchor and stopping
        at the first such gap isolates just the real value.
        """
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda w: (w.line_num, w.x))
        # NOTE: tried a cross-line wrap allowance here (mirroring the
        # 5-flag run's own precedent), gated to only fire when the
        # run's leading word is leftmost on its own line - fixed the
        # targeted case (STM Name "chhayag kprincess10" wrapping before
        # STM Code's anchor on 752.jpeg record 29610274) but measured a
        # much larger net regression on Shipper Name (63.6% -> 52.7%):
        # its search window is noisier (leftover unconsumed text from
        # other fields, per this function's own docstring), and the
        # leftmost-word-on-next-line condition is satisfied far more
        # often there by unrelated leading words ("joan.", "wan",
        # "unitLani", address fragments) than by genuine wraps. Reverted at
        # the time to an unconditional same-line boundary for all three
        # callers (Shipper Name, STM Name, City_2) rather than trading a
        # worse field for a better one.
        #
        # Re-added since, but opt-in per caller (allow_line_wrap, default
        # False) instead of shared behavior - STM Name specifically asked
        # for it again (confirmed real, November dump USER 06/1758.jpeg
        # records 412944/412945: "Lianlieb Tattyf" and "Jsobell
        # Scott.Shipman" each wrap mid-name onto the next physical line,
        # STM Name silently blank both times), and Shipper Name's own
        # extraction has since changed enough (the city_2-reclaim fallback
        # below, the dup_name_tokens filter) that re-measuring found no
        # reason to believe the original regression would still reproduce
        # there - but Shipper Name and City_2 are left at the proven-safe
        # default rather than assuming that without new evidence of their
        # own.
        leftmost_ids: set[int] = set()
        if allow_line_wrap:
            by_line: dict[int, list[OCRWord]] = {}
            for w in ordered:
                by_line.setdefault(w.line_num, []).append(w)
            for line_words in by_line.values():
                leftmost_ids.add(id(min(line_words, key=lambda w: w.x)))
        run = [ordered[-1]]
        for w in reversed(ordered[:-1]):
            if w.line_num != run[0].line_num:
                if allow_line_wrap and id(w) in leftmost_ids:
                    run.insert(0, w)
                    continue
                break
            gap = run[0].x - w.right
            # Widened from 0.8x to 1.0x, then to 1.25x - same real,
            # confirmed need as _find_name_run's identical widening (see
            # that function's own comment): a genuine 2-word Shipper/STM
            # Name's real inter-word gap measured 12px against a 12px word
            # height (ratio 1.0, exactly at the old cutoff's edge) in one
            # real record ("cherrylynne edikis"). Verified via direct
            # ground-truth comparison (DataM (4).xml, November dump) that
            # 1.0x was STILL too tight for two more real cases: "pymz
            # tczyy" (gap 13px / height 11px, ratio 1.18) and "ankhhathor
            # tchadb" (gap 12px / height 11px, ratio 1.09) - both were
            # isolated into single-word runs, failing the >=2 check and
            # falling through to an unbounded fallback that then grabbed
            # unrelated text (confirmed: Card Name's own "Platinum Card"
            # ended up as STM Name instead).
            if gap > max(w.height, run[0].height) * 1.25:
                break
            run.insert(0, w)
        return run[-max_count:]

    def _extract_names(
        self,
        words,
        consumed,
        candidates,
        claim,
        state_2_key=None,
        city2_start_key=None,
        stm_code_start_key=None,
        policy_no_start_key=None,
        city2_mw=None,
    ) -> None:
        """Per BN_Workflow.pdf: Billing Name must match Customer Name (rule
        16), and Name_P_Holder is explicitly "Same as Customer Name" (rule
        30) - both are true duplicates of the primary name, not independent
        values. STM Name (rule 31) is a genuinely separate name.

        Shipper Name (rule 17) is genuinely independent, and its values have
        no shape signal (mixed case, underscores, e.g. "weksler_m kotowskia
        Lani"). It is located by schema ORDER instead: it is the name-like text
        printed after Billing Name and before City_2 in reading order (City_2
        is the field that structurally follows it - using City_2's own start
        position as the right boundary when known, falling back to State_2's
        position otherwise, since City_2 sometimes isn't itself found/consumed).
        STM Name similarly sits right before STM Code in schema order, so
        STM Code's own position (when found) anchors its right boundary too.
        Both searches take only the trailing contiguous run nearest their
        right-hand anchor (see _trailing_contiguous_run) rather than every
        name-shaped word in the window, since real Shipper/STM names have no
        shape of their own and the window can span other fields' leftover
        unconsumed text. Verified against real ground-truth data that the
        window also catches place-name noise the form prints in ALL CAPS
        (city/street fragments City_2 extraction failed to consume) - those
        are excluded outright, since every confirmed real Shipper/STM Name
        value has lowercase letters somewhere in it and never prints in all
        capitals.
        """
        primary = self._find_name_run(words, consumed)
        if not primary:
            return
        mw, value, repeated = primary
        # A name confirmed by exact repetition across multiple name-slot
        # positions is strong evidence even from a single OCR pass - boosted
        # so it can win cross-pass aggregation against other passes that
        # only found a shape-based (Title Case) guess (see _find_name_run).
        conf = min(100.0, average_confidence(mw) * 1.3) if repeated else None
        claim("customer_name", mw, value, value, confidence_override=conf)

        dup_note = "Source does not clearly distinguish this from Customer Name - duplicated, please verify"
        claim("billing_name", mw, value, value, note=dup_note, confidence_override=conf)
        claim("name_p_holder", mw, value, value, note=dup_note, confidence_override=conf)

        # Prefer City_2's own position (tightest, most accurate boundary),
        # then State_2's, then - verified against real ground-truth data
        # that City_2 and State_2 are BOTH missing in a large fraction of
        # records (neither found at all, not just wrong) - fall back to
        # Policy No's position (found independently via hash-code
        # detection, structurally after Shipper Name/City_2/State_2/Zip_2/
        # Country_2/PhNo_2/the 5 flags). This widens the search window
        # substantially, but the trailing-contiguous-run walk already
        # isolates just the real name regardless of window width (the same
        # mechanism that made the tight City_2 case work).
        # Verified against a real image crop (record 241114, 09112.jpeg):
        # the form reprints Billing Name/Name_P_Holder's own text directly
        # on the SAME line as Shipper Name and STM Name ("Enkay1 Matschek
        # Fredrick Beckman", "Enkay1 Matschek Janica Louise Fallon") as part
        # of the same duplicate-field pattern already confirmed for dates -
        # that redundant occurrence is never consumed elsewhere, so it sits
        # right there as more Title-Case, non-all-caps "name-shaped" text,
        # indistinguishable by shape or gap width alone from the real
        # Shipper/STM Name that follows it. Since the duplicate name's own
        # text is already known at this point, excluding its exact tokens
        # is a precise, narrow filter - not a guess at what a city or name
        # "looks like" (a US place-name reference would not have caught
        # this: "Matschek" is a surname, not a place).
        dup_name_tokens = {t.lower() for t in value.split()}
        anchor = max((w.line_num, w.x) for w in mw)
        # City_2/State_2 (when valid - genuinely ahead of Billing Name's
        # own position) are structurally reliable, TIGHT boundaries and
        # stay preferred first, same as before. Only when NEITHER is
        # available does this fall back to a hash-coded anchor
        # (Policy No/STM Code) - and now tries BOTH, keeping the FARTHER
        # of the two VALID ones rather than assuming Policy No alone is
        # always safe to use. Real, confirmed bug (November dataset,
        # MI_New T(132).jpeg record 1): Policy No's own hash-coded token
        # was sometimes printed BEFORE Shipper Name's real value in this
        # document family (not safely after it, as originally assumed for
        # a different, already-verified family) - picking it anyway built
        # an inverted/empty window (anchor > right_boundary) at best, or a
        # window that cut off before the real name at worst, so Shipper
        # Name came back silently blank. STM Code, the OTHER
        # independently-detected hash-coded anchor, was still genuinely
        # past the real name in that same record. Taking the FARTHER of
        # the two (not the nearer) matches this fallback's own original
        # design intent, documented below: a hash-coded position is
        # expected to widen the window substantially, with the trailing-
        # contiguous-run walk doing the real precision work, not the
        # window's own tightness.
        if city2_start_key is not None and city2_start_key > anchor:
            right_boundary = city2_start_key
        elif state_2_key is not None and state_2_key > anchor:
            right_boundary = state_2_key
        else:
            hash_candidates = [
                k for k in (policy_no_start_key, stm_code_start_key) if k is not None and k > anchor
            ]
            # MIN, not MAX, of the valid candidates - confirmed via direct
            # ground-truth comparison (DataM (4).xml) that MAX over-widens
            # the window when BOTH Policy No and STM Code sit on the same
            # later line as STM Name's own real value: the window then
            # reaches straight past Shipper Name's true boundary and the
            # trailing-run walk picks STM Name's text (closer to the
            # window's far edge) instead, swapping the two fields' values.
            # Safe against the ORIGINAL reason this used MAX (Policy No
            # printed before Shipper Name's real value, in a different
            # record/family): the `k > anchor` filter above already drops
            # an out-of-order Policy No from `hash_candidates` entirely in
            # that case, leaving only STM Code - MIN and MAX are identical
            # on a single-element list, so that fix is untouched.
            right_boundary = min(hash_candidates) if hash_candidates else None
        if right_boundary is not None:
            candidates_ship = [
                w
                for w in words
                if id(w) not in consumed
                and anchor < (w.line_num, w.x) < right_boundary
                # Verified against record 241018 (dev set): "Bruce D
                # Mcdonald" - a bare single-letter middle initial with no
                # period, a confirmed real name shape in this dataset
                # (First MiddleInitial Last, no punctuation) - was
                # invisible to this filter two ways at once: the regex
                # required 2+ characters, and a lone capital letter
                # trivially satisfies isupper() the same way an ALL-CAPS
                # noise word like "WINSLOW" does, even though the isupper
                # check exists to exclude THOSE, not initials. With "D"
                # excluded, the gap measurement jumped straight from
                # "Bruce" to "Mcdonald", exceeding the contiguous-run
                # threshold and breaking the name in two. Both relaxed:
                # a single character is now a valid candidate, and the
                # ALL-CAPS exclusion only applies to words long enough to
                # actually be place-name noise. Counting ALPHABETIC
                # characters specifically, not raw length, for that last
                # check: a period-punctuated initial like "B." has
                # len()==2 but only one real letter, and
                # "B.".isupper()==True (Python ignores the non-cased
                # period) - the original len>=2 guard still wrongly
                # excluded this shape (confirmed via "Claude B. Vicars",
                # record 241017, dev set).
                # Same precedent and reasoning as _find_name_run's
                # allow_alnum=True: real Shipper Name values in this schema
                # sometimes embed digits (e.g. "Rogertmoorec21", "Bohemian3"
                # - originally confirmed in new-layout ground truth). The
                # original letters-only class silently excluded every one
                # of these from the candidate pool, leaving shipper_tokens
                # empty and Shipper Name entirely unclaimed. Confirmed the
                # identical need in old-layout November-dump data too (not
                # new-layout-specific after all): once City_2's own
                # extraction was fixed to stop silently absorbing stray
                # text, several records' real Shipper Name value turned out
                # to be a single digit-laden synthetic token (e.g.
                # "celis86"), which the letters-only class was rejecting
                # outright, leaving the candidate pool empty.
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", w.text)
                and w.text.upper() not in NOISE_TOKENS
                and not (sum(ch.isalpha() for ch in w.text) >= 2 and w.text.isupper())
                and w.text.lower() not in dup_name_tokens
            ]
            shipper_tokens = self._trailing_contiguous_run(candidates_ship)
            # Verified against record 241029 (dev set): Shipper Name
            # itself sometimes wraps across the line break ("David" ends
            # one line, "Martowska" starts the next), the same way
            # City_2 does - _trailing_contiguous_run can't cross a line
            # boundary, so the wrap left only "Martowska" (1 word),
            # failing the >=2 check entirely. A dedicated, narrowly-
            # scoped recovery (not a change to the shared helper, which
            # also serves the unrelated city_2 recovery below): when
            # exactly one token remains, look for a continuation on the
            # line directly above, reusing the SAME already-filtered
            # candidates_ship pool (not a fresh raw-word search) - this
            # is the key difference from an earlier, reverted city_2 fix
            # that searched raw ALL-CAPS words on the row above and
            # ended up stealing unrelated fields' text. Restricted to
            # the rightmost candidate on that line, and only when the
            # remaining token is itself the leftmost candidate on its
            # own line (confirming it genuinely starts a wrapped run,
            # not a mid-row fragment).
            if len(shipper_tokens) == 1:
                only = shipper_tokens[0]
                same_line_candidates = [w for w in candidates_ship if w.line_num == only.line_num]
                if same_line_candidates and min(same_line_candidates, key=lambda w: w.x).x == only.x:
                    prev_line_candidates = [w for w in candidates_ship if w.line_num == only.line_num - 1]
                    if prev_line_candidates:
                        wrap_word = max(prev_line_candidates, key=lambda w: w.x)
                        shipper_tokens = [wrap_word, only]
            if len(shipper_tokens) >= 2:
                shipper_value = _smart_join(shipper_tokens)
                claim("shipper_name", shipper_tokens, shipper_value, shipper_value)

            # City_2 sometimes wraps onto a different visual line than
            # State_2 (verified against real ground-truth data: Shipper
            # Name + City_2 share one line, State_2 starts the next), which
            # _extract_city's same-line-as-state check can't reach. Recover
            # it the same way Shipper Name itself was recovered: the
            # trailing run between the end of Shipper Name and State_2's
            # own position - not restricted to ALL CAPS (verified against
            # record 241074: "livingston" wraps this way in ordinary
            # lowercase, same as the Title Case minority
            # _extract_city_fallback already accounts for in the same-line
            # case). Not nested inside the shipper_tokens>=2 check above:
            # verified against the same record that Shipper Name can be
            # genuinely absent (its own row held only an explicit "N/A"
            # annotation) while City_2 still wrapped onto that row - the
            # two are coupled by row position, not by whether Shipper Name
            # itself was found, so this uses the primary name's own end
            # position as a fallback left boundary when shipper_tokens is
            # empty.
            if candidates.get("city_2") is None and state_2_key is not None:
                city2_left = max((x.line_num, x.x) for x in shipper_tokens) if shipper_tokens else anchor
                city2_candidates = [
                    w
                    for w in words
                    if id(w) not in consumed
                    and city2_left < (w.line_num, w.x) < state_2_key
                    and self._looks_alphabetic_city_word(w.text)
                    and not self._is_city_noise_word(w.text)
                ]
                city2_run = self._trailing_contiguous_run(city2_candidates)
                if city2_run:
                    city2_value = _smart_join(city2_run)
                    claim("city_2", city2_run, city2_value, city2_value)

        # City_2's own extraction (above, much earlier in this method) has
        # no concept of "stop, the rest belongs to Shipper Name" - when
        # nothing above managed to locate Shipper Name at all, a MULTI-WORD
        # City_2 match is itself the strongest available evidence that it
        # absorbed Shipper Name's own text too (a real US city name sits in
        # City_2 ALONE in the large majority of real records; Shipper Name,
        # by contrast, is reliably 2+ words and has no shape signal of its
        # own to find separately once City_2's greedy backward walk has
        # already consumed every word up to State_2). Confirmed real
        # (November dump, USER 06/1758.jpeg record 412948 and multiple
        # USER 10/USER 13 records): City_2 absorbed "Lkgbvayalil Sbowman
        # Ramsey" (3 words) with Shipper Name entirely missing; the real
        # City_2 is just "Ramsey", the LAST word, with "Lkgbvayalil
        # Sbowman" belonging to Shipper Name. Reassigns the leading words
        # to Shipper Name and shrinks City_2 to its own last word, rather
        # than leaving Shipper Name silently blank. Deliberately checked
        # LAST (after every other recovery above has had its chance) and
        # gated on Shipper Name still being completely unclaimed - this
        # only fires when nothing else found a better explanation.
        if candidates.get("shipper_name") is None and city2_mw and len(city2_mw) > 1:
            reclaimed = sorted(city2_mw, key=lambda w: (w.line_num, w.x))
            new_shipper = reclaimed[:-1]
            new_city2 = [reclaimed[-1]]
            if len(new_shipper) >= 2:
                shipper_value = _smart_join(new_shipper)
                claim("shipper_name", new_shipper, shipper_value, shipper_value)
                city2_value = new_city2[0].text
                claim("city_2", new_city2, city2_value, city2_value)

        if stm_code_start_key is not None:
            candidates_stm = [
                w
                for w in words
                if id(w) not in consumed
                and (w.line_num, w.x) < stm_code_start_key
                # This filter required a bare w.text.isalpha(), stricter than
                # the equivalent, already-proven shipper-name filter just
                # above (which allows a trailing period/underscore and
                # counts ALPHABETIC characters rather than raw length for
                # the all-caps/noise exclusion, fixed earlier for "B."-style
                # middle initials - see that filter's own comment). Verified
                # directly against two real source images this was the same
                # gap-inflation bug there: "G." (record 241037, 09104.jpeg:
                # "G. Michael Robinson", 4px real gaps throughout) and "&"
                # (record 241039, same image: "Carl & Bernadette Perry", 3px
                # real gaps throughout) were both silently dropped from the
                # candidate pool for failing isalpha(), which then made
                # _trailing_contiguous_run measure an artificially wide gap
                # to whatever real word was kept next and stop there,
                # truncating the name. Aligned to the shipper filter's shape
                # plus "&" (not needed by shipper, but confirmed needed here).
                # Same digit-allowance fix as the shipper-name filter above
                # (kept in sync per this function's own comment that the
                # two filters are meant to be equivalent), and confirmed
                # needed in old-layout November-dump data too for the same
                # reason: real STM Name values in this schema are
                # sometimes a single digit-laden synthetic token, which the
                # original letters-only class rejected outright.
                and (w.text == "&" or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", w.text))
                and w.text.upper() not in NOISE_TOKENS
                and not (sum(ch.isalpha() for ch in w.text) >= 2 and w.text.isupper())
                and w.text.lower() not in dup_name_tokens
                # Real, confirmed bug (DataM (4).xml records 29610288/
                # 29610337/29610349): "Platinum Card"/"Dankort Card" -
                # CARD Name's own fragments - are both Title-Case and
                # alphabetic, so with no better candidate nearby they get
                # swept into STM Name instead. "CARD" itself is never part
                # of a real STM Name in this schema; excluding it (and any
                # full CARD_TYPES phrase) is a vocabulary-based filter, the
                # same precedent as the existing ALL-CAPS place-name and
                # dup-name-token exclusions just above.
                and w.text.upper() != "CARD"
                and w.text.upper() not in pat.CARD_TYPES
            ]
            stm_tokens = self._trailing_contiguous_run(candidates_stm, allow_line_wrap=True)
            # NEW-LAYOUT ONLY: same wrapped-line recovery as the
            # shipper-name case above, for the same reason (verified
            # directly, e.g. record 402428: "Kscabell" ends one line,
            # "Ssubherwal" starts the next - ground truth "Kscabell
            # Ssubherwal", only "Ssubherwal" was kept without this). Old
            # layout's own Shipper Name wrap case (see above, record
            # 241029) is a separate, pre-existing fix not affected by this.
            if self.layout == "new" and len(stm_tokens) == 1:
                only = stm_tokens[0]
                same_line_candidates = [w for w in candidates_stm if w.line_num == only.line_num]
                if same_line_candidates and min(same_line_candidates, key=lambda w: w.x).x == only.x:
                    prev_line_candidates = [w for w in candidates_stm if w.line_num == only.line_num - 1]
                    if prev_line_candidates:
                        wrap_word = max(prev_line_candidates, key=lambda w: w.x)
                        stm_tokens = [wrap_word, only]
            if len(stm_tokens) >= 2:
                stm_value = _smart_join(stm_tokens)
                claim("stm_name", stm_tokens, stm_value, stm_value)
                return

        # Verified against record 241044 (dev set): with no stm_code
        # anchor to search against, this fallback re-runs the same
        # repeat-detection used for the primary name from scratch - with
        # no awareness that the primary name's own duplicate-printed
        # occurrence is still sitting unconsumed elsewhere (only the ONE
        # occurrence actually claimed by customer_name gets consumed), so
        # it re-discovers the already-claimed name as "repeated" again
        # and wrongly assigns it to stm_name too. Excluding the known
        # duplicate name's own tokens here, the same way the stm_code-
        # anchored path above already does.
        secondary_words = [
            w for w in words
            if w.text.lower() not in dup_name_tokens
            and w.text.upper() != "CARD"
            and w.text.upper() not in pat.CARD_TYPES
        ]
        secondary = self._find_name_run(secondary_words, consumed, allow_alnum=False)
        if secondary:
            mw2, value2, repeated2 = secondary
            conf2 = min(100.0, average_confidence(mw2) * 1.3) if repeated2 else None
            claim("stm_name", mw2, value2, value2, confidence_override=conf2)


class AIExtractionEngine(ExtractionEngine):
    """Placeholder for an optional, future AI/LLM-assisted extraction engine.

    Not wired into the application and not used by default. Sending document
    content to an external AI provider requires an explicit, separate
    integration decision - see app/core/config.py `ai_extraction_enabled` -
    and must never happen implicitly. Left unimplemented intentionally.
    """

    def extract(self, ocr_result: OCRPassResult) -> dict[str, FieldCandidate]:
        raise NotImplementedError(
            "AI-assisted extraction is not enabled in this prototype. "
            "Enable and implement explicitly via app/core/config.py if approved."
        )


class HybridExtractionEngine(ExtractionEngine):
    """Runs the rule-based engine first; would defer to AIExtractionEngine only
    for fields the rule-based pass could not resolve, if/when that engine is
    implemented and enabled. Currently behaves identically to RuleBasedExtractionEngine.
    """

    def __init__(self) -> None:
        self._rule_based = RuleBasedExtractionEngine()

    def extract(self, ocr_result: OCRPassResult) -> dict[str, FieldCandidate]:
        return self._rule_based.extract(ocr_result)
