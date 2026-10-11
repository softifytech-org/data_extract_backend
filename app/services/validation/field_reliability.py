"""Per-field confidence thresholds for auto-acceptance, calibrated from
measured held-out accuracy - NOT a static guess.

The rest of the validation pipeline uses one global confidence bar
(Settings.confidence_high_threshold) for every field equally. That's wrong:
a field that measures 95%+ exact-match accuracy against real ground truth
across many held-out documents has earned the right to be auto-accepted at
a lower raw confidence score than a field that measures 20% - requiring the
SAME bar for both either makes the reliable field get needlessly flagged
for review constantly, or lets the unreliable field slip through just
because its OCR confidence happened to look fine.

THRESHOLDS below are deliberately left at the global default (meaning: no
calibration applied yet) until a real held-out benchmark run populates
them via `update_from_benchmark`. Never hand-set a number here without a
measured run backing it - that's exactly the "estimate reported as
measured accuracy" this project has repeatedly had to correct course on.
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "field_reliability_thresholds.json"

# Fields whose measured false-acceptance rate (of values currently marked
# VERIFIED that are actually wrong) exceeded the agreed 5% safety limit by a
# wide margin on held-out data, even though their raw score already clears
# the global confidence bar - meaning the score does not reliably predict
# correctness for these fields (see PHASE5_FINAL_REPORT.md: stm_name 38-55%
# FAR, shipper_name 17-43%, stm_code 30-36%, customer_name/billing_name/
# name_p_holder 25-32%, res_address 22-24%, email_address 13-16%, zip_2 12-
# 15%). remark is included unconditionally: it is derived from the other 43
# fields' own validation status and has no independent confidence signal of
# its own. Routed to mandatory review regardless of score - raising the
# threshold further does not help when a wrong value already clears it most
# of the time (see services/validation/confidence.py's ocr_agreement
# component, which is architecturally maxed for any single-OCR-pass source
# and so cannot discriminate correct from incorrect here). Only add a field
# here backed by a measured FAR run, never a guess - the whole point of this
# list is to stop pretending the score means something it doesn't.
#
# policy_no added after a FRESH held-out confirmation (round 8, 25 images
# never used for tuning) measured 24.5% FAR - well above both the 5% bar
# and the earlier dev-set-only estimate of 11.8%. Exactly the dev-set-vs-
# held-out gap this project's discipline exists to catch: a field must not
# be treated as safe on development-set results alone (see RESULTS_SUMMARY.md
# round 8).
ALWAYS_REVIEW_FIELDS: frozenset[str] = frozenset({
    "remark",
    "stm_name",
    "shipper_name",
    "stm_code",
    "customer_name",
    "billing_name",
    "name_p_holder",
    "res_address",
    "email_address",
    "zip_2",
    "policy_no",
})


def load_thresholds() -> dict[str, float]:
    """field_name -> confidence score (0-100) required for auto-accept.
    A field with no entry here uses the global default
    (Settings.confidence_high_threshold) - i.e. uncalibrated, not
    "known reliable"."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_thresholds(thresholds: dict[str, float]) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, sort_keys=True)


def threshold_for_field(field_name: str, default: float) -> float:
    thresholds = load_thresholds()
    return thresholds.get(field_name, default)


def calibrate_from_field_accuracy(
    field_accuracy: dict[str, float],
    *,
    min_records: dict[str, int] | None = None,
    min_sample_size: int = 30,
) -> dict[str, float]:
    """Derives an auto-accept threshold per field from measured held-out
    exact-match accuracy (0-100 per field, from the benchmark scorer).

    A field is only calibrated below the global default if it has BOTH a
    high measured accuracy AND enough held-out samples to trust that
    number (min_sample_size) - a field that looks 100% accurate on 3
    records proves nothing. Fields below the sample-size bar, or below a
    96% measured-accuracy bar, are left uncalibrated (fall back to the
    global default, i.e. still require full confidence + human review).

    The mapping from measured accuracy to a confidence threshold is
    intentionally conservative: even a field measured at 99% accuracy
    still requires a respectable 80 confidence score to auto-accept -
    this lowers the bar for a PROVEN-reliable field, it does not remove
    the bar entirely.
    """
    min_records = min_records or {}
    thresholds: dict[str, float] = {}
    for field, acc in field_accuracy.items():
        n = min_records.get(field, 0)
        if n < min_sample_size:
            continue
        if acc >= 99.0:
            thresholds[field] = 75.0
        elif acc >= 96.0:
            thresholds[field] = 85.0
        elif acc >= 90.0:
            thresholds[field] = 92.0
        # Below 90% measured accuracy: no calibration: fields such as
        # Res Address / Shipper Name stay at the global default (95) and
        # therefore continue to need review on effectively every record,
        # which is the correct, honest behavior until they're actually fixed.
    return thresholds
