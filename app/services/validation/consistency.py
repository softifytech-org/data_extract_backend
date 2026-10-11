"""Cross-field / business-rule validation.

Answers "are these values consistent with each other?" - a field can pass
its own format check and still be wrong, which only shows up when compared
against a related field (e.g. an OCR-mangled total that doesn't match
quantity x unit price). Extensible: add another function + register it in
run_consistency_checks to add a rule without touching anything else.

Note on _check_tablets_cost: the source document's Cost is authoritative
and is NOT expected to always equal tablets x pill_rate exactly (real
billing can include discounts, fees, etc.) - a mismatch is flagged as a
"possible inconsistency" for human review, never auto-corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CURRENCY_TOLERANCE = 0.5  # dollars - allows for small rounding/OCR noise


def _as_float(value: str) -> float:
    # Currency values now keep their "$" (source-fidelity - see
    # extraction/parser.normalize_currency); strip it just for this
    # internal arithmetic check, never for the stored value itself.
    return float(value[1:] if value.startswith("$") else value)


@dataclass
class ConsistencyResult:
    field_validity: dict[str, bool] = field(default_factory=dict)  # field_name -> business_rule_valid
    warnings: list[str] = field(default_factory=list)


def _check_tablets_cost(values: dict[str, str | None], result: ConsistencyResult) -> None:
    tablets, rate, cost = values.get("tablets"), values.get("pill_rate"), values.get("cost")
    if not (tablets and rate and cost):
        return
    try:
        expected = int(tablets) * _as_float(rate)
        actual = _as_float(cost)
    except ValueError:
        return
    consistent = abs(expected - actual) <= CURRENCY_TOLERANCE
    for f in ("tablets", "pill_rate", "cost"):
        result.field_validity[f] = consistent
    if not consistent:
        result.warnings.append(
            f"tablets x pill_rate ({tablets} x {rate} = {expected:.2f}) does not match "
            f"cost ({actual:.2f}) - possible amount inconsistency"
        )


def _check_cost_shipping_total(values: dict[str, str | None], result: ConsistencyResult) -> None:
    cost, shipping, total = values.get("cost"), values.get("shipping_cost"), values.get("total_amount")
    if not (cost and shipping and total):
        return
    try:
        expected = _as_float(cost) + _as_float(shipping)
        actual = _as_float(total)
    except ValueError:
        return
    consistent = abs(expected - actual) <= CURRENCY_TOLERANCE
    for f in ("cost", "shipping_cost", "total_amount"):
        result.field_validity[f] = result.field_validity.get(f, True) and consistent
    if not consistent:
        result.warnings.append(
            f"cost + shipping_cost ({cost} + {shipping} = {expected:.2f}) does not match "
            f"total_amount ({actual:.2f})"
        )


def _check_customer_billing_name(values: dict[str, str | None], result: ConsistencyResult) -> None:
    # BN_WorkFlow.pdf rule 16: "Customer Name and Billing Name must match" -
    # a hard equality requirement, not the best-effort duplication guess
    # _extract_names() falls back to when only one name run is found.
    customer, billing = values.get("customer_name"), values.get("billing_name")
    if not (customer and billing):
        return
    consistent = customer.strip().casefold() == billing.strip().casefold()
    result.field_validity["customer_name"] = consistent
    result.field_validity["billing_name"] = consistent
    if not consistent:
        result.warnings.append(
            f"customer_name ({customer}) does not match billing_name ({billing}) - they must be identical"
        )


RULES = (_check_tablets_cost, _check_cost_shipping_total, _check_customer_billing_name)


def run_consistency_checks(values: dict[str, str | None]) -> ConsistencyResult:
    result = ConsistencyResult()
    for rule in RULES:
        rule(values, result)
    return result
