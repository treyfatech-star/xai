"""Request validation for the REST API (plain-English error messages)."""

from __future__ import annotations

from datetime import date
from typing import Any

from procure.config import load_rules_config, load_settings
from procure.data import geo
from procure.domain import ProcurementRequest


class ValidationError(ValueError):
    """Raised with a list of problems in the submitted data."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


REQUIRED = ["item", "item_category", "quantity", "delivery_state", "delivery_lga", "method", "budget"]


def parse_recommend(body: dict[str, Any] | None) -> ProcurementRequest:
    """Validate a POST /recommend body and build a ProcurementRequest."""
    body = body or {}
    errors = [f"'{k}' is required" for k in REQUIRED if body.get(k) in (None, "")]
    if errors:
        raise ValidationError(errors)
    try:
        qty, budget = float(body["quantity"]), float(body["budget"])
        if qty <= 0 or budget <= 0:
            errors.append("quantity and budget must be positive")
    except (TypeError, ValueError):
        errors.append("quantity and budget must be numbers")
        qty = budget = 0.0
    state = geo.canonical_state(body["delivery_state"])
    if state is None:
        errors.append(f"delivery_state must be one of {load_settings()['region']['states']}")
    lga = geo.canonical_lga(state, body["delivery_lga"]) if state else None
    if state and lga is None:
        errors.append(f"delivery_lga '{body['delivery_lga']}' is not a known LGA of {state}")
    methods = list(load_rules_config()["params"]["method_thresholds"])
    if body["method"] not in methods:
        errors.append(f"method must be one of {methods}")
    try:
        d = date.fromisoformat(str(body.get("date") or date.today().isoformat()))
    except ValueError:
        errors.append("date must be YYYY-MM-DD")
        d = date.today()
    k = body.get("k_max")
    if k is not None and (not isinstance(k, int) or k < 1):
        errors.append("k_max must be a positive whole number")
    if errors:
        raise ValidationError(errors)
    return ProcurementRequest(item=str(body["item"]), item_category=str(body["item_category"]), quantity=qty,
                              delivery_state=state, delivery_lga=lga, method=body["method"], budget=budget, date=d,
                              k_max=k, candidate_supplier_ids=body.get("candidate_supplier_ids"),
                              bids=body.get("bids"))
