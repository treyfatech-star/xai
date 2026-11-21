"""Turn SHAP values into plain-English sentences with units, for non-technical officers.

Example: "Delay is predicted at 19 days, 6 days longer than a typical order, mainly because of
rainy-season delivery (May to October) to Gwoza and 31 security incidents within 50 km of
Gwoza in the last 90 days."

No jargon: no "SHAP", "log-odds" or feature names appear in the text.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from procure.config import load_settings

FEATURE_LABELS: dict[str, str] = {
    "log_item_ref_price": "Typical item price", "log_quantity": "Order size", "n_bidders": "Number of bidders",
    "supplier_experience": "Supplier experience", "past_on_time_rate": "Past on-time rate",
    "past_failure_rate": "Past failure rate", "past_quality_mean": "Past inspection score",
    "past_price_index": "Supplier's past prices", "distance_km": "Distance to delivery point",
    "same_state": "Supplier in same state", "rainy_season": "Rainy season",
    "conflict_events_90d": "Nearby security incidents", "conflict_fatalities_90d": "Nearby conflict deaths",
    "item_category": "Item category", "item": "Item", "method": "Procurement method",
    "delivery_state": "Delivery state",
}

METHOD_NAMES = {"open_competitive": "open competitive bidding", "selective_tendering": "selective tendering",
                "request_for_quotation": "request for quotation", "direct_procurement": "direct procurement"}


def _pct_above(v: float) -> str:
    r = float(np.exp(v)) - 1
    return f"{abs(r):.0%} {'above' if r >= 0 else 'below'} the typical price"


PHRASES: dict[str, Callable[[Any, dict[str, Any]], str]] = {
    "log_item_ref_price": lambda v, c: f"the item's typical price of N{np.exp(v):,.0f}",
    "log_quantity": lambda v, c: f"the order size of {np.exp(v):,.0f} units",
    "n_bidders": lambda v, c: f"competition from {v:.0f} eligible suppliers",
    "supplier_experience": lambda v, c: f"the supplier's {v:.0f} earlier contracts",
    "past_on_time_rate": lambda v, c: f"the supplier's past on-time delivery rate of {v:.0%}",
    "past_failure_rate": lambda v, c: f"the supplier's past failure rate of {v:.0%}",
    "past_quality_mean": lambda v, c: f"the supplier's average inspection score of {v:.1f} out of 5",
    "past_price_index": lambda v, c: f"the supplier's past prices being {_pct_above(v)}",
    "distance_km": lambda v, c: (f"the supplier being based in {c['delivery_lga']} itself" if v < 1
                                 else f"the {v:.0f} km distance from {c.get('supplier_lga', 'the supplier')} to {c['delivery_lga']}"),
    "same_state": lambda v, c: "the supplier being based in the same state" if v >= 0.5
    else "the supplier being based in another state",
    "rainy_season": lambda v, c: f"rainy-season delivery (May to October) to {c['delivery_lga']}" if v >= 0.5
    else f"dry-season delivery to {c['delivery_lga']}",
    "conflict_events_90d": lambda v, c: f"{v:.0f} security incidents within 50 km of {c['delivery_lga']} in the last 90 days",
    "conflict_fatalities_90d": lambda v, c: f"{v:.0f} conflict deaths near {c['delivery_lga']} in the last 90 days",
    "item_category": lambda v, c: f"the item category ({str(v).replace('_', ' ')})",
    "item": lambda v, c: f"the item ({v})",
    "method": lambda v, c: f"the use of {METHOD_NAMES.get(str(v), str(v).replace('_', ' '))}",
    "delivery_state": lambda v, c: f"delivery to {v} State",
}


def phrase(feature: str, value: Any, ctx: dict[str, Any]) -> str:
    """Plain-English description of a feature value."""
    fn = PHRASES.get(feature)
    return fn(value, ctx) if fn else f"{FEATURE_LABELS.get(feature, feature)} = {value}"


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-z))


def effect_text(model: str, shap_value: float, margin: float, failure_weight: float) -> tuple[str, float]:
    """Size of one feature's effect in natural units: (+4% price, +3 days, +5 points risk)."""
    from procure.models.train import correct_prob

    if model == "cost":
        eff = float(np.exp(shap_value) - 1) * 100
        return (f"{'adds' if eff >= 0 else 'saves'} about {abs(eff):.0f}% on the price", eff)
    if model == "delay":
        return (f"{'adds' if shap_value >= 0 else 'saves'} about {abs(shap_value):.1f} days", shap_value)
    p_with = correct_prob(np.array([_sigmoid(margin)]), failure_weight)[0]
    p_without = correct_prob(np.array([_sigmoid(margin - shap_value)]), failure_weight)[0]
    pp = float(p_with - p_without) * 100
    return (f"{'raises' if pp >= 0 else 'lowers'} the risk by about {abs(pp):.1f} percentage points", pp)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def narrate_model(model: str, prediction: float, baseline: float, shap_row: dict[str, float],
                  values: dict[str, Any], ctx: dict[str, Any], margin: float, failure_weight: float,
                  k: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Headline sentence + top-k reason records for one model and one supplier.

    baseline is in natural units: typical item price (cost), typical delay (days), typical risk (prob).
    """
    cfg = load_settings()["explain"]
    k = k or cfg["top_k"]
    exclude = set(cfg.get("narrative_exclude", {}).get(model, []))
    ranked = sorted((f for f in shap_row if f not in exclude), key=lambda f: -abs(shap_row[f]))[:k]
    total = prediction - baseline
    reasons, same, opposite = [], [], []
    for f in ranked:
        txt, size = effect_text(model, shap_row[f], margin, failure_weight)
        p = phrase(f, values.get(f), ctx)
        reasons.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "value": _plain(values.get(f)),
                        "shap": float(shap_row[f]), "effect": round(size, 2),
                        "sentence": f"{p[0].upper() + p[1:]} {txt}."})
        (same if np.sign(size) == np.sign(total) or total == 0 else opposite).append(p)
    if model == "cost":
        pct = (prediction / baseline - 1) * 100 if baseline else 0.0
        head = (f"Unit price is predicted at N{prediction:,.0f}, {abs(pct):.0f}% "
                f"{'above' if pct >= 0 else 'below'} the typical N{baseline:,.0f} for this item")
    elif model == "delay":
        head = (f"Delay is predicted at {prediction:.0f} days, {abs(total):.0f} days "
                f"{'longer' if total >= 0 else 'shorter'} than a typical order")
    else:
        head = (f"Risk of failure (more than 30 days late, short delivery or rejection) is {prediction:.0%}, "
                f"{'higher' if total >= 0 else 'lower'} than the typical {baseline:.0%}")
    text = head
    if same:
        text += f", mainly because of {_join(same)}"
    if opposite:
        text += f"; {_join(opposite)} {'works' if len(opposite) == 1 else 'work'} the other way"
    return text + ".", reasons


def _plain(v: Any) -> Any:
    """JSON-friendly scalar."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v
