"""Core domain objects shared by the rule, prediction, optimisation and explanation layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as Date
from typing import Any
import uuid


@dataclass
class ProcurementRequest:
    """A procurement request entered by an officer. Money is in naira (2024 prices)."""

    item: str
    item_category: str
    quantity: float
    delivery_state: str
    delivery_lga: str
    method: str
    budget: float
    date: Date
    k_max: int | None = None
    candidate_supplier_ids: list[str] | None = None
    bids: dict[str, float] | None = None          # optional supplier_id -> quoted unit price
    request_id: str = field(default_factory=lambda: f"REQ-{uuid.uuid4().hex[:10]}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        d = asdict(self)
        d["date"] = str(self.date)
        return d


@dataclass
class ExcludedSupplier:
    """A supplier removed by the rule layer, with every rule that excluded it."""

    supplier_id: str
    rule_ids: list[str]
    reasons: list[str]


@dataclass
class RuleFlag:
    """A warning raised by the rule layer (supplier stays eligible)."""

    rule_id: str
    reason: str
    supplier_id: str | None = None


@dataclass
class SupplierAllocation:
    """One supplier's share of an option, with predictions and plain-English reasons."""

    supplier_id: str
    share: float
    quantity: float
    unit_cost: float
    total_cost: float
    delay_days: float
    failure_prob: float
    quality: float
    reasons: dict[str, list[str]] = field(default_factory=dict)     # model -> sentences
    shap_top: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class Option:
    """One recommended way to fill the order (a point on the Pareto front)."""

    label: str
    rank: int
    total_cost: float
    worst_delay_days: float
    quality: float
    failure_prob: float
    within_budget: bool
    allocations: list[SupplierAllocation]
    summary: str = ""
    also_best_for: list[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Everything the officer sees for one request. The officer always makes the decision."""

    recommendation_id: str
    request: ProcurementRequest
    options: list[Option]
    excluded: list[ExcludedSupplier]
    flags: list[RuleFlag]
    pareto_front: list[dict[str, float]]
    n_candidates: int
    n_eligible: int
    runtime_seconds: float
    model_version: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        d = asdict(self)
        d["request"] = self.request.to_dict()
        return d
