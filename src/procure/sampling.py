"""Turn historical (test-year) transactions into realistic procurement requests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.config import load_settings
from procure.domain import ProcurementRequest
from procure.rules.engine import apply_rules
from procure.store import DataStore


def request_from_transaction(row: pd.Series) -> ProcurementRequest:
    """Request mirroring a real contract; budget = real contract value x budget_slack."""
    slack = load_settings()["optimise"]["budget_slack"]
    return ProcurementRequest(
        item=row["item"], item_category=row["item_category"], quantity=float(row["quantity"]),
        delivery_state=row["delivery_state"], delivery_lga=row["delivery_lga"], method=row["method"],
        budget=float(row["unit_cost_real"] * row["quantity"] * slack), date=pd.Timestamp(row["date"]).date(),
        request_id=f"REQ-{row['ocid']}")


def sample_requests(store: DataStore, pool: pd.DataFrame, n: int, seed: int | None = None
                    ) -> list[tuple[ProcurementRequest, pd.Series]]:
    """Sample up to n requests from pool whose rule-filtered candidate set has >= min_candidates."""
    cfg = load_settings()["optimise"]
    rng = np.random.default_rng(load_settings()["seed"] if seed is None else seed)
    out = []
    for i in rng.permutation(len(pool)):
        row = pool.iloc[i]
        req = request_from_transaction(row)
        res = apply_rules(req, store.candidates(req))
        if len(res.eligible) >= cfg["min_candidates"]:
            out.append((req, row))
        if len(out) >= n:
            break
    return out
