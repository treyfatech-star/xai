"""Rule-layer demo (make rules-demo): print why each supplier is excluded or flagged."""

from __future__ import annotations

import pandas as pd

from procure.domain import ProcurementRequest
from procure.rules.engine import apply_rules
from procure.store import DataStore


def demo_request() -> ProcurementRequest:
    """A realistic request: rice for Gwoza (Borno) in the 2024 rainy season."""
    return ProcurementRequest(item="Rice 50kg bag", item_category="food", quantity=400, delivery_state="Borno",
                              delivery_lga="Gwoza", method="selective_tendering", budget=45_000_000,
                              date=pd.Timestamp("2024-07-15").date(), request_id="REQ-DEMO-RULES")


def main() -> None:
    """Run the rule layer on the demo request and print reasons."""
    store = DataStore.from_processed()
    req = demo_request()
    cands = store.candidates(req)
    res = apply_rules(req, cands)
    print(f"Request {req.request_id}: {req.quantity:.0f} x {req.item} to {req.delivery_lga}, {req.delivery_state} "
          f"({req.method}, budget N{req.budget:,.0f})")
    print(f"Candidates: {len(cands)} | eligible: {len(res.eligible)} | excluded: {len(res.excluded)} | flags: {len(res.flags)}\n")
    print("EXCLUDED")
    for ex in res.excluded:
        for rid, why in zip(ex.rule_ids, ex.reasons):
            print(f"  [{rid}] {why}")
    print("\nFLAGS")
    for f in res.flags:
        print(f"  [{f.rule_id}] {f.reason}")
    print("\nELIGIBLE: " + ", ".join(res.eligible["supplier_id"]))


if __name__ == "__main__":
    main()
