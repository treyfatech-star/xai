"""Back-test on real 2024 contracts: what would the system have recommended, and how does its
expected cost and failure risk compare with the supplier that was actually chosen?

Both sides are scored with the same trained models (the counterfactual outcome of a supplier
that was not chosen cannot be observed), so savings are EXPECTED savings under the models.
The realised outcome of the actual award is reported alongside for calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.config import load_settings, path, set_seed
from procure.models.predict import Predictor
from procure.optimise.nsga import run_nsga2
from procure.optimise.problem import ProcurementProblem, build_predictions, evaluate_shares
from procure.optimise.select import select_options
from procure.rules.engine import apply_rules
from procure.sampling import request_from_transaction
from procure.store import DataStore


def _actual_supplier(store: DataStore, pred: Predictor, req, row: pd.Series) -> dict[str, float] | None:
    """Model-expected cost/delay/failure for the supplier that actually won, plus rule status."""
    from dataclasses import replace

    cand = store.candidates(replace(req, candidate_supplier_ids=[row["supplier_id"]], bids=None))
    if cand.empty:
        return None
    rules = apply_rules(req, cand)
    p = pred.predict(store.feature_rows(req, cand)).iloc[0]
    return {"cost": float(p["unit_cost"] * req.quantity), "delay": float(p["delay_days"]),
            "fail": float(p["failure_prob"]), "excluded": bool(rules.excluded),
            "flagged": any(f.supplier_id for f in rules.flags)}


def run_backtest(n: int | None = None) -> pd.DataFrame:
    """One row per back-tested contract."""
    s = load_settings()
    set_seed()
    n = n or s["experiments"]["n_backtest"]
    store, pred = DataStore.from_processed(), Predictor()
    test = pd.read_parquet(path("processed_dir") / "test.parquet")
    rng = np.random.default_rng(s["seed"])
    rows = []
    for i in rng.permutation(len(test)):
        if len(rows) >= n:
            break
        row = test.iloc[i]
        req = request_from_transaction(row)
        elig = apply_rules(req, store.candidates(req)).eligible
        if len(elig) < s["optimise"]["min_candidates"]:
            continue
        act = _actual_supplier(store, pred, req, row)
        if act is None:
            continue
        sp, _ = build_predictions(store, pred, req, elig)
        front = run_nsga2(ProcurementProblem(sp))
        sel = {x.label: x for x in select_options(front.F)}
        best = select_options(front.F)[0]
        F, _ = evaluate_shares(sp, front.X[best.index][None, :])
        low = next((x for x in select_options(front.F) if "lowest_cost" in [x.label, *x.also_best_for]), best)
        FL, _ = evaluate_shares(sp, front.X[low.index][None, :])
        rows.append({
            "ocid": row["ocid"], "item_category": row["item_category"], "delivery_state": row["delivery_state"],
            "n_eligible": len(elig), "actual_cost_expected": act["cost"], "actual_fail_expected": act["fail"],
            "actual_delay_expected": act["delay"], "actual_failed_realised": int(row["failed"]),
            "actual_excluded_by_rules": act["excluded"], "actual_flagged": act["flagged"],
            "rec_cost": float(F[0, 0]), "rec_fail": float(F[0, 3]), "rec_delay": float(F[0, 1]),
            "rec_n_suppliers": int((front.X[best.index] > 0).sum()), "rec_label": best.label,
            "lowcost_cost": float(FL[0, 0]), "lowcost_fail": float(FL[0, 3]), "feasible": front.feasible,
        })
    df = pd.DataFrame(rows)
    df["cost_saving_pct"] = 100 * (df.actual_cost_expected - df.rec_cost) / df.actual_cost_expected
    df["risk_reduction_pp"] = 100 * (df.actual_fail_expected - df.rec_fail)
    df["lowcost_saving_pct"] = 100 * (df.actual_cost_expected - df.lowcost_cost) / df.actual_cost_expected
    df["lowcost_risk_reduction_pp"] = 100 * (df.actual_fail_expected - df.lowcost_fail)
    return df


def summarise_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Thesis table for the back-test."""
    def ci(x: pd.Series) -> str:
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        return f"{m:+.1f} [{m - 1.96 * se:+.1f}, {m + 1.96 * se:+.1f}]"

    return pd.DataFrame([
        {"Quantity": "Contracts back-tested", "Value": f"{len(df)}"},
        {"Quantity": "Top-ranked option: expected cost saving, % (95% CI)", "Value": ci(df.cost_saving_pct)},
        {"Quantity": "Top-ranked option: failure-risk reduction, points (95% CI)", "Value": ci(df.risk_reduction_pp)},
        {"Quantity": "Lowest-cost option: expected cost saving, % (95% CI)", "Value": ci(df.lowcost_saving_pct)},
        {"Quantity": "Lowest-cost option: failure-risk reduction, points (95% CI)", "Value": ci(df.lowcost_risk_reduction_pp)},
        {"Quantity": "Top option cheaper AND less risky than actual award (%)",
         "Value": f"{100 * ((df.cost_saving_pct > 0) & (df.risk_reduction_pp > 0)).mean():.0f}"},
        {"Quantity": "Actual awards to suppliers the rules would exclude (%)",
         "Value": f"{100 * df.actual_excluded_by_rules.mean():.0f}"},
        {"Quantity": "Actual awards flagged for high past failure (%)", "Value": f"{100 * df.actual_flagged.mean():.0f}"},
        {"Quantity": "Mean expected failure risk of actual awards (%)", "Value": f"{100 * df.actual_fail_expected.mean():.1f}"},
        {"Quantity": "Realised failure rate of actual awards (%)", "Value": f"{100 * df.actual_failed_realised.mean():.1f}"},
    ])
