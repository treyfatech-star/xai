"""recommend(request) -> RecommendationResult: rules -> predictions -> NSGA-II -> selection -> explanations.

Run `python -m procure.pipeline` (make demo) for one end-to-end example in the terminal.
Persistence (model_output, audit_log) is done by procure.service so this module stays pure.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from procure.config import load_settings
from procure.domain import Option, ProcurementRequest, RecommendationResult, SupplierAllocation
from procure.explain.narrate import narrate_model
from procure.explain.shap_explain import ShapExplainer
from procure.models.predict import Predictor
from procure.models.train import MODEL_NAMES, correct_prob
from procure.optimise.nsga import run_nsga2
from procure.optimise.problem import ProcurementProblem, build_predictions, evaluate_shares
from procure.optimise.select import select_options
from procure.rules.engine import apply_rules
from procure.store import DataStore

OPTION_TITLES = {"lowest_cost": "Lowest cost", "fastest": "Fastest delivery", "most_reliable": "Most reliable",
                 "balanced": "Balanced", "best_quality": "Best quality", "alternative": "Alternative"}


@dataclass
class Engine:
    """Loaded data + models, reused across requests."""

    store: DataStore
    predictor: Predictor
    explainer: ShapExplainer

    @classmethod
    def from_disk(cls) -> "Engine":
        """Load processed data and the trained bundle."""
        pred = Predictor()
        return cls(DataStore.from_processed(), pred, ShapExplainer(pred.bundle))


@lru_cache(maxsize=1)
def default_engine() -> Engine:
    """Process-wide engine (loaded on first use)."""
    return Engine.from_disk()


def _explain_allocations(engine: Engine, request: ProcurementRequest, eligible: pd.DataFrame,
                         allocations: list[tuple[int, str, float]]) -> dict[tuple[int, str], dict]:
    """SHAP + narration for every (option, supplier, share) in one batched call."""
    if not allocations:
        return {}
    idx = {s: i for i, s in enumerate(eligible["supplier_id"])}
    sel = eligible.iloc[[idx[s] for _, s, _ in allocations]].reset_index(drop=True)
    rows = []
    for j, (_, sid, share) in enumerate(allocations):
        rows.append(engine.store.feature_rows(request, eligible, np.array([share * request.quantity])).iloc[[idx[sid]]])
    feats = pd.concat(rows, ignore_index=True)
    x = engine.predictor.transform(feats)
    shap = engine.explainer.explain(x)
    margins = engine.explainer.margins(x)
    pred = engine.predictor.predict_transformed(x)
    b = engine.predictor.bundle
    base = {"cost": engine.store.item_ref_price(request),
            "delay": b.trained_on.get("delay_mean", engine.explainer.base_value("delay")),
            "failure": b.trained_on.get("failure_rate", float(correct_prob(
                np.array([1 / (1 + np.exp(-engine.explainer.base_value("failure")))]), b.failure_weight)[0]))}
    out = {}
    for j, (opt, sid, _) in enumerate(allocations):
        ctx = {"delivery_lga": request.delivery_lga, "supplier_lga": sel.loc[j, "supplier_lga"] or "the supplier"}
        values = feats.iloc[j].to_dict()
        reasons, tops = {}, {}
        for m, col in zip(MODEL_NAMES, ["unit_cost", "delay_days", "failure_prob"]):
            text, top = narrate_model(m, float(pred.loc[j, col]), float(base[m]), shap[m].iloc[j].to_dict(),
                                      values, ctx, float(margins[m][j]), b.failure_weight)
            reasons[m] = [text] + [r["sentence"] for r in top]
            tops[m] = top
        out[(opt, sid)] = {"reasons": reasons, "shap_top": tops}
    return out


def _summary(opt: Option) -> str:
    """One-line plain-English summary of an option."""
    parts = " / ".join(f"{a.share:.0%} {a.supplier_id}" for a in opt.allocations)
    budget = "within budget" if opt.within_budget else "OVER BUDGET"
    return (f"{OPTION_TITLES.get(opt.label, opt.label)}: {parts}. Total N{opt.total_cost:,.0f} ({budget}), "
            f"slowest delivery about {opt.worst_delay_days:.0f} days late, "
            f"{opt.failure_prob:.0%} chance that at least one supplier fails, average quality {opt.quality:.1f}/5.")


def recommend(request: ProcurementRequest, engine: Engine | None = None) -> RecommendationResult:
    """Full recommendation for one request. The officer always makes the final decision."""
    t0 = time.perf_counter()
    engine = engine or default_engine()
    cfg = load_settings()
    rec_id = f"REC-{uuid.uuid4().hex[:12]}"
    cands = engine.store.candidates(request)
    rules = apply_rules(request, cands)
    eligible = rules.eligible
    common = dict(recommendation_id=rec_id, request=request, excluded=rules.excluded, flags=rules.flags,
                  n_candidates=len(cands), n_eligible=len(eligible), model_version=engine.predictor.version)
    if eligible.empty:
        return RecommendationResult(options=[], pareto_front=[], runtime_seconds=time.perf_counter() - t0,
                                    message="No supplier is eligible for this request; see the exclusion reasons.",
                                    **common)
    sp, _ = build_predictions(engine.store, engine.predictor, request, eligible)
    problem = ProcurementProblem(sp, request.k_max)
    front = run_nsga2(problem)
    selections = select_options(front.F)
    flags_by_sup: dict[str, list[str]] = {}
    for f in rules.flags:
        if f.supplier_id:
            flags_by_sup.setdefault(f.supplier_id, []).append(f.reason)
    todo = [(r, sp.supplier_ids[i], float(front.X[s.index, i]))
            for r, s in enumerate(selections) for i in np.flatnonzero(front.X[s.index] > 0)]
    expl = _explain_allocations(engine, request, eligible, todo)
    options = []
    for r, s in enumerate(selections):
        x = front.X[s.index]
        F, G = evaluate_shares(sp, x[None, :])
        c, d, fp = sp.at(x[None, :])
        allocs = []
        for i in np.flatnonzero(x > 0):
            sid = sp.supplier_ids[i]
            e = expl[(r, sid)]
            allocs.append(SupplierAllocation(
                supplier_id=sid, share=float(x[i]), quantity=float(x[i] * request.quantity),
                unit_cost=float(c[0, i]), total_cost=float(c[0, i] * x[i] * request.quantity),
                delay_days=float(d[0, i]), failure_prob=float(fp[0, i]), quality=float(sp.quality[i]),
                reasons=e["reasons"], shap_top=e["shap_top"], flags=flags_by_sup.get(sid, [])))
        allocs.sort(key=lambda a: -a.share)
        opt = Option(label=s.label, rank=r + 1, total_cost=float(F[0, 0]), worst_delay_days=float(F[0, 1]),
                     quality=float(-F[0, 2]), failure_prob=float(F[0, 3]), within_budget=bool(G[0, 0] <= 0),
                     allocations=allocs, also_best_for=s.also_best_for)
        opt.summary = _summary(opt)
        options.append(opt)
    n_front = cfg["explain"]["max_front_points_returned"]
    keep = np.linspace(0, len(front.F) - 1, min(n_front, len(front.F))).astype(int)
    pareto = [{"cost": float(f[0]), "worst_delay": float(f[1]), "quality": float(-f[2]), "failure_prob": float(f[3])}
              for f in front.F[keep]]
    msg = "" if front.feasible else ("No allocation fits the budget. The closest option is shown; "
                                     "consider raising the budget or reducing the quantity.")
    return RecommendationResult(options=options, pareto_front=pareto, runtime_seconds=time.perf_counter() - t0,
                                message=msg, **common)


def print_result(res: RecommendationResult) -> None:
    """Human-readable terminal output."""
    r = res.request
    print(f"{res.recommendation_id}: {r.quantity:,.0f} x {r.item} to {r.delivery_lga}, {r.delivery_state} "
          f"| budget N{r.budget:,.0f} | {res.n_eligible}/{res.n_candidates} suppliers eligible "
          f"| {res.runtime_seconds:.1f}s | model {res.model_version}")
    if res.message:
        print("NOTE:", res.message)
    for ex in res.excluded:
        print(f"  excluded {ex.supplier_id}: " + " ".join(ex.reasons))
    for f in res.flags:
        print(f"  flag [{f.rule_id}] {f.reason}")
    for o in res.options:
        extra = f" (also {', '.join(OPTION_TITLES[b] for b in o.also_best_for)})" if o.also_best_for else ""
        print(f"\n#{o.rank} {o.summary}{extra}")
        for a in o.allocations:
            print(f"   {a.supplier_id}: {a.quantity:,.0f} units at N{a.unit_cost:,.0f}")
            for m in MODEL_NAMES:
                print(f"     - {a.reasons[m][0]}")


if __name__ == "__main__":
    from procure.rules.demo import demo_request

    print_result(recommend(demo_request()))
