"""Optimiser benchmark (make optimise): NSGA-II vs baselines on sampled 2024 requests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from procure.config import load_settings, path, set_seed
from procure.models.predict import Predictor
from procure.optimise import baselines as obl
from procure.optimise.metrics import generalised_spread, hypervolume, reference_bounds
from procure.optimise.nsga import nondominated, run_nsga2
from procure.optimise.problem import ProcurementProblem, build_predictions
from procure.reporting import save_table
from procure.rules.engine import apply_rules
from procure.sampling import sample_requests
from procure.store import DataStore


def run_benchmark(n_requests: int | None = None, max_candidates: int | None = None,
                  k: int | None = None) -> pd.DataFrame:
    """One row per (request, method) with HV, spread, front size, runtime, evaluations.
    max_candidates / k override the config (used by the scalability experiment)."""
    s = load_settings()
    set_seed()
    store = DataStore.from_processed()
    pred = Predictor()
    test = pd.read_parquet(path("processed_dir") / "test.parquet")
    old_max = s["optimise"]["max_candidates"]
    if max_candidates:
        s["optimise"]["max_candidates"] = max_candidates
    try:
        reqs = sample_requests(store, test, n_requests or s["optimise"]["n_eval_requests"])
        reqs = [(r, row, apply_rules(r, store.candidates(r)).eligible) for r, row in reqs]
    finally:
        s["optimise"]["max_candidates"] = old_max
    ref = s["optimise"]["hv_ref_point"]
    rows = []
    for req, _, elig in reqs:
        sp, _ = build_predictions(store, pred, req, elig)
        fronts = []
        for fn in [run_nsga2, obl.weighted_sum, obl.random_search]:
            fronts.append(fn(ProcurementProblem(sp, k)))
        fronts.append(obl.lowest_bid(ProcurementProblem(sp, k), elig["bid_unit_price"].to_numpy()))
        if not any(f.feasible for f in fronts):
            continue
        lo, hi = reference_bounds(fronts)
        union = np.vstack([f.F for f in fronts if f.feasible])
        union = union[nondominated(union)]
        for f in fronts:
            rows.append({"request_id": req.request_id, "n_suppliers": sp.n, "k": min(k or s["optimise"]["k_max"], sp.n),
                         "method": f.method,
                         "hypervolume": hypervolume(f, lo, hi, ref), "spread": generalised_spread(f, lo, hi, union),
                         "front_size": len(f.F) if f.feasible else 0, "feasible": f.feasible,
                         "runtime_s": f.runtime, "evaluations": f.n_evals})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per method."""
    g = df.groupby("method", sort=False)
    out = pd.DataFrame({
        "Method": list(g.groups),
        "Hypervolume": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.hypervolume.mean(), g.hypervolume.std())],
        "Spread": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.spread.mean(), g.spread.std())],
        "Front size": [f"{a:.1f}" for a in g.front_size.mean()],
        "Feasible (%)": [f"{100 * a:.0f}" for a in g.feasible.mean()],
        "Run time (s)": [f"{a:.2f}" for a in g.runtime_s.mean()],
        "Evaluations": [f"{a:,.0f}" for a in g.evaluations.mean()],
    })
    return out


def significance(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Wilcoxon signed-rank test of per-request hypervolume, NSGA-II vs each baseline."""
    from scipy.stats import wilcoxon

    wide = df.pivot(index="request_id", columns="method", values="hypervolume")
    out = {}
    for m in wide.columns.drop("NSGA-II"):
        d = wide["NSGA-II"] - wide[m]
        p = float(wilcoxon(d, zero_method="zsplit").pvalue) if (d != 0).any() else 1.0
        out[m] = {"mean_diff": float(d.mean()), "nsga_ge_share": float((d >= -1e-12).mean()), "wilcoxon_p": p}
    return out


def main() -> None:
    """CLI: standard and large-search-space settings; save CSV/LaTeX/JSON; print exit check."""
    s = load_settings()["optimise"]
    out = path("results_dir")
    settings = {"standard": {}, "large": dict(s["scalability"])}
    report = {}
    for name, kw in settings.items():
        df = run_benchmark(**kw)
        df.to_csv(out / f"optimiser_per_request_{name}.csv", index=False)
        table = summarise(df)
        cap = (f"NSGA-II vs baselines over {df.request_id.nunique()} sampled 2024 requests "
               f"({name}: up to {kw.get('max_candidates', s['max_candidates'])} suppliers, k = {kw.get('k', s['k_max'])})")
        save_table(table, f"optimiser_{name}", cap)
        mean_hv = df.groupby("method").hypervolume.mean()
        sig = significance(df)
        report[name] = {"mean_hv": mean_hv.to_dict(), "tests": sig, "n_requests": int(df.request_id.nunique()),
                        "mean_suppliers": float(df.n_suppliers.mean())}
        save_table(pd.DataFrame([{"Baseline": m, "Mean HV difference": f"{v['mean_diff']:+.4f}",
                                  "NSGA-II >= baseline (%)": f"{100 * v['nsga_ge_share']:.0f}",
                                  "Wilcoxon p": f"{v['wilcoxon_p']:.3g}"} for m, v in sig.items()]),
                   f"optimiser_tests_{name}", f"NSGA-II minus baseline hypervolume ({name} setting)")
        print(f"\n=== {name} setting (mean {df.n_suppliers.mean():.1f} eligible suppliers) ===")
        print(table.to_string(index=False))
        print("EXIT CHECK (mean normalised hypervolume):")
        for m, v in sig.items():
            ok = mean_hv["NSGA-II"] > mean_hv[m]
            print(f"  {'PASS' if ok else 'FAIL'} NSGA-II {mean_hv['NSGA-II']:.4f} vs {m} {mean_hv[m]:.4f} "
                  f"(diff {v['mean_diff']:+.4f}, NSGA-II >= on {100 * v['nsga_ge_share']:.0f}%, Wilcoxon p={v['wilcoxon_p']:.3g})")
    (out / "optimiser.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
