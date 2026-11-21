"""Reproduce every thesis number, table and figure from raw data with one command:

    make experiments

Steps: data build -> Optuna tuning + training -> prediction metrics vs baselines -> optimiser vs
baselines (standard + large settings) -> back-test on 2024 contracts -> SHAP plots + SHAP-LIME
agreement -> end-to-end latency -> figures -> user-study pack -> summary.json.
All outputs go to experiments/results/ (tables as CSV + LaTeX, figures as 300-dpi PNG).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procure import figures  # noqa: E402
from procure.backtest import run_backtest, summarise_backtest  # noqa: E402
from procure.config import load_settings, path, set_seed  # noqa: E402
from procure.data import build  # noqa: E402
from procure.explain import lime_check  # noqa: E402
from procure.models import evaluate, train  # noqa: E402
from procure.optimise import benchmark  # noqa: E402
from procure.pipeline import Engine, recommend  # noqa: E402
from procure.reporting import save_table  # noqa: E402
from procure.rules.demo import demo_request  # noqa: E402
from procure.sampling import sample_requests  # noqa: E402
from procure.user_study import write_pack  # noqa: E402


def step(name: str, timings: dict[str, float]):
    """Context-free timer helper: returns a function that records elapsed time for `name`."""
    t0 = time.perf_counter()
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
    return lambda: timings.__setitem__(name, round(time.perf_counter() - t0, 1))


def main() -> None:
    """Run the whole experimental pipeline."""
    s = load_settings()
    set_seed()
    out = path("results_dir")
    timings: dict[str, float] = {}

    done = step("1. Data pipeline", timings)
    build.main()
    done()

    done = step("2. Train (Optuna) + 3. prediction metrics vs baselines", timings)
    train.main()
    evaluate.main()
    done()

    done = step("4. Optimiser vs baselines", timings)
    benchmark.main()
    done()

    done = step("5. Back-test on 2024 contracts", timings)
    bt = run_backtest()
    bt.to_csv(out / "backtest_per_contract.csv", index=False)
    bt_table = summarise_backtest(bt)
    save_table(bt_table, "backtest", "Back-test: recommended option vs actual award (model-expected, 2024)")
    print(bt_table.to_string(index=False))
    done()

    done = step("6. SHAP global plots + SHAP-LIME agreement", timings)
    lime_check.main()
    done()

    done = step("7. End-to-end latency + example recommendation", timings)
    eng = Engine.from_disk()
    test = pd.read_parquet(path("processed_dir") / "test.parquet")
    lat = [recommend(r, eng).runtime_seconds
           for r, _ in sample_requests(eng.store, test, s["experiments"]["latency_requests"], seed=s["seed"] + 1)]
    demo = recommend(demo_request(), eng).to_dict()
    (out / "example_recommendation.json").write_text(json.dumps(demo, indent=2, default=str))
    lat_table = pd.DataFrame([{"Requests": len(lat), "Mean (s)": f"{np.mean(lat):.2f}",
                               "95th percentile (s)": f"{np.percentile(lat, 95):.2f}", "Max (s)": f"{np.max(lat):.2f}",
                               "Within budget (%)": f"{100 * np.mean(np.array(lat) < s['api']['latency_budget_seconds']):.0f}"}])
    save_table(lat_table, "latency", "End-to-end recommendation time (rules, prediction, NSGA-II, explanations)")
    print(lat_table.to_string(index=False))
    done()

    done = step("8. Figures + user-study pack", timings)
    figs = [figures.data_overview(pd.read_parquet(path("processed_dir") / "transactions.parquet")),
            figures.prediction_comparison(pd.read_csv(out / "prediction_per_seed.csv")),
            figures.optimiser_boxplots({n: pd.read_csv(out / f"optimiser_per_request_{n}.csv") for n in ["standard", "large"]}),
            figures.backtest_scatter(bt),
            figures.pareto_example(demo["pareto_front"], demo["options"], demo["request"]["budget"]),
            figures.latency_hist(lat, s["api"]["latency_budget_seconds"])]
    pack = write_pack()
    print("figures:", *figs, sep="\n  ")
    print("user-study pack:", pack)
    done()

    save_table(pd.DataFrame([{"Step": k, "Seconds": v} for k, v in timings.items()]), "run_times",
               "Wall-clock time of each experiment step")
    summary = {
        "prediction": json.loads((out / "prediction.json").read_text())["beats_baselines"],
        "prediction_bootstrap": json.loads((out / "prediction.json").read_text())["bootstrap_xgb_minus_baseline"],
        "optimiser": json.loads((out / "optimiser.json").read_text()),
        "backtest": {r["Quantity"]: r["Value"] for _, r in bt_table.iterrows()},
        "shap_lime_agreement": json.loads((out / "shap_lime_agreement.json").read_text()),
        "latency_seconds": {"mean": float(np.mean(lat)), "p95": float(np.percentile(lat, 95))},
        "step_seconds": timings,
        "simulated_data": not __import__("procure.data.demo_raw", fromlist=["x"]).raw_dir_has_real_data(path("raw_dir")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nAll results written to", out)


if __name__ == "__main__":
    main()
