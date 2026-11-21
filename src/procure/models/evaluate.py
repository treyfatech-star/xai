"""Evaluate XGBoost models vs baselines on the 2024 test set (real-only and full), 5 seeds."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, mean_absolute_error, recall_score, roc_auc_score

from procure.config import load_settings, path, set_seed
from procure.models import baselines as bl
from procure.models.predict import Predictor
from procure.models.train import fit_bundle
from procure.reporting import save_table

PRIMARY = {"cost": ("MAE", "min"), "delay": ("MAE", "min"), "failure": ("AUC", "max")}


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and MAPE (MAPE over rows with |y| >= 1 to avoid dividing by ~0 delays)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    nz = np.abs(y) >= 1
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAPE": float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100)}


def classification_metrics(y: np.ndarray, prob: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """AUC-ROC, F1, recall (at the configured threshold) and Brier score (corrected probabilities)."""
    return {"AUC": float(roc_auc_score(y, prob)), "F1": float(f1_score(y, label)),
            "Recall": float(recall_score(y, label)), "Brier": float(brier_score_loss(y, prob))}


def _score_all(models: dict[str, Any], sets: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Score a dict {(target, model_name): predictor_fn} on each evaluation set."""
    thr = load_settings()["models"]["classification_threshold"]
    rows = []
    for (tgt, name), fn in models.items():
        for set_name, df in sets.items():
            if tgt == "cost":
                m = regression_metrics(df["unit_cost_real"], fn(df))
            elif tgt == "delay":
                m = regression_metrics(df["delay_days"], fn(df))
            else:
                prob, score = fn(df)
                m = classification_metrics(df["failed"], prob, (score >= thr).astype(int))
            rows.append({"target": tgt, "model": name, "eval_set": set_name, **m})
    return rows


def run_evaluation() -> dict[str, Any]:
    """Fit every model for each seed on train+val, score on the test sets, aggregate mean/std."""
    s = load_settings()
    set_seed()
    d = path("processed_dir")
    trv = pd.concat([pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet")], ignore_index=True)
    test = pd.read_parquet(d / "test.parquet")
    test_full = pd.concat([test, pd.read_parquet(d / "test_synthetic.parquet")], ignore_index=True)
    sets = {"real_only": test, "full": test_full}
    params = json.loads((path("models_dir") / "best_params.json").read_text())
    det = {"median": bl.MedianPerCategory().fit(trv), "linear": bl.LinearCost().fit(trv),
           "supplier_mean": bl.PerSupplierMeanDelay().fit(trv), "logistic": bl.LogisticFailure().fit(trv)}
    rows: list[dict[str, Any]] = []
    preds: dict[tuple[str, str], np.ndarray] = {}
    for seed in s["models"]["eval_seeds"]:
        pred = Predictor(fit_bundle(trv, params, seed))
        rf = bl.RandomForestDelay(seed).fit(trv)

        def fail_fn(df: pd.DataFrame, p: Predictor = pred) -> tuple[np.ndarray, np.ndarray]:
            x = p.transform(df)
            score = p.bundle.models["failure"].predict_proba(x)[:, 1]
            return p.predict_transformed(x)["failure_prob"].to_numpy(), score

        models = {
            ("cost", "XGBoost"): lambda df, p=pred: p.predict(df)["unit_cost"].to_numpy(),
            ("cost", "Median per category"): det["median"].predict,
            ("cost", "Linear regression"): det["linear"].predict,
            ("delay", "XGBoost"): lambda df, p=pred: p.predict(df)["delay_days"].to_numpy(),
            ("delay", "Per-supplier mean"): det["supplier_mean"].predict,
            ("delay", "Random forest"): rf.predict,
            ("failure", "XGBoost"): fail_fn,
            ("failure", "Logistic regression"): lambda df: (det["logistic"].predict(df), det["logistic"].score(df)),
        }
        for r in _score_all(models, sets):
            rows.append({**r, "seed": seed})
        if seed == s["seed"]:
            for key, fn in models.items():
                out = fn(test)
                preds[key] = out[0] if isinstance(out, tuple) else out
    raw = pd.DataFrame(rows)
    metrics = [c for c in raw.columns if c not in {"target", "model", "eval_set", "seed"}]
    agg = raw.groupby(["target", "model", "eval_set"], sort=False)[metrics].agg(["mean", "std"])
    return {"raw": raw, "agg": agg, "sizes": {k: len(v) for k, v in sets.items()},
            "bootstrap": paired_bootstrap(test, preds)}


def paired_bootstrap(test: pd.DataFrame, preds: dict[tuple[str, str], np.ndarray]) -> dict[str, dict[str, Any]]:
    """95% CI of (XGBoost - baseline) on the primary metric, resampling test rows (seed-42 models).
    For MAE a negative difference favours XGBoost; for AUC a positive one does."""
    n_boot = load_settings()["models"]["bootstrap_n"]
    rng = np.random.default_rng(load_settings()["seed"])
    ycol = {"cost": "unit_cost_real", "delay": "delay_days", "failure": "failed"}
    out: dict[str, dict[str, Any]] = {}
    n = len(test)
    idx = rng.integers(0, n, size=(n_boot, n))
    for tgt, (metric, _) in PRIMARY.items():
        y = test[ycol[tgt]].to_numpy(float)
        stat = (lambda yy, pp: roc_auc_score(yy, pp)) if metric == "AUC" else (lambda yy, pp: np.mean(np.abs(yy - pp)))
        for (t, name), p in preds.items():
            if t != tgt or name == "XGBoost":
                continue
            px = preds[(tgt, "XGBoost")]
            diffs = np.array([stat(y[i], px[i]) - stat(y[i], p[i]) for i in idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out[f"{tgt} vs {name}"] = {"metric": metric, "diff": float(stat(y, px) - stat(y, p)),
                                       "ci95": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0)}
    return out


def beats_baselines(agg: pd.DataFrame, eval_set: str = "real_only") -> dict[str, dict[str, Any]]:
    """For each target, does XGBoost beat every baseline on the primary metric?"""
    out = {}
    for tgt, (metric, direction) in PRIMARY.items():
        sub = agg.xs(eval_set, level="eval_set").xs(tgt, level="target")[(metric, "mean")]
        xgb_v = float(sub["XGBoost"])
        others = sub.drop("XGBoost")
        better = all(xgb_v < v if direction == "min" else xgb_v > v for v in others)
        out[tgt] = {"metric": metric, "xgboost": xgb_v, "baselines": others.to_dict(), "beats_all": better}
    return out


def format_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Thesis table: mean ± std per metric."""
    rows = []
    for (tgt, model, es), r in agg.iterrows():
        row = {"Target": tgt, "Model": model, "Test set": es}
        for m in sorted({c[0] for c in agg.columns}, key=list(dict.fromkeys(c[0] for c in agg.columns)).index):
            if pd.isna(r[(m, "mean")]):
                continue
            mean, sd = r[(m, "mean")], r[(m, "std")]
            fmt = ",.0f" if (tgt == "cost" and m in {"MAE", "RMSE"}) else ".3f" if tgt == "failure" else ".2f"
            row[m] = f"{mean:{fmt}} ± {0 if pd.isna(sd) else sd:{fmt}}"
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def main() -> None:
    """CLI (make train, second step): write prediction.json and LaTeX/CSV tables."""
    res = run_evaluation()
    agg = res["agg"]
    check = {es: beats_baselines(agg, es) for es in ["real_only", "full"]}
    out = path("results_dir")
    payload = {"test_sizes": res["sizes"], "seeds": load_settings()["models"]["eval_seeds"],
               "metrics": {f"{t}|{m}|{e}": {f"{k}_{s}": v for (k, s), v in r.items()} for (t, m, e), r in agg.iterrows()},
               "beats_baselines": check, "bootstrap_xgb_minus_baseline": res["bootstrap"]}
    (out / "prediction.json").write_text(json.dumps(payload, indent=2, default=float))
    res["raw"].to_csv(out / "prediction_per_seed.csv", index=False)
    for tgt in ["cost", "delay", "failure"]:
        t = format_table(agg.xs(tgt, level="target", drop_level=False))
        save_table(t, f"prediction_{tgt}", f"Test-set (2024) {tgt} prediction, mean ± std over 5 seeds")
    save_table(format_table(agg), "prediction_all", "Prediction layer vs baselines on the 2024 test set")
    print(format_table(agg).to_string(index=False))
    print("\nEXIT CHECK (primary metric, real-only 2024 test set):")
    for tgt, c in check["real_only"].items():
        status = "PASS" if c["beats_all"] else "FAIL"
        print(f"  {status} {tgt:8s} XGBoost {c['metric']}={c['xgboost']:.4g} vs "
              + ", ".join(f"{k}={v:.4g}" for k, v in c["baselines"].items()))
    print("\nPaired bootstrap, XGBoost minus baseline (95% CI, real-only test):")
    for k, b in res["bootstrap"].items():
        print(f"  {k:32s} {b['metric']} diff {b['diff']:+.4g}  CI [{b['ci95'][0]:+.4g}, {b['ci95'][1]:+.4g}]"
              f"  {'significant' if b['significant'] else 'not significant'}")
    save_table(pd.DataFrame([{"Comparison": k, "Metric": b["metric"], "Difference": f"{b['diff']:+.4g}",
                              "95% CI": f"[{b['ci95'][0]:+.4g}, {b['ci95'][1]:+.4g}]",
                              "Significant": "yes" if b["significant"] else "no"} for k, b in res["bootstrap"].items()]),
               "prediction_bootstrap", "Paired bootstrap of XGBoost minus baseline on the 2024 real test set")


if __name__ == "__main__":
    main()
