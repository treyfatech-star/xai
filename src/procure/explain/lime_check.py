"""LIME as a secondary agreement check for SHAP (make explain).

For n sampled 2024 test predictions per model, compare the top-3 original features by |SHAP|
with the top-3 by |LIME weight| (LIME run on the same transformed matrix, weights summed back
to original features). Agreement = |overlap| / 3, averaged. Also saves global SHAP plots.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from procure.config import load_settings, path, set_seed
from procure.explain.shap_explain import ShapExplainer, global_importance, global_summary_plots
from procure.models.predict import Predictor
from procure.models.train import MODEL_NAMES
from procure.reporting import save_table


def _top(values: pd.Series, k: int) -> set[str]:
    """Names of the k largest |values|."""
    return set(values.abs().sort_values(ascending=False).index[:k])


def lime_agreement(predictor: Predictor, explainer: ShapExplainer, train_x: np.ndarray, test_x: np.ndarray,
                   n: int | None = None, num_samples: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Per-prediction top-k SHAP/LIME overlap for each model."""
    cfg = load_settings()["explain"]
    seed = load_settings()["seed"] if seed is None else seed
    n = n or cfg["lime_n_predictions"]
    num_samples = num_samples or cfg["lime_num_samples"]
    k = cfg["top_k"]
    rng = np.random.default_rng(seed)
    rows_idx = rng.choice(len(test_x), size=min(n, len(test_x)), replace=False)
    names = predictor.bundle.transformed_features
    orig = pd.Series(predictor.bundle.original_features, index=names)
    m = predictor.bundle.models
    fns = {"cost": lambda z: m["cost"].predict(z), "delay": lambda z: m["delay"].predict(z),
           "failure": lambda z: m["failure"].predict_proba(z)}
    out = []
    for model in MODEL_NAMES:
        mode = "classification" if model == "failure" else "regression"
        lime = LimeTabularExplainer(train_x, feature_names=names, mode=mode, discretize_continuous=False,
                                    random_state=seed)
        sv = explainer.shap_values(test_x[rows_idx], model)
        t0 = time.perf_counter()
        for j, r in enumerate(rows_idx):
            exp = lime.explain_instance(test_x[r], fns[model], num_features=len(names), num_samples=num_samples,
                                        labels=(1,) if mode == "classification" else None)
            pairs = dict(exp.as_map()[1])  # LIME stores the explained output under label 1 in both modes
            w = np.array([pairs.get(i, 0.0) for i in range(len(names))])
            # LIME fits on standardised columns, so weight x standardised deviation is the local
            # contribution of each column - the quantity comparable to a SHAP attribution.
            zscore = (test_x[r] - lime.scaler.mean_) / np.where(lime.scaler.scale_ > 0, lime.scaler.scale_, 1.0)
            contrib = pd.Series(w * zscore, index=names)
            lime_orig = contrib.groupby(orig.values).sum()
            overlap = len(_top(sv.iloc[j], k) & _top(lime_orig, k)) / k
            out.append({"model": model, "row": int(r), "agreement": overlap,
                        "shap_top": sorted(_top(sv.iloc[j], k)), "lime_top": sorted(_top(lime_orig, k))})
        out[-1]["seconds"] = time.perf_counter() - t0
    return pd.DataFrame(out)


def main() -> None:
    """CLI: SHAP global plots + LIME agreement table/JSON."""
    set_seed()
    d = path("processed_dir")
    pred = Predictor()
    expl = ShapExplainer(pred.bundle)
    trv = pd.concat([pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet")], ignore_index=True)
    test = pd.read_parquet(d / "test.parquet")
    train_x, test_x = pred.transform(trv), pred.transform(test)
    plots = global_summary_plots(expl, test_x)
    imp = global_importance(expl, test_x).round(4).reset_index(names="feature")
    save_table(imp, "shap_global_importance", "Mean absolute SHAP value per feature on the 2024 test set")
    agr = lime_agreement(pred, expl, train_x, test_x)
    agr.to_csv(path("results_dir") / "lime_agreement_per_prediction.csv", index=False)
    summ = agr.groupby("model", sort=False).agreement.agg(["mean", "std", "count"]).reset_index()
    summ["full_agreement_%"] = agr.groupby("model", sort=False).agreement.apply(lambda s: 100 * (s == 1).mean()).values
    table = pd.DataFrame({"Model": summ["model"], "Top-3 agreement": [f"{a:.2f} ± {b:.2f}" for a, b in zip(summ["mean"], summ["std"])],
                          "All 3 agree (%)": [f"{v:.0f}" for v in summ["full_agreement_%"]],
                          "Predictions": summ["count"]})
    save_table(table, "shap_lime_agreement", "Top-3 feature agreement between SHAP and LIME (2024 test set)")
    (path("results_dir") / "shap_lime_agreement.json").write_text(json.dumps(
        {r["model"]: {"mean": r["mean"], "std": r["std"]} for _, r in summ.iterrows()}, indent=2))
    print(table.to_string(index=False))
    print("Saved plots:", *plots, sep="\n  ")


if __name__ == "__main__":
    main()
