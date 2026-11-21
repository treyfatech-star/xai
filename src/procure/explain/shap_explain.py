"""SHAP (TreeExplainer) for the three XGBoost models.

SHAP values are computed on the transformed (one-hot) matrix and summed back to the original
features, so an officer sees "item category" rather than "item_category_fuel". Units:
cost model -> log naira (converted to % effects in narrate.py), delay model -> days,
failure model -> log-odds of the weighted classifier (converted to percentage points).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from procure.models.train import MODEL_NAMES, ModelBundle


class ShapExplainer:
    """TreeExplainer per model plus aggregation to original features."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.explainers = {m: shap.TreeExplainer(bundle.models[m]) for m in MODEL_NAMES}
        self.original = list(dict.fromkeys(bundle.original_features))
        orig = np.array(bundle.original_features)
        self._agg = np.stack([(orig == f).astype(float) for f in self.original], axis=1)  # (n_trans, n_orig)

    def base_value(self, model: str) -> float:
        """Expected model output (log cost, days, or log-odds)."""
        ev = self.explainers[model].expected_value
        return float(np.ravel(ev)[-1])

    def shap_values(self, x: np.ndarray, model: str) -> pd.DataFrame:
        """SHAP values aggregated to original features: (rows, original features)."""
        sv = self.explainers[model].shap_values(x)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        return pd.DataFrame(sv @ self._agg, columns=self.original)

    def explain(self, x: np.ndarray) -> dict[str, pd.DataFrame]:
        """SHAP frames for all three models."""
        return {m: self.shap_values(x, m) for m in MODEL_NAMES}

    def margins(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Raw model outputs (same scale as SHAP values)."""
        m = self.bundle.models
        return {"cost": m["cost"].predict(x), "delay": m["delay"].predict(x),
                "failure": m["failure"].predict(x, output_margin=True)}


def top_features(row: pd.Series, k: int, exclude: list[str] | None = None) -> list[dict[str, Any]]:
    """Top-k features by |SHAP| for one prediction (optionally excluding some features)."""
    r = row.drop(labels=[c for c in (exclude or []) if c in row.index])
    order = r.abs().sort_values(ascending=False).index[:k]
    return [{"feature": f, "shap": float(r[f])} for f in order]


def global_importance(explainer: ShapExplainer, x: np.ndarray) -> pd.DataFrame:
    """Mean |SHAP| per original feature and model (for global summary plots/tables)."""
    return pd.DataFrame({m: explainer.shap_values(x, m).abs().mean() for m in MODEL_NAMES})


def global_summary_plots(explainer: ShapExplainer, x: np.ndarray, out_prefix: str = "shap_global") -> list[str]:
    """Save a bar chart of mean |SHAP| per original feature for each model (300 dpi)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from procure.explain.narrate import FEATURE_LABELS
    from procure.reporting import save_figure

    imp = global_importance(explainer, x)
    units = {"cost": "mean |SHAP| (log naira)", "delay": "mean |SHAP| (days)", "failure": "mean |SHAP| (log-odds)"}
    paths = []
    for m in MODEL_NAMES:
        s = imp[m].sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.barh([FEATURE_LABELS.get(f, f) for f in s.index], s.values, color="#2a6f97")
        ax.set_xlabel(units[m])
        ax.set_title(f"What drives the {m} model (2024 test set)")
        ax.spines[["top", "right"]].set_visible(False)
        paths.append(str(save_figure(fig, f"{out_prefix}_{m}")))
    return paths
