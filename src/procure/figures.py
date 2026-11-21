"""Thesis figures (300-dpi PNG). Categorical colours follow a validated colour-blind-safe order
(fixed order, never cycled); one y-axis per chart; recessive grid; legends present."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from procure.reporting import save_figure  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
METHOD_ORDER = ["NSGA-II", "Random search", "Weighted-sum GA", "Lowest bid wins"]


def _style(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light grid behind marks, secondary ink for ticks."""
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def prediction_comparison(per_seed: pd.DataFrame) -> str:
    """Primary metric per target and model on the real-only 2024 test set (mean ± std over seeds)."""
    primary = {"cost": ("MAE", "Unit-cost MAE (N, 2024)"), "delay": ("MAE", "Delay MAE (days)"),
               "failure": ("AUC", "Failure AUC-ROC (higher is better)")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, (tgt, (metric, title)) in zip(axes, primary.items()):
        d = per_seed[(per_seed.target == tgt) & (per_seed.eval_set == "real_only")]
        g = d.groupby("model", sort=False)[metric].agg(["mean", "std"]).fillna(0)
        colors = [SERIES[0] if m == "XGBoost" else SERIES[1 + i % 3] for i, m in enumerate(g.index)]
        ax.bar(range(len(g)), g["mean"], yerr=g["std"], color=colors, width=0.6, capsize=3,
               edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK2, "lw": 1})
        ax.set_xticks(range(len(g)), [m.replace(" ", "\n") for m in g.index], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        for i, v in enumerate(g["mean"]):
            ax.annotate(f"{v:,.3g}", (i, v), ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 3),
                        textcoords="offset points")
        if metric == "AUC":
            ax.set_ylim(0.5, 1.0)
        _style(ax)
    fig.tight_layout()
    return str(save_figure(fig, "fig_prediction_comparison"))


def optimiser_boxplots(results: dict[str, pd.DataFrame]) -> str:
    """Hypervolume distribution per method, one panel per benchmark setting (shared scale)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 3.8), sharey=True, facecolor=SURFACE)
    for ax, (name, df) in zip(np.atleast_1d(axes), results.items()):
        data = [df.loc[df.method == m, "hypervolume"].to_numpy() for m in METHOD_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "lw": 1.5},
                        whiskerprops={"color": INK2}, capprops={"color": INK2},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK2, "markeredgecolor": "none"})
        for patch, c in zip(bp["boxes"], SERIES):
            patch.set(facecolor=c, alpha=0.85, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(range(1, 5), [m.replace(" ", "\n") for m in METHOD_ORDER], fontsize=8)
        ax.set_title(f"{name.capitalize()} setting (mean {df.n_suppliers.mean():.0f} suppliers)", fontsize=10,
                     color=INK, loc="left")
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Normalised hypervolume", color=INK2)
    fig.tight_layout()
    return str(save_figure(fig, "fig_optimiser_hypervolume"))


def backtest_scatter(df: pd.DataFrame) -> str:
    """Expected cost saving vs failure-risk reduction of the top option, per back-tested contract."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.scatter(df.cost_saving_pct, df.risk_reduction_pp, s=36, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
               label="Contract (top-ranked option vs actual award)")
    ax.set_xlabel("Expected cost saving (%)", color=INK2)
    ax.set_ylabel("Failure-risk reduction (percentage points)", color=INK2)
    ax.set_title("Back-test on real 2024 contracts", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return str(save_figure(fig, "fig_backtest"))


def pareto_example(front: list[dict], options: list[dict], budget: float) -> str:
    """Example Pareto front: cost vs worst delay, colour = failure risk (sequential), options starred."""
    pf = pd.DataFrame(front)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    sc = ax.scatter(pf.cost / 1e6, pf.worst_delay, c=pf.failure_prob, cmap="Blues", s=28, edgecolor=SURFACE, linewidth=1)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Probability any supplier fails", color=INK2)
    for o in options:
        ax.scatter(o["total_cost"] / 1e6, o["worst_delay_days"], marker="*", s=220, color=SERIES[1], edgecolor=INK, lw=0.6,
                   zorder=3)
        ax.annotate(f"#{o['rank']} {o['label'].replace('_', ' ')}", (o["total_cost"] / 1e6, o["worst_delay_days"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    if budget <= 1.5 * pf.cost.max():
        ax.axvline(budget / 1e6, color=INK2, ls="--", lw=1)
    ax.set_xlabel("Total expected cost (N million, 2024 prices)", color=INK2)
    ax.set_ylabel("Slowest delivery (days late)", color=INK2)
    ax.set_title("Pareto front for the demonstration request", fontsize=10, color=INK, loc="left")
    _style(ax)
    return str(save_figure(fig, "fig_pareto_example"))


def latency_hist(seconds: list[float], budget: float) -> str:
    """End-to-end recommend() latency."""
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=SURFACE)
    ax.hist(seconds, bins=15, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axvline(budget, color=SERIES[7], ls="--", lw=1.5, label=f"{budget:.0f} s requirement")
    ax.set_xlabel("End-to-end time per request (s)", color=INK2)
    ax.set_ylabel("Requests", color=INK2)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return str(save_figure(fig, "fig_latency"))


def data_overview(tx: pd.DataFrame) -> str:
    """Contracts per year, split real vs synthetic outcomes (stacked, 2px gaps)."""
    g = tx.groupby([tx.date.dt.year, "is_synthetic"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURFACE)
    ax.bar(g.index, g.get(False, 0), color=SERIES[0], edgecolor=SURFACE, linewidth=2, label="Recorded outcomes")
    ax.bar(g.index, g.get(True, 0), bottom=g.get(False, 0), color=SERIES[1], edgecolor=SURFACE, linewidth=2,
           label="Synthetic outcomes")
    ax.set_ylabel("Contracts", color=INK2)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return str(save_figure(fig, "fig_data_overview"))
