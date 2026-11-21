"""Multi-objective supplier-allocation problem for one request (pymoo).

Decision variables x_i in [0, 1] = share of the order quantity Q given to eligible supplier i,
with sum x = 1 and at most k suppliers (enforced by ShareRepair).

  f1 expected cost         = sum_i x_i Q c_i(x_i Q)                 (naira, 2024 prices)
  f2 worst delay           = max_{i: x_i > 0} d_i(x_i Q)            (days)
  f3 negative quality      = - sum_i x_i q_i                        (score 1-5)
  f4 any-failure prob.     = 1 - prod_{i: x_i > 0} (1 - p_i(x_i Q))
  g1 budget                = f1 - B <= 0

Predictions depend on the quantity a supplier receives, so they are pre-computed once on a
share grid (config optimise.quantity_grid) and linearly interpolated during the search; this
keeps an NSGA-II run to a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling

from procure.config import load_settings
from procure.domain import ProcurementRequest
from procure.models.predict import Predictor
from procure.store import DataStore

OBJECTIVES: tuple[str, ...] = ("cost", "worst_delay", "neg_quality", "failure_prob")


@dataclass
class SupplierPredictions:
    """Model predictions for each eligible supplier on a grid of order shares."""

    supplier_ids: list[str]
    grid: np.ndarray            # (g,) shares of Q
    unit_cost: np.ndarray       # (n, g) naira per unit
    delay: np.ndarray           # (n, g) days
    fail: np.ndarray            # (n, g) probability
    quality: np.ndarray         # (n,) expected quality score
    quantity: float
    budget: float

    @property
    def n(self) -> int:
        """Number of suppliers."""
        return len(self.supplier_ids)

    def at(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate (unit_cost, delay, fail) at shares X (pop, n)."""
        X = np.atleast_2d(X)
        c, d, f = (np.empty_like(X, dtype=float) for _ in range(3))
        for i in range(self.n):
            xi = np.clip(X[:, i], self.grid[0], self.grid[-1])
            c[:, i] = np.interp(xi, self.grid, self.unit_cost[i])
            d[:, i] = np.interp(xi, self.grid, self.delay[i])
            f[:, i] = np.interp(xi, self.grid, self.fail[i])
        return c, d, f


def build_predictions(store: DataStore, predictor: Predictor, request: ProcurementRequest,
                      eligible: pd.DataFrame) -> tuple[SupplierPredictions, pd.DataFrame]:
    """Predict cost/delay/failure for each eligible supplier at every grid quantity.
    Returns the predictions and the feature rows (reused by the explanation layer)."""
    grid = np.asarray(load_settings()["optimise"]["quantity_grid"], float)
    rows = store.feature_rows(request, eligible, grid * float(request.quantity))
    p = predictor.predict(rows)
    n, g = len(eligible), len(grid)
    shape = lambda col: p[col].to_numpy().reshape(g, n).T  # rows are grouped by quantity  # noqa: E731
    sp = SupplierPredictions(
        supplier_ids=list(eligible["supplier_id"]), grid=grid, unit_cost=shape("unit_cost"),
        delay=shape("delay_days"), fail=shape("failure_prob"),
        quality=eligible["past_quality_mean"].to_numpy(float), quantity=float(request.quantity),
        budget=float(request.budget))
    return sp, rows


def evaluate_shares(sp: SupplierPredictions, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Objectives F (pop, 4) and constraint G (pop, 1) for already-repaired shares X."""
    X = np.atleast_2d(X)
    used = X > 0
    c, d, f = sp.at(X)
    f1 = np.sum(X * sp.quantity * c, axis=1)
    f2 = np.max(np.where(used, d, 0.0), axis=1)
    f3 = -np.sum(X * sp.quality[None, :], axis=1)
    f4 = 1.0 - np.prod(np.where(used, 1.0 - f, 1.0), axis=1)
    return np.column_stack([f1, f2, f3, f4]), (f1 - sp.budget)[:, None]


def repair_shares(X: np.ndarray, k: int, min_share: float) -> np.ndarray:
    """Keep the k largest shares, drop shares below min_share, renormalise to sum 1.
    A row with no positive share becomes 100% to its first supplier."""
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0, None).copy()
    if X.shape[1] > k:
        drop = np.argsort(-X, axis=1, kind="stable")[:, k:]
        np.put_along_axis(X, drop, 0.0, axis=1)
    X[X.sum(axis=1) <= 0, 0] = 1.0
    X = X / X.sum(axis=1, keepdims=True)
    top = np.argmax(X, axis=1)
    X[(X > 0) & (X < min_share)] = 0.0
    X[np.arange(len(X)), top] = np.maximum(X[np.arange(len(X)), top], 1e-12)
    return X / X.sum(axis=1, keepdims=True)


class ShareRepair(Repair):
    """pymoo repair operator enforcing sum = 1 and at most k suppliers."""

    def __init__(self, k: int, min_share: float) -> None:
        super().__init__()
        self.k, self.min_share = k, min_share

    def _do(self, problem: Problem, X: np.ndarray, **kwargs: object) -> np.ndarray:
        return repair_shares(X, self.k, self.min_share)


class SparseSampling(Sampling):
    """Initial population: every single-supplier solution, then random 1..k supplier mixes
    with Dirichlet shares. Also used by the random-search baseline so both start equal."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def _do(self, problem: Problem, n_samples: int, random_state: np.random.Generator | None = None,
            **kwargs: object) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng(load_settings()["seed"])
        n = problem.n_var
        X = np.zeros((n_samples, n))
        m = min(n, n_samples)
        X[np.arange(m), np.arange(m)] = 1.0
        for r in range(m, n_samples):
            size = int(rng.integers(1, min(self.k, n) + 1))
            idx = rng.choice(n, size=size, replace=False)
            X[r, idx] = rng.dirichlet(np.ones(size))
        return X


class ProcurementProblem(Problem):
    """pymoo Problem wrapping SupplierPredictions (4 objectives, 1 budget constraint)."""

    def __init__(self, sp: SupplierPredictions, k: int | None = None) -> None:
        cfg = load_settings()["optimise"]
        self.sp = sp
        self.k = min(k or cfg["k_max"], sp.n)
        self.min_share = cfg["min_share"]
        self.n_evals = 0
        super().__init__(n_var=sp.n, n_obj=4, n_ieq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args: object, **kwargs: object) -> None:
        Xr = repair_shares(X, self.k, self.min_share)
        out["F"], out["G"] = evaluate_shares(self.sp, Xr)
        self.n_evals += len(X)
