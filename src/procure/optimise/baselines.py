"""Optimiser baselines with the same evaluation budget as NSGA-II.

* weighted_sum: pymoo GA on a weighted sum of normalised objectives, repeated for
  weighted_sum_n_weights weight vectors (budget split evenly); returns the set of optima.
* random_search: same number of repaired random samples (same sparse sampler as NSGA-II).
* lowest_bid: 100% of the order to the supplier with the lowest quoted unit price.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from procure.config import load_settings
from procure.optimise.nsga import FrontResult, finalise
from procure.optimise.problem import ProcurementProblem, ShareRepair, SparseSampling, evaluate_shares, repair_shares


def budget_evals() -> int:
    """NSGA-II evaluation budget (pop_size x n_gen) used for every baseline."""
    c = load_settings()["optimise"]
    return int(c["pop_size"] * c["n_gen"])


def objective_bounds(problem: ProcurementProblem) -> tuple[np.ndarray, np.ndarray]:
    """Ideal/nadir estimate from all single-supplier solutions (for scalarisation)."""
    F, _ = evaluate_shares(problem.sp, np.eye(problem.n_var))
    lo, hi = F.min(axis=0), F.max(axis=0)
    return lo, np.where(hi - lo > 0, hi, lo + 1.0)


class _Scalarised(Problem):
    """Single-objective view: sum_j w_j * normalised f_j, same budget constraint."""

    def __init__(self, base: ProcurementProblem, w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        self.base, self.w, self.lo, self.hi = base, w, lo, hi
        super().__init__(n_var=base.n_var, n_obj=1, n_ieq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args: object, **kwargs: object) -> None:
        Xr = repair_shares(X, self.base.k, self.base.min_share)
        F, G = evaluate_shares(self.base.sp, Xr)
        out["F"] = ((F - self.lo) / (self.hi - self.lo)) @ self.w
        out["G"] = G
        self.base.n_evals += len(X)


def weight_vectors(n: int, seed: int) -> np.ndarray:
    """Equal weights plus n-1 Dirichlet-sampled weight vectors."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.full(4, 0.25), rng.dirichlet(np.ones(4), size=max(n - 1, 0))])


def weighted_sum(problem: ProcurementProblem, seed: int | None = None) -> FrontResult:
    """GA on weighted sums; equal total evaluations to NSGA-II."""
    cfg = load_settings()["optimise"]
    seed = load_settings()["seed"] if seed is None else seed
    W = weight_vectors(cfg["weighted_sum_n_weights"], seed)
    pop = cfg["baseline_ga_pop"]
    n_gen = max(1, budget_evals() // (len(W) * pop))
    lo, hi = objective_bounds(problem)
    t0 = time.perf_counter()
    xs = []
    for j, w in enumerate(W):
        algo = GA(pop_size=pop, sampling=SparseSampling(problem.k), crossover=SBX(prob=cfg["sbx_prob"], eta=cfg["sbx_eta"]),
                  mutation=PM(eta=cfg["pm_eta"]), repair=ShareRepair(problem.k, problem.min_share), eliminate_duplicates=True)
        res = minimize(_Scalarised(problem, w, lo, hi), algo, ("n_gen", n_gen), seed=seed + j, verbose=False)
        xs.append(np.atleast_2d(res.X if res.X is not None else res.pop.get("X")[:1]))
    return finalise(problem, np.vstack(xs), "Weighted-sum GA", time.perf_counter() - t0)


def random_search(problem: ProcurementProblem, seed: int | None = None) -> FrontResult:
    """Budget-matched random sampling, keep feasible non-dominated points."""
    seed = load_settings()["seed"] if seed is None else seed
    t0 = time.perf_counter()
    X = SparseSampling(problem.k)._do(problem, budget_evals(), random_state=np.random.default_rng(seed))
    problem.n_evals += len(X)
    return finalise(problem, X, "Random search", time.perf_counter() - t0)


def lowest_bid(problem: ProcurementProblem, bid_unit_prices: np.ndarray | None = None) -> FrontResult:
    """Award everything to the lowest quoted (or predicted full-order) unit price."""
    t0 = time.perf_counter()
    prices = bid_unit_prices if bid_unit_prices is not None else problem.sp.unit_cost[:, -1]
    X = np.zeros((1, problem.n_var))
    X[0, int(np.argmin(prices))] = 1.0
    F, G = evaluate_shares(problem.sp, X)
    return FrontResult(X, F, G, bool(G[0, 0] <= 0), 1, time.perf_counter() - t0, "Lowest bid wins")
