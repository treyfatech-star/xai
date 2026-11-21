"""NSGA-II runner (pop 100, 200 generations, SBX 0.9, polynomial mutation, seeded)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from procure.config import load_settings
from procure.optimise.problem import ProcurementProblem, ShareRepair, SparseSampling, repair_shares


@dataclass
class FrontResult:
    """A set of solutions returned by any optimiser (Pareto set for NSGA-II)."""

    X: np.ndarray
    F: np.ndarray
    G: np.ndarray
    feasible: bool
    n_evals: int
    runtime: float
    method: str


def nondominated(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of F (minimisation)."""
    if len(F) == 0:
        return np.array([], dtype=int)
    return NonDominatedSorting().do(F, only_non_dominated_front=True)


def unique_rows(X: np.ndarray, F: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Indices of rows with distinct objective vectors."""
    _, idx = np.unique(np.round(F, decimals), axis=0, return_index=True)
    return np.sort(idx)


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance (boundary points get +inf)."""
    n, m = F.shape
    cd = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j] - F[order[0], j]
        cd[order[0]] = cd[order[-1]] = np.inf
        if span > 0:
            cd[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return cd


def truncate(F: np.ndarray, max_size: int) -> np.ndarray:
    """Indices kept after iteratively removing the most crowded points (chunks of 10%)."""
    keep = np.arange(len(F))
    while len(keep) > max_size:
        cd = crowding_distance(F[keep])
        n_drop = max(1, min(len(keep) - max_size, len(keep) // 10))
        keep = np.delete(keep, np.argsort(cd, kind="stable")[:n_drop])
    return keep


def finalise(problem: ProcurementProblem, X: np.ndarray, method: str, runtime: float) -> FrontResult:
    """Repair, evaluate, keep feasible non-dominated distinct points (or least-violating if none)."""
    from procure.optimise.problem import evaluate_shares

    X = repair_shares(X, problem.k, problem.min_share)
    F, G = evaluate_shares(problem.sp, X)
    ok = G[:, 0] <= 0
    if ok.any():
        X, F, G = X[ok], F[ok], G[ok]
        nd = nondominated(F)
        X, F, G = X[nd], F[nd], G[nd]
        u = unique_rows(X, F)
        X, F, G = X[u], F[u], G[u]
        t = truncate(F, load_settings()["optimise"]["archive_max_size"])
        return FrontResult(X[t], F[t], G[t], True, problem.n_evals, runtime, method)
    best = np.argsort(G[:, 0])[:1]
    return FrontResult(X[best], F[best], G[best], False, problem.n_evals, runtime, method)


class ArchiveCallback(Callback):
    """External archive: remembers every evaluated solution so the returned front is the
    non-dominated set of ALL evaluations (same treatment as the random-search baseline),
    not only the final population of pop_size points."""

    def __init__(self) -> None:
        super().__init__()
        self.X: list[np.ndarray] = []

    def notify(self, algorithm: object) -> None:
        off = getattr(algorithm, "off", None)
        src = off if off is not None and len(off) else algorithm.pop
        self.X.append(src.get("X"))


def run_nsga2(problem: ProcurementProblem, seed: int | None = None, archive: bool = True) -> FrontResult:
    """Run NSGA-II with the configured operators and return the feasible Pareto front
    (of the external archive by default, or of the final population if archive=False)."""
    cfg = load_settings()["optimise"]
    seed = load_settings()["seed"] if seed is None else seed
    algo = NSGA2(pop_size=cfg["pop_size"], sampling=SparseSampling(problem.k),
                 crossover=SBX(prob=cfg["sbx_prob"], eta=cfg["sbx_eta"]), mutation=PM(eta=cfg["pm_eta"]),
                 repair=ShareRepair(problem.k, problem.min_share), eliminate_duplicates=True)
    cb = ArchiveCallback()
    t0 = time.perf_counter()
    res = minimize(problem, algo, ("n_gen", cfg["n_gen"]), seed=seed, verbose=False, save_history=False, callback=cb)
    X = np.vstack(cb.X + [res.pop.get("X")]) if archive else res.pop.get("X")
    return finalise(problem, X, "NSGA-II", time.perf_counter() - t0)
