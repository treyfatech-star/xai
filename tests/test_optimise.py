"""Phase 4 tests: repair operator, objectives, non-dominated front, selection, baselines, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from procure.optimise import baselines as obl
from procure.optimise.metrics import generalised_spread, hypervolume, reference_bounds
from procure.optimise.nsga import FrontResult, crowding_distance, nondominated, run_nsga2, truncate
from procure.optimise.problem import ProcurementProblem, SupplierPredictions, evaluate_shares, repair_shares
from procure.optimise.select import select_options
from procure.config import load_settings


def make_sp(n: int = 5, budget: float = 1e9, seed: int = 0) -> SupplierPredictions:
    rng = np.random.default_rng(seed)
    grid = np.array(load_settings()["optimise"]["quantity_grid"], float)
    g = len(grid)
    base = rng.uniform(80, 120, n)
    return SupplierPredictions(
        supplier_ids=[f"S{i}" for i in range(n)], grid=grid,
        unit_cost=base[:, None] * (1 - 0.05 * grid[None, :]), delay=rng.uniform(2, 30, (n, 1)) * np.ones((1, g)),
        fail=rng.uniform(0.05, 0.4, (n, 1)) * np.ones((1, g)), quality=rng.uniform(2.5, 4.5, n),
        quantity=100.0, budget=budget)


@pytest.fixture(autouse=True)
def small_nsga(monkeypatch):
    """Shrink NSGA-II for fast tests (config values are restored afterwards)."""
    cfg = load_settings()["optimise"]
    monkeypatch.setitem(cfg, "pop_size", 30)
    monkeypatch.setitem(cfg, "n_gen", 20)
    monkeypatch.setitem(cfg, "weighted_sum_n_weights", 3)
    monkeypatch.setitem(cfg, "baseline_ga_pop", 20)


def test_repair_sum_and_k_limit() -> None:
    X = np.random.default_rng(1).random((50, 8))
    X[0] = 0
    R = repair_shares(X, k=3, min_share=0.05)
    assert np.allclose(R.sum(axis=1), 1) and ((R > 0).sum(axis=1) <= 3).all()
    assert R[(R > 0)].min() >= 0.05 - 1e-12 and R[0, 0] == 1.0


def test_objectives_by_hand() -> None:
    sp = make_sp(2)
    sp.unit_cost[:] = [[10.0] * len(sp.grid), [20.0] * len(sp.grid)]
    sp.delay[:] = [[5.0] * len(sp.grid), [9.0] * len(sp.grid)]
    sp.fail[:] = [[0.1] * len(sp.grid), [0.2] * len(sp.grid)]
    sp.quality[:] = [4.0, 3.0]
    F, G = evaluate_shares(sp, np.array([[0.5, 0.5], [1.0, 0.0]]))
    assert np.allclose(F[0], [1500.0, 9.0, -3.5, 1 - 0.9 * 0.8])
    assert np.allclose(F[1], [1000.0, 5.0, -4.0, 0.1])  # unused supplier does not count
    assert np.allclose(G[:, 0], F[:, 0] - sp.budget)


def test_nsga_front_is_non_dominated_and_feasible() -> None:
    sp = make_sp(6, budget=11_000)
    res = run_nsga2(ProcurementProblem(sp))
    assert res.feasible and len(res.F) > 1
    assert len(nondominated(res.F)) == len(res.F)
    for i in range(len(res.F)):
        dominated = (np.all(res.F <= res.F[i], axis=1) & np.any(res.F < res.F[i], axis=1)).any()
        assert not dominated
    assert (res.F[:, 0] <= sp.budget + 1e-6).all()
    assert np.allclose(res.X.sum(axis=1), 1) and ((res.X > 0).sum(axis=1) <= 3).all()


def test_nsga_is_reproducible() -> None:
    sp = make_sp(5)
    a, b = run_nsga2(ProcurementProblem(sp)), run_nsga2(ProcurementProblem(sp))
    assert np.allclose(a.F, b.F)


def test_infeasible_budget_returns_least_violation() -> None:
    res = run_nsga2(ProcurementProblem(make_sp(4, budget=1.0)))
    assert not res.feasible and len(res.F) == 1


def test_select_four_options_ranked() -> None:
    F = np.array([[1, 9, -1, 0.9], [9, 1, -1, 0.9], [9, 9, -1, 0.1], [4, 4, -3, 0.4], [8, 8, -1, 0.8]], float)
    sel = select_options(F)
    labels = {s.label: s.index for s in sel}
    assert labels == {"balanced": 3, "lowest_cost": 0, "fastest": 1, "most_reliable": 2}
    assert sel[0].label == "balanced"


def test_select_merges_duplicates_and_fills_minimum() -> None:
    F = np.array([[1, 1, -5, 0.1], [2, 2, -4, 0.2], [3, 3, -3, 0.3]], float)
    sel = select_options(F)
    assert sel[0].index == 0
    assert {sel[0].label, *sel[0].also_best_for} == {"lowest_cost", "fastest", "most_reliable", "balanced"}
    assert len(sel) == 3


def test_baselines_and_metrics() -> None:
    sp = make_sp(6)
    fronts = [run_nsga2(ProcurementProblem(sp)), obl.weighted_sum(ProcurementProblem(sp)),
              obl.random_search(ProcurementProblem(sp)), obl.lowest_bid(ProcurementProblem(sp))]
    lo, hi = reference_bounds(fronts)
    union = np.vstack([f.F for f in fronts])
    union = union[nondominated(union)]
    hv = [hypervolume(f, lo, hi, 1.1) for f in fronts]
    assert all(0 <= h <= 1.1 ** 4 for h in hv) and hv[0] >= hv[3]
    assert fronts[3].F.shape == (1, 4) and np.isnan(generalised_spread(fronts[3], lo, hi, union))
    assert hypervolume(FrontResult(np.zeros((1, 2)), np.zeros((1, 4)), np.ones((1, 1)), False, 1, 0, "x"), lo, hi, 1.1) == 0


def test_truncation_keeps_extremes() -> None:
    F = np.random.default_rng(3).random((200, 4))
    keep = truncate(F, 50)
    assert len(keep) == 50
    for j in range(4):
        assert np.argmin(F[:, j]) in keep
    assert np.isinf(crowding_distance(F[:2])).all()
