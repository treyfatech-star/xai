"""Front-quality metrics: normalised hypervolume and generalised spread (lower = more even)."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV

from procure.optimise.nsga import FrontResult, nondominated


def reference_bounds(fronts: list[FrontResult]) -> tuple[np.ndarray, np.ndarray]:
    """Ideal and nadir of the union of all methods' feasible points (per request)."""
    F = np.vstack([f.F for f in fronts if f.feasible and len(f.F)])
    nd = F[nondominated(F)]
    return nd.min(axis=0), nd.max(axis=0)


def normalised(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Scale objectives so the union front spans [0, 1] in every objective."""
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (F - lo) / span


def hypervolume(front: FrontResult, lo: np.ndarray, hi: np.ndarray, ref: float) -> float:
    """Hypervolume of a front in normalised space with reference point (ref, ..., ref).
    Infeasible results score 0."""
    if not front.feasible or len(front.F) == 0:
        return 0.0
    return float(HV(ref_point=np.full(front.F.shape[1], ref))(normalised(front.F, lo, hi)))


def generalised_spread(front: FrontResult, lo: np.ndarray, hi: np.ndarray, union: np.ndarray) -> float:
    """Generalised spread (Zhou et al., 2006) against the extreme points of the union front.
    Returns NaN for fronts with fewer than 2 points."""
    if not front.feasible or len(front.F) < 2:
        return float("nan")
    S = normalised(front.F, lo, hi)
    U = normalised(union, lo, hi)
    extremes = U[np.argmin(U, axis=0)]
    d_ext = sum(np.min(np.linalg.norm(S - e, axis=1)) for e in extremes)
    D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    mean = nn.mean()
    return float((d_ext + np.abs(nn - mean).sum()) / (d_ext + len(S) * mean))
