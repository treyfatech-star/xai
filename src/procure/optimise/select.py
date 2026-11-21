"""Pick officer-facing options from the Pareto front.

lowest_cost = min f1, fastest = min f2, most_reliable = min f4, balanced = closest (Euclidean)
to the ideal point after min-max normalising each objective over the front. Options are
ranked by that same distance (balanced first). If two labels pick the same allocation they
are merged; if fewer than min_options remain, the next closest distinct points are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from procure.config import load_settings

LABEL_OBJECTIVE = {"lowest_cost": 0, "fastest": 1, "best_quality": 2, "most_reliable": 3}


@dataclass
class Selection:
    """One chosen front point."""

    label: str
    index: int
    distance: float
    also_best_for: list[str] = field(default_factory=list)


def normalise(F: np.ndarray) -> np.ndarray:
    """Min-max normalise columns to [0, 1] (constant columns become 0)."""
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (F - lo) / span


def _argmin_tiebreak(F: np.ndarray, col: int) -> int:
    """argmin of one objective, ties broken by the normalised sum of all objectives."""
    N = normalise(F)
    order = np.lexsort((N.sum(axis=1), np.round(F[:, col], 9)))
    return int(order[0])


def select_options(F: np.ndarray, labels: list[str] | None = None) -> list[Selection]:
    """Return ranked selections (rank = order in list)."""
    cfg = load_settings()["explain"]
    labels = labels or cfg["options"]
    if len(F) == 0:
        return []
    dist = np.linalg.norm(normalise(F), axis=1)
    chosen: dict[int, Selection] = {}
    for lab in labels:
        idx = int(np.argmin(dist)) if lab == "balanced" else _argmin_tiebreak(F, LABEL_OBJECTIVE[lab])
        if idx in chosen:
            chosen[idx].also_best_for.append(lab)
        else:
            chosen[idx] = Selection(lab, idx, float(dist[idx]))
    for idx in np.argsort(dist):
        if len(chosen) >= min(cfg["min_options"], len(F)):
            break
        if int(idx) not in chosen:
            chosen[int(idx)] = Selection("alternative", int(idx), float(dist[idx]))
    ranked = sorted(chosen.values(), key=lambda s: s.distance)
    return ranked[: cfg["max_options"]]
