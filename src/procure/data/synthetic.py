"""Synthetic outcomes, generated ONLY where real outcomes are missing.

Distributions (config: synthetic.*) are conditioned on rainy season, conflict exposure and
distance. Any row with at least one generated outcome gets is_synthetic = True; real values
are never overwritten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.config import load_settings
from procure.data.clean import OUTCOME_COLS
from procure.data.features import compute_failed


def expected_delay(df: pd.DataFrame) -> np.ndarray:
    """Mean delay (days) implied by the configured conditional model."""
    c = load_settings()["synthetic"]["delay"]
    conflict = np.minimum(df["conflict_events_90d"].to_numpy(float), c["conflict_cap_events"])
    return (c["base_mean_days"] + c["rainy_season_add_days"] * df["rainy_season"].to_numpy(float)
            + c["per_conflict_event_days"] * conflict + c["per_100km_days"] * df["distance_km"].to_numpy(float) / 100)


def generate_outcomes(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Fill missing delay_days, quality_score, short_delivered, rejected; set is_synthetic and failed."""
    s = load_settings()
    c = s["synthetic"]
    rng = np.random.default_rng(s["seed"] if seed is None else seed)
    out = df.copy()
    n = len(out)
    cap = c["delay"]["conflict_cap_events"]
    conflict = np.minimum(out["conflict_events_90d"].to_numpy(float), cap)
    shape = c["delay"]["gamma_shape"]
    delay = np.round(rng.gamma(shape, expected_delay(out) / shape, size=n))
    q = c["quality"]
    quality = np.clip(rng.normal(q["mean"] + q["rainy_season_effect"] * out["rainy_season"].to_numpy(float)
                                 + q["per_conflict_event"] * conflict, q["sd"], size=n), q["min"], q["max"]).round(1)
    sd = c["short_delivery"]
    p_short = sd["base_prob"] + sd["per_conflict_event"] * conflict + sd["per_100km"] * out["distance_km"].to_numpy(float) / 100
    short = rng.random(n) < np.clip(p_short, 0, 1)
    generated = {"delay_days": delay, "quality_score": quality, "short_delivered": short}
    synth = np.zeros(n, dtype=bool)
    for col, vals in generated.items():
        miss = out[col].isna().to_numpy()
        out[col] = out[col].astype(object) if col == "short_delivered" else out[col]
        out.loc[miss, col] = vals[miss]
        synth |= miss
    miss_rej = out["rejected"].isna().to_numpy()
    out["rejected"] = out["rejected"].astype(object)
    out.loc[miss_rej, "rejected"] = out.loc[miss_rej, "quality_score"].astype(float) < c["rejection_quality_below"]
    synth |= miss_rej
    out["delay_days"] = out["delay_days"].astype(float)
    out["quality_score"] = out["quality_score"].astype(float)
    out["short_delivered"] = out["short_delivered"].astype(bool)
    out["rejected"] = out["rejected"].astype(bool)
    out["is_synthetic"] = synth
    out["failed"] = compute_failed(out)
    assert out[OUTCOME_COLS].notna().all().all()
    return out
