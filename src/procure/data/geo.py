"""Geography helpers: canonical state/LGA names, coordinates, distances, conflict counts."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from procure.config import load_geo, load_settings

EARTH_RADIUS_KM = 6371.0088  # physical constant, not a tunable parameter


def _key(name: object) -> str:
    """Lower-case, strip punctuation/extra spaces for name matching."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _state_lookup() -> dict[str, str]:
    """Map every known spelling of a state to its canonical name."""
    out: dict[str, str] = {}
    for canon, aliases in load_geo()["state_aliases"].items():
        out[_key(canon)] = canon
        for a in aliases:
            out[_key(a)] = canon
    return out


@lru_cache(maxsize=1)
def _lga_lookup() -> dict[tuple[str, str], str]:
    """Map (state, any LGA spelling) to canonical LGA name."""
    out: dict[tuple[str, str], str] = {}
    for state, lgas in load_geo()["lgas"].items():
        for canon, info in lgas.items():
            out[(state, _key(canon))] = canon
            for a in info.get("aliases", []):
                out[(state, _key(a))] = canon
    return out


def canonical_state(name: object) -> str | None:
    """Return the canonical state name, or None if not one of the six North-East states."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    return _state_lookup().get(_key(name))


def canonical_lga(state: str | None, name: object) -> str | None:
    """Return the canonical LGA name within a state, or None when unknown."""
    if state is None or name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    return _lga_lookup().get((state, _key(name)))


@lru_cache(maxsize=1)
def lga_table() -> pd.DataFrame:
    """DataFrame with columns state, lga, lat, lon."""
    rows = [
        {"state": st, "lga": lga, "lat": float(i["lat"]), "lon": float(i["lon"])}
        for st, lgas in load_geo()["lgas"].items()
        for lga, i in lgas.items()
    ]
    return pd.DataFrame(rows)


def state_capital_lga(state: str) -> str:
    """First LGA listed for the state (the state capital in geo.yaml)."""
    return next(iter(load_geo()["lgas"][state]))


def coords(state: pd.Series, lga: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised lookup of (lat, lon) arrays for state/LGA series (NaN if unknown)."""
    t = lga_table().set_index(["state", "lga"])
    idx = pd.MultiIndex.from_arrays([state.astype(object), lga.astype(object)])
    sub = t.reindex(idx)
    return sub["lat"].to_numpy(dtype=float), sub["lon"].to_numpy(dtype=float)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km (vectorised)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def conflict_counts(
    state: pd.Series,
    lga: pd.Series,
    dates: pd.Series,
    events: pd.DataFrame,
    radius_km: float | None = None,
    window_days: int | None = None,
) -> pd.DataFrame:
    """Count conflict events and fatalities within radius_km of each LGA in the window_days
    strictly BEFORE each date (window [date - window_days, date)).

    events needs columns event_date, latitude, longitude, fatalities.
    Returns DataFrame (same index as inputs) with conflict_events_90d, conflict_fatalities_90d.
    """
    cfg = load_settings()["data"]
    radius_km = cfg["conflict_radius_km"] if radius_km is None else radius_km
    window_days = cfg["conflict_window_days"] if window_days is None else window_days
    out = pd.DataFrame(index=state.index, data={"conflict_events_90d": 0.0, "conflict_fatalities_90d": 0.0})
    if events is None or len(events) == 0:
        return out
    ev_dates = pd.to_datetime(events["event_date"]).to_numpy(dtype="datetime64[D]")
    ev_lat = events["latitude"].to_numpy(float)
    ev_lon = events["longitude"].to_numpy(float)
    ev_fat = events["fatalities"].fillna(0).to_numpy(float)
    t = lga_table()
    d = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    window = np.timedelta64(int(window_days), "D")
    keys = pd.Series(list(zip(state, lga)), index=state.index)
    for (st, lg), idx in keys.groupby(keys).groups.items():
        row = t[(t.state == st) & (t.lga == lg)]
        if row.empty:
            continue
        dist = haversine_km(row.lat.iloc[0], row.lon.iloc[0], ev_lat, ev_lon)
        mask = dist <= radius_km
        order = np.argsort(ev_dates[mask])
        sd = ev_dates[mask][order]
        cum_fat = np.concatenate([[0.0], np.cumsum(ev_fat[mask][order])])
        pos = out.index.get_indexer(idx)
        td = d[pos]
        hi = np.searchsorted(sd, td, side="left")
        lo = np.searchsorted(sd, td - window, side="left")
        out.iloc[pos, 0] = hi - lo
        out.iloc[pos, 1] = cum_fat[hi] - cum_fat[lo]
    return out
