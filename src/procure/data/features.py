"""Feature engineering.

Two stages so that synthetic outcomes can be conditioned on context:
1. add_context_features: things known when the contract is signed (cost in 2024 naira,
   distance, rainy season, conflict in the prior 90 days within 50 km, ...).
2. add_history_features: supplier track record computed ONLY from earlier transactions.
   Outcome-based rates (on-time, failure, quality) use a transaction only after its outcome
   would have been known: signing date + outcome_lag_days + delay_days < current date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.config import load_settings
from procure.data import geo

HISTORY_FEATURES: list[str] = [
    "supplier_experience", "past_on_time_rate", "past_failure_rate", "past_quality_mean", "past_price_index",
]


def cpi_factor(dates: pd.Series, macro: pd.DataFrame) -> pd.Series:
    """Multiplier converting nominal naira at each date into base-year (2024) naira."""
    base = str(load_settings()["data"]["cpi_base_year"])
    cpi = macro.set_index("month")["cpi"]
    base_cpi = cpi[[m for m in cpi.index if m.startswith(base)]].mean()
    month = pd.to_datetime(dates).dt.strftime("%Y-%m")
    c = month.map(cpi)
    c = c.fillna(cpi.iloc[-1])  # months beyond the macro file use the latest CPI
    return base_cpi / c


def add_context_features(df: pd.DataFrame, macro: pd.DataFrame, acled: pd.DataFrame) -> pd.DataFrame:
    """Add cost, geography, season and conflict features known at signing time."""
    cfg = load_settings()["data"]
    out = df.reset_index(drop=True).copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["unit_cost_nominal"] = out["value"] / out["quantity"]
    out["unit_cost_real"] = out["unit_cost_nominal"] * cpi_factor(out["date"], macro)
    out["log_quantity"] = np.log(out["quantity"].astype(float))
    slat, slon = geo.coords(out["supplier_state"], out["supplier_lga"])
    dlat, dlon = geo.coords(out["delivery_state"], out["delivery_lga"])
    out["distance_km"] = geo.haversine_km(slat, slon, dlat, dlon)
    out["distance_km"] = out["distance_km"].fillna(out["distance_km"].median())
    out["same_state"] = (out["supplier_state"] == out["delivery_state"]).astype(int)
    out["rainy_season"] = out["month"].isin(cfg["rainy_months"]).astype(int)
    cc = geo.conflict_counts(out["delivery_state"], out["delivery_lga"], out["date"], acled)
    out[["conflict_events_90d", "conflict_fatalities_90d"]] = cc.to_numpy()
    return out


def compute_failed(df: pd.DataFrame) -> pd.Series:
    """failed = late by more than failure_delay_days OR short-delivered OR rejected."""
    lim = load_settings()["data"]["failure_delay_days"]
    return ((df["delay_days"] > lim) | df["short_delivered"].astype(bool) | df["rejected"].astype(bool)).astype(int)


def item_reference_prices(df: pd.DataFrame) -> pd.Series:
    """Median real unit cost per item using reference (training) years only."""
    ref = df[df["year"].isin(load_settings()["data"]["reference_years"])]
    s = ref.groupby("item")["unit_cost_real"].median()
    return s.reindex(df["item"].unique()).fillna(df.groupby("item")["unit_cost_real"].median())


def _asof_sums(events: pd.DataFrame, event_date: str, value_cols: list[str], queries: pd.DataFrame) -> pd.DataFrame:
    """For each query (supplier_id, date), sum value_cols over that supplier's events whose
    event_date is STRICTLY before the query date. Returns columns n + value_cols."""
    ev = events[["supplier_id", event_date] + value_cols].dropna(subset=[event_date]).copy()
    ev = ev.sort_values(event_date, kind="mergesort")
    ev["n"] = 1.0
    cs = ev.groupby("supplier_id")[["n"] + value_cols].cumsum()
    ev[["n"] + value_cols] = cs
    ev = ev.rename(columns={event_date: "_edate"})
    ev["_edate"] = ev["_edate"].astype("datetime64[ns]")
    q = queries[["supplier_id", "date"]].copy()
    q["date"] = q["date"].astype("datetime64[ns]")
    q["_qid"] = np.arange(len(q))
    q = q.sort_values("date", kind="mergesort")
    m = pd.merge_asof(q, ev, left_on="date", right_on="_edate", by="supplier_id",
                      allow_exact_matches=False, direction="backward")
    m = m.sort_values("_qid")
    res = m[["n"] + value_cols].fillna(0.0)
    res.index = queries.index
    return res


def history_for(queries: pd.DataFrame, transactions: pd.DataFrame, ref_prices: pd.Series) -> pd.DataFrame:
    """Supplier history features for arbitrary (supplier_id, date) queries, using only
    transactions dated (or with outcomes known) strictly before each query date."""
    cfg = load_settings()["data"]
    k = cfg["history_prior_strength"]
    pri = cfg["history_priors"]
    tx = transactions.copy()
    tx["on_time"] = (tx["delay_days"] <= cfg["on_time_tolerance_days"]).astype(float)
    tx["known_date"] = tx["date"] + pd.to_timedelta(cfg["outcome_lag_days"] + tx["delay_days"].fillna(0), unit="D")
    tx["price_index"] = np.log(tx["unit_cost_real"] / tx["item"].map(ref_prices))
    signed = _asof_sums(tx, "date", ["price_index"], queries)
    known = _asof_sums(tx.dropna(subset=["failed"]), "known_date", ["on_time", "failed", "quality_score"], queries)
    out = pd.DataFrame(index=queries.index)
    out["supplier_experience"] = signed["n"]
    out["past_price_index"] = (k * pri["price_index"] + signed["price_index"]) / (k + signed["n"])
    out["past_on_time_rate"] = (k * pri["on_time_rate"] + known["on_time"]) / (k + known["n"])
    out["past_failure_rate"] = (k * pri["failure_rate"] + known["failed"]) / (k + known["n"])
    out["past_quality_mean"] = (k * pri["quality_mean"] + known["quality_score"]) / (k + known["n"])
    out["past_outcomes_known"] = known["n"]
    return out


def add_history_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Attach supplier history features to every transaction; returns (frame, item ref prices)."""
    ref = item_reference_prices(df)
    hist = history_for(df[["supplier_id", "date"]], df, ref)
    out = df.copy()
    out[hist.columns] = hist
    out["log_item_ref_price"] = np.log(out["item"].map(ref).astype(float))
    return out, ref
