"""DataStore: supplier registry, candidate suppliers and request features as of a date.

All supplier history is computed with features.history_for, i.e. only from transactions
strictly before the request date, exactly as in training (no train/serve skew).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from procure.config import load_settings, path
from procure.data import geo
from procure.data.features import history_for
from procure.domain import ProcurementRequest

STATUS_COLS: list[str] = ["registered", "debarred", "tax_clearance", "cac"]


@dataclass
class DataStore:
    """In-memory view of processed data used at recommendation time."""

    transactions: pd.DataFrame
    ref_prices: pd.Series
    acled: pd.DataFrame

    @classmethod
    def from_processed(cls) -> "DataStore":
        """Load from data/processed/*.parquet (run `make data` first)."""
        d = path("processed_dir")
        tx = pd.read_parquet(d / "transactions.parquet")
        ref = pd.read_parquet(d / "ref_prices.parquet").set_index("item")["ref_price"]
        acled = pd.read_parquet(d / "acled.parquet")
        return cls(tx, ref, acled)

    def registry(self) -> pd.DataFrame:
        """Latest known profile per supplier (location, compliance status, categories served)."""
        tx = self.transactions.sort_values("date")
        g = tx.groupby("supplier_id")
        reg = g[["supplier_state", "supplier_lga"] + STATUS_COLS].last()
        reg["categories"] = g["item_category"].agg(lambda s: sorted(set(s)))
        reg["display_name"] = g["supplier_name"].agg(lambda s: s.str.strip().mode().iloc[0])
        return reg.reset_index()

    def candidates(self, request: ProcurementRequest) -> pd.DataFrame:
        """Candidate suppliers with compliance status, history as of request.date and bid value."""
        cfg = load_settings()["optimise"]
        tx = self.transactions
        past = tx[tx["date"] < pd.Timestamp(request.date)]
        reg = self.registry()
        if request.candidate_supplier_ids:
            reg = reg[reg["supplier_id"].isin(request.candidate_supplier_ids)]
        else:
            served = past.loc[past["item_category"] == request.item_category, "supplier_id"].value_counts()
            reg = reg[reg["supplier_id"].isin(served.index[: cfg["max_candidates"]])]
        reg = reg.reset_index(drop=True)
        q = pd.DataFrame({"supplier_id": reg["supplier_id"], "date": pd.Timestamp(request.date)})
        hist = history_for(q, tx, self.ref_prices)
        out = pd.concat([reg, hist.reset_index(drop=True)], axis=1)
        ref = self.item_ref_price(request)
        bids = request.bids or {}
        out["bid_unit_price"] = [bids.get(s, ref * np.exp(pi)) for s, pi in zip(out["supplier_id"], out["past_price_index"])]
        out["bid_value"] = out["bid_unit_price"] * float(request.quantity)
        return out

    def item_ref_price(self, request: ProcurementRequest) -> float:
        """Typical 2024-naira unit price of the item (training-years median; category median if new)."""
        ref = float(self.ref_prices.get(request.item, np.nan))
        if np.isnan(ref):
            tx = self.transactions
            past = tx[(tx["date"] < pd.Timestamp(request.date)) & (tx["item_category"] == request.item_category)]
            ref = float(past["unit_cost_real"].median())
        return ref

    def request_context(self, request: ProcurementRequest) -> dict[str, float]:
        """Delivery-side context known on the request date (season and conflict exposure)."""
        rainy = int(pd.Timestamp(request.date).month in load_settings()["data"]["rainy_months"])
        cc = geo.conflict_counts(pd.Series([request.delivery_state]), pd.Series([request.delivery_lga]),
                                 pd.Series([pd.Timestamp(request.date)]), self.acled)
        return {"rainy_season": rainy, "conflict_events_90d": float(cc.iloc[0, 0]),
                "conflict_fatalities_90d": float(cc.iloc[0, 1])}

    def feature_rows(self, request: ProcurementRequest, suppliers: pd.DataFrame,
                     quantities: np.ndarray | None = None) -> pd.DataFrame:
        """Model feature rows: one per (supplier, quantity). quantities defaults to the full order."""
        quantities = np.array([request.quantity]) if quantities is None else np.asarray(quantities, float)
        ctx = self.request_context(request)
        dlat, dlon = geo.coords(pd.Series([request.delivery_state]), pd.Series([request.delivery_lga]))
        slat, slon = geo.coords(suppliers["supplier_state"], suppliers["supplier_lga"])
        dist = geo.haversine_km(slat, slon, dlat[0], dlon[0])
        base = pd.DataFrame({
            "supplier_id": suppliers["supplier_id"].to_numpy(),
            "n_bidders": float(len(suppliers)),
            "supplier_experience": suppliers["supplier_experience"].to_numpy(),
            "past_on_time_rate": suppliers["past_on_time_rate"].to_numpy(),
            "past_failure_rate": suppliers["past_failure_rate"].to_numpy(),
            "past_quality_mean": suppliers["past_quality_mean"].to_numpy(),
            "past_price_index": suppliers["past_price_index"].to_numpy(),
            "distance_km": np.nan_to_num(dist, nan=float(np.nanmedian(dist)) if np.isfinite(dist).any() else 0.0),
            "same_state": (suppliers["supplier_state"] == request.delivery_state).astype(int).to_numpy(),
            "log_item_ref_price": float(np.log(self.item_ref_price(request))),
            "item_category": request.item_category, "item": request.item, "method": request.method,
            "delivery_state": request.delivery_state, "delivery_lga": request.delivery_lga, **ctx,
        })
        rows = [base.assign(quantity=q, log_quantity=np.log(q)) for q in quantities]
        return pd.concat(rows, ignore_index=True)
