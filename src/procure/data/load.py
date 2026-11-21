"""Load raw procurement (NOCOPO/OCDS JSON or CSV), ACLED, macro and supplier-registry files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from procure.config import load_settings, path
from procure.data import geo

CONTRACT_COLUMNS: list[str] = [
    "ocid", "date", "buyer", "supplier_name", "supplier_state", "supplier_lga", "item", "item_category",
    "quantity", "unit", "value", "currency", "method", "n_bidders", "delivery_state", "delivery_lga",
    "delay_days", "quality_score", "short_delivered", "rejected",
]


def _get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Safely walk nested dicts/lists: _get(r, 'awards', 0, 'value', 'amount')."""
    for k in keys:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return default
        if d is None:
            return default
    return d


def flatten_ocds(package: dict[str, Any]) -> pd.DataFrame:
    """Flatten an OCDS release package (one row per release/first item/first award).

    Standard OCDS fields are used where they exist. Delivery outcomes are not part of the core
    OCDS schema; they are read from an extension block contracts[0].implementation.x_outcome
    when present and are otherwise left missing (to be filled by synthetic.py).
    """
    rows = []
    for r in package.get("releases", []):
        outcome = _get(r, "contracts", 0, "implementation", "x_outcome", default={})
        rows.append({
            "ocid": r.get("ocid"),
            "date": _get(r, "contracts", 0, "dateSigned") or r.get("date"),
            "buyer": _get(r, "buyer", "name"),
            "supplier_name": _get(r, "awards", 0, "suppliers", 0, "name"),
            "supplier_state": _get(r, "awards", 0, "suppliers", 0, "address", "region"),
            "supplier_lga": _get(r, "awards", 0, "suppliers", 0, "address", "locality"),
            "item": _get(r, "tender", "items", 0, "description"),
            "item_category": _get(r, "tender", "items", 0, "classification", "description"),
            "quantity": _get(r, "tender", "items", 0, "quantity"),
            "unit": _get(r, "tender", "items", 0, "unit", "name"),
            "value": _get(r, "awards", 0, "value", "amount"),
            "currency": _get(r, "awards", 0, "value", "currency"),
            "method": _get(r, "tender", "procurementMethodDetails") or _get(r, "tender", "procurementMethod"),
            "n_bidders": _get(r, "tender", "numberOfTenderers"),
            "delivery_state": _get(r, "tender", "items", 0, "deliveryAddress", "region"),
            "delivery_lga": _get(r, "tender", "items", 0, "deliveryAddress", "locality"),
            "delay_days": outcome.get("delayDays"),
            "quality_score": outcome.get("qualityScore"),
            "short_delivered": outcome.get("shortDelivered"),
            "rejected": outcome.get("rejected"),
        })
    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)


def load_contracts(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read every contracts CSV and OCDS JSON in raw_dir into one frame with a `source` column."""
    raw_dir = raw_dir or path("raw_dir")
    pats = load_settings()["raw_files"]
    frames = []
    for f in sorted(raw_dir.glob(pats["contracts_csv"])):
        df = pd.read_csv(f, dtype={"ocid": str})
        frames.append(df.reindex(columns=CONTRACT_COLUMNS).assign(source=f.name))
    for f in sorted(raw_dir.glob(pats["ocds_json"])):
        with open(f, encoding="utf-8") as fh:
            frames.append(flatten_ocds(json.load(fh)).assign(source=f.name))
    if not frames:
        raise FileNotFoundError(f"No contract files matching {pats['contracts_csv']} / {pats['ocds_json']} in {raw_dir}")
    frames = [f.astype(object) for f in frames]
    return pd.concat(frames, ignore_index=True)


def filter_region(df: pd.DataFrame, state_col: str = "delivery_state") -> pd.DataFrame:
    """Keep only rows whose state is one of the six North-East states (any spelling)."""
    canon = df[state_col].map(geo.canonical_state)
    return df[canon.notna()].copy()


def load_acled(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read ACLED CSV(s), keep North-East events, return event_date/lat/lon/fatalities/admin1."""
    raw_dir = raw_dir or path("raw_dir")
    files = sorted(raw_dir.glob(load_settings()["raw_files"]["acled_csv"]))
    if not files:
        return pd.DataFrame(columns=["event_date", "latitude", "longitude", "fatalities", "admin1"])
    ev = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ev = filter_region(ev, "admin1")
    ev["admin1"] = ev["admin1"].map(geo.canonical_state)
    ev["event_date"] = pd.to_datetime(ev["event_date"], format="mixed", dayfirst=False)
    ev["fatalities"] = pd.to_numeric(ev["fatalities"], errors="coerce").fillna(0)
    return ev.reset_index(drop=True)


def load_macro(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read monthly macro series: month (YYYY-MM), cpi, exchange_rate_ngn_usd, fuel_price_ngn_litre."""
    raw_dir = raw_dir or path("raw_dir")
    files = sorted(raw_dir.glob(load_settings()["raw_files"]["macro_csv"]))
    if not files:
        raise FileNotFoundError(f"No macro file in {raw_dir}")
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m["month"] = pd.to_datetime(m["month"]).dt.strftime("%Y-%m")
    return m.drop_duplicates("month", keep="last").sort_values("month").reset_index(drop=True)


def load_registry(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read the supplier registry (BPP/CAC/tax status). Empty frame if not provided."""
    raw_dir = raw_dir or path("raw_dir")
    files = sorted(raw_dir.glob(load_settings()["raw_files"]["supplier_registry_csv"]))
    cols = ["supplier_name", "state", "lga", "registered", "debarred", "tax_clearance", "cac"]
    if not files:
        return pd.DataFrame(columns=cols)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).reindex(columns=cols)


def load_all(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load every raw source, with contracts filtered to the North-East."""
    return {
        "contracts": filter_region(load_contracts(raw_dir)),
        "acled": load_acled(raw_dir),
        "macro": load_macro(raw_dir),
        "registry": load_registry(raw_dir),
    }
