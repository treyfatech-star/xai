"""Cleaning: dates, currency, names, supplier IDs, de-duplication, imputation, outlier flags.

Anonymisation (Nigeria Data Protection Act 2023): supplier and buyer names are replaced by
salted hashes (supplier_id, buyer_id). The readable supplier name is kept only in the
access-controlled supplier table, never in model outputs or logs.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd

from procure.config import load_settings
from procure.data import geo

_SUFFIX_MAP = {
    "limited": "ltd", "nigeria": "nig", "nig.": "nig", "enterprises": "ent", "enterprise": "ent",
    "company": "co", "services": "svc", "service": "svc", "and": "&",
}
OUTCOME_COLS: list[str] = ["delay_days", "quality_score", "short_delivered", "rejected"]


def normalise_supplier_name(name: object) -> str:
    """Lower-case, strip punctuation and standardise company suffixes ('Limited' -> 'ltd')."""
    s = re.sub(r"[^a-z0-9& ]", " ", str(name).lower())
    toks = [_SUFFIX_MAP.get(t, t) for t in s.split()]
    return " ".join(toks)


def _salt() -> str:
    """Anonymisation salt from the environment (falls back to a dev salt)."""
    d = load_settings()["data"]
    return os.environ.get(d["anonymisation_salt_env"], d["anonymisation_default_salt"])


def make_id(prefix: str, *parts: object) -> str:
    """Deterministic salted hash ID, e.g. SUP-1a2b3c4d."""
    raw = "|".join(str(p) for p in parts) + "|" + _salt()
    return f"{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def supplier_id(name: object, state: object) -> str:
    """Supplier ID from normalised name + canonical state (name + state match)."""
    return make_id("SUP", normalise_supplier_name(name), geo.canonical_state(state) or str(state))


def _to_bool(v: object) -> object:
    """Parse booleans from mixed raw representations; keep missing as NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t", "1.0"}:
        return True
    if s in {"false", "0", "no", "n", "f", "0.0"}:
        return False
    return np.nan


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse mixed date strings: ISO (YYYY-MM-DD...) as year-first, everything else as
    day-first (Nigerian dd/mm/yyyy convention). Unparseable values become NaT."""
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    out[~iso] = pd.to_datetime(s[~iso], format="mixed", dayfirst=True, errors="coerce")
    return out


def standardise(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, convert currency to naira, canonicalise states/LGAs and categories."""
    out = df.copy()
    out["date"] = parse_dates(out["date"])
    out = out[out["date"].notna()].copy()
    out["date"] = out["date"].dt.normalize()
    for c in ["quantity", "value", "n_bidders", "delay_days", "quality_score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["short_delivered", "rejected"]:
        out[c] = out[c].map(_to_bool).astype(object)
    month = out["date"].dt.strftime("%Y-%m")
    fx = macro.set_index("month")["exchange_rate_ngn_usd"]
    cur = out["currency"].fillna(load_settings()["data"]["base_currency"]).astype(str).str.upper().str.strip()
    usd = cur.eq("USD")
    out.loc[usd, "value"] = out.loc[usd, "value"] * month[usd].map(fx)
    out["currency"] = load_settings()["data"]["base_currency"]
    out["delivery_state"] = out["delivery_state"].map(geo.canonical_state)
    out["supplier_state"] = out["supplier_state"].map(geo.canonical_state)
    out["delivery_lga"] = [geo.canonical_lga(s, l) for s, l in zip(out["delivery_state"], out["delivery_lga"])]
    out["supplier_lga"] = [geo.canonical_lga(s, l) for s, l in zip(out["supplier_state"], out["supplier_lga"])]
    out["item"] = out["item"].astype(str).str.strip()
    out["item_category"] = out["item_category"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    out["method"] = out["method"].where(out["method"].notna(), np.nan)
    out["method"] = out["method"].map(lambda m: m if pd.isna(m) else str(m).strip().lower().replace(" ", "_"))
    return out[out["delivery_state"].notna() & out["value"].gt(0) & out["quantity"].gt(0)].copy()


def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add supplier_id (name + state match) and buyer_id; drop raw buyer name."""
    out = df.copy()
    out["supplier_id"] = [supplier_id(n, s) for n, s in zip(out["supplier_name"], out["supplier_state"])]
    out["buyer_id"] = [make_id("BUY", b) for b in out["buyer"]]
    return out.drop(columns=["buyer"])


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates on the configured key (supplier_id, item, date, value)."""
    keys = load_settings()["data"]["dedup_keys"]
    tmp = df.assign(_v=df["value"].round(0))
    keys = ["_v" if k == "value" else k for k in keys]
    return tmp.drop_duplicates(subset=keys, keep="first").drop(columns="_v")


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median/mode imputation of predictors with is_imputed flags. Outcomes are NOT imputed."""
    out = df.copy()
    flags = pd.DataFrame(False, index=out.index, columns=["n_bidders", "method", "supplier_lga", "delivery_lga"])
    m = out["method"].isna()
    out.loc[m, "method"] = out["method"].mode().iloc[0]
    flags["method"] = m
    b = out["n_bidders"].isna()
    med = out.groupby("method")["n_bidders"].transform("median").fillna(out["n_bidders"].median())
    out.loc[b, "n_bidders"] = med[b]
    flags["n_bidders"] = b
    for col, st in [("supplier_lga", "supplier_state"), ("delivery_lga", "delivery_state")]:
        miss = out[col].isna()
        if col == "supplier_lga":
            mode = out.groupby("supplier_id")[col].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
            out.loc[miss, col] = out.loc[miss, "supplier_id"].map(mode)
        still = out[col].isna() & out[st].notna()
        out.loc[still, col] = out.loc[still, st].map(geo.state_capital_lga)
        flags[col] = miss
    out["is_imputed"] = flags.any(axis=1)
    out["imputed_cols"] = flags.apply(lambda r: ",".join(r.index[r.values]), axis=1)
    return out


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag (keep) cost and delay outliers beyond k*IQR (log unit price within item; delay overall)."""
    k = load_settings()["data"]["outlier_iqr_k"]
    out = df.copy()
    lp = np.log(out["value"] / out["quantity"])

    def _iqr_flag(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return (s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))

    out["cost_outlier"] = lp.groupby(out["item"]).transform(_iqr_flag).astype(bool)
    d = out["delay_days"]
    out["delay_outlier"] = _iqr_flag(d.fillna(d.median())) & d.notna()
    return out


def merge_registry(df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Attach compliance status. Suppliers missing from the registry are treated as non-compliant
    (False), which is conservative: the rule layer will exclude and explain them."""
    out = df.copy()
    cols = ["registered", "debarred", "tax_clearance", "cac"]
    if registry.empty:
        for c in cols:
            out[c] = False
        out["in_registry"] = False
        return out
    reg = registry.copy()
    reg["supplier_id"] = [supplier_id(n, s) for n, s in zip(reg["supplier_name"], reg["state"])]
    for c in cols:
        reg[c] = reg[c].map(_to_bool)
    reg = reg.drop_duplicates("supplier_id", keep="last").set_index("supplier_id")[cols]
    out = out.join(reg, on="supplier_id")
    out["in_registry"] = out["registered"].notna()
    out["registered"] = out["registered"].fillna(False).astype(bool)
    out["debarred"] = out["debarred"].fillna(False).astype(bool)
    out["tax_clearance"] = out["tax_clearance"].fillna(False).astype(bool)
    out["cac"] = out["cac"].fillna(False).astype(bool)
    return out


def clean(contracts: pd.DataFrame, macro: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline in a fixed order."""
    df = standardise(contracts, macro)
    df = assign_ids(df)
    df = deduplicate(df)
    df = impute(df)
    df = flag_outliers(df)
    df = merge_registry(df, registry)
    df["has_real_outcome"] = df[OUTCOME_COLS].notna().all(axis=1)
    return df.sort_values(["date", "ocid"]).reset_index(drop=True)
