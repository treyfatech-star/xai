"""SIMULATED raw data for development and demonstration.

Writes NOCOPO/OCDS-like contracts (CSV + a small OCDS JSON package), an ACLED-like
conflict file, a macro file and a supplier registry into data/raw. It is used only when
data/raw contains no real files; the file names follow the patterns the loader expects and a
marker file records that the folder is simulated. The generating "true process" is documented in
config/demo_catalog.yaml so the thesis can describe the simulation. Results produced on this
data are illustrative and must be re-run on real data before being reported as findings.

The generator deliberately injects messiness that clean.py must handle: name variants,
mixed date formats, state/LGA spelling variants, USD-denominated rows, duplicates and
missing values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from procure.config import load_settings, path, resolve, _read_yaml
from procure.data import geo

DEMO_MARKER = "SIMULATED_DATA_README.txt"


def _catalog() -> dict[str, Any]:
    """Load config/demo_catalog.yaml."""
    return _read_yaml(resolve("config/demo_catalog.yaml"))


def _period_value(table: dict, ts: pd.Timestamp) -> float:
    """Look up a yearly value with optional H1/H2 split keys (e.g. '2023H2')."""
    half = "H1" if ts.month <= 6 else "H2"
    key = f"{ts.year}{half}"
    return float(table[key] if key in table else table[ts.year])


def make_macro(cat: dict[str, Any]) -> pd.DataFrame:
    """Monthly CPI (compounded from annual inflation), exchange rate and fuel price."""
    months = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    cpi, level = [], 100.0
    for m in months:
        level *= (1 + cat["annual_inflation"][m.year]) ** (1 / 12)
        cpi.append(level)
    return pd.DataFrame({
        "month": months.strftime("%Y-%m"),
        "cpi": np.round(cpi, 3),
        "exchange_rate_ngn_usd": [_period_value(cat["exchange_rate"], m) for m in months],
        "fuel_price_ngn_litre": [_period_value(cat["fuel_price"], m) for m in months],
    })


def make_acled(cat: dict[str, Any], rng: np.random.Generator, per_year: int) -> pd.DataFrame:
    """Conflict events concentrated in Borno/Yobe/northern Adamawa hot-spot LGAs."""
    lgas = geo.lga_table().copy()
    sw = cat["conflict_state_weights"]
    hot = cat["conflict_hot_lgas"]
    w = lgas.apply(lambda r: sw[r.state] * hot.get(r.lga, 1.0), axis=1).to_numpy()
    w = w / w.sum()
    rows = []
    for year in range(2018, 2025):
        n = per_year
        pick = rng.choice(len(lgas), size=n, p=w)
        days = rng.integers(0, 365, size=n)
        jit = cat["conflict_jitter_deg"]
        for i, d in zip(pick, days):
            r = lgas.iloc[i]
            rows.append({
                "event_date": (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"),
                "year": year,
                "event_type": rng.choice(["Battles", "Violence against civilians", "Explosions/Remote violence"]),
                "admin1": r.state,
                "admin2": r.lga,
                "latitude": round(r.lat + rng.uniform(-jit, jit), 4),
                "longitude": round(r.lon + rng.uniform(-jit, jit), 4),
                "fatalities": int(rng.poisson(cat["conflict_mean_fatalities"])),
            })
    ev = pd.DataFrame(rows)
    # a few events outside the region to exercise the state filter
    extra = ev.sample(20, random_state=1).assign(admin1="Kano", latitude=12.0, longitude=8.5)
    return pd.concat([ev, extra], ignore_index=True)


def make_suppliers(cat: dict[str, Any], rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Supplier registry with latent reliability/quality/markup (latent columns kept separately)."""
    states = list(cat["state_weights"])
    names: set[str] = set()
    rows = []
    cats = list(cat["items"])
    while len(rows) < n:
        name = f"{rng.choice(cat['name_prefixes'])} {rng.choice(cat['name_middles'])} {rng.choice(cat['name_suffixes'])}"
        if name in names:
            continue
        names.add(name)
        st = str(rng.choice(states))
        lga = str(rng.choice(geo.lga_table().query("state == @st").lga))
        a, b = cat["simulation"]["reliability_beta"]
        rel = float(rng.beta(a, b))
        rows.append({
            "supplier_name": name, "state": st, "lga": lga,
            "registered": bool(rng.random() > 0.05), "debarred": bool(rng.random() < 0.03),
            "tax_clearance": bool(rng.random() > 0.07), "cac": bool(rng.random() > 0.04),
            "_reliability": rel,
            "_quality": float(np.clip(2.6 + 1.8 * rel + rng.normal(0, 0.3), 1.5, 4.9)),
            "_markup": float(np.exp(rng.normal(0, cat["simulation"]["supplier_markup_sd"]))),
            "_categories": list(rng.choice(cats, size=int(rng.integers(1, 4)), replace=False)),
        })
    return pd.DataFrame(rows)


def _variant(name: str, rng: np.random.Generator) -> str:
    """Return a messy spelling variant of a supplier name."""
    choice = rng.integers(0, 4)
    if choice == 0:
        return name.upper()
    if choice == 1:
        return name.replace("Limited", "Ltd.").replace("Ltd", "Limited")
    if choice == 2:
        return name + "."
    return "  " + name.lower() + " "


def make_transactions(cat: dict[str, Any], sup: pd.DataFrame, macro: pd.DataFrame,
                      acled: pd.DataFrame, rng: np.random.Generator, cfg: dict[str, Any]) -> pd.DataFrame:
    """Contracts with outcomes generated from the documented true process."""
    sim = cat["simulation"]
    n = cfg["n_transactions"]
    years = cfg["years"]
    # mild growth in contract volume over time
    yw = np.linspace(0.8, 1.2, len(years))
    yr = rng.choice(years, size=n, p=yw / yw.sum())
    dates = pd.to_datetime([f"{y}-01-01" for y in yr]) + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
    states = list(cat["state_weights"])
    sw = np.array([cat["state_weights"][s] for s in states])
    dstate = rng.choice(states, size=n, p=sw / sw.sum())
    lgas = geo.lga_table()
    dlga = np.array([rng.choice(lgas.query("state == @s").lga.to_numpy()) for s in dstate])
    cats = list(cat["items"])
    icat = rng.choice(cats, size=n)
    items = np.array([rng.choice(list(cat["items"][c])) for c in icat])
    cpi = macro.set_index("month")["cpi"]
    cpi_2024 = cpi[[m for m in cpi.index if m.startswith("2024")]].mean()
    conflict = geo.conflict_counts(pd.Series(dstate), pd.Series(dlga), pd.Series(dates), acled)
    lga_index = {(r.state, r.lga): j for j, r in enumerate(lgas.itertuples())}
    dist_mat = geo.haversine_km(lgas.lat.to_numpy()[:, None], lgas.lon.to_numpy()[:, None],
                                lgas.lat.to_numpy()[None, :], lgas.lon.to_numpy()[None, :])
    pools = {c: sup[sup._categories.apply(lambda cs, k=c: k in cs)].reset_index(drop=True) for c in cats}
    rainy_months = load_settings()["data"]["rainy_months"]
    reject_below = load_settings()["synthetic"]["rejection_quality_below"]
    rows = []
    for i in range(n):
        pool = pools[icat[i]]
        w = np.where(pool.state.to_numpy() == dstate[i], sim["supplier_same_state_weight"], 1.0)
        s = pool.iloc[rng.choice(len(pool), p=w / w.sum())]
        info = cat["items"][icat[i]][items[i]]
        lo, hi = info["qty"]
        qty = int(np.clip(np.exp(rng.uniform(np.log(lo), np.log(hi))), lo, hi))
        dist = float(dist_mat[lga_index[(s.state, s.lga)], lga_index[(dstate[i], dlga[i])]])
        cev = float(conflict.iloc[i, 0])
        est_value = info["price"] * qty
        if est_value < 5e6:
            method = rng.choice(["request_for_quotation", "direct_procurement", "selective_tendering"], p=[0.6, 0.2, 0.2])
        elif est_value < 5e7:
            method = rng.choice(["selective_tendering", "open_competitive", "direct_procurement"], p=[0.5, 0.4, 0.1])
        else:
            method = rng.choice(["open_competitive", "selective_tendering", "direct_procurement"], p=[0.85, 0.1, 0.05])
        bidders = {"open_competitive": rng.integers(3, 11), "selective_tendering": rng.integers(2, 6),
                   "request_for_quotation": 3, "direct_procurement": 1}[method]
        real_price = (info["price"] * s._markup * qty ** sim["quantity_discount_exp"]
                      * (1 + sim["remote_premium_per_km"] * dist)
                      * (1 + sim["conflict_premium_per_event"] * min(cev, 40))
                      * (1 + (sim["direct_method_premium"] if method == "direct_procurement" else 0))
                      * (1 - sim["bidder_discount_per_bidder"] * (min(int(bidders), 8) - 1))
                      * np.exp(rng.normal(0, sim["price_noise_sd"])))
        month = dates[i].strftime("%Y-%m")
        nominal = real_price * cpi[month] / cpi_2024
        rainy = dates[i].month in rainy_months
        mean_delay = (sim["delay_base_days"] + sim["delay_per_km"] * dist + sim["delay_rainy_days"] * rainy
                      + sim["delay_per_conflict_event"] * min(cev, 40)
                      + sim["delay_unreliability_days"] * (1 - s._reliability)
                      + cat["category_lead_days"][icat[i]])
        k = sim["delay_gamma_shape"]
        delay = int(round(rng.gamma(k, mean_delay / k)))
        quality = float(np.clip(rng.normal(s._quality - 0.15 * rainy - 0.01 * min(cev, 40), sim["quality_sd"]), 1, 5))
        p_short = sim["short_base"] + sim["short_unreliability"] * (1 - s._reliability) + sim["short_per_conflict"] * min(cev, 40)
        short = bool(rng.random() < p_short)
        rows.append({
            "ocid": f"ocds-ne-{i:06d}", "date": dates[i], "buyer": f"MDA-{rng.integers(1, 40):03d}",
            "supplier_name": s.supplier_name, "supplier_state": s.state, "supplier_lga": s.lga,
            "item": items[i], "item_category": icat[i], "quantity": qty, "unit": "unit",
            "value": round(nominal * qty, 2), "currency": "NGN", "method": method, "n_bidders": float(bidders),
            "delivery_state": dstate[i], "delivery_lga": dlga[i],
            "delay_days": float(delay), "quality_score": round(quality, 1),
            "short_delivered": short, "rejected": bool(quality < reject_below),
        })
    df = pd.DataFrame(rows)
    df[["short_delivered", "rejected", "method"]] = df[["short_delivered", "rejected", "method"]].astype(object)
    # ---- inject messiness ----
    miss = df.date.dt.year.map(cfg["missing_outcome_frac"]).to_numpy() > rng.random(n)
    df.loc[miss, ["delay_days", "quality_score", "short_delivered", "rejected"]] = np.nan
    v = rng.random(n) < cfg["name_variant_prob"]
    df.loc[v, "supplier_name"] = [_variant(x, rng) for x in df.loc[v, "supplier_name"]]
    usd = rng.random(n) < cfg["usd_row_frac"]
    fx = macro.set_index("month")["exchange_rate_ngn_usd"]
    df.loc[usd, "value"] = (df.loc[usd, "value"] / df.loc[usd, "date"].dt.strftime("%Y-%m").map(fx)).round(2)
    df.loc[usd, "currency"] = "USD"
    df.loc[rng.random(n) < cfg["missing_bidders_frac"], "n_bidders"] = np.nan
    df.loc[rng.random(n) < cfg["missing_method_frac"], "method"] = np.nan
    st_var = rng.random(n) < 0.1
    df.loc[st_var, "delivery_state"] = df.loc[st_var, "delivery_state"].str.upper() + " STATE"
    fmt = rng.random(n) < 0.1
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.loc[fmt, "date"] = pd.to_datetime(df.loc[fmt, "date"]).dt.strftime("%d/%m/%Y")
    dup = df.sample(frac=cfg["duplicate_frac"], random_state=7)
    return pd.concat([df, dup], ignore_index=True)


def to_ocds(df: pd.DataFrame) -> dict[str, Any]:
    """Convert flat contract rows to a minimal OCDS release package."""
    releases = []
    for r in df.itertuples():
        releases.append({
            "ocid": r.ocid, "date": str(r.date), "tag": ["contract"],
            "buyer": {"name": r.buyer},
            "tender": {
                "procurementMethodDetails": None if pd.isna(r.method) else r.method,
                "numberOfTenderers": None if pd.isna(r.n_bidders) else int(r.n_bidders),
                "items": [{"description": r.item, "classification": {"description": r.item_category},
                           "quantity": r.quantity, "unit": {"name": r.unit},
                           "deliveryAddress": {"region": r.delivery_state, "locality": r.delivery_lga}}],
            },
            "awards": [{"suppliers": [{"name": r.supplier_name,
                                       "address": {"region": r.supplier_state, "locality": r.supplier_lga}}],
                        "value": {"amount": r.value, "currency": r.currency}}],
            "contracts": [{"dateSigned": str(r.date), "implementation": {"x_outcome": {
                "delayDays": None if pd.isna(r.delay_days) else float(r.delay_days),
                "qualityScore": None if pd.isna(r.quality_score) else float(r.quality_score),
                "shortDelivered": None if pd.isna(r.short_delivered) else bool(r.short_delivered),
                "rejected": None if pd.isna(r.rejected) else bool(r.rejected)}}}],
        })
    return {"uri": "simulated", "publisher": {"name": "simulated"}, "releases": releases}


def raw_dir_has_real_data(raw: Path) -> bool:
    """True when data/raw contains data files that this generator did not write."""
    files = [p for p in raw.iterdir() if p.is_file() and not p.name.startswith(".")]
    return bool(files) and not (raw / DEMO_MARKER).exists()


def generate(raw: Path | None = None, force: bool = False, overrides: dict[str, Any] | None = None) -> Path:
    """Write simulated raw files into data/raw (no-op if real data already present).

    overrides replaces keys of settings['demo_raw'] (used by tests for small fixtures).
    """
    raw = raw or path("raw_dir")
    raw.mkdir(parents=True, exist_ok=True)
    if raw_dir_has_real_data(raw):
        return raw
    if (raw / DEMO_MARKER).exists() and not force:
        return raw
    s = load_settings()
    cfg = {**s["demo_raw"], **(overrides or {})}
    cat = _catalog()
    rng = np.random.default_rng(s["seed"])
    macro = make_macro(cat)
    acled = make_acled(cat, rng, cfg["acled_events_per_year"])
    sup = make_suppliers(cat, rng, cfg["n_suppliers"])
    tx = make_transactions(cat, sup, macro, acled, rng, cfg)
    n_json = max(1, len(tx) // 30)
    macro.to_csv(raw / "macro_ng.csv", index=False)
    acled.to_csv(raw / "acled_northeast.csv", index=False)
    sup.drop(columns=[c for c in sup.columns if c.startswith("_")]).to_csv(raw / "supplier_registry.csv", index=False)
    tx.iloc[n_json:].to_csv(raw / "contracts_nocopo.csv", index=False)
    with open(raw / "ocds_nocopo_sample.json", "w", encoding="utf-8") as fh:
        json.dump(to_ocds(tx.iloc[:n_json]), fh)
    (raw / DEMO_MARKER).write_text(
        "All files in this folder were SIMULATED by procure.data.demo_raw for development.\n"
        "Delete them (and this file) and add real NOCOPO/OCDS, ACLED and macro files to use real data.\n")
    return raw


if __name__ == "__main__":
    print(generate(force=True))
