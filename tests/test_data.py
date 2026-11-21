"""Phase 1 tests: loading, cleaning, leakage-free features, synthetic rows, temporal split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.config import load_settings
from procure.data import clean, dictionary, features, geo, load


def test_no_rows_after_2022_in_train(small_frames: dict) -> None:
    assert small_frames["train"]["date"].dt.year.max() <= 2022
    assert set(small_frames["val"]["date"].dt.year) == {2023}
    assert set(small_frames["test"]["date"].dt.year) == {2024}


def test_no_synthetic_rows_in_test_when_real_outcomes_exist(small_frames: dict) -> None:
    test = small_frames["test"]
    assert len(test) > 0 and not test["is_synthetic"].any()
    assert small_frames["test_synthetic"]["is_synthetic"].all()


def test_supplier_history_uses_only_earlier_rows(small_frames: dict) -> None:
    """Brute-force recomputation for a sample of rows must match the vectorised features."""
    tx = small_frames["transactions"]
    cfg = load_settings()["data"]
    k, pri = cfg["history_prior_strength"], cfg["history_priors"]
    known_date = tx["date"] + pd.to_timedelta(cfg["outcome_lag_days"] + tx["delay_days"], unit="D")
    for i in tx.sample(40, random_state=0).index:
        row = tx.loc[i]
        same = tx["supplier_id"] == row.supplier_id
        earlier = tx[same & (tx["date"] < row.date)]
        assert row.supplier_experience == len(earlier)
        known = tx[same & (known_date < row.date)]
        exp_fail = (k * pri["failure_rate"] + known["failed"].sum()) / (k + len(known))
        assert np.isclose(row.past_failure_rate, exp_fail)
        assert (known["date"] < row.date).all()


def test_history_ignores_future_changes(small_frames: dict) -> None:
    """Changing outcomes in 2024 must not change any 2023 history feature."""
    tx = small_frames["transactions"]
    ref = features.item_reference_prices(tx)
    base = features.history_for(tx[["supplier_id", "date"]], tx, ref)
    tampered = tx.copy()
    fut = tampered["date"].dt.year == 2024
    tampered.loc[fut, "failed"] = 1 - tampered.loc[fut, "failed"]
    tampered.loc[fut, "delay_days"] = 0
    new = features.history_for(tx[["supplier_id", "date"]], tampered, ref)
    early = tx["date"].dt.year <= 2023
    pd.testing.assert_frame_equal(base[early], new[early])


def test_synthetic_never_overwrites_real(small_raw, small_frames: dict) -> None:
    raw = load.load_all(small_raw)["contracts"]
    tx = small_frames["transactions"]
    real = tx[~tx["is_synthetic"]]
    merged = real.merge(raw[["ocid", "delay_days"]].drop_duplicates("ocid"), on="ocid", suffixes=("", "_raw"))
    assert np.allclose(merged["delay_days"], pd.to_numeric(merged["delay_days_raw"]))
    assert tx["is_synthetic"].sum() == (~tx["has_real_outcome"]).sum()


def test_supplier_name_normalisation_and_ids() -> None:
    a = clean.supplier_id("Sahel Global Limited", "Borno")
    assert a == clean.supplier_id("  SAHEL GLOBAL LTD. ", "BORNO STATE")
    assert a != clean.supplier_id("Sahel Global Limited", "Yobe")


def test_currency_dates_states_and_dedup() -> None:
    macro = pd.DataFrame({"month": ["2021-03"], "cpi": [100.0], "exchange_rate_ngn_usd": [400.0],
                          "fuel_price_ngn_litre": [165.0]})
    raw = pd.DataFrame({
        "ocid": ["a", "b", "c"], "date": ["05/03/2021", "2021-03-05", "2021-03-05"], "buyer": ["x"] * 3,
        "supplier_name": ["Arewa Star Ltd", "AREWA STAR LIMITED", "Arewa Star Ltd"],
        "supplier_state": ["Gombe"] * 3, "supplier_lga": ["Gombe"] * 3, "item": ["Cement 50kg bag"] * 3,
        "item_category": ["construction"] * 3, "quantity": [10, 10, 10], "unit": ["bag"] * 3,
        "value": [100.0, 100.0, 1.0], "currency": ["NGN", "NGN", "USD"], "method": ["open_competitive"] * 3,
        "n_bidders": [3, 3, np.nan], "delivery_state": ["GOMBE STATE"] * 3, "delivery_lga": ["gombe lga"] * 3,
        "delay_days": [1, 1, 1], "quality_score": [4, 4, 4], "short_delivered": [False] * 3, "rejected": [False] * 3,
    })
    st = clean.standardise(raw, macro)
    assert (st["date"] == pd.Timestamp("2021-03-05")).all()
    assert st.loc[2, "value"] == 400.0 and set(st["delivery_state"]) == {"Gombe"}
    out = clean.deduplicate(clean.assign_ids(st))
    assert len(out) == 2  # rows a and b are the same supplier/item/date/value
    imp = clean.impute(out)
    assert imp["n_bidders"].notna().all() and imp["is_imputed"].tolist() == [False, True]


def test_flatten_ocds() -> None:
    pkg = {"releases": [{"ocid": "x1", "date": "2022-01-01", "buyer": {"name": "B"},
                         "tender": {"procurementMethodDetails": "open_competitive", "numberOfTenderers": 4,
                                    "items": [{"description": "Diesel (litre)", "quantity": 100,
                                               "classification": {"description": "fuel"},
                                               "deliveryAddress": {"region": "Yobe", "locality": "Potiskum"}}]},
                         "awards": [{"suppliers": [{"name": "S"}], "value": {"amount": 5, "currency": "NGN"}}]}]}
    df = load.flatten_ocds(pkg)
    assert df.loc[0, "n_bidders"] == 4 and df.loc[0, "delivery_lga"] == "Potiskum"
    assert pd.isna(df.loc[0, "delay_days"])


def test_region_filter_and_conflict_window() -> None:
    df = pd.DataFrame({"delivery_state": ["Borno", "Kano", "yobe state"]})
    assert len(load.filter_region(df)) == 2
    ev = pd.DataFrame({"event_date": pd.to_datetime(["2022-01-01", "2022-03-31", "2022-04-01"]),
                       "latitude": [11.83] * 3, "longitude": [13.15] * 3, "fatalities": [1, 2, 4]})
    c = geo.conflict_counts(pd.Series(["Borno"]), pd.Series(["Maiduguri"]), pd.Series([pd.Timestamp("2022-04-01")]), ev)
    assert c.iloc[0, 0] == 2 and c.iloc[0, 1] == 3  # same-day event excluded; window is 90 days before


def test_data_dictionary(small_frames: dict, tmp_path) -> None:
    out = dictionary.export_dictionary(small_frames["transactions"], tmp_path / "dd.csv")
    dd = pd.read_csv(out)
    assert {"name", "type", "source", "meaning"} <= set(dd.columns)
    assert not dd["meaning"].eq("(undocumented)").any()
