"""Temporal split: train 2019-2022, validation 2023, test 2024 (never random)."""

from __future__ import annotations

import pandas as pd

from procure.config import load_settings


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by contract year per config.

    Synthetic rows are removed from the test set whenever the test years contain any real
    outcomes; those rows are returned separately as 'test_synthetic' so metrics can be
    reported on real-only (test) and full (test + test_synthetic) data.
    """
    sp = load_settings()["split"]
    year = df["date"].dt.year
    train = df[year.isin(sp["train_years"])]
    val = df[year.isin(sp["val_years"])]
    test_all = df[year.isin(sp["test_years"])]
    if (~test_all["is_synthetic"]).any():
        test = test_all[~test_all["is_synthetic"]]
        test_syn = test_all[test_all["is_synthetic"]]
    else:
        test, test_syn = test_all, test_all.iloc[0:0]
    return {k: v.reset_index(drop=True) for k, v in
            {"train": train, "val": val, "test": test, "test_synthetic": test_syn}.items()}
