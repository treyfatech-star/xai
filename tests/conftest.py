"""Shared small fixtures (fast: a few hundred simulated transactions)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("PROCURE_QUICK", "1")

from procure.data import build, demo_raw  # noqa: E402

SMALL = {"n_transactions": 900, "n_suppliers": 40, "acled_events_per_year": 150}


@pytest.fixture(scope="session")
def small_raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with a small simulated raw dataset."""
    raw = tmp_path_factory.mktemp("raw")
    demo_raw.generate(raw, force=True, overrides=SMALL)
    return raw


@pytest.fixture(scope="session")
def small_frames(small_raw: Path) -> dict[str, pd.DataFrame]:
    """Processed frames built from the small raw dataset."""
    return build.build_dataset(small_raw)

TINY_PARAMS = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2, "subsample": 1.0,
               "colsample_bytree": 1.0, "min_child_weight": 1.0, "reg_lambda": 1.0}


@pytest.fixture(scope="session")
def tiny_bundle(small_frames: dict):
    """A quickly trained bundle on the small fixture (train + val years)."""
    from procure.models.train import fit_bundle

    trv = pd.concat([small_frames["train"], small_frames["val"]], ignore_index=True)
    return fit_bundle(trv, {k: TINY_PARAMS for k in ("cost", "delay", "failure")}, seed=42)


@pytest.fixture(scope="session")
def tiny_engine(small_frames: dict, tiny_bundle):
    """Recommendation engine on the small fixture data + tiny models."""
    from procure.explain.shap_explain import ShapExplainer
    from procure.models.predict import Predictor
    from procure.pipeline import Engine
    from procure.store import DataStore

    ref = small_frames["ref_prices"].set_index("item")["ref_price"]
    store = DataStore(small_frames["transactions"], ref, small_frames["acled"])
    return Engine(store, Predictor(tiny_bundle), ShapExplainer(tiny_bundle))


@pytest.fixture(scope="session")
def sample_request(small_frames: dict, tiny_engine):
    """A 2024 request with at least 3 eligible suppliers."""
    from procure.sampling import sample_requests

    return sample_requests(tiny_engine.store, small_frames["test"], 1)[0][0]
