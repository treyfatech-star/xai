"""Phase 3 tests: shared preprocessing, probability correction, bundle I/O, metrics, baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.models import baselines as bl
from procure.models.evaluate import classification_metrics, regression_metrics
from procure.models.predict import Predictor
from procure.models.train import ModelBundle, correct_prob, original_feature_of, pos_weight, tune


def test_three_models_share_one_preprocessor(tiny_bundle) -> None:
    n = len(tiny_bundle.transformed_features)
    for m in tiny_bundle.models.values():
        assert m.n_features_in_ == n
    assert len(tiny_bundle.original_features) == n
    assert set(original_feature_of(["item_category_fuel", "distance_km"])) == {"item_category", "distance_km"}


def test_predictions_are_in_natural_units(tiny_bundle, small_frames) -> None:
    p = Predictor(tiny_bundle).predict(small_frames["test"])
    assert (p["unit_cost"] > 0).all() and (p["delay_days"] >= 0).all()
    assert p["failure_prob"].between(0, 1).all()
    ratio = p["unit_cost"] / small_frames["test"]["unit_cost_real"]
    assert 0.2 < ratio.median() < 5  # log-target model returns naira, not log-naira


def test_probability_correction_undoes_weighting() -> None:
    # with weight w, a true probability p maps to score s = w p / (w p + 1 - p)
    p, w = np.array([0.05, 0.2, 0.5]), 4.0
    s = w * p / (w * p + 1 - p)
    assert np.allclose(correct_prob(s, w), p)
    assert pos_weight(np.array([0, 0, 0, 1])) == 3.0


def test_bundle_roundtrip(tiny_bundle, tmp_path, small_frames) -> None:
    tiny_bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path / "bundle.joblib")
    assert loaded.version == tiny_bundle.version and (tmp_path / "metadata.json").exists()
    a = Predictor(tiny_bundle).predict(small_frames["test"].head(20))
    b = Predictor(loaded).predict(small_frames["test"].head(20))
    pd.testing.assert_frame_equal(a, b)


def test_metrics() -> None:
    r = regression_metrics(np.array([10.0, 0.0, 20.0]), np.array([12.0, 1.0, 20.0]))
    assert np.isclose(r["MAE"], 1.0) and np.isclose(r["MAPE"], 10.0)  # zero target excluded from MAPE
    c = classification_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.6, 0.4]), np.array([0, 1, 1, 0]))
    assert c["AUC"] == 1.0 and c["F1"] == 1.0 and c["Brier"] < 0.15


def test_baselines_fit_predict(small_frames) -> None:
    tr, te = small_frames["train"], small_frames["test"]
    for b in [bl.MedianPerCategory(), bl.LinearCost(), bl.PerSupplierMeanDelay(), bl.RandomForestDelay(42),
              bl.LogisticFailure()]:
        p = b.fit(tr).predict(te)
        assert p.shape == (len(te),) and np.isfinite(p).all()


def test_optuna_tuning_runs_on_time_series_folds(small_frames) -> None:
    trv = pd.concat([small_frames["train"], small_frames["val"]], ignore_index=True)
    params, score = tune("delay", trv, n_trials=1, seed=42)
    assert "max_depth" in params and score > 0
