"""Prediction helpers on a trained ModelBundle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from procure.models.train import ModelBundle, correct_prob


class Predictor:
    """Wraps a ModelBundle: transform once, predict cost (naira), delay (days), failure probability."""

    def __init__(self, bundle: ModelBundle | None = None) -> None:
        self.bundle = bundle or ModelBundle.load()

    @property
    def version(self) -> str:
        """Model version string stored with the bundle."""
        return self.bundle.version

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the shared preprocessor."""
        return self.bundle.preprocessor.transform(df)

    def predict_transformed(self, x: np.ndarray) -> pd.DataFrame:
        """Predictions from an already-transformed matrix."""
        m = self.bundle.models
        return pd.DataFrame({
            "unit_cost": np.exp(m["cost"].predict(x)),
            "delay_days": np.clip(m["delay"].predict(x), 0, None),
            "failure_prob": correct_prob(m["failure"].predict_proba(x)[:, 1], self.bundle.failure_weight),
        })

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predictions for raw feature rows (index preserved)."""
        out = self.predict_transformed(self.transform(df))
        out.index = df.index
        return out
