"""Baselines for the prediction layer.

cost:    median unit cost per item category; linear regression (log target, shared features)
delay:   per-supplier mean delay; random forest
failure: logistic regression (class-balanced, probabilities corrected like the XGBoost model)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from procure.config import load_settings
from procure.models.train import build_preprocessor, correct_prob, pos_weight, target


class Baseline(Protocol):
    """fit(df) / predict(df) on raw frames; predictions in natural units (naira, days, prob)."""

    def fit(self, df: pd.DataFrame) -> "Baseline": ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


class MedianPerCategory:
    """Cost = median real unit cost of the item category in training data."""

    def fit(self, df: pd.DataFrame) -> "MedianPerCategory":
        self.med = df.groupby("item_category")["unit_cost_real"].median()
        self.glob = float(df["unit_cost_real"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["item_category"].map(self.med).fillna(self.glob).to_numpy(float)


class LinearCost:
    """Ordinary least squares on log cost with the shared preprocessor."""

    def fit(self, df: pd.DataFrame) -> "LinearCost":
        self.pre = build_preprocessor().fit(df)
        self.m = LinearRegression().fit(self.pre.transform(df), target(df, "cost"))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.m.predict(self.pre.transform(df)))


class PerSupplierMeanDelay:
    """Delay = supplier's mean delay in training data (global mean for unseen suppliers)."""

    def fit(self, df: pd.DataFrame) -> "PerSupplierMeanDelay":
        self.mean = df.groupby("supplier_id")["delay_days"].mean()
        self.glob = float(df["delay_days"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["supplier_id"].map(self.mean).fillna(self.glob).to_numpy(float)


class RandomForestDelay:
    """Random forest regressor on the shared features."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "RandomForestDelay":
        c = load_settings()["models"]["random_forest"]
        self.pre = build_preprocessor().fit(df)
        self.m = RandomForestRegressor(n_estimators=c["n_estimators"], min_samples_leaf=c["min_samples_leaf"],
                                       random_state=self.seed, n_jobs=-1).fit(self.pre.transform(df), target(df, "delay"))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.m.predict(self.pre.transform(df)), 0, None)


class LogisticFailure:
    """Class-balanced logistic regression; returns corrected probabilities."""

    def fit(self, df: pd.DataFrame) -> "LogisticFailure":
        c = load_settings()["models"]["logistic"]
        self.pre = build_preprocessor().fit(df)
        y = target(df, "failure")
        self.w = pos_weight(y)
        self.m = LogisticRegression(max_iter=c["max_iter"], C=c["C"], class_weight="balanced").fit(self.pre.transform(df), y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Raw (weighted) score, used for the 0.5 classification threshold."""
        return self.m.predict_proba(self.pre.transform(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return correct_prob(self.score(df), self.w)
