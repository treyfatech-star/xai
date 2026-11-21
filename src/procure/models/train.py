"""Prediction layer training: three XGBoost models (cost, delay, failure) sharing ONE
ColumnTransformer, tuned with Optuna using expanding-window time-series CV on train+val.

Key choices (see DECISIONS.md):
* Cost is modelled on log(unit_cost_real): prices span 4 orders of magnitude across items, so
  errors are relative; predictions are exponentiated back to naira.
* The failure model uses scale_pos_weight = negatives/positives. Re-weighting inflates the
  odds by that factor, so probabilities are corrected analytically (p = s / (s + w(1-s)))
  before they are used in the optimiser or scored with the Brier score.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from procure import __version__
from procure.config import load_settings, path, quick_mode, set_seed

log = logging.getLogger("procure.models")
MODEL_NAMES: tuple[str, ...] = ("cost", "delay", "failure")


def feature_lists() -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature names from config."""
    f = load_settings()["features"]
    return list(f["numeric"]), list(f["categorical"])


def build_preprocessor() -> ColumnTransformer:
    """The single shared preprocessing pipeline (imputation, scaling, one-hot encoding)."""
    num, cat = feature_lists()
    mf = load_settings()["features"]["onehot_min_frequency"]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=mf,
                                                   sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)


def original_feature_of(transformed_names: list[str]) -> list[str]:
    """Map each transformed column (e.g. 'item_category_fuel') back to its source feature."""
    num, cat = feature_lists()
    out = []
    for n in transformed_names:
        if n in num:
            out.append(n)
            continue
        out.append(next(c for c in sorted(cat, key=len, reverse=True) if n.startswith(c + "_")))
    return out


def target(df: pd.DataFrame, name: str) -> np.ndarray:
    """Training target for a model (log cost for the cost model)."""
    m = load_settings()["models"][name]
    y = df[m["target"]].to_numpy(float)
    return np.log(y) if m["log_target"] else y


def pos_weight(y: np.ndarray) -> float:
    """scale_pos_weight = negatives / positives."""
    pos = float(np.sum(y == 1))
    return float(np.sum(y == 0)) / max(pos, 1.0)


def correct_prob(score: np.ndarray, weight: float) -> np.ndarray:
    """Undo the odds inflation from positive-class weighting: p = s / (s + w (1 - s))."""
    s = np.clip(score, 1e-9, 1 - 1e-9)
    return s / (s + weight * (1 - s))


def make_xgb(name: str, params: dict[str, Any], seed: int, spw: float | None = None) -> xgb.XGBModel:
    """Instantiate an XGBoost regressor/classifier with fixed + tuned params."""
    fixed = load_settings()["models"]["xgb_fixed"]
    if load_settings()["models"][name]["task"] == "classification":
        return xgb.XGBClassifier(**fixed, **params, random_state=seed, scale_pos_weight=spw,
                                 eval_metric="logloss")
    return xgb.XGBRegressor(**fixed, **params, random_state=seed, objective="reg:squarederror")


def _suggest(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyper-parameters from the configured search space."""
    sp = load_settings()["models"]["optuna"]["space"]
    return {
        "n_estimators": trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
        "min_child_weight": trial.suggest_float("min_child_weight", *sp["min_child_weight"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *sp["reg_lambda"], log=True),
    }


def cv_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]:
    """Pre-transform each expanding-window fold (preprocessor refit on each fold's train part)."""
    folds = []
    for f in load_settings()["models"]["cv_folds"]:
        tr = df[df["year"].isin(f["train"])]
        va = df[df["year"].isin(f["valid"])]
        pre = build_preprocessor().fit(tr)
        folds.append((pre.transform(tr), pre.transform(va), tr, va))
    return folds


def tune(name: str, df: pd.DataFrame, n_trials: int, seed: int) -> tuple[dict[str, Any], float]:
    """Optuna TPE search; objective = mean validation score over the time-series folds
    (MAE for regression, 1 - AUC for classification). Returns (best params, best score)."""
    folds = cv_folds(df)
    clf = load_settings()["models"][name]["task"] == "classification"

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        scores = []
        for xtr, xva, tr, va in folds:
            ytr, yva = target(tr, name), target(va, name)
            m = make_xgb(name, params, seed, pos_weight(ytr) if clf else None)
            m.fit(xtr, ytr)
            if clf:
                scores.append(1 - roc_auc_score(yva, m.predict_proba(xva)[:, 1]))
            else:
                scores.append(mean_absolute_error(yva, m.predict(xva)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


@dataclass
class ModelBundle:
    """Preprocessor + three models + metadata, saved together so versions never drift."""

    preprocessor: ColumnTransformer
    models: dict[str, xgb.XGBModel]
    params: dict[str, dict[str, Any]]
    failure_weight: float
    transformed_features: list[str]
    original_features: list[str]
    version: str
    trained_on: dict[str, Any] = field(default_factory=dict)
    cv_scores: dict[str, float] = field(default_factory=dict)
    feature_means: dict[str, float] = field(default_factory=dict)

    def save(self, out: Path | None = None) -> Path:
        """Write models/bundle.joblib and models/metadata.json."""
        out = out or path("models_dir")
        joblib.dump(self, out / "bundle.joblib")
        meta = {"version": self.version, "params": self.params, "failure_weight": self.failure_weight,
                "features": feature_lists(), "transformed_features": self.transformed_features,
                "trained_on": self.trained_on, "cv_scores": self.cv_scores}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out / "bundle.joblib"

    @staticmethod
    def load(p: Path | None = None) -> "ModelBundle":
        """Load models/bundle.joblib."""
        return joblib.load(p or path("models_dir") / "bundle.joblib")


def fit_bundle(df: pd.DataFrame, params: dict[str, dict[str, Any]], seed: int,
               cv_scores: dict[str, float] | None = None) -> ModelBundle:
    """Fit the shared preprocessor and the three models on df with given params."""
    pre = build_preprocessor().fit(df)
    x = pre.transform(df)
    names = list(pre.get_feature_names_out())
    y_fail = target(df, "failure")
    w = pos_weight(y_fail)
    models = {}
    for name in MODEL_NAMES:
        m = make_xgb(name, params[name], seed, w if name == "failure" else None)
        m.fit(x, target(df, name))
        models[name] = m
    num, _ = feature_lists()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return ModelBundle(
        preprocessor=pre, models=models, params=params, failure_weight=w, transformed_features=names,
        original_features=original_feature_of(names), version=f"{__version__}+{stamp}-seed{seed}",
        trained_on={"rows": len(df), "years": sorted(df["year"].unique().tolist()),
                    "synthetic_share": float(df["is_synthetic"].mean()),
                    "delay_mean": float(df["delay_days"].mean()), "failure_rate": float(df["failed"].mean())},
        cv_scores=cv_scores or {}, feature_means={c: float(df[c].mean()) for c in num},
    )


def main() -> None:
    """CLI (make train): tune each model, fit final bundle on train+val, save."""
    s = load_settings()
    seed = set_seed()
    d = path("processed_dir")
    df = pd.concat([pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet")], ignore_index=True)
    n_trials = s["models"]["quick_n_trials"] if quick_mode() else s["models"]["optuna"]["n_trials"]
    params, scores = {}, {}
    for name in MODEL_NAMES:
        t0 = time.time()
        params[name], scores[name] = tune(name, df, n_trials, seed)
        (path("models_dir") / "best_params.json").write_text(json.dumps(params, indent=2))
        print(f"tuned {name:8s} ({n_trials} trials, {time.time() - t0:5.1f}s): CV score {scores[name]:.4f} {params[name]}")
    # import via the package so the pickled class path is procure.models.train.ModelBundle,
    # not __main__.ModelBundle (this file runs as __main__ under `python -m`)
    from procure.models import train as trained_module

    bundle = trained_module.fit_bundle(df, params, seed, scores)
    out = bundle.save()
    (path("models_dir") / "best_params.json").write_text(json.dumps(params, indent=2))
    print(f"saved {out} (version {bundle.version}, {len(bundle.transformed_features)} transformed features)")


if __name__ == "__main__":
    main()
