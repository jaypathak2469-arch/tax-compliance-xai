"""Model 1 — Logistic Regression baseline.

Per the approved plan, transaction_anomaly_count is dropped for this model
only (collinear_drop_for_linear in config.yaml): it is exactly
transaction_anomaly_ratio * transaction_count, and linear models are affected
by that redundancy in a way tree models are not.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev

MODEL_NAME = "logistic_regression"


def build_lr_preprocessor(cfg: dict) -> ColumnTransformer:
    """A dedicated preprocessor that excludes the collinear column."""
    f = cfg["features"]
    drop = set(cfg["collinear_drop_for_linear"])
    numeric = [
        c
        for c in (f["numeric_tax"] + f["numeric_txn_existing"] + f["numeric_txn_engineered"])
        if c not in drop
    ]
    categorical = f["categorical"]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(cfg: dict, C: float, seed: int) -> Pipeline:
    pre = build_lr_preprocessor(cfg)
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def select_hyperparams(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Small real grid search selected on validation macro F1. No test data touched."""
    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_val = val_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()

    results = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        pipe = build_pipeline(cfg, C=C, seed=cfg["seed"])
        pipe.fit(train_df, y_train)
        proba = pipe.predict_proba(val_df)
        pred = proba.argmax(axis=1)
        m = ev.full_classification_metrics(y_val, pred, proba)
        results.append({"C": C, "val_macro_f1": m["macro_f1"], "val_high_f1": m["high_risk"]["f1"]})

    best = max(results, key=lambda r: r["val_macro_f1"])
    return {"C": best["C"], "grid_results": results}


def train_and_evaluate(cfg: dict, train_df, val_df, test_df) -> dict:
    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_val = val_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_test = test_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()

    sel = select_hyperparams(cfg, train_df, val_df)
    best_C = sel["C"]

    t0 = time.time()
    pipe = build_pipeline(cfg, C=best_C, seed=cfg["seed"])
    pipe.fit(train_df, y_train)
    train_time = time.time() - t0

    n_iter = pipe.named_steps["clf"].n_iter_.tolist()
    converged = all(it < pipe.named_steps["clf"].max_iter for it in n_iter)

    val_proba = pipe.predict_proba(val_df)
    val_metrics = ev.full_classification_metrics(y_val, val_proba.argmax(1), val_proba)

    test_proba = pipe.predict_proba(test_df)
    test_metrics = ev.full_classification_metrics(y_test, test_proba.argmax(1), test_proba)

    n_features = pipe.named_steps["pre"].get_feature_names_out().shape[0]

    return {
        "model_name": MODEL_NAME,
        "hyperparameters": {
            "C": best_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 3000,
            "random_state": cfg["seed"],
        },
        "hyperparameter_search": sel["grid_results"],
        "collinear_column_dropped": cfg["collinear_drop_for_linear"],
        "n_features": int(n_features),
        "training_time_seconds": train_time,
        "convergence": {"n_iter": n_iter, "converged": converged, "max_iter": 3000},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pipeline": pipe,
        "test_y_true": y_test,
        "test_y_proba": test_proba,
    }


def cv_pipeline_factory(cfg: dict, best_C: float):
    def factory():
        return build_pipeline(cfg, C=best_C, seed=cfg["seed"])
    return factory
