"""Model 3 — XGBoost baseline.

XGBoost's multi:softprob objective has no class_weight parameter, so class
imbalance is handled via per-sample weights derived from the same train-only
balanced weights used elsewhere (results/metrics/class_weights.json).
Regularization (shallow depth, subsampling, min_child_weight) is set
deliberately conservative given only 26 High-risk training examples.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev

MODEL_NAME = "xgboost"


def sample_weights_from_counts(y: np.ndarray, weight_vector: list[float]) -> np.ndarray:
    w = np.array(weight_vector)
    return w[y]


def build_pipeline(preprocessor, seed: int, **xgb_params) -> Pipeline:
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        **xgb_params,
    )
    return Pipeline([("pre", clone(preprocessor)), ("clf", clf)])


def select_hyperparams(preprocessor, cfg, train_df, val_df, weight_vector) -> dict:
    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_val = val_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    sw_train = sample_weights_from_counts(y_train, weight_vector)

    grid = [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 2.0},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5, "reg_lambda": 2.0},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0},
    ]
    results = []
    for g in grid:
        pipe = build_pipeline(preprocessor, seed=cfg["seed"], **g)
        pipe.fit(train_df, y_train, clf__sample_weight=sw_train)
        proba = pipe.predict_proba(val_df)
        pred = proba.argmax(1)
        m = ev.full_classification_metrics(y_val, pred, proba)
        results.append({**g, "val_macro_f1": m["macro_f1"], "val_high_f1": m["high_risk"]["f1"]})

    best = max(results, key=lambda r: r["val_macro_f1"])
    keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"]
    return {"best": {k: best[k] for k in keys}, "grid_results": results}


def train_and_evaluate(cfg: dict, preprocessor, train_df, val_df, test_df, weight_vector: list[float]) -> dict:
    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_val = val_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_test = test_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    sw_train = sample_weights_from_counts(y_train, weight_vector)

    sel = select_hyperparams(preprocessor, cfg, train_df, val_df, weight_vector)
    best = sel["best"]

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline(preprocessor, seed=cfg["seed"], **best)
        pipe.fit(train_df, y_train, clf__sample_weight=sw_train)
    train_time = time.time() - t0

    val_proba = pipe.predict_proba(val_df)
    val_metrics = ev.full_classification_metrics(y_val, val_proba.argmax(1), val_proba)
    test_proba = pipe.predict_proba(test_df)
    test_metrics = ev.full_classification_metrics(y_test, test_proba.argmax(1), test_proba)

    n_features = pipe.named_steps["pre"].get_feature_names_out().shape[0]

    return {
        "model_name": MODEL_NAME,
        "hyperparameters": {**best, "objective": "multi:softprob", "sample_weighting": "train-only balanced weights", "random_state": cfg["seed"]},
        "hyperparameter_search": sel["grid_results"],
        "n_features": int(n_features),
        "training_time_seconds": train_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pipeline": pipe,
        "test_y_true": y_test,
        "test_y_proba": test_proba,
    }
