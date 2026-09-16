"""Model 2 — Random Forest baseline. Uses the shared Phase 2 preprocessor
(full 25-feature set, no columns dropped)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev

MODEL_NAME = "random_forest"


def build_pipeline(preprocessor, n_estimators: int, max_depth, min_samples_leaf: int, seed: int) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    # clone the fitted-structure preprocessor's config, not its fitted state
    from sklearn.base import clone
    return Pipeline([("pre", clone(preprocessor)), ("clf", clf)])


def select_hyperparams(preprocessor, cfg, train_df, val_df) -> dict:
    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_val = val_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()

    grid = [
        {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 3},
    ]
    results = []
    for g in grid:
        pipe = build_pipeline(preprocessor, seed=cfg["seed"], **g)
        pipe.fit(train_df, y_train)
        proba = pipe.predict_proba(val_df)
        pred = proba.argmax(1)
        m = ev.full_classification_metrics(y_val, pred, proba)
        results.append({**g, "val_macro_f1": m["macro_f1"], "val_high_f1": m["high_risk"]["f1"]})

    best = max(results, key=lambda r: r["val_macro_f1"])
    return {"best": {k: best[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")},
            "grid_results": results}


def train_and_evaluate(cfg: dict, preprocessor, train_df, val_df, test_df) -> dict:
    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_val = val_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_test = test_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()

    sel = select_hyperparams(preprocessor, cfg, train_df, val_df)
    best = sel["best"]

    t0 = time.time()
    pipe = build_pipeline(preprocessor, seed=cfg["seed"], **best)
    pipe.fit(train_df, y_train)
    train_time = time.time() - t0

    val_proba = pipe.predict_proba(val_df)
    val_metrics = ev.full_classification_metrics(y_val, val_proba.argmax(1), val_proba)
    test_proba = pipe.predict_proba(test_df)
    test_metrics = ev.full_classification_metrics(y_test, test_proba.argmax(1), test_proba)

    n_features = pipe.named_steps["pre"].get_feature_names_out().shape[0]

    return {
        "model_name": MODEL_NAME,
        "hyperparameters": {**best, "class_weight": "balanced", "random_state": cfg["seed"], "n_jobs": -1},
        "hyperparameter_search": sel["grid_results"],
        "n_features": int(n_features),
        "training_time_seconds": train_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pipeline": pipe,
        "test_y_true": y_test,
        "test_y_proba": test_proba,
    }
