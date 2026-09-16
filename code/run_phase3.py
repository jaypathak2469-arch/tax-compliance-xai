"""PHASE 3 — baseline model training and evaluation.

Trains Logistic Regression, Random Forest, and XGBoost using the exact Phase 2
artifacts. Hyperparameters are selected on the validation set only. The test
set is touched exactly once per model, for final reporting. Repeated stratified
5-fold CV is run on the training split (with preprocessing refit inside each
fold) to give a stable estimate for the 26-sample High class.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
import evaluate as ev
import train_logistic as tlr
import train_random_forest as trf
import train_xgboost as txgb
import visualization as viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("phase3")

N_CV_SPLITS = 5
N_CV_REPEATS = 5


def run_repeated_cv_generic(name, pipeline_factory, train_df, y_train, cfg, weight_vector=None):
    """Repeated stratified 5-fold CV. A NEW pipeline (incl. preprocessor) is
    fit inside every fold on that fold's training portion only."""
    rskf = RepeatedStratifiedKFold(n_splits=N_CV_SPLITS, n_repeats=N_CV_REPEATS, random_state=cfg["seed"])
    fold_metrics = []
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fold_i, (tr_idx, va_idx) in enumerate(rskf.split(train_df, y_train)):
            pipe = pipeline_factory()
            tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            if weight_vector is not None:
                sw = np.array(weight_vector)[ytr]
                pipe.fit(tr_df, ytr, clf__sample_weight=sw)
            else:
                pipe.fit(tr_df, ytr)
            proba = pipe.predict_proba(va_df)
            pred = proba.argmax(1)
            fold_metrics.append(ev.full_classification_metrics(yva, pred, proba))
    cv_time = time.time() - t0
    summary = ev.summarize_cv(fold_metrics)
    log.info("%s: CV done, %d folds, %.1fs", name, len(fold_metrics), cv_time)
    return {"n_folds": len(fold_metrics), "cv_time_seconds": cv_time, "summary": summary}


def main():
    cfg = dl.load_config()
    np.random.seed(cfg["seed"])
    root = dl.project_root()
    proc_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    metrics_dir = root / cfg["paths"]["metrics_dir"]
    fig_dir = root / cfg["paths"]["figures_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(proc_dir / "train_raw.csv")
    val_df = pd.read_csv(proc_dir / "val_raw.csv")
    test_df = pd.read_csv(proc_dir / "test_raw.csv")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    class_weights = json.load(open(metrics_dir / "class_weights.json"))
    weight_vector = class_weights["weight_vector"]

    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]
    y_train = train_df[target].map({c: i for i, c in enumerate(classes)}).to_numpy()

    log.info("train=%d val=%d test=%d | class counts train=%s",
             len(train_df), len(val_df), len(test_df), class_weights["counts"])

    results = {}

    # ---- Logistic Regression ----
    log.info("training logistic regression")
    lr_out = tlr.train_and_evaluate(cfg, train_df, val_df, test_df)
    lr_cv = run_repeated_cv_generic(
        "logistic_regression",
        tlr.cv_pipeline_factory(cfg, lr_out["hyperparameters"]["C"]),
        train_df, y_train, cfg,
    )
    lr_out["cv"] = lr_cv
    results["logistic_regression"] = lr_out
    joblib.dump(lr_out["pipeline"], models_dir / "logistic_regression.joblib")

    # ---- Random Forest ----
    log.info("training random forest")
    rf_out = trf.train_and_evaluate(cfg, preprocessor, train_df, val_df, test_df)
    best_rf = rf_out["hyperparameters"]

    def rf_factory():
        return trf.build_pipeline(
            preprocessor,
            n_estimators=best_rf["n_estimators"],
            max_depth=best_rf["max_depth"],
            min_samples_leaf=best_rf["min_samples_leaf"],
            seed=cfg["seed"],
        )

    rf_cv = run_repeated_cv_generic("random_forest", rf_factory, train_df, y_train, cfg)
    rf_out["cv"] = rf_cv
    results["random_forest"] = rf_out
    joblib.dump(rf_out["pipeline"], models_dir / "random_forest.joblib")

    # ---- XGBoost ----
    log.info("training xgboost")
    xgb_out = txgb.train_and_evaluate(cfg, preprocessor, train_df, val_df, test_df, weight_vector)
    best_xgb = xgb_out["hyperparameters"]
    xgb_keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"]

    def xgb_factory():
        return txgb.build_pipeline(preprocessor, seed=cfg["seed"], **{k: best_xgb[k] for k in xgb_keys})

    xgb_cv = run_repeated_cv_generic("xgboost", xgb_factory, train_df, y_train, cfg, weight_vector=weight_vector)
    xgb_out["cv"] = xgb_cv
    results["xgboost"] = xgb_out
    joblib.dump(xgb_out["pipeline"], models_dir / "xgboost.joblib")

    # ---- Persist metrics ----
    def strip(o):
        return {k: v for k, v in o.items() if k not in ("pipeline", "test_y_true", "test_y_proba")}

    metrics_out = {name: strip(o) for name, o in results.items()}
    with open(metrics_dir / "phase3_metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2)

    # Flat comparison table
    rows = []
    for name, o in results.items():
        t = o["test_metrics"]
        rows.append({
            "model": name,
            "accuracy": t["accuracy"],
            "macro_f1": t["macro_f1"],
            "weighted_f1": t["weighted_f1"],
            "roc_auc_macro_ovr": t["roc_auc_macro_ovr"],
            "pr_auc_macro_mean": t["pr_auc_macro_mean_of_valid"],
            "high_precision": t["high_risk"]["precision"],
            "high_recall": t["high_risk"]["recall"],
            "high_f1": t["high_risk"]["f1"],
            "high_pr_auc": t["high_risk_pr_auc"],
            "cv_macro_f1_mean": o["cv"]["summary"]["macro_f1"]["mean"] if o["cv"]["summary"]["macro_f1"] else None,
            "cv_macro_f1_std": o["cv"]["summary"]["macro_f1"]["std"] if o["cv"]["summary"]["macro_f1"] else None,
            "cv_high_f1_mean": o["cv"]["summary"]["High_f1"]["mean"],
            "cv_high_f1_std": o["cv"]["summary"]["High_f1"]["std"],
            "training_time_seconds": o["training_time_seconds"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(metrics_dir / "phase3_comparison.csv", index=False)

    # ---- Figures ----
    log.info("generating figures")
    train_counts = class_weights["counts"]
    viz.plot_class_distribution(train_counts, fig_dir / "class_distribution_train.png")

    model_test_metrics = {name: o["test_metrics"] for name, o in results.items()}
    for name, o in results.items():
        viz.plot_confusion_matrix(o["test_metrics"]["confusion_matrix"], name, fig_dir / f"confusion_matrix_{name}.png")

    viz.plot_metric_comparison(model_test_metrics, "accuracy", fig_dir / "comparison_accuracy.png")
    viz.plot_metric_comparison(model_test_metrics, "macro_f1", fig_dir / "comparison_macro_f1.png")
    viz.plot_metric_comparison(model_test_metrics, "weighted_f1", fig_dir / "comparison_weighted_f1.png")

    viz.plot_per_class_metric(model_test_metrics, "precision", fig_dir / "per_class_precision.png")
    viz.plot_per_class_metric(model_test_metrics, "recall", fig_dir / "per_class_recall.png")
    viz.plot_per_class_metric(model_test_metrics, "f1", fig_dir / "per_class_f1.png")

    y_test = results["logistic_regression"]["test_y_true"]  # identical across models (same test split)
    proba_by_model = {name: o["test_y_proba"] for name, o in results.items()}
    for idx, cname in enumerate(classes):
        viz.plot_roc_curves(y_test, proba_by_model, idx, cname, fig_dir / f"roc_{cname.lower()}.png")
        viz.plot_pr_curves(y_test, proba_by_model, idx, cname, fig_dir / f"pr_{cname.lower()}.png")

    log.info("phase 3 complete")
    return comparison, metrics_out


if __name__ == "__main__":
    comp, _ = main()
    pd.set_option("display.width", 200)
    print("\n=== TEST SET COMPARISON ===")
    print(comp.round(4).to_string(index=False))
