"""Shared evaluation utilities used by every Phase 3+ model.

All metrics are computed from actual predictions on actual arrays — nothing
here fabricates or hardcodes a result.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

CLASSES = ["Low", "Medium", "High"]  # index 0,1,2 — must match config.yaml


def full_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """y_proba: (n_samples, 3) probability matrix, columns ordered Low, Medium, High."""
    n_classes = len(CLASSES)
    labels = list(range(n_classes))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        CLASSES[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # One-vs-rest ROC-AUC / PR-AUC. Only defined for classes present in y_true.
    y_true_bin = label_binarize(y_true, classes=labels)
    roc_auc, pr_auc = {}, {}
    present = sorted(set(y_true.tolist()))
    for i in labels:
        cname = CLASSES[i]
        if i not in present or y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == len(y_true):
            roc_auc[cname] = None
            pr_auc[cname] = None
            continue
        try:
            roc_auc[cname] = float(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))
        except ValueError:
            roc_auc[cname] = None
        try:
            pr_auc[cname] = float(average_precision_score(y_true_bin[:, i], y_proba[:, i]))
        except ValueError:
            pr_auc[cname] = None

    valid_roc = [v for v in roc_auc.values() if v is not None]
    valid_pr = [v for v in pr_auc.values() if v is not None]

    try:
        macro_roc_auc_ovr = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        macro_roc_auc_ovr = None

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": CLASSES,
        "roc_auc_per_class_ovr": roc_auc,
        "pr_auc_per_class_ovr": pr_auc,
        "roc_auc_macro_ovr": macro_roc_auc_ovr,
        "roc_auc_macro_mean_of_valid": float(np.mean(valid_roc)) if valid_roc else None,
        "pr_auc_macro_mean_of_valid": float(np.mean(valid_pr)) if valid_pr else None,
        "high_risk": per_class["High"],
        "high_risk_roc_auc": roc_auc["High"],
        "high_risk_pr_auc": pr_auc["High"],
        "text_report": classification_report(
            y_true, y_pred, labels=labels, target_names=CLASSES, zero_division=0
        ),
    }


def summarize_cv(fold_metrics: list[dict]) -> dict:
    """Mean/std across repeated-CV folds for the scalar metrics."""
    keys = ["accuracy", "macro_f1", "weighted_f1", "roc_auc_macro_mean_of_valid", "pr_auc_macro_mean_of_valid"]
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_folds": len(vals)} if vals else None

    for cls in CLASSES:
        for metric in ["precision", "recall", "f1"]:
            vals = [m["per_class"][cls][metric] for m in fold_metrics]
            out[f"{cls}_{metric}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        roc_vals = [m["roc_auc_per_class_ovr"][cls] for m in fold_metrics if m["roc_auc_per_class_ovr"][cls] is not None]
        pr_vals = [m["pr_auc_per_class_ovr"][cls] for m in fold_metrics if m["pr_auc_per_class_ovr"][cls] is not None]
        out[f"{cls}_roc_auc"] = (
            {"mean": float(np.mean(roc_vals)), "std": float(np.std(roc_vals)), "n_folds_defined": len(roc_vals)}
            if roc_vals else {"mean": None, "std": None, "n_folds_defined": 0}
        )
        out[f"{cls}_pr_auc"] = (
            {"mean": float(np.mean(pr_vals)), "std": float(np.std(pr_vals)), "n_folds_defined": len(pr_vals)}
            if pr_vals else {"mean": None, "std": None, "n_folds_defined": 0}
        )
    return out
