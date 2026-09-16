"""Matplotlib figures for Phase 3 (and reusable in later phases)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Low", "Medium", "High"]


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(counts: dict, path: Path, title: str = "Class distribution (overall_risk)"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.set_ylabel("count")
    ax.set_title(title)
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _save(fig, path)


def plot_confusion_matrix(cm: list[list[int]], model_name: str, path: Path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_metric_comparison(model_metrics: dict, metric_key: str, path: Path, title: str = None):
    """model_metrics: {model_name: metrics_dict}. metric_key: 'accuracy'|'macro_f1'|'weighted_f1'."""
    names = list(model_metrics.keys())
    vals = [model_metrics[n][metric_key] for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(names, vals, color="#4C72B0")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric_key)
    ax.set_title(title or f"Model comparison — {metric_key}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=15)
    _save(fig, path)


def plot_per_class_metric(model_metrics: dict, metric: str, path: Path):
    """Grouped bar: per-class metric ('precision'|'recall'|'f1') across models."""
    names = list(model_metrics.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = np.arange(len(CLASSES))
    for i, name in enumerate(names):
        vals = [model_metrics[name]["per_class"][c][metric] for c in CLASSES]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-class {metric} by model")
    ax.legend()
    _save(fig, path)


def plot_training_curves(train_series: list[float], val_series: list[float], ylabel: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(train_series) + 1)
    ax.plot(epochs, train_series, label="train")
    ax.plot(epochs, val_series, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def plot_high_risk_group_bar(model_names: list[str], metric_dict: dict, path: Path):
    """metric_dict: {'precision': [...], 'recall': [...], 'f1': [...]} aligned to model_names."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.25
    x = np.arange(len(model_names))
    for i, (metric, vals) in enumerate(metric_dict.items()):
        ax.bar(x + i * width, vals, width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_title("High-risk precision / recall / F1 by model")
    ax.legend()
    _save(fig, path)


def plot_roc_curves(y_true: np.ndarray, proba_by_model: dict, class_idx: int, class_name: str, path: Path):
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=[0, 1, 2])[:, class_idx]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        ax.text(0.5, 0.5, "undefined (single class in fold)", ha="center", va="center")
    else:
        for name, proba in proba_by_model.items():
            fpr, tpr, _ = roc_curve(y_bin, proba[:, class_idx])
            auc = roc_auc_score(y_bin, proba[:, class_idx])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {class_name} vs rest")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_pr_curves(y_true: np.ndarray, proba_by_model: dict, class_idx: int, class_name: str, path: Path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=[0, 1, 2])[:, class_idx]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if y_bin.sum() == 0:
        ax.text(0.5, 0.5, "undefined (no positives)", ha="center", va="center")
    else:
        for name, proba in proba_by_model.items():
            prec, rec, _ = precision_recall_curve(y_bin, proba[:, class_idx])
            ap = average_precision_score(y_bin, proba[:, class_idx])
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        base = y_bin.mean()
        ax.axhline(base, color="k", linestyle="--", linewidth=0.8, label=f"baseline ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {class_name} vs rest")
    ax.legend(fontsize=8)
    _save(fig, path)
