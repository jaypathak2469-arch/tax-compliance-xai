from __future__ import annotations
import numpy as np
import torch

IMMUTABLE = ["age", "previous_default_count", "transaction_count"]
MONOTONIC_DECREASE_ONLY = ["outstanding_dues", "late_filing_count", "transaction_anomaly_ratio"]
BOUNDS = {"deduction_claim_ratio": (0.0, 1.0), "transaction_anomaly_ratio": (0.0, 1.0)}
CATEGORICAL_PREFIX = "income_sources_"
DERIVED_COUNT = "transaction_anomaly_count"


def raw_from_z(z: torch.Tensor, feature_to_idx: dict, scaler_stats) -> dict:
    out = {}
    for f, j in feature_to_idx.items():
        if f in scaler_stats.index:
            out[f] = float(z[j].detach().cpu().item() * scaler_stats.loc[f, "scale"] + scaler_stats.loc[f, "mean"])
    return out


def z_from_raw_value(raw: float, feature: str, scaler_stats) -> float:
    return (float(raw) - float(scaler_stats.loc[feature, "mean"])) / float(scaler_stats.loc[feature, "scale"])


def project_constraints(x: torch.Tensor, x0: torch.Tensor, feature_to_idx: dict, scaler_stats) -> torch.Tensor:
    """Project a standardized candidate onto the Phase-5 feasibility set."""
    x = x.clone()
    # Immutable features.
    for f in IMMUTABLE:
        x[feature_to_idx[f]] = x0[feature_to_idx[f]]

    # Preserve income_sources one-hot category.
    for f in feature_to_idx:
        if f.startswith(CATEGORICAL_PREFIX):
            x[feature_to_idx[f]] = x0[feature_to_idx[f]]

    # Monotonic and bounded constraints in raw units.
    for f in MONOTONIC_DECREASE_ONLY:
        j = feature_to_idx[f]
        raw = x[j] * float(scaler_stats.loc[f, "scale"]) + float(scaler_stats.loc[f, "mean"])
        raw0 = x0[j] * float(scaler_stats.loc[f, "scale"]) + float(scaler_stats.loc[f, "mean"])
        raw = torch.minimum(raw, raw0)
        if f in BOUNDS:
            lo, hi = BOUNDS[f]
            raw = torch.clamp(raw, lo, hi)
        x[j] = (raw - float(scaler_stats.loc[f, "mean"])) / float(scaler_stats.loc[f, "scale"])

    # Other bounded features.
    for f, (lo, hi) in BOUNDS.items():
        j = feature_to_idx[f]
        raw = x[j] * float(scaler_stats.loc[f, "scale"]) + float(scaler_stats.loc[f, "mean"])
        raw = torch.clamp(raw, lo, hi)
        x[j] = (raw - float(scaler_stats.loc[f, "mean"])) / float(scaler_stats.loc[f, "scale"])

    # Derived anomaly count = ratio * immutable transaction_count.
    rj, cj, tj = (feature_to_idx["transaction_anomaly_ratio"],
                  feature_to_idx["transaction_anomaly_count"],
                  feature_to_idx["transaction_count"])
    ratio = x[rj] * float(scaler_stats.loc["transaction_anomaly_ratio", "scale"]) + float(scaler_stats.loc["transaction_anomaly_ratio", "mean"])
    count = ratio * (x0[tj] * float(scaler_stats.loc["transaction_count", "scale"]) + float(scaler_stats.loc["transaction_count", "mean"]))
    x[cj] = (count - float(scaler_stats.loc["transaction_anomaly_count", "mean"])) / float(scaler_stats.loc["transaction_anomaly_count", "scale"])
    return x


def decode_income_source(z: torch.Tensor, feature_to_idx: dict) -> str | None:
    cats = [(f, float(z[j].detach().cpu().item())) for f, j in feature_to_idx.items() if f.startswith(CATEGORICAL_PREFIX)]
    return max(cats, key=lambda t: t[1])[0].replace(CATEGORICAL_PREFIX, "") if cats else None


def validate_constraints(cf: torch.Tensor, x0: torch.Tensor, feature_to_idx: dict, scaler_stats, tol: float = 1e-5):
    issues = []
    if not torch.isfinite(cf).all():
        issues.append("nonfinite")
        return False, issues

    def raw(z, f):
        j = feature_to_idx[f]
        return float(z[j].detach().cpu().item() * scaler_stats.loc[f, "scale"] + scaler_stats.loc[f, "mean"])

    for f in IMMUTABLE:
        if abs(raw(cf, f) - raw(x0, f)) > tol * max(1.0, abs(raw(x0, f))):
            issues.append(f"immutable:{f}")
    for f in MONOTONIC_DECREASE_ONLY:
        if raw(cf, f) > raw(x0, f) + tol * max(1.0, abs(raw(x0, f))):
            issues.append(f"monotonic:{f}")
    for f, (lo, hi) in BOUNDS.items():
        v = raw(cf, f)
        if v < lo - tol or v > hi + tol:
            issues.append(f"bound:{f}")

    ratio = raw(cf, "transaction_anomaly_ratio")
    count = raw(cf, "transaction_anomaly_count")
    transactions = raw(cf, "transaction_count")
    if abs(count - ratio * transactions) > 1e-4 * max(1.0, abs(count), abs(ratio * transactions)):
        issues.append("derived:transaction_anomaly_count")

    # Categorical one-hot validity and invariance.
    cat = [f for f in feature_to_idx if f.startswith(CATEGORICAL_PREFIX)]
    if cat:
        c0 = np.array([float(x0[feature_to_idx[f]].detach().cpu()) for f in cat])
        c1 = np.array([float(cf[feature_to_idx[f]].detach().cpu()) for f in cat])
        if not np.array_equal(np.argmax(c0), np.argmax(c1)) or np.max(np.abs(c1 - np.round(c1))) > 1e-4:
            issues.append("categorical:income_sources")
    return len(issues) == 0, issues
