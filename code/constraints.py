from __future__ import annotations

import numpy as np
import torch

# Phase-5 feasibility rules retained.
IMMUTABLE = ["age", "previous_default_count", "transaction_count"]
MONOTONIC_DECREASE_ONLY = [
    "outstanding_dues",
    "late_filing_count",
    "transaction_anomaly_ratio",
]

# Domain-support bounds for live counterfactual recourse.
# These bounds are based on the observed support of the project dataset and
# semantic limits. They are application/model guardrails, not legal or
# regulatory limits.
BOUNDS = {
    "annual_income": (150000.0, 7815700.0),
    "late_filing_count": (0.0, 6.0),
    "outstanding_dues": (0.0, 490700.0),
    "income_growth_rate": (-0.50, 0.901),
    "deduction_claim_ratio": (0.0, 1.0),
    "total_transaction_value": (50.0, 301312.72),
    "average_transaction_value": (50.0, 37472.50),
    "transaction_anomaly_count": (0.0, 11.0),
    "transaction_anomaly_ratio": (0.0, 1.0),
    "avg_anomaly_score": (0.0, 1.0),
    "avg_location_risk_score": (0.0, 1.0),
    "avg_device_risk_score": (0.0, 1.0),
    "avg_transaction_frequency": (1.0, 10.0),
    "std_transaction_amount": (0.0, 74495.630550),
    "max_transaction_amount": (50.0, 237907.82),
}

CATEGORICAL_PREFIX = "income_sources_"
DERIVED_COUNT = "transaction_anomaly_count"


def raw_from_z(z: torch.Tensor, feature_to_idx: dict, scaler_stats) -> dict:
    out = {}
    for f, j in feature_to_idx.items():
        if f in scaler_stats.index:
            out[f] = float(
                z[j].detach().cpu().item() * scaler_stats.loc[f, "scale"]
                + scaler_stats.loc[f, "mean"]
            )
    return out


def z_from_raw_value(raw: float, feature: str, scaler_stats) -> float:
    return (float(raw) - float(scaler_stats.loc[feature, "mean"])) / float(
        scaler_stats.loc[feature, "scale"]
    )


def _raw_value(x: torch.Tensor, feature: str, feature_to_idx: dict, scaler_stats) -> torch.Tensor:
    j = feature_to_idx[feature]
    return x[j] * float(scaler_stats.loc[feature, "scale"]) + float(
        scaler_stats.loc[feature, "mean"]
    )


def _set_raw_value(x: torch.Tensor, feature: str, raw: torch.Tensor, scaler_stats, feature_to_idx: dict) -> None:
    j = feature_to_idx[feature]
    x[j] = (raw - float(scaler_stats.loc[feature, "mean"])) / float(
        scaler_stats.loc[feature, "scale"]
    )


def project_constraints(x: torch.Tensor, x0: torch.Tensor, feature_to_idx: dict, scaler_stats) -> torch.Tensor:
    """Project a standardized candidate onto the live domain-feasibility set."""
    x = x.clone()

    # Immutable features.
    for f in IMMUTABLE:
        x[feature_to_idx[f]] = x0[feature_to_idx[f]]

    # Preserve the original income-source category.
    for f in feature_to_idx:
        if f.startswith(CATEGORICAL_PREFIX):
            x[feature_to_idx[f]] = x0[feature_to_idx[f]]

    # Apply domain bounds to all configured bounded numeric features.
    for f, (lo, hi) in BOUNDS.items():
        if f not in feature_to_idx:
            continue
        raw = _raw_value(x, f, feature_to_idx, scaler_stats)
        raw = torch.clamp(raw, lo, hi)
        _set_raw_value(x, f, raw, scaler_stats, feature_to_idx)

    # Monotonic decrease-only rules are applied after bounds.
    for f in MONOTONIC_DECREASE_ONLY:
        if f not in feature_to_idx:
            continue
        raw = _raw_value(x, f, feature_to_idx, scaler_stats)
        raw0 = _raw_value(x0, f, feature_to_idx, scaler_stats)
        raw = torch.minimum(raw, raw0)
        if f in BOUNDS:
            lo, hi = BOUNDS[f]
            raw = torch.clamp(raw, lo, hi)
        _set_raw_value(x, f, raw, scaler_stats, feature_to_idx)

    # transaction_anomaly_count is derived from the immutable transaction_count
    # and anomaly ratio; never optimise it independently.
    rj = feature_to_idx["transaction_anomaly_ratio"]
    cj = feature_to_idx["transaction_anomaly_count"]
    tj = feature_to_idx["transaction_count"]
    ratio = _raw_value(x, "transaction_anomaly_ratio", feature_to_idx, scaler_stats)
    transactions = _raw_value(x0, "transaction_count", feature_to_idx, scaler_stats)
    count = ratio * transactions
    if "transaction_anomaly_count" in BOUNDS:
        lo, hi = BOUNDS["transaction_anomaly_count"]
        count = torch.clamp(count, lo, hi)
    _set_raw_value(x, "transaction_anomaly_count", count, scaler_stats, feature_to_idx)
    return x


def decode_income_source(z: torch.Tensor, feature_to_idx: dict) -> str | None:
    cats = [
        (f, float(z[j].detach().cpu().item()))
        for f, j in feature_to_idx.items()
        if f.startswith(CATEGORICAL_PREFIX)
    ]
    return max(cats, key=lambda t: t[1])[0].replace(CATEGORICAL_PREFIX, "") if cats else None


def validate_constraints(cf: torch.Tensor, x0: torch.Tensor, feature_to_idx: dict, scaler_stats, tol: float = 1e-5):
    issues = []
    if not torch.isfinite(cf).all():
        issues.append("nonfinite")
        return False, issues

    def raw(z, f):
        return float(_raw_value(z, f, feature_to_idx, scaler_stats).detach().cpu().item())

    for f in IMMUTABLE:
        if abs(raw(cf, f) - raw(x0, f)) > tol * max(1.0, abs(raw(x0, f))):
            issues.append(f"immutable:{f}")

    for f in MONOTONIC_DECREASE_ONLY:
        if raw(cf, f) > raw(x0, f) + tol * max(1.0, abs(raw(x0, f))):
            issues.append(f"monotonic:{f}")

    for f, (lo, hi) in BOUNDS.items():
        if f not in feature_to_idx:
            continue
        v = raw(cf, f)
        if v < lo - tol or v > hi + tol:
            issues.append(f"bound:{f}")

    ratio = raw(cf, "transaction_anomaly_ratio")
    count = raw(cf, "transaction_anomaly_count")
    transactions = raw(cf, "transaction_count")
    if abs(count - ratio * transactions) > 1e-4 * max(1.0, abs(count), abs(ratio * transactions)):
        issues.append("derived:transaction_anomaly_count")

    cat = [f for f in feature_to_idx if f.startswith(CATEGORICAL_PREFIX)]
    if cat:
        c0 = np.array([float(x0[feature_to_idx[f]].detach().cpu()) for f in cat])
        c1 = np.array([float(cf[feature_to_idx[f]].detach().cpu()) for f in cat])
        if np.argmax(c0) != np.argmax(c1) or np.max(np.abs(c1 - np.round(c1))) > 1e-4:
            issues.append("categorical:income_sources")

    return len(issues) == 0, issues
