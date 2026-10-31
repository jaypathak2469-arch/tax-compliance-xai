
from __future__ import annotations

import numpy as np
import torch


# Features that must never change during counterfactual generation.
IMMUTABLE = [
    "age",
    "previous_default_count",
    "transaction_count",
    "tax_return_filed",
]

# Features that may decrease but must never increase.
MONOTONIC_DECREASE_ONLY = [
    "outstanding_dues",
    "late_filing_count",
    "transaction_anomaly_ratio",
]

# Observed dataset-support bounds.
# These are project/model guardrails, not legal or regulatory thresholds.
BOUNDS = {
    "age": (21.0, 65.0),
    "annual_income": (150000.0, 7815700.0),
    "late_filing_count": (0.0, 6.0),
    "outstanding_dues": (0.0, 490700.0),
    "previous_default_count": (0.0, 3.0),
    "income_growth_rate": (-0.50, 0.901),
    "deduction_claim_ratio": (0.01, 0.80),
    "transaction_count": (1.0, 122.0),
    "total_transaction_value": (50.0, 301312.72),
    "average_transaction_value": (50.0, 37472.50),
    "transaction_anomaly_count": (0.0, 11.0),
    "transaction_anomaly_ratio": (0.0, 1.0),
    "avg_anomaly_score": (0.0, 0.304),
    "avg_location_risk_score": (0.011, 0.545667),
    "avg_device_risk_score": (0.022, 0.589),
    "avg_transaction_frequency": (1.0, 10.0),
    "std_transaction_amount": (0.0, 74495.630550),
    "max_transaction_amount": (50.0, 237907.82),
}

BINARY_FEATURES = [
    "tax_return_filed",
]

CATEGORICAL_PREFIX = "income_sources_"
DERIVED_COUNT = "transaction_anomaly_count"


def _raw_value(
    x: torch.Tensor,
    feature: str,
    feature_to_idx: dict,
    scaler_stats,
) -> torch.Tensor:
    """Convert one standardized feature into its original scale."""

    j = feature_to_idx[feature]

    return (
        x[j] * float(scaler_stats.loc[feature, "scale"])
        + float(scaler_stats.loc[feature, "mean"])
    )


def _set_raw_value(
    x: torch.Tensor,
    feature: str,
    raw: torch.Tensor,
    scaler_stats,
    feature_to_idx: dict,
) -> None:
    """Set one feature using its original-scale value."""

    j = feature_to_idx[feature]

    x[j] = (
        raw - float(scaler_stats.loc[feature, "mean"])
    ) / float(scaler_stats.loc[feature, "scale"])


def raw_from_z(
    z: torch.Tensor,
    feature_to_idx: dict,
    scaler_stats,
) -> dict:
    """Convert standardized features into original-scale values."""

    out = {}

    for feature, j in feature_to_idx.items():
        if feature in scaler_stats.index:
            out[feature] = float(
                z[j].detach().cpu().item()
                * scaler_stats.loc[feature, "scale"]
                + scaler_stats.loc[feature, "mean"]
            )

    return out


def z_from_raw_value(
    raw: float,
    feature: str,
    scaler_stats,
) -> float:
    """Convert an original-scale value into standardized form."""

    return (
        float(raw) - float(scaler_stats.loc[feature, "mean"])
    ) / float(scaler_stats.loc[feature, "scale"])


def _check_required_features(feature_to_idx: dict) -> None:
    """Ensure all features required by the constraints are available."""

    required = (
        set(IMMUTABLE)
        | set(MONOTONIC_DECREASE_ONLY)
        | set(BOUNDS)
        | set(BINARY_FEATURES)
        | {
            "transaction_anomaly_ratio",
            "transaction_anomaly_count",
            "transaction_count",
        }
    )

    missing = sorted(required - set(feature_to_idx))

    if missing:
        raise ValueError(
            f"Missing required constraint features: {missing}"
        )

    missing_scaler = sorted(
        feature
        for feature in required
        if feature not in feature_to_idx
        or feature not in getattr(
            _check_required_features, "_scaler_index", []
        )
    )

    # Scaler-index validation is performed in the public functions.


def project_constraints(
    x: torch.Tensor,
    x0: torch.Tensor,
    feature_to_idx: dict,
    scaler_stats,
) -> torch.Tensor:
    """Project a standardized candidate onto the feasible set."""

    required = (
        set(IMMUTABLE)
        | set(MONOTONIC_DECREASE_ONLY)
        | set(BOUNDS)
        | set(BINARY_FEATURES)
    )

    missing = sorted(required - set(feature_to_idx))

    if missing:
        raise ValueError(
            f"Missing required constraint features: {missing}"
        )

    missing_scaler = sorted(
        feature
        for feature in required
        if feature not in scaler_stats.index
    )

    if missing_scaler:
        raise ValueError(
            f"Missing scaler statistics: {missing_scaler}"
        )

    x = x.clone()

    # 1. Preserve immutable features.
    for feature in IMMUTABLE:
        x[feature_to_idx[feature]] = x0[feature_to_idx[feature]]

    # 2. Preserve original categorical income-source values.
    for feature in feature_to_idx:
        if feature.startswith(CATEGORICAL_PREFIX):
            x[feature_to_idx[feature]] = x0[feature_to_idx[feature]]

    
    # 3. Apply numerical bounds with an inward numerical margin.
    # This avoids float32 inverse-scaling round-off placing values
    # marginally outside the configured limits.
    for feature, (lower, upper) in BOUNDS.items():

        # Immutable features are already pinned to their original values.
        if feature in IMMUTABLE:
            continue

        # The derived anomaly count is calculated separately below.
        if feature == DERIVED_COUNT:
            continue

        raw = _raw_value(
            x, feature, feature_to_idx, scaler_stats
        )

        scale = abs(float(scaler_stats.loc[feature, "scale"]))

        margin = min(
            max(scale * 1e-6, 1e-8),
            (upper - lower) * 1e-3,
        )

        safe_lower = lower + margin
        safe_upper = upper - margin

        raw = torch.clamp(
            raw,
            min=safe_lower,
            max=safe_upper,
        )

        _set_raw_value(
            x, feature, raw, scaler_stats, feature_to_idx
        )


    # 4. Apply monotonic decrease-only constraints.
    for feature in MONOTONIC_DECREASE_ONLY:
        raw = _raw_value(
            x, feature, feature_to_idx, scaler_stats
        )

        raw0 = _raw_value(
            x0, feature, feature_to_idx, scaler_stats
        )

        raw = torch.minimum(raw, raw0)

        lower, upper = BOUNDS[feature]
        raw = torch.clamp(raw, lower, upper)

        _set_raw_value(
            x, feature, raw, scaler_stats, feature_to_idx
        )

    # 5. Enforce the derived anomaly-count relationship.
    # The count must equal:
    # anomaly_ratio * immutable_transaction_count
    #
    # Since the count cannot exceed 11, cap the ratio conditionally.
    ratio = _raw_value(
        x,
        "transaction_anomaly_ratio",
        feature_to_idx,
        scaler_stats,
    )

    transactions = _raw_value(
        x0,
        "transaction_count",
        feature_to_idx,
        scaler_stats,
    )

    max_count = BOUNDS[DERIVED_COUNT][1]

    max_ratio = torch.clamp(
        max_count / transactions,
        max=1.0,
    )

    ratio = torch.minimum(ratio, max_ratio)

    # Reapply the monotonic condition after the conditional cap.
    original_ratio = _raw_value(
        x0,
        "transaction_anomaly_ratio",
        feature_to_idx,
        scaler_stats,
    )

    ratio = torch.minimum(ratio, original_ratio)

    _set_raw_value(
        x,
        "transaction_anomaly_ratio",
        ratio,
        scaler_stats,
        feature_to_idx,
    )

    # Calculate the derived count exactly.
    count = ratio * transactions

    _set_raw_value(
        x,
        DERIVED_COUNT,
        count,
        scaler_stats,
        feature_to_idx,
    )

    return x


def decode_income_source(
    z: torch.Tensor,
    feature_to_idx: dict,
) -> str | None:
    """Decode the preserved income-source category."""

    categories = [
        (feature, float(z[j].detach().cpu().item()))
        for feature, j in feature_to_idx.items()
        if feature.startswith(CATEGORICAL_PREFIX)
    ]

    if not categories:
        return None

    return (
        max(categories, key=lambda item: item[1])[0]
        .replace(CATEGORICAL_PREFIX, "")
    )


def validate_constraints(
    cf: torch.Tensor,
    x0: torch.Tensor,
    feature_to_idx: dict,
    scaler_stats,
    tol: float = 1e-5,
):
    """Validate immutable, monotonic, bounded, binary,
    derived and categorical constraints.
    """

    issues = []

    required = (
        set(IMMUTABLE)
        | set(MONOTONIC_DECREASE_ONLY)
        | set(BOUNDS)
        | set(BINARY_FEATURES)
    )

    missing = sorted(required - set(feature_to_idx))

    if missing:
        return False, [f"missing_features:{missing}"]

    missing_scaler = sorted(
        feature
        for feature in required
        if feature not in scaler_stats.index
    )

    if missing_scaler:
        return False, [f"missing_scaler:{missing_scaler}"]

    if not torch.isfinite(cf).all():
        return False, ["nonfinite"]

    def raw(z, feature):
        return float(
            _raw_value(
                z, feature, feature_to_idx, scaler_stats
            ).detach().cpu().item()
        )

    # 1. Immutable checks.
    for feature in IMMUTABLE:
        value = raw(cf, feature)
        original = raw(x0, feature)

        if abs(value - original) > (
            tol * max(1.0, abs(original))
        ):
            issues.append(f"immutable:{feature}")

    # 2. Monotonic checks.
    for feature in MONOTONIC_DECREASE_ONLY:
        value = raw(cf, feature)
        original = raw(x0, feature)

        if value > original + (
            tol * max(1.0, abs(original))
        ):
            issues.append(f"monotonic:{feature}")

    # 3. Numerical bounds.
    for feature, (lower, upper) in BOUNDS.items():
        value = raw(cf, feature)

        if value < lower - tol or value > upper + tol:
            issues.append(f"bound:{feature}")

    # 4. Binary feature checks.
    for feature in BINARY_FEATURES:
        value = raw(cf, feature)

        if min(
            abs(value),
            abs(value - 1.0),
        ) > tol:
            issues.append(f"binary:{feature}")

    # 5. Derived anomaly-count relationship.
    ratio = raw(cf, "transaction_anomaly_ratio")
    count = raw(cf, "transaction_anomaly_count")
    transactions = raw(cf, "transaction_count")

    expected_count = ratio * transactions

    if abs(count - expected_count) > (
        1e-4
        * max(1.0, abs(count), abs(expected_count))
    ):
        issues.append("derived:transaction_anomaly_count")

    # 6. Conditional anomaly-ratio/count bound.
    if count > BOUNDS[DERIVED_COUNT][1] + tol:
        issues.append("derived:transaction_anomaly_count_bound")

    # 7. Preserve categorical income-source values.
    categories = [
        feature
        for feature in feature_to_idx
        if feature.startswith(CATEGORICAL_PREFIX)
    ]

    if categories:
        original = np.array([
            float(x0[feature_to_idx[feature]].detach().cpu())
            for feature in categories
        ])

        candidate = np.array([
            float(cf[feature_to_idx[feature]].detach().cpu())
            for feature in categories
        ])

        if (
            np.argmax(original) != np.argmax(candidate)
            or np.max(np.abs(candidate - np.round(candidate))) > 1e-4
            or np.max(np.abs(original - candidate)) > 1e-4
        ):
            issues.append("categorical:income_sources")

    return len(issues) == 0, issues
