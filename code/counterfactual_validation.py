
from __future__ import annotations

import numpy as np
import torch

from constraints import (
    validate_constraints,
    IMMUTABLE,
    MONOTONIC_DECREASE_ONLY,
    BOUNDS,
    BINARY_FEATURES,
)


def validate_prediction_target(
    model,
    cf,
    target=0,
    threshold=0.5,
):
    """Check whether the target class is predicted."""

    with torch.no_grad():
        probabilities = torch.softmax(
            model(cf.unsqueeze(0)), dim=1
        )[0].detach().cpu().numpy()

    predicted_class = int(np.argmax(probabilities))

    return predicted_class == target, probabilities


def validate_counterfactual(
    model,
    cf,
    x0,
    feature_to_idx,
    scaler_stats,
    target=0,
):
    """Validate feasibility and target-class achievement."""

    feasible, issues = validate_constraints(
        cf,
        x0,
        feature_to_idx,
        scaler_stats,
    )

    reached, probabilities = validate_prediction_target(
        model,
        cf,
        target,
    )

    return {
        "feasible": feasible,
        "target_reached": reached,
        "issues": issues,
        "probabilities": probabilities,
        "finite": bool(torch.isfinite(cf).all().item()),
    }


def _set_raw(
    cf,
    feature,
    value,
    feature_to_idx,
    scaler_stats,
):
    """Set a feature using its original, unscaled value."""

    j = feature_to_idx[feature]

    mean = float(scaler_stats.loc[feature, "mean"])
    scale = float(scaler_stats.loc[feature, "scale"])

    cf[j] = (float(value) - mean) / scale

    return cf


def _raw(
    cf,
    feature,
    feature_to_idx,
    scaler_stats,
):
    """Read a feature in its original scale."""

    j = feature_to_idx[feature]

    return (
        float(cf[j].detach().cpu().item())
        * float(scaler_stats.loc[feature, "scale"])
        + float(scaler_stats.loc[feature, "mean"])
    )


def run_negative_tests(
    x0,
    feature_to_idx,
    scaler_stats,
):
    """
    Deliberately corrupt a valid input and verify that
    the constraint validator rejects each invalid candidate.

    Returns:
        List of (test_name, rejected, issues) tuples.
    """

    tests = []

    def check(test_name, cf):
        valid, issues = validate_constraints(
            cf,
            x0,
            feature_to_idx,
            scaler_stats,
        )

        tests.append(
            (test_name, not valid, issues)
        )

    # 1. Immutable age corruption.
    cf = x0.clone()

    _set_raw(
        cf,
        "age",
        _raw(x0, "age", feature_to_idx, scaler_stats) + 1.0,
        feature_to_idx,
        scaler_stats,
    )

    check("immutable_age_corruption", cf)

    # 2. Immutable tax-return status corruption.
    cf = x0.clone()

    original_status = _raw(
        x0, "tax_return_filed",
        feature_to_idx, scaler_stats,
    )

    corrupted_status = 1.0 if original_status < 0.5 else 0.0

    _set_raw(
        cf,
        "tax_return_filed",
        corrupted_status,
        feature_to_idx,
        scaler_stats,
    )

    check("immutable_tax_return_corruption", cf)

    # 3. Monotonic outstanding-dues increase.
    cf = x0.clone()

    _set_raw(
        cf,
        "outstanding_dues",
        _raw(x0, "outstanding_dues", feature_to_idx, scaler_stats)
        + 1000.0,
        feature_to_idx,
        scaler_stats,
    )

    check("monotonic_dues_increase", cf)

    # 4. Numerical deduction-ratio bound violation.
    cf = x0.clone()

    lower, upper = BOUNDS["deduction_claim_ratio"]

    invalid_value = upper + 0.5

    _set_raw(
        cf,
        "deduction_claim_ratio",
        invalid_value,
        feature_to_idx,
        scaler_stats,
    )

    check("deduction_ratio_out_of_bounds", cf)

    # 5. Binary feature corruption.
    cf = x0.clone()

    _set_raw(
        cf,
        "tax_return_filed",
        0.5,
        feature_to_idx,
        scaler_stats,
    )

    check("binary_tax_return_corruption", cf)

    # 6. Derived anomaly-count relationship corruption.
    cf = x0.clone()

    _set_raw(
        cf,
        "transaction_anomaly_count",
        _raw(
            x0,
            "transaction_anomaly_count",
            feature_to_idx,
            scaler_stats,
        ) + 1.0,
        feature_to_idx,
        scaler_stats,
    )

    check("derived_anomaly_count_corruption", cf)

    # 7. Conditional anomaly-count upper-bound violation.
    # Keep the ratio within [0, 1], but make ratio * transactions > 11.
    cf = x0.clone()

    transactions = _raw(
        x0, "transaction_count",
        feature_to_idx, scaler_stats,
    )

    if transactions > 11.0:
        invalid_ratio = min(
            1.0,
            11.0 / transactions + 0.05,
        )

        _set_raw(
            cf,
            "transaction_anomaly_ratio",
            invalid_ratio,
            feature_to_idx,
            scaler_stats,
        )

        _set_raw(
            cf,
            "transaction_anomaly_count",
            invalid_ratio * transactions,
            feature_to_idx,
            scaler_stats,
        )

        check("conditional_anomaly_count_limit", cf)

    return tests
