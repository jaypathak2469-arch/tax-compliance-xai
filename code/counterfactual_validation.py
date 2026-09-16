from __future__ import annotations
import numpy as np
import torch
from constraints import validate_constraints, IMMUTABLE, MONOTONIC_DECREASE_ONLY, BOUNDS


def validate_prediction_target(model, cf, target=0, threshold=0.5):
    with torch.no_grad():
        p = torch.softmax(model(cf.unsqueeze(0)), dim=1)[0].numpy()
    return int(np.argmax(p)) == target, p


def validate_counterfactual(model, cf, x0, feature_to_idx, scaler_stats, target=0):
    feasible, issues = validate_constraints(cf, x0, feature_to_idx, scaler_stats)
    reached, probs = validate_prediction_target(model, cf, target)
    return {
        "feasible": feasible,
        "target_reached": reached,
        "issues": issues,
        "probabilities": probs,
        "finite": bool(torch.isfinite(cf).all().item())
    }


def run_negative_tests(x0, feature_to_idx, scaler_stats):
    tests = []
    # Immutable age corruption.
    cf = x0.clone(); cf[feature_to_idx["age"]] += 1.0
    ok, issues = validate_constraints(cf, x0, feature_to_idx, scaler_stats)
    tests.append(("immutable_age_corruption", not ok, issues))
    # Monotonic dues increase.
    cf = x0.clone(); j = feature_to_idx["outstanding_dues"]
    raw = float(cf[j]) * float(scaler_stats.loc["outstanding_dues", "scale"]) + float(scaler_stats.loc["outstanding_dues", "mean"])
    raw += 1000.0
    cf[j] = (raw - float(scaler_stats.loc["outstanding_dues", "mean"])) / float(scaler_stats.loc["outstanding_dues", "scale"])
    ok, issues = validate_constraints(cf, x0, feature_to_idx, scaler_stats)
    tests.append(("monotonic_dues_increase", not ok, issues))
    # Bound violation.
    cf = x0.clone(); j = feature_to_idx["deduction_claim_ratio"]
    raw = 1.5
    cf[j] = (raw - float(scaler_stats.loc["deduction_claim_ratio", "mean"])) / float(scaler_stats.loc["deduction_claim_ratio", "scale"])
    ok, issues = validate_constraints(cf, x0, feature_to_idx, scaler_stats)
    tests.append(("deduction_ratio_out_of_bounds", not ok, issues))
    # Derived relationship corruption.
    cf = x0.clone(); j = feature_to_idx["transaction_anomaly_count"]
    cf[j] += 1.0
    ok, issues = validate_constraints(cf, x0, feature_to_idx, scaler_stats)
    tests.append(("derived_anomaly_count_corruption", not ok, issues))
    return tests
