"""Input validation and constraint reporting.

Two distinct jobs live here:

1. validate_input() checks a raw form submission before it reaches the model —
   required fields, numeric types, finiteness, valid category. This is new
   code, because the existing pipeline assumes clean CSV rows and never had to
   validate free-form input.

2. constraint_report() re-groups the output of the EXISTING
   code/constraints.py::validate_constraints() into the six labelled checks the
   UI shows. It does not reimplement constraint logic — it calls the existing
   function and parses its issue strings.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import datasets, pipeline

OK, FAIL = "ok", "fail"


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    detail: str


def validate_input(raw: dict) -> list[str]:
    """Return a list of human-readable problems. Empty list means the input is
    safe to encode and predict on."""
    issues: list[str] = []
    num, cat = pipeline.numeric_and_categorical_columns()

    for feature in num:
        if feature not in raw or raw[feature] is None:
            issues.append(f"{datasets.label(feature)} is missing.")
            continue
        try:
            value = float(raw[feature])
        except (TypeError, ValueError):
            issues.append(f"{datasets.label(feature)} must be numeric.")
            continue
        if value != value or value in (float("inf"), float("-inf")):  # NaN / Inf
            issues.append(f"{datasets.label(feature)} is not a finite number.")

    for feature in cat:
        value = raw.get(feature)
        valid = datasets.income_source_categories()
        if not value:
            issues.append(f"{datasets.label(feature)} is missing.")
        elif valid and value not in valid:
            issues.append(f"{datasets.label(feature)} '{value}' is not a recognised category.")

    if "outstanding_dues" in raw and _is_number(raw.get("outstanding_dues")) and raw["outstanding_dues"] < 0:
        issues.append("Outstanding dues cannot be negative.")
    if "age" in raw and _is_number(raw.get("age")) and not (0 < raw["age"] < 130):
        issues.append("Age is outside a plausible range.")
    if "deduction_claim_ratio" in raw and _is_number(raw.get("deduction_claim_ratio")):
        v = raw["deduction_claim_ratio"]
        if not (0.0 <= v <= 1.0):
            issues.append("Deduction claim ratio must be between 0 and 1.")

    ratio, count = raw.get("transaction_anomaly_ratio"), raw.get("transaction_count")
    declared = raw.get("transaction_anomaly_count")
    if _is_number(ratio) and _is_number(count) and _is_number(declared):
        expected = ratio * count
        if abs(declared - expected) > 1e-4 * max(1.0, abs(expected)):
            issues.append(
                "Anomalous transactions does not equal anomaly ratio × transaction count."
            )
    return issues


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def constraint_report(feasible: bool, issues: list[str], finite: bool) -> list[Check]:
    """Turn the existing validate_constraints() output into six labelled checks.

    `feasible` and `issues` come directly from constraints.validate_constraints();
    `finite` from counterfactual_validation.validate_counterfactual(). No new
    constraint logic — this only groups and labels what the existing check
    already found.
    """
    def has(prefix: str) -> list[str]:
        return [i for i in issues if i.startswith(prefix)]

    groups = [
        ("Immutable constraints", "immutable:"),
        ("Monotonic constraints", "monotonic:"),
        ("Bounded constraints", "bound:"),
        ("Derived relationship", "derived:"),
        ("Categorical validity", "categorical:"),
    ]
    checks = []
    for label, prefix in groups:
        hit = has(prefix)
        if hit:
            checks.append(Check(label, FAIL, ", ".join(f.split(":", 1)[1] for f in hit)))
        else:
            checks.append(Check(label, OK, "satisfied"))
    checks.append(
        Check("No NaN / Inf", OK if finite else FAIL, "all values finite" if finite else "non-finite value found")
    )
    return checks


def overall_valid(checks: list[Check]) -> bool:
    return all(c.state == OK for c in checks)
