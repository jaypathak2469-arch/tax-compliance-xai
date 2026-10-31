
"""Data quality assessment page for TAX-XAI."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from services import datasets, validation


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _quality_report(raw: dict) -> dict:
    numeric = datasets.NUMERIC_FEATURES
    required = numeric + [datasets.CATEGORICAL_FEATURE]

    # 1. Completeness
    completeness = (
        sum(
            raw.get(f) is not None
            and not (
                isinstance(raw.get(f), str)
                and not raw.get(f).strip()
            )
            for f in required
        ) / len(required)
        if required else 0.0
    )

    # 2. Numeric validity
    numeric_valid = (
        sum(_finite(raw.get(f)) for f in numeric) / len(numeric)
        if numeric else 0.0
    )

    # 3. Category validity
    categories = datasets.income_source_categories()
    category_valid = (
        1.0 if raw.get("income_sources") in categories else 0.0
    ) if categories else 0.0

    # 4. Range checks: apply only defensible field-specific rules.
    range_checks = []

    def check_range(feature, condition):
        value = raw.get(feature)
        if _finite(value):
            range_checks.append(condition(float(value)))

    check_range("age", lambda v: 0 < v < 130)
    check_range("annual_income", lambda v: v >= 0)
    check_range("outstanding_dues", lambda v: v >= 0)
    check_range("deduction_claim_ratio", lambda v: 0 <= v <= 1)
    check_range("transaction_anomaly_ratio", lambda v: 0 <= v <= 1)

    # Binary feature
    check_range("tax_return_filed", lambda v: v in (0, 1))

    # Count fields must be non-negative integers.
    count_features = [
        "late_filing_count",
        "previous_default_count",
        "transaction_count",
        "transaction_anomaly_count",
    ]

    for feature in count_features:
        check_range(
            feature,
            lambda v: v >= 0 and v.is_integer()
        )

    # Transaction amounts and frequency cannot be negative.
    non_negative_features = [
        "total_transaction_value",
        "average_transaction_value",
        "avg_transaction_frequency",
        "std_transaction_amount",
        "max_transaction_amount",
    ]

    for feature in non_negative_features:
        check_range(feature, lambda v: v >= 0)

    range_score = (
        sum(range_checks) / len(range_checks)
        if range_checks else 0.0
    )

    # 5. Consistency: assess explicit relationships only.
    consistency_checks = []

    transaction_count = raw.get("transaction_count")
    anomaly_count = raw.get("transaction_anomaly_count")
    anomaly_ratio = raw.get("transaction_anomaly_ratio")

    if _finite(transaction_count) and _finite(anomaly_count):
        consistency_checks.append(
            float(anomaly_count) <= float(transaction_count)
        )

    if (
        _finite(transaction_count)
        and _finite(anomaly_count)
        and _finite(anomaly_ratio)
        and float(transaction_count) >= 0
    ):
        expected_ratio = (
            float(anomaly_count) / float(transaction_count)
            if float(transaction_count) > 0 else 0.0
        )

        tolerance = 1e-4 * max(1.0, abs(expected_ratio))

        consistency_checks.append(
            abs(float(anomaly_ratio) - expected_ratio) <= tolerance
        )

    consistency_score = (
        sum(consistency_checks) / len(consistency_checks)
        if consistency_checks else 0.0
    )

    # Existing application validation messages.
    issues = validation.validate_input(raw)

    # Preserve the original scoring weights.
    overall = (
        0.25 * completeness
        + 0.25 * numeric_valid
        + 0.15 * category_valid
        + 0.20 * range_score
        + 0.15 * consistency_score
    )

    return {
        "overall": overall,
        "completeness": completeness,
        "numeric_validity": numeric_valid,
        "category_validity": category_valid,
        "range_checks": range_score,
        "consistency": consistency_score,
        "issues": issues,
    }


def _status(score: float) -> str:
    if score >= 0.90:
        return "Excellent"
    if score >= 0.75:
        return "Good"
    if score >= 0.60:
        return "Needs review"
    return "Poor"


def render() -> None:
    st.markdown("## Data quality")
    st.caption(
        "A pre-assessment quality screen for the live model inputs. "
        "The score is an application-level data-quality indicator, not a "
        "statistical confidence score or a measure of taxpayer compliance."
    )

    profiles = datasets.load_modelled_profiles()
    if profiles.empty:
        st.error("The modelled profile dataset is unavailable.")
        return

    latest = st.session_state.get("latest_assessment") or {}
    ids = profiles["profile_id"].tolist()
    latest_id = latest.get("profile_id")
    default_index = ids.index(latest_id) if latest_id in ids else 0

    selected_id = st.selectbox(
        "Select taxpayer",
        ids,
        index=default_index,
        key="quality_profile_id",
    )

    row = profiles.loc[
        profiles["profile_id"] == selected_id
    ].iloc[0]

    raw = {
        feature: row[feature]
        for feature in (
            datasets.NUMERIC_FEATURES
            + [datasets.CATEGORICAL_FEATURE]
        )
    }

    if latest_id == selected_id and latest.get("raw_input"):
        raw = dict(latest["raw_input"])

    report = _quality_report(raw)

    st.markdown("### Quality score")
    score_cols = st.columns(2)

    with score_cols[0]:
        st.metric(
            "Overall data quality",
            f"{report['overall']:.0%}"
        )

    with score_cols[1]:
        st.metric(
            "Assessment",
            _status(report["overall"])
        )

    st.progress(report["overall"])

    st.markdown("### Quality dimensions")

    dimensions = [
        (
            "Completeness",
            report["completeness"],
            "Required model inputs present",
        ),
        (
            "Numeric validity",
            report["numeric_validity"],
            "Numeric values are finite",
        ),
        (
            "Category validity",
            report["category_validity"],
            "Income-source category is recognised",
        ),
        (
            "Range checks",
            report["range_checks"],
            "Field-specific plausibility and bound checks",
        ),
        (
            "Consistency",
            report["consistency"],
            "Transaction count and anomaly ratio relationships",
        ),
    ]

    table = pd.DataFrame(
        [
            {
                "Dimension": name,
                "Score": f"{score:.0%}",
                "Status": _status(score),
                "Meaning": meaning,
            }
            for name, score, meaning in dimensions
        ]
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    if report["issues"]:
        st.markdown("### Issues requiring review")
        for issue in report["issues"]:
            st.warning(issue)
    else:
        st.success(
            "No issues were detected by the application's input validation rules."
        )

    st.markdown("### Interpretation")
    st.info(
        "The quality score summarizes completeness, finite numeric values, "
        "category validity, field-specific range checks, and explicit "
        "transaction consistency checks. The range checks use selected "
        "field-specific rules rather than treating observed dataset ranges "
        "as universal business limits. This score does not certify that "
        "underlying information is factually correct, and it does not "
        "change the trained model."
    )
