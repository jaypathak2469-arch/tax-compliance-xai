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
    required = datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]

    completeness = sum(
        raw.get(f) is not None and not (isinstance(raw.get(f), str) and not raw.get(f).strip())
        for f in required
    ) / len(required)

    numeric = datasets.NUMERIC_FEATURES
    numeric_valid = sum(_finite(raw.get(f)) for f in numeric) / len(numeric)

    categories = datasets.income_source_categories()
    category_valid = (
        1.0 if raw.get("income_sources") in categories else 0.0
    ) if categories else 0.0

    range_checks = []
    for feature in numeric:
        value = raw.get(feature)
        if not _finite(value):
            continue
        if feature == "age":
            range_checks.append(0 < float(value) < 130)
        elif feature == "outstanding_dues":
            range_checks.append(float(value) >= 0)
        elif feature == "deduction_claim_ratio":
            range_checks.append(0 <= float(value) <= 1)
        elif feature == "transaction_anomaly_ratio":
            range_checks.append(0 <= float(value) <= 1)
        else:
            range_checks.append(True)

    range_score = sum(range_checks) / len(range_checks) if range_checks else 0.0

    issues = validation.validate_input(raw)
    consistency_score = 1.0 if not issues else max(0.0, 1.0 - len(issues) / 6.0)

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

    row = profiles.loc[profiles["profile_id"] == selected_id].iloc[0]
    raw = {
        feature: row[feature]
        for feature in datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]
    }

    if latest_id == selected_id and latest.get("raw_input"):
        raw = dict(latest["raw_input"])

    report = _quality_report(raw)

    st.markdown("### Quality score")
    score_cols = st.columns(2)
    with score_cols[0]:
        st.metric("Overall data quality", f"{report['overall']:.0%}")
    with score_cols[1]:
        st.metric("Assessment", _status(report["overall"]))

    st.progress(report["overall"])

    st.markdown("### Quality dimensions")

    dimensions = [
        ("Completeness", report["completeness"], "Required model inputs present"),
        ("Numeric validity", report["numeric_validity"], "Numeric values are finite"),
        ("Category validity", report["category_validity"], "Income-source category is recognised"),
        ("Range checks", report["range_checks"], "Basic plausibility/bound checks"),
        ("Consistency", report["consistency"], "Existing validation rules pass"),
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
    st.dataframe(table, use_container_width=True, hide_index=True)

    if report["issues"]:
        st.markdown("### Issues requiring review")
        for issue in report["issues"]:
            st.warning(issue)
    else:
        st.success("No issues were detected by the application's input validation rules.")

    st.markdown("### Interpretation")
    st.info(
        "The quality score summarizes completeness, finite numeric values, "
        "category validity, basic range checks, and the application's existing "
        "validation rules. It does not certify that the underlying information "
        "is factually correct, and it does not change the trained model."
    )
