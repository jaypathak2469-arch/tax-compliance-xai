"""Interactive what-if risk simulator for TAX-XAI.

The simulator changes raw model inputs and immediately evaluates the existing
preprocessor + trained MLP. It is a model-behaviour tool, not a causal or
real-world tax recommendation engine.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from services import datasets, predictor, validation


INTEGER_FEATURES = {
    "age",
    "tax_return_filed",
    "late_filing_count",
    "previous_default_count",
    "transaction_count",
    "transaction_anomaly_count",
}

DEFAULT_EDITABLE = [
    "annual_income",
    "late_filing_count",
    "outstanding_dues",
    "income_growth_rate",
    "deduction_claim_ratio",
    "transaction_anomaly_count",
    "transaction_anomaly_ratio",
]


def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback


def _format_probability(value: float) -> str:
    return f"{value:.1%}"


def _risk_badge(risk: str) -> None:
    if risk == "Low":
        st.success("LOW RISK")
    elif risk == "Medium":
        st.warning("MEDIUM RISK")
    else:
        st.error("HIGH RISK")


def _probability_table(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for i, risk in enumerate(datasets.CLASSES):
        before_p = float(before["probs"][i])
        after_p = float(after["probs"][i])
        rows.append({
            "Risk": risk,
            "Before": before_p,
            "After": after_p,
            "Change": after_p - before_p,
        })
    return pd.DataFrame(rows)


def _render_probability_bars(before: dict, after: dict) -> None:
    st.markdown("#### Probability movement")
    data = _probability_table(before, after).set_index("Risk")[["Before", "After"]]
    st.bar_chart(data, y=["Before", "After"], height=260, use_container_width=True)


def _raw_value_input(feature: str, current, minimum, maximum, key: str):
    if feature in INTEGER_FEATURES:
        lo = int(math.floor(minimum))
        hi = int(math.ceil(maximum))
        value = int(round(_safe_float(current)))
        lo = min(lo, value)
        hi = max(hi, value)
        if lo == hi:
            return value
        return st.slider(
            datasets.label(feature),
            min_value=lo,
            max_value=hi,
            value=max(lo, min(hi, value)),
            step=1,
            key=key,
        )

    lo = _safe_float(minimum)
    hi = _safe_float(maximum)
    value = _safe_float(current)

    lo = min(lo, value)
    hi = max(hi, value)

    if abs(hi - lo) < 1e-12:
        return value

    # Ratios/scores benefit from a small step; financial amounts get a
    # human-readable step.
    if "ratio" in feature or "score" in feature or "growth" in feature:
        step = 0.01
        value = round(value, 2)
    else:
        span = hi - lo
        step = max(1.0, round(span / 100.0, 2))

    return st.slider(
        datasets.label(feature),
        min_value=float(lo),
        max_value=float(hi),
        value=float(max(lo, min(hi, value))),
        step=float(step),
        key=key,
    )


def render() -> None:
    st.markdown("## What-if simulator")
    st.caption(
        "Explore how changing model inputs affects the existing live risk "
        "prediction. Values are evaluated by the same fitted preprocessor and "
        "trained MLP used elsewhere in the application."
    )

    modelled = datasets.load_modelled_profiles()
    if modelled.empty:
        st.error("The modelled profile dataset is unavailable.")
        return

    profile_ids = modelled["profile_id"].tolist()
    latest = st.session_state.get("latest_assessment") or {}
    latest_id = latest.get("profile_id")

    default_index = profile_ids.index(latest_id) if latest_id in profile_ids else 0

    st.markdown("### 1. Select taxpayer")

    selected_id = st.selectbox(
        "Taxpayer profile",
        profile_ids,
        index=default_index,
        key="what_if_profile_id",
    )

    stored_row = modelled.loc[
        modelled["profile_id"] == selected_id
    ].iloc[0]

    raw_before = {
        feature: stored_row[feature]
        for feature in datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]
    }

    # If the user just assessed a profile manually, use that exact assessment
    # when it matches the selected profile. Otherwise use persisted profile data.
    if latest_id == selected_id and latest.get("raw_input"):
        raw_before = dict(latest["raw_input"])

    input_issues = validation.validate_input(raw_before)
    if input_issues:
        st.error("Selected profile cannot be simulated because its input is invalid.")
        for issue in input_issues:
            st.write(f"• {issue}")
        return

    try:
        before = predictor.predict(raw_before)
    except Exception as exc:
        st.error(f"Could not run the baseline prediction: {exc}")
        return

    st.markdown("### 2. Current prediction")
    top = st.columns(4)
    with top[0]:
        st.metric("Profile", selected_id)
    with top[1]:
        st.metric("Current risk", before["pred_class"])
    with top[2]:
        st.metric("Low probability", _format_probability(float(before["probs"][0])))
    with top[3]:
        st.metric("Confidence", _format_probability(float(max(before["probs"]))))

    _risk_badge(before["pred_class"])

    ranges = datasets.feature_ranges()
    if ranges.empty:
        st.error("Observed feature ranges are unavailable.")
        return

    range_lookup = ranges.set_index("feature")

    editable = [
        f for f in DEFAULT_EDITABLE
        if f in datasets.NUMERIC_FEATURES and f in raw_before and f in range_lookup.index
    ]

    with st.expander("Choose additional features to change", expanded=False):
        available = [
            f for f in datasets.NUMERIC_FEATURES
            if f in range_lookup.index
        ]
        selected_features = st.multiselect(
            "Features",
            available,
            default=editable,
            format_func=datasets.label,
            key="what_if_features",
        )

    if not selected_features:
        st.info("Select at least one feature to create a what-if scenario.")
        return

    st.markdown("### 3. Build a what-if scenario")
    st.caption(
        "Slider limits are the observed minimum/maximum values in the modelled "
        "dataset. They are not regulatory limits or causal feasibility bounds."
    )

    scenario = dict(raw_before)

    groups = {
        "Taxpayer information": [
            f for f in selected_features
            if f in datasets.FEATURE_GROUPS["Taxpayer information"]
        ],
        "Compliance information": [
            f for f in selected_features
            if f in datasets.FEATURE_GROUPS["Compliance information"]
        ],
        "Transaction behaviour": [
            f for f in selected_features
            if f in datasets.FEATURE_GROUPS["Transaction behaviour"]
        ],
    }

    for group_name, features in groups.items():
        if not features:
            continue
        st.markdown(f"#### {group_name}")
        cols = st.columns(2)
        for index, feature in enumerate(features):
            with cols[index % 2]:
                stats = range_lookup.loc[feature]
                scenario[feature] = _raw_value_input(
                    feature,
                    raw_before[feature],
                    stats["min"],
                    stats["max"],
                    key=f"what_if_{selected_id}_{feature}",
                )

    # Categorical feature can be explored too, while staying within the
    # categories already seen by the fitted encoder.
    categories = datasets.income_source_categories()
    if categories:
        current_source = raw_before.get("income_sources")
        if current_source not in categories:
            current_source = categories[0]
        scenario["income_sources"] = st.selectbox(
            "Income sources",
            categories,
            index=categories.index(current_source),
            key=f"what_if_{selected_id}_income_sources",
        )

    scenario_issues = validation.validate_input(scenario)

    st.markdown("### 4. Compare")

    if scenario_issues:
        st.error("The what-if scenario failed validation.")
        for issue in scenario_issues:
            st.write(f"• {issue}")
        return

    try:
        after = predictor.predict(scenario)
    except Exception as exc:
        st.error(f"Could not evaluate the what-if scenario: {exc}")
        return

    changed = []
    for feature in datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]:
        before_value = raw_before.get(feature)
        after_value = scenario.get(feature)
        try:
            different = float(before_value) != float(after_value)
        except (TypeError, ValueError):
            different = before_value != after_value
        if different:
            changed.append({
                "Feature": datasets.label(feature),
                "Before": before_value,
                "After": after_value,
            })

    result_cols = st.columns(2)
    with result_cols[0]:
        st.markdown("#### Before")
        _risk_badge(before["pred_class"])
        st.metric("Prediction", before["pred_class"])
        st.write(
            "Low:", _format_probability(float(before["probs"][0])),
            " · Medium:", _format_probability(float(before["probs"][1])),
            " · High:", _format_probability(float(before["probs"][2])),
        )

    with result_cols[1]:
        st.markdown("#### After")
        _risk_badge(after["pred_class"])
        st.metric("Prediction", after["pred_class"])
        st.write(
            "Low:", _format_probability(float(after["probs"][0])),
            " · Medium:", _format_probability(float(after["probs"][1])),
            " · High:", _format_probability(float(after["probs"][2])),
        )

    _render_probability_bars(before, after)

    if before["pred_class"] != after["pred_class"]:
        st.success(
            f"Model prediction changed from **{before['pred_class']}** "
            f"to **{after['pred_class']}** in this scenario."
        )
    else:
        st.info(
            f"The model prediction remains **{after['pred_class']}** under this scenario."
        )

    if changed:
        st.markdown("#### Changed inputs")
        st.dataframe(
            pd.DataFrame(changed),
            use_container_width=True,
            hide_index=True,
        )

        table = _probability_table(before, after)
        table["Before"] = table["Before"].map(_format_probability)
        table["After"] = table["After"].map(_format_probability)
        table["Change"] = table["Change"].map(
            lambda x: f"{x:+.1%}"
        )
        st.markdown("#### Probability changes")
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.warning(
        "This simulator shows model-level sensitivity to hypothetical input "
        "changes. It does not establish causality and is not tax, legal, "
        "financial, or regulatory advice."
    )
