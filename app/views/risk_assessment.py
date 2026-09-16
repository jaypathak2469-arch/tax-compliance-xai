"""Risk Assessment — existing taxpayer or new client.

Two assessment paths are supported:

1. Existing Taxpayer
   Select a taxpayer from the Phase 2 modelling population and run the
   existing trained MLP through the live pipeline.

2. New Client
   Manually enter the features expected by the existing preprocessing
   pipeline and run the same trained MLP.

Both paths save the result as ``latest_assessment`` so the existing Recourse
workflow continues to work without changing the model or Phase 5 artifacts.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from services import (
    datasets,
    pipeline,
    predictor,
    records,
    validation,
    explanation,
)

from ui import components as c


INT_FEATURES = {
    "age",
    "late_filing_count",
    "previous_default_count",
    "transaction_count",
    "transaction_anomaly_count",
    "tax_return_filed",
}

CURRENCY = {
    "annual_income",
    "outstanding_dues",
    "total_transaction_value",
    "average_transaction_value",
    "max_transaction_amount",
    "std_transaction_amount",
}


def _defaults() -> dict:
    """Return dataset medians as sensible manual-entry defaults."""

    ranges = datasets.feature_ranges()

    if ranges.empty:
        return {}

    return dict(
        zip(
            ranges["feature"],
            ranges["median"],
        )
    )


def _number_input(
    feature: str,
    default: float,
    ranges: pd.DataFrame,
):
    """Render one numeric input."""

    label = datasets.label(feature)

    row = ranges[
        ranges["feature"] == feature
    ]

    hint = ""

    if not row.empty:
        lo = float(row["min"].iloc[0])
        hi = float(row["max"].iloc[0])

        hint = (
            f"Observed in the dataset: "
            f"{lo:,.4g} to {hi:,.4g}"
        )

    if feature == "tax_return_filed":
        return int(
            st.selectbox(
                label,
                [1, 0],
                index=0 if default >= 0.5 else 1,
                format_func=lambda value: (
                    "Yes" if value else "No"
                ),
                help="Whether a return was filed for the period.",
            )
        )

    if feature in INT_FEATURES:
        return int(
            st.number_input(
                label,
                value=int(round(default)),
                step=1,
                format="%d",
                help=hint,
            )
        )

    step = (
        100.0
        if feature in CURRENCY
        else 0.01
    )

    fmt = (
        "%.2f"
        if feature in CURRENCY
        else "%.4f"
    )

    return float(
        st.number_input(
            label,
            value=float(default),
            step=step,
            format=fmt,
            help=hint,
        )
    )


def _store_latest_assessment(
    *,
    profile_id: str,
    raw_input: dict,
    x0,
    probs,
    pred_class: str,
    record_id: str | None,
    source: str,
    dataset_label: str | None = None,
    split: str | None = None,
) -> None:
    """Store an assessment for use by Recourse and other pages."""

    st.session_state["latest_assessment"] = {
        "profile_id": profile_id,
        "raw_input": raw_input,
        "x0": x0,
        "probs": probs,
        "pred_class": pred_class,
        "record_id": record_id,
        "source": source,
        "dataset_label": dataset_label,
        "split": split,
        "timestamp": dt.datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    st.session_state["recourse_prefill_profile"] = profile_id

    if profile_id != "MANUAL":
        st.session_state["recourse_profile_id"] = profile_id


def _run_prediction(
    *,
    profile_id: str,
    source: str,
    raw_input: dict,
    dataset_label: str | None = None,
    split: str | None = None,
):
    """Run the existing live prediction pipeline."""

    try:
        result = predictor.predict(raw_input)

    except pipeline.PipelineError as exc:
        c.notice(
            "Assessment failed",
            str(exc),
            kind="stop",
        )
        return None

    record = records.record_assessment(
        profile_id=profile_id,
        source=source,
        raw_input=result["raw_input"],
        probs=result["probs"],
        pred_class=result["pred_class"],
        dataset_label=dataset_label,
        split=split,
    )

    record_id = (
        record.get("record_id")
        if record
        else None
    )

    _store_latest_assessment(
        profile_id=profile_id,
        raw_input=result["raw_input"],
        x0=result["x0"],
        probs=result["probs"],
        pred_class=result["pred_class"],
        record_id=record_id,
        source=source,
        dataset_label=dataset_label,
        split=split,
    )

    st.session_state["latest_cf"] = None

    return result


def _show_explanation(result) -> None:
    """Display local model-level explanation."""

    try:
        explanation_result = (
            explanation.explain_prediction(
                result["x0"],
                predicted_index=result["pred_idx"],
            )
        )

    except Exception as exc:
        c.notice(
            "Explanation unavailable",
            (
                "The risk prediction is still valid, but the local "
                f"model explanation could not be generated: "
                f"{type(exc).__name__}: {exc}"
            ),
            kind="caution",
        )
        return

    c.spacer(0.7)

    c.card_open(
        "Why this prediction?",
        glass=False,
        note=(
            "Local model-level feature contributions "
            "for this individual assessment."
        ),
    )

    c.notice(
        "Model-related explanation",
        (
            "These contributions describe how the trained MLP responds "
            "to this individual assessment. They are not causal effects "
            "and should not be interpreted as guarantees that changing "
            "a feature will change real-world taxpayer risk."
        ),
        kind="info",
    )

    c.spacer(0.5)

    st.markdown("### Top contributing features")

    top = explanation.top_features(
        explanation_result,
        limit=8,
    )

    if not top:
        st.info(
            "No feature contributions were available."
        )

    else:
        for item in top:

            feature = item["feature"]
            value = float(
                item["attribution"]
            )

            label = datasets.label(
                feature
            )

            if value > 0:
                direction = (
                    "↑ Toward predicted class"
                )
            elif value < 0:
                direction = (
                    "↓ Away from predicted class"
                )
            else:
                direction = "Neutral"

            left, middle, right = st.columns(
                [3, 2, 1]
            )

            with left:
                st.write(
                    f"**{label}**"
                )

            with middle:
                st.caption(direction)

            with right:
                st.write(
                    f"{value:+.4f}"
                )

    c.spacer(0.4)

    st.caption(
        f"Method: {explanation_result['method']}. "
        "Attributions are calculated locally for the selected prediction."
    )

    c.card_close()


def _show_prediction(result) -> None:
    """Display live model probabilities and explanation."""

    probs = {
        cls: float(probability)
        for cls, probability in zip(
            pipeline.CLASSES,
            result["probs"],
        )
    }

    predicted_class = result[
        "pred_class"
    ]

    badge_color = {
        "Low": "green",
        "Medium": "orange",
        "High": "red",
    }.get(
        predicted_class,
        "gray",
    )

    c.spacer(0.8)

    c.card_open(
        "Risk assessment result",
        glass=False,
        note="Live output from the existing trained MLP.",
    )

    header_left, header_right = st.columns(
        [4, 1]
    )

    with header_left:

        st.markdown(
            '<div class="section-title">'
            'Predicted risk'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:2rem;'
            'font-weight:800;margin-top:.3rem;">'
            f"{predicted_class}"
            "</div>",
            unsafe_allow_html=True,
        )

    with header_right:

        st.badge(
            f"{predicted_class} risk",
            color=badge_color,
        )

    probability_columns = st.columns(3)

    for column, risk_class in zip(
        probability_columns,
        pipeline.CLASSES,
    ):

        probability = probs.get(
            risk_class,
            0.0,
        )

        with column:

            st.metric(
                f"{risk_class} risk",
                f"{probability * 100:.2f}%",
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0,
                    ),
                    1.0,
                )
            )

    c.card_close()

    # ---------------------------------------------------------------
    # WHY THIS PREDICTION
    # ---------------------------------------------------------------

    _show_explanation(result)

    c.spacer(0.7)

    if predicted_class in {
        "Medium",
        "High",
    }:

        c.notice(
            "Counterfactual explanation available",
            (
                "This Medium/High assessment can be sent to Recourse "
                "to generate a constrained counterfactual toward Low risk."
            ),
            kind="caution",
        )

    else:

        c.notice(
            "Low-risk assessment",
            (
                "The live model already predicts Low risk, so a "
                "Medium/High → Low counterfactual is not required."
            ),
            kind="info",
        )


def _existing_taxpayer_mode(
    profiles: pd.DataFrame,
) -> None:
    """Existing taxpayer assessment workflow."""

    ids = profiles[
        "profile_id"
    ].tolist()

    if not ids:
        c.notice(
            "No taxpayers available",
            "The modelling population could not be loaded.",
            kind="stop",
        )
        return

    default_id = (
        st.session_state.get(
            "assess_existing_profile"
        )
        if st.session_state.get(
            "assess_existing_profile"
        ) in ids
        else (
            "TAXP04291"
            if "TAXP04291" in ids
            else ids[0]
        )
    )

    chosen = st.selectbox(
        f"Existing taxpayer — {len(ids):,} profiles",
        ids,
        index=ids.index(default_id),
        key="assess_existing_profile",
    )

    row = profiles[
        profiles["profile_id"] == chosen
    ].iloc[0]

    c.spacer(0.5)

    meta_left, meta_right = st.columns(2)

    with meta_left:
        st.caption("Dataset split")
        st.write(str(row["split"]))

    with meta_right:
        st.caption("Recorded dataset label")
        st.write(str(row["overall_risk"]))

    c.spacer(0.5)

    c.notice(
        "Existing taxpayer",
        (
            "This uses the exact feature values stored in the Phase 2 "
            "modelling population. The recorded overall_risk value is "
            "a dataset label, not the live model prediction."
        ),
        kind="info",
    )

    c.spacer(0.6)

    for section, features in (
        datasets.FEATURE_GROUPS.items()
    ):

        items = []

        for feature in features:

            if feature not in row.index:
                continue

            value = row[feature]

            if feature == "tax_return_filed":

                display_value = (
                    "Yes"
                    if float(value) >= 0.5
                    else "No"
                )

            elif isinstance(value, str):

                display_value = value

            else:

                numeric_value = float(value)

                if feature in CURRENCY:

                    display_value = (
                        f"{numeric_value:,.2f}"
                    )

                elif numeric_value.is_integer():

                    display_value = (
                        f"{int(numeric_value):,}"
                    )

                else:

                    display_value = (
                        f"{numeric_value:,.4f}"
                    )

            items.append(
                (
                    datasets.label(feature),
                    display_value,
                )
            )

        c.panel(
            section,
            c.stat_grid_html(
                items,
                columns=4,
            ),
            glass=False,
            delay=2,
        )

        c.spacer(0.7)

    c.spacer(0.4)

    assess = st.button(
        "Assess selected taxpayer",
        type="primary",
        use_container_width=True,
        key="assess_existing_button",
    )

    if assess:

        raw = {
            feature: row[feature]
            for feature in (
                datasets.NUMERIC_FEATURES
                + [datasets.CATEGORICAL_FEATURE]
            )
        }

        result = _run_prediction(
            profile_id=chosen,
            source="profile",
            raw_input=raw,
            dataset_label=row["overall_risk"],
            split=row["split"],
        )

        if result is not None:

            st.session_state[
                "assessment_message"
            ] = (
                f"Live assessment completed for {chosen}."
            )

    latest = st.session_state.get(
        "latest_assessment"
    )

    if latest is None:
        return

    if latest.get("profile_id") != chosen:
        return

    result = {
        "x0": latest["x0"],
        "probs": latest["probs"],
        "pred_idx": int(
            latest["probs"].argmax()
        ),
        "pred_class": latest["pred_class"],
    }

    _show_prediction(result)

    if latest["pred_class"] in {
        "Medium",
        "High",
    }:

        c.spacer(0.5)

        if st.button(
            "Continue to Counterfactual Recourse →",
            type="primary",
            use_container_width=True,
            key="existing_to_recourse",
        ):

            st.session_state[
                "recourse_prefill_profile"
            ] = chosen

            st.session_state[
                "nav_page_request"
            ] = "Recourse"

            st.rerun()


def _new_client_mode(
    ranges: pd.DataFrame,
) -> None:
    """Manual new-client assessment workflow."""

    defaults = _defaults()

    c.notice(
        "New client",
        (
            "Enter the taxpayer attributes used by the existing trained "
            "model. The application will validate the input and run the "
            "live MLP without retraining or changing any model artifact."
        ),
        kind="info",
    )

    c.spacer(0.5)

    c.card_open(
        "Taxpayer information",
        glass=False,
        note="All fields below are model inputs.",
    )

    raw_input: dict = {}

    st.markdown("### Tax & compliance")

    tax_features = [
        "age",
        "annual_income",
        "income_sources",
        "tax_return_filed",
        "late_filing_count",
        "outstanding_dues",
        "previous_default_count",
        "income_growth_rate",
        "deduction_claim_ratio",
    ]

    for feature in tax_features:

        if feature == "income_sources":

            options = (
                datasets.income_source_categories()
            )

            if not options:
                options = ["Salary"]

            current_value = (
                st.session_state.get(
                    "assess_income_sources"
                )
            )

            if current_value not in options:
                current_value = options[0]

            raw_input[feature] = st.selectbox(
                datasets.label(feature),
                options,
                index=options.index(
                    current_value
                ),
                key="assess_income_sources",
            )

        else:

            raw_input[feature] = _number_input(
                feature,
                defaults.get(
                    feature,
                    0.0,
                ),
                ranges,
            )

    st.markdown(
        "### Transaction behaviour"
    )

    transaction_features = [
        "transaction_count",
        "total_transaction_value",
        "average_transaction_value",
        "transaction_anomaly_count",
        "transaction_anomaly_ratio",
        "avg_anomaly_score",
        "avg_location_risk_score",
        "avg_device_risk_score",
        "avg_transaction_frequency",
        "std_transaction_amount",
        "max_transaction_amount",
    ]

    for feature in transaction_features:

        raw_input[feature] = _number_input(
            feature,
            defaults.get(
                feature,
                0.0,
            ),
            ranges,
        )

    c.card_close()

    c.spacer(0.6)

    validation_result = (
        validation.validate_input(
            raw_input
        )
    )

    if validation_result:

        if isinstance(
            validation_result,
            dict,
        ):

            errors = validation_result.get(
                "errors",
                [],
            )

            if errors:

                c.notice(
                    "Input validation",
                    "\n".join(
                        str(error)
                        for error in errors
                    ),
                    kind="stop",
                )

    assess = st.button(
        "Assess New Client",
        type="primary",
        use_container_width=True,
        key="assess_new_client_button",
    )

    if assess:

        if (
            isinstance(
                validation_result,
                dict,
            )
            and validation_result.get(
                "errors"
            )
        ):
            return

        try:

            with st.spinner(
                "Running the trained MLP…"
            ):

                result = _run_prediction(
                    profile_id="MANUAL",
                    source="manual",
                    raw_input=raw_input,
                )

        except Exception as exc:

            c.notice(
                "Assessment failed",
                f"{type(exc).__name__}: {exc}",
                kind="stop",
            )
            return

        if result is not None:

            st.session_state[
                "assessment_message"
            ] = (
                "New-client assessment completed."
            )

    latest = st.session_state.get(
        "latest_assessment"
    )

    if latest is None:
        return

    if latest.get("profile_id") != "MANUAL":
        return

    result = {
        "x0": latest["x0"],
        "probs": latest["probs"],
        "pred_idx": int(
            latest["probs"].argmax()
        ),
        "pred_class": latest["pred_class"],
    }

    _show_prediction(result)

    if latest["pred_class"] in {
        "Medium",
        "High",
    }:

        c.spacer(0.5)

        if st.button(
            "Continue to Counterfactual Recourse →",
            type="primary",
            use_container_width=True,
            key="manual_to_recourse",
        ):

            st.session_state[
                "recourse_prefill_profile"
            ] = "MANUAL"

            st.session_state[
                "nav_page_request"
            ] = "Recourse"

            st.rerun()


def render() -> None:
    """Render the complete Risk Assessment page."""

    c.page_header(
        "Risk Assessment",
        (
            "Assess an existing taxpayer or create a new-client "
            "assessment using the same trained MLP prediction pipeline."
        ),
    )

    c.spacer(0.5)

    live, reason = pipeline.status()

    if not live:

        c.notice(
            "Live pipeline unavailable",
            reason,
            kind="stop",
        )
        return

    ranges = datasets.feature_ranges()

    if ranges.empty:

        c.notice(
            "Feature metadata unavailable",
            (
                "The application could not load the "
                "observed feature ranges."
            ),
            kind="stop",
        )
        return

    profiles = datasets.load_modelled_profiles()

    if profiles.empty:

        c.notice(
            "Profile population unavailable",
            (
                "The persisted Phase 2 modelling population "
                "could not be loaded."
            ),
            kind="stop",
        )
        return

    st.markdown(
        "### Assessment type"
    )

    mode = st.radio(
        "Choose how you want to assess the taxpayer",
        [
            "Existing Taxpayer",
            "New Client",
        ],
        horizontal=True,
        key="risk_assessment_mode",
        label_visibility="collapsed",
    )

    c.spacer(0.5)

    if mode == "Existing Taxpayer":

        _existing_taxpayer_mode(
            profiles,
        )

    else:

        _new_client_mode(
            ranges,
        )