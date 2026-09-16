"""Risk Assessment — live taxpayer risk scoring.

The page collects taxpayer inputs, runs the real trained MLP through the
existing preprocessing/prediction pipeline, stores the latest assessment in
Streamlit session state, and makes that assessment available to Recourse.

Assess -> latest_assessment -> Recourse
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from services import datasets, pipeline, predictor, records, validation
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
    """Return median values for each feature."""
    ranges = datasets.feature_ranges()

    if ranges.empty:
        return {}

    return dict(zip(ranges["feature"], ranges["median"]))


def _number_input(
    feature: str,
    default: float,
    ranges: pd.DataFrame,
):
    """Render one input using observed dataset ranges as guidance."""

    label = datasets.label(feature)

    row = ranges[ranges["feature"] == feature]

    hint = ""

    if not row.empty:
        lo = float(row["min"].iloc[0])
        hi = float(row["max"].iloc[0])
        hint = f"Observed in the dataset: {lo:,.4g} to {hi:,.4g}"

    # Tax return filed is represented as 1 / 0.
    if feature == "tax_return_filed":
        return int(
            st.selectbox(
                label,
                [1, 0],
                index=0 if default >= 0.5 else 1,
                format_func=lambda v: "Yes" if v else "No",
                help="Whether a return was filed for the period.",
            )
        )

    # Integer features.
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

    # Continuous features.
    step = 100.0 if feature in CURRENCY else 0.01
    fmt = "%.2f" if feature in CURRENCY else "%.4f"

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
) -> None:
    """Store the exact assessment used by the Recourse page."""

    # This is the important Assess -> Recourse hand-off.
    st.session_state["latest_assessment"] = {
        "profile_id": profile_id,
        "raw_input": raw_input,
        "x0": x0,
        "probs": probs,
        "pred_class": pred_class,
        "record_id": record_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }

    # Tell Recourse which taxpayer should be pre-selected when the user
    # switches to an existing-profile view.
    st.session_state["recourse_prefill_profile"] = profile_id

    # Keep a persistent selection for the Recourse selectbox.
    st.session_state["recourse_profile_id"] = profile_id


def _risk_badge(pred_class: str) -> str:
    """Return the CSS badge class for a risk class."""

    return {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }.get(pred_class, "neutral")


def render() -> None:

    c.page_header(
        "Assess a taxpayer",
        "Enter a profile to score it against the trained MLP. Fields match the "
        "features the existing preprocessing pipeline expects — nothing more, "
        "nothing fewer.",
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
            "The application could not load the observed feature ranges.",
            kind="stop",
        )
        return

    defaults = _defaults()

    # ------------------------------------------------------------------
    # INPUT FORM
    # ------------------------------------------------------------------

    c.card_open(
        "Taxpayer information",
        glass=False,
        note="Enter the taxpayer attributes used by the trained model.",
    )

    raw_input: dict = {}

    # Tax features.
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
            options = datasets.income_source_categories()

            if not options:
                options = ["Salary"]

            default_value = (
                st.session_state.get("assess_income_sources")
                if st.session_state.get("assess_income_sources") in options
                else options[0]
            )

            raw_input[feature] = st.selectbox(
                datasets.label(feature),
                options,
                index=options.index(default_value),
                key="assess_income_sources",
            )
        else:
            raw_input[feature] = _number_input(
                feature,
                defaults.get(feature, 0.0),
                ranges,
            )

    st.markdown("### Transaction behaviour")

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
            defaults.get(feature, 0.0),
            ranges,
        )

    c.card_close()

    c.spacer(0.8)

    # ------------------------------------------------------------------
    # VALIDATE INPUT
    # ------------------------------------------------------------------

    validation_result = validation.validate_input(raw_input)

    if validation_result:
        # The existing validation service may return either a boolean,
        # dictionary, list, or message depending on implementation.
        # Only display an error when it clearly reports invalid input.
        if isinstance(validation_result, dict):
            errors = validation_result.get("errors", [])
            if errors:
                c.notice(
                    "Input validation",
                    "\n".join(str(x) for x in errors),
                    kind="stop",
                )

    # ------------------------------------------------------------------
    # ASSESS BUTTON
    # ------------------------------------------------------------------

    assess = st.button(
        "Assess Risk",
        type="primary",
        use_container_width=True,
    )

    if assess:

        try:
            with st.spinner("Running the trained MLP…"):

                prediction = predictor.predict(raw_input)

            # Support the application's existing predictor return format.
            if isinstance(prediction, dict):
                probs = prediction.get("probs")
                pred_class = prediction.get("pred_class")

                # Some predictor implementations use "class".
                if pred_class is None:
                    pred_class = prediction.get("class")

                # Some implementations use "probabilities".
                if probs is None:
                    probs = prediction.get("probabilities")

            else:
                raise pipeline.PipelineError(
                    "Unexpected prediction result returned by the model."
                )

            if probs is None or pred_class is None:
                raise pipeline.PipelineError(
                    "Prediction result is missing probabilities or risk class."
                )

            # Convert probabilities to a plain list of floats so they can
            # safely live in Streamlit session state.
            probs = [float(x) for x in probs]

            # Encode the exact raw input through the real preprocessing
            # pipeline for counterfactual use.
            x0 = pipeline.encode(raw_input)

            # ----------------------------------------------------------
            # CREATE ONE ASSESSMENT RECORD
            # ----------------------------------------------------------

            record = records.record_assessment(
                profile_id="MANUAL",
                source="manual",
                raw_input=raw_input,
                probs=probs,
                pred_class=pred_class,
                dataset_label=None,
            )

            record_id = record.get("record_id") if record else None

            # ----------------------------------------------------------
            # SAVE AS LATEST ASSESSMENT
            # ----------------------------------------------------------

            _store_latest_assessment(
                profile_id="MANUAL",
                raw_input=raw_input,
                x0=x0,
                probs=probs,
                pred_class=pred_class,
                record_id=record_id,
            )

            st.session_state["latest_cf"] = None

            st.session_state["assessment_message"] = (
                "This assessment is now available in Recourse under "
                "'Latest risk assessment'."
            )

        except pipeline.PipelineError as exc:

            c.notice(
                "Assessment failed",
                str(exc),
                kind="stop",
            )
            return

        except Exception as exc:

            c.notice(
                "Assessment failed",
                f"{type(exc).__name__}: {exc}",
                kind="stop",
            )
            return

    # ------------------------------------------------------------------
    # DISPLAY LATEST RESULT
    # ------------------------------------------------------------------

    latest = st.session_state.get("latest_assessment")

    if latest is None:
        c.spacer(0.8)

        c.placeholder(
            "Risk assessment",
            "Enter the taxpayer information above and click Assess Risk.",
        )

        return

    probs = {
        cls: float(prob)
        for cls, prob in zip(
            pipeline.CLASSES,
            latest["probs"],
        )
    }

    pred_class = latest["pred_class"]

    c.spacer(0.8)

    c.card_open(
        "Risk assessment result",
        glass=False,
        note="Live output from the selected trained MLP.",
    )

    # --------------------------------------------------------------
    # HEADER
    # --------------------------------------------------------------

    badge_color = {
        "Low": "green",
        "Medium": "orange",
        "High": "red",
    }.get(pred_class, "gray")

    header_left, header_right = st.columns([4, 1])

    with header_left:
        st.markdown(
            '<div class="section-title">Predicted risk</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:2rem;font-weight:800;margin-top:.3rem;">'
            f'{pred_class}</div>',
            unsafe_allow_html=True,
        )

    with header_right:
        st.badge(
            f"{pred_class} risk",
            color=badge_color,
        )

    # --------------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------------

    cols = st.columns(3)

    for col, cls in zip(cols, pipeline.CLASSES):

        probability = probs.get(cls, 0.0)

        with col:

            st.metric(
                f"{cls} risk",
                f"{probability * 100:.2f}%",
            )

            st.progress(
                min(max(probability, 0.0), 1.0),
            )

    c.spacer(0.6)

    # --------------------------------------------------------------
    # TAXPAYER / SESSION INFORMATION
    # --------------------------------------------------------------

    info_cols = st.columns(3)

    with info_cols[0]:
        st.caption("Assessment source")
        st.write("Manual entry")

    with info_cols[1]:
        st.caption("Assessment ID")
        st.write(latest.get("record_id") or "Session assessment")

    with info_cols[2]:
        st.caption("Recourse")
        if pred_class in {"Medium", "High"}:
            st.write("Available → Low")
        else:
            st.write("Not required for Low risk")

    c.card_close()

    # ------------------------------------------------------------------
    # RECOURSE HAND-OFF MESSAGE
    # ------------------------------------------------------------------

    if pred_class in {"Medium", "High"}:

        c.spacer(0.7)

        c.notice(
            "Ready for counterfactual explanation",
            "This Medium/High assessment has been saved as the latest "
            "assessment. Open Recourse and select 'Latest risk assessment' "
            "to generate a constrained counterfactual toward Low.",
            kind="info",
        )

    else:

        c.spacer(0.7)

        c.notice(
            "Low-risk assessment",
            "A Medium/High → Low counterfactual is not required for this "
            "assessment.",
            kind="info",
        )

    # ------------------------------------------------------------------
    # DEBUG-FRIENDLY SESSION CONFIRMATION
    # ------------------------------------------------------------------

    st.caption(
        f"Current assessment: {latest['profile_id']} • "
        f"{latest['pred_class']} risk"
    )
