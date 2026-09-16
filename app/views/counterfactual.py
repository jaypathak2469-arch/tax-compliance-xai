"""Counterfactual XAI — constrained recourse toward the Low-risk class.

Live generation calls the existing code/counterfactual.py::generate_counterfactual()
and code/counterfactual_validation.py::validate_counterfactual() through
services/explainer.py. No second counterfactual algorithm exists in this
application.

Every live result is tagged "live model output"; the evidence panel at the
bottom of the page is tagged "precomputed phase 5 result" and reads straight
from results/metrics/ — the two are never merged into one number.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from services import datasets, explainer, pipeline, predictor, records
from ui import components as c


LIMITATION = (
    "These counterfactual changes represent mathematically feasible changes under "
    "the configured model constraints. They are not regulatory, financial, or tax "
    "advice. The current constraints do not enforce realistic domain ranges for "
    "every mutable numerical feature, so some generated values can fall outside "
    "the ranges observed in the data."
)


def _profile_raw(profile_id: str) -> dict:
    """Load the raw model features for one stored taxpayer profile."""

    df = datasets.load_modelled_profiles()

    row = df[df["profile_id"] == profile_id].iloc[0]

    features = datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]

    return (
        {f: row[f] for f in features},
        row["overall_risk"],
        row["split"],
    )


def _assess(raw: dict) -> dict:
    """Run one raw profile through the live prediction pipeline."""

    result = predictor.predict(raw)

    return {
        "raw_input": result["raw_input"],
        "x0": result["x0"],
        "probs": result["probs"],
        "pred_class": result["pred_class"],
    }


def render() -> None:
    c.page_header(
        "Counterfactual recourse",
        "For a Medium or High assessment, find the nearest profile the model would "
        "classify as Low while respecting the feasibility constraints.",
    )

    c.notice(
        "Model-level counterfactual recourse",
        LIMITATION,
        kind="caution",
    )

    c.spacer(1.1)

    # -------------------------------------------------------------------------
    # LIVE PIPELINE STATUS
    # -------------------------------------------------------------------------

    live, reason = pipeline.status()

    if not live:
        c.notice(
            "Live pipeline unavailable",
            reason,
            kind="stop",
        )
        c.spacer(1.0)
        return

    # -------------------------------------------------------------------------
    # LOAD MODELLED PROFILES
    # -------------------------------------------------------------------------

    profiles = datasets.load_modelled_profiles()

    if profiles.empty:
        c.notice(
            "No profiles available",
            "The split files under data/processed were not found.",
            kind="stop",
        )
        return

    # -------------------------------------------------------------------------
    # SOURCE SELECTION
    #
    # There are two ways to enter recourse:
    #
    # 1. Existing taxpayer profile
    # 2. Latest risk assessment
    #
    # The selected profile is persisted in session state so changing the
    # source option does NOT reset the taxpayer to TAXP04291.
    # -------------------------------------------------------------------------

    c.card_open(
        "Choose a case",
        glass=False,
        note="Either a profile from the modelled population, or the most "
             "recent assessment from this session.",
    )

    left, right = st.columns([2, 1])

    with left:
        source = st.radio(
            "Source",
            ["Existing taxpayer profile", "Latest risk assessment"],
            horizontal=True,
            label_visibility="collapsed",
            key="recourse_source",
        )

    # -------------------------------------------------------------------------
    # PROFILE SELECTION
    # -------------------------------------------------------------------------

    prefill = st.session_state.pop(
        "recourse_prefill_profile",
        None,
    )

    ids = profiles["profile_id"].tolist()

    # If Profiles page sent us here with a specific taxpayer, preserve it.
    if prefill and prefill in ids:
        st.session_state["recourse_profile_id"] = prefill

    # Keep any existing selection only when it is still a valid profile.
    # This prevents Streamlit session state from retaining a stale taxpayer ID.
    current = st.session_state.get("recourse_profile_id")
    if current not in ids:
        st.session_state["recourse_profile_id"] = (
            "TAXP04291"
            if "TAXP04291" in ids
            else ids[0]
        )
    with right:
        if source == "Existing taxpayer profile":

            pid = st.selectbox(
                "Profile",
                ids,
                key="recourse_profile_id",
                label_visibility="collapsed",
            )

    c.card_close()

    c.spacer(1.0)

    # -------------------------------------------------------------------------
    # BUILD CURRENT CASE
    # -------------------------------------------------------------------------

    case = None

    try:

        # ---------------------------------------------------------------------
        # OPTION 1: EXISTING TAXPAYER PROFILE
        # ---------------------------------------------------------------------

        if source == "Existing taxpayer profile":

            pid = st.session_state["recourse_profile_id"]

            raw, dataset_label, split = _profile_raw(pid)

            with st.spinner(
                "Scoring the selected profile against the live model…"
            ):
                case = _assess(raw)

            case["profile_id"] = pid
            case["dataset_label"] = dataset_label
            case["split"] = split

            # -------------------------------------------------------------
            # IMPORTANT:
            # Treat the selected existing profile's live prediction as the
            # current session assessment.
            #
            # This makes "Latest risk assessment" useful immediately after
            # selecting a taxpayer here.
            # -------------------------------------------------------------

            record = records.record_assessment(
                profile_id=pid,
                source="profile",
                raw_input=case["raw_input"],
                probs=case["probs"],
                pred_class=case["pred_class"],
                dataset_label=dataset_label,
                split=split,
            )

            st.session_state["latest_assessment"] = {
                "profile_id": pid,
                "raw_input": case["raw_input"],
                "x0": case["x0"],
                "probs": case["probs"],
                "pred_class": case["pred_class"],
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "record_id": record["record_id"],
            }

            case["record_id"] = record["record_id"]

        # ---------------------------------------------------------------------
        # OPTION 2: LATEST RISK ASSESSMENT
        # ---------------------------------------------------------------------

        else:

            if "latest_assessment" not in st.session_state:

                c.notice(
                    "No assessment yet",
                    "Score a taxpayer on the Assess page first, or choose an "
                    "existing profile above.",
                    kind="info",
                )

                return

            last = st.session_state["latest_assessment"]

            case = {
                "raw_input": last["raw_input"],
                "x0": last["x0"],
                "probs": last["probs"],
                "pred_class": last["pred_class"],
                "profile_id": last["profile_id"],
                "dataset_label": None,
                "record_id": last.get("record_id"),
            }

            # Keep the existing-profile selector synchronized with the
            # taxpayer represented by the latest assessment.
            if case["profile_id"] in ids:
                st.session_state["recourse_profile_id"] = case["profile_id"]

    except pipeline.PipelineError as exc:

        c.notice(
            "Live prediction unavailable",
            str(exc),
            kind="stop",
        )

        return

    # -------------------------------------------------------------------------
    # CURRENT ASSESSMENT
    # -------------------------------------------------------------------------

    probs = {
        cls: float(p)
        for cls, p in zip(
            pipeline.CLASSES,
            case["probs"],
        )
    }

    pred_class = case["pred_class"]

    badge_kind = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }[pred_class]

    header_bits = [
        f'<span class="badge badge-{badge_kind}">'
        f"{pred_class} risk"
        f"</span>"
    ]

    if case.get("dataset_label"):
        header_bits.append(
            c.badge(
                f"dataset label: {case['dataset_label']}",
                "neutral",
            )
        )

    c.card_open(delay=1)

    st.markdown(
        f'<div style="display:flex;align-items:center;'
        f'justify-content:space-between;margin-bottom:.9rem">'
        f'<p class="section-title">Current assessment — '
        f'{case["profile_id"]}</p>'
        f'{c.source_tag(live=True)}'
        f"</div>"
        f'<div style="display:flex;gap:.5rem;margin-bottom:.9rem">'
        f'{"".join(header_bits)}'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        c.probability_rows_html(
            probs,
            predicted=pred_class,
        ),
        unsafe_allow_html=True,
    )

    c.card_close()

    c.spacer(1.0)

    # -------------------------------------------------------------------------
    # LOW-RISK CASE
    # -------------------------------------------------------------------------

    if pred_class == "Low":

        c.notice(
            "Recourse not applicable",
            "This profile is already predicted Low risk.",
            kind="info",
        )

        _evidence_panel()

        return

    # -------------------------------------------------------------------------
    # MEDIUM / HIGH CASE
    # -------------------------------------------------------------------------

    generate = st.button(
        "Generate constrained counterfactual",
        type="primary",
        use_container_width=True,
    )

    if generate:

        stages_ph = st.empty()

        # Honest progression through the actual stages performed by the
        # live generation call.
        _render_stage_track(
            stages_ph,
            active=1,
        )

        time.sleep(0.2)

        _render_stage_track(
            stages_ph,
            active=2,
        )

        try:

            result = explainer.generate(
                case["x0"],
                target=0,
                constrained=True,
            )

        except pipeline.PipelineError as exc:

            stages_ph.empty()

            c.notice(
                "Generation failed",
                str(exc),
                kind="stop",
            )

            return

        _render_stage_track(
            stages_ph,
            active=3,
        )

        time.sleep(0.2)

        _render_stage_track(
            stages_ph,
            active=4,
        )

        time.sleep(0.2)

        stages_ph.empty()

        # ---------------------------------------------------------------------
        # RECORD MANAGEMENT
        # ---------------------------------------------------------------------

        record_id = case.get("record_id")

        if record_id is None or records.get(record_id) is None:

            new_record = records.record_assessment(
                profile_id=case["profile_id"],
                source=(
                    "profile"
                    if case.get("dataset_label")
                    else "session"
                ),
                raw_input=case["raw_input"],
                probs=case["probs"],
                pred_class=case["pred_class"],
                dataset_label=case.get("dataset_label"),
            )

            record_id = new_record["record_id"]

            if source == "Latest risk assessment":
                st.session_state["latest_assessment"]["record_id"] = record_id

        records.attach_counterfactual(
            record_id,
            result,
        )

        # Store the latest counterfactual for this specific taxpayer.
        st.session_state["latest_cf"] = {
            **result,
            "profile_id": case["profile_id"],
            "record_id": record_id,
        }

    # -------------------------------------------------------------------------
    # SHOW COUNTERFACTUAL RESULT
    # -------------------------------------------------------------------------

    if (
        "latest_cf" in st.session_state
        and st.session_state["latest_cf"]["profile_id"]
        == case["profile_id"]
    ):

        _render_cf_result(
            st.session_state["latest_cf"]
        )

    else:

        c.spacer(1.0)

        c.placeholder(
            "Before and after probabilities, feature-change table and "
            "L0 / L1 / L2 / steps / runtime",
            "click Generate above",
        )

        c.spacer(1.0)

        c.placeholder(
            "Constraint validation report",
            "click Generate above",
        )

    _evidence_panel()


def _render_stage_track(placeholder, active: int) -> None:
    """Render counterfactual generation stages without raw HTML."""

    stages = [
        "Prepare input",
        "Optimise counterfactual",
        "Check constraints",
        "Validate result",
    ]

    with placeholder.container():
        st.markdown("#### Counterfactual generation")

        cols = st.columns(4)

        for i, stage in enumerate(stages, start=1):
            with cols[i - 1]:
                label = "Stage {} — {}".format(i, stage)

                if i < active:
                    st.success(label)
                elif i == active:
                    st.info(label)
                else:
                    st.caption(label)

def _render_cf_result(
    result: dict,
) -> None:

    from services import validation

    c.spacer(1.0)

    status_kind = (
        "low"
        if result["success"]
        else "high"
    )

    source_class = pipeline.CLASSES[
        int(result["p0"].argmax())
    ]

    source_kind = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }[source_class]

    c.card_open(delay=1)

    st.markdown(
        f'<div style="display:flex;align-items:center;'
        f'justify-content:space-between;margin-bottom:.9rem">'
        f'<p class="section-title">Counterfactual result</p>'
        f'{c.source_tag(live=True)}'
        f"</div>"
        f'<div style="display:flex;align-items:center;'
        f'gap:1.2rem;margin-bottom:1rem">'
        f'<span class="badge badge-{source_kind}">'
        f"current: {source_class.lower()}"
        f"</span>"
        f'<span style="color:var(--primary-soft);font-size:1.3rem">'
        f"→"
        f"</span>"
        f'<span class="badge badge-low">'
        f"target: low"
        f"</span>"
        f'<span class="badge badge-{status_kind}" '
        f'style="margin-left:auto">'
        f'{result["status"]}'
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<p class="section-note" style="margin-bottom:.4rem">'
            "Before"
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            c.probability_rows_html(
                {
                    cls: float(p)
                    for cls, p in zip(
                        pipeline.CLASSES,
                        result["p0"],
                    )
                }
            ),
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<p class="section-note" style="margin-bottom:.4rem">'
            "After"
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            c.probability_rows_html(
                {
                    cls: float(p)
                    for cls, p in zip(
                        pipeline.CLASSES,
                        result["probs"],
                    )
                },
                predicted=result["predicted_class"],
            ),
            unsafe_allow_html=True,
        )

    c.card_close()

    c.spacer(1.0)

    stats = [
        (
            "L0 (features changed)",
            str(result["l0"]),
        ),
        (
            "L1 distance",
            f"{result['l1']:.4f}",
        ),
        (
            "L2 distance",
            f"{result['l2']:.4f}",
        ),
        (
            "Optimisation steps",
            str(result["optimization_steps"]),
        ),
        (
            "Runtime",
            f"{result['runtime_seconds'] * 1000:.2f} ms",
        ),
        (
            "Final objective",
            f"{result['final_objective']:.4f}",
        ),
    ]

    c.panel(
        "Optimisation summary",
        c.stat_grid_html(
            stats,
            columns=3,
        ),
        glass=False,
        delay=2,
    )

    c.spacer(1.0)

    changed = result["changed_features"]

    if changed:

        rows = []

        for feature in changed:

            before = result["raw_before"][feature]
            after = result["raw_after"][feature]

            rows.append(
                {
                    "Feature": datasets.label(feature),
                    "Original": before,
                    "Counterfactual": after,
                    "Change": after - before,
                }
            )

        table = pd.DataFrame(rows)

        c.card_open(
            "Feature changes",
            glass=False,
            note=(
                f"{len(changed)} of "
                f"{len(datasets.NUMERIC_FEATURES)} numeric "
                "features moved."
            ),
        )

        st.dataframe(
            table.set_index("Feature"),
            use_container_width=True,
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    format="%.4f",
                )
                for col in [
                    "Original",
                    "Counterfactual",
                    "Change",
                ]
            },
        )

        c.card_close()

    else:

        c.notice(
            "No features changed",
            "The optimiser did not move any feature.",
            kind="info",
        )

    c.spacer(1.0)

    checks = validation.constraint_report(
        result["feasible"],
        result["issues"],
        result["finite"],
    )

    overall = validation.overall_valid(checks)

    c.panel(
        "Constraint validation",
        c.status_lines_html(checks)
        + (
            f'<div style="margin-top:.8rem">'
            f'{c.badge("VALID", "low") if overall else c.badge("INVALID", "high")}'
            f"</div>"
        ),
        note=(
            "Computed by the existing constraints.validate_constraints(), "
            "grouped by category for display."
        ),
        delay=3,
    )


def _evidence_panel() -> None:

    c.spacer(1.3)

    summary = datasets.counterfactual_summary()

    if summary.empty:
        return

    constrained = summary[
        summary["mode"] == "constrained"
    ]

    items = []

    for _, row in constrained.iterrows():

        items.append(
            (
                f"{row['source_risk']} → Low",
                f"{row['success_rate'] * 100:.0f}% "
                f"of {int(row['n'])}",
            )
        )

    items.append(
        (
            "Mean optimisation steps",
            f"{constrained['mean_steps'].mean():.1f}",
        )
    )

    items.append(
        (
            "Mean runtime per case",
            f"{constrained['mean_runtime_seconds'].mean() * 1000:.1f} ms",
        )
    )

    c.panel(
        "Existing Phase 5 evidence",
        c.stat_grid_html(
            items,
            columns=4,
        ),
        note=(
            "Read from results/metrics/counterfactual_summary.csv. "
            "These are completed experiment results, not live output."
        ),
        delay=4,
    )

    st.markdown(
        c.source_tag(live=False),
        unsafe_allow_html=True,
    )
