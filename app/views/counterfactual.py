
"""Counterfactual XAI — constrained recourse toward the Low-risk class.

Live generation calls the existing code/counterfactual.py::generate_counterfactual()
and code/counterfactual_validation.py::validate_counterfactual() through
services/explainer.py. No second counterfactual algorithm exists in this
application. Every live result is tagged "live model output"; the evidence panel
at the bottom of the page is tagged "precomputed phase 5 result" and reads
straight from results/metrics/ — the two are never merged into one number.
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
    df = datasets.load_modelled_profiles()
    row = df[df["profile_id"] == profile_id].iloc[0]
    features = datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]
    return {f: row[f] for f in features}, row["overall_risk"], row["split"]


def _assess(raw: dict) -> dict:
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

    c.notice("Model-level counterfactual recourse", LIMITATION, kind="caution")

    with st.expander("Live recourse domain bounds", expanded=False):
        st.caption(
            "These guardrails use the observed support of the project data plus "
            "simple semantic limits. They are not legal, tax, or regulatory thresholds."
        )
        try:
            pipeline.paths.ensure_code_on_path()
            from constraints import BOUNDS

            bounds_rows = [
                {"Feature": f, "Minimum": lo, "Maximum": hi}
                for f, (lo, hi) in BOUNDS.items()
            ]
            st.dataframe(
                pd.DataFrame(bounds_rows),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.caption(
                f"Bounds are enforced by the live recourse engine. "
                f"Details unavailable in the UI: {exc}"
            )

    c.spacer(1.1)

    live, reason = pipeline.status()

    if not live:
        c.notice("Live pipeline unavailable", reason, kind="stop")
        c.spacer(1.0)
        return

    # --- source selection ---------------------------------------------------

    c.card_open(
        "Choose a case",
        glass=False,
        note="Either a profile from the modelled population, or the most "
             "recent assessment from this session.",
    )

    left, right = st.columns([2, 1])
    profiles = datasets.load_modelled_profiles()

    with left:
        source = st.radio(
            "Source",
            ["Existing taxpayer profile", "Latest risk assessment"],
            horizontal=True,
            label_visibility="collapsed",
        )

    case = None
    prefill = st.session_state.pop("recourse_prefill_profile", None)

    with right:
        if source == "Existing taxpayer profile" and not profiles.empty:
            default = (
                prefill
                if prefill in profiles["profile_id"].values
                else (
                    "TAXP04291"
                    if "TAXP04291" in profiles["profile_id"].values
                    else None
                )
            )

            ids = profiles["profile_id"].tolist()

            pid = st.selectbox(
                "Profile",
                ids,
                index=ids.index(default) if default else 0,
                label_visibility="collapsed",
            )

    c.card_close()
    c.spacer(1.0)

    try:
        if source == "Existing taxpayer profile":
            if profiles.empty:
                c.notice(
                    "No profiles available",
                    "Split files not found.",
                    kind="stop",
                )
                return

            raw, dataset_label, split = _profile_raw(pid)

            with st.spinner(
                "Scoring the selected profile against the live model…"
            ):
                case = _assess(raw)

            case["profile_id"] = pid
            case["dataset_label"] = dataset_label

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

    except pipeline.PipelineError as exc:
        c.notice("Live prediction unavailable", str(exc), kind="stop")
        return

    # --- current state -----------------------------------------------------

    probs = {
        cls: float(p)
        for cls, p in zip(pipeline.CLASSES, case["probs"])
    }

    badge_kind = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }[case["pred_class"]]

    header_bits = [
        f'<span class="badge badge-{badge_kind}">'
        f'{case["pred_class"]} risk</span>'
    ]

    if case.get("dataset_label"):
        header_bits.append(
            c.badge(f"dataset label: {case['dataset_label']}", "neutral")
        )

    c.card_open(delay=1)

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:.9rem"><p class="section-title">Current assessment — '
        f'{case["profile_id"]}</p>{c.source_tag(live=True)}</div>'
        f'<div style="display:flex;gap:.5rem;margin-bottom:.9rem">'
        f'{"".join(header_bits)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        c.probability_rows_html(
            probs,
            predicted=case["pred_class"],
        ),
        unsafe_allow_html=True,
    )

    c.card_close()
    c.spacer(1.0)

    if case["pred_class"] == "Low":
        c.notice(
            "Recourse not applicable",
            "This profile is already predicted Low risk.",
            kind="info",
        )
        return

    generate = st.button(
        "Generate constrained counterfactual",
        type="primary",
    )

    if generate:
        stages_ph = st.empty()

        # Display the real stages performed by the live recourse workflow.
        _render_stage_track(stages_ph, active=1)
        time.sleep(0.2)

        _render_stage_track(stages_ph, active=2)

        try:
            result = explainer.generate(
                case["x0"],
                target=0,
                constrained=True,
            )

        except pipeline.PipelineError as exc:
            stages_ph.empty()
            c.notice("Generation failed", str(exc), kind="stop")
            return

        _render_stage_track(stages_ph, active=3)
        time.sleep(0.2)

        _render_stage_track(stages_ph, active=4)
        time.sleep(0.2)

        stages_ph.empty()

        # Create or reuse the assessment record.
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

        records.attach_counterfactual(record_id, result)

        st.session_state["latest_cf"] = {
            **result,
            "profile_id": case["profile_id"],
            "record_id": record_id,
        }

    if (
        "latest_cf" in st.session_state
        and st.session_state["latest_cf"]["profile_id"] == case["profile_id"]
    ):
        _render_cf_result(st.session_state["latest_cf"])

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
    stages = [
        "Prepare input",
        "Optimise counterfactual",
        "Check constraints",
        "Validate result",
    ]

    cells = []

    for i, stage in enumerate(stages, 1):
        state = (
            "border-color:rgba(129,140,248,.5);"
            "background:rgba(99,102,241,.1)"
            if i <= active
            else "border:1px dashed var(--border);"
                 "background:var(--surface-2)"
        )

        cells.append(
            f"""<div style="flex:1;min-width:150px;padding:.7rem .85rem;
                     border-radius:10px;{state}">
                  <div class="eyebrow">Stage {i}</div>
                  <div style="font-weight:550;font-size:.88rem;
                              margin-top:.3rem">{stage}</div>
                </div>"""
        )

    placeholder.markdown(
        f'<div class="glass"><div style="display:flex;gap:.7rem;flex-wrap:wrap">'
        f'{"".join(cells)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_cf_result(result: dict) -> None:
    from services import validation

    c.spacer(1.0)

    # Recourse is valid only when the target is reached AND
    # the generated counterfactual satisfies the constraints.
    valid_recourse = result["success"] and result["feasible"]

    status_kind = "low" if valid_recourse else "high"

    if valid_recourse:
        display_status = "VALID RECOURSE"
    elif result["success"]:
        display_status = "INVALID RECOURSE"
    else:
        display_status = "UNSUCCESSFUL"

    source_class = pipeline.CLASSES[int(result["p0"].argmax())]

    source_kind = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }[source_class]

    c.card_open(delay=1)

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:.9rem"><p class="section-title">Counterfactual result</p>'
        f'{c.source_tag(live=True)}</div>'
        f'<div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1rem">'
        f'<span class="badge badge-{source_kind}">'
        f'current: {source_class.lower()}</span>'
        f'<span style="color:var(--primary-soft);font-size:1.3rem">→</span>'
        f'<span class="badge badge-low">target: low</span>'
        f'<span class="badge badge-{status_kind}" style="margin-left:auto">'
        f'{display_status}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<p class="section-note" style="margin-bottom:.4rem">Before</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            c.probability_rows_html(
                {
                    cls: float(p)
                    for cls, p in zip(pipeline.CLASSES, result["p0"])
                }
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<p class="section-note" style="margin-bottom:.4rem">After</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            c.probability_rows_html(
                {
                    cls: float(p)
                    for cls, p in zip(pipeline.CLASSES, result["probs"])
                },
                predicted=result["predicted_class"],
            ),
            unsafe_allow_html=True,
        )

    c.card_close()
    c.spacer(1.0)

    stats = [
        ("L0 (features changed)", str(result["l0"])),
        ("L1 distance", f"{result['l1']:.4f}"),
        ("L2 distance", f"{result['l2']:.4f}"),
        ("Optimisation steps", str(result["optimization_steps"])),
        ("Runtime", f"{result['runtime_seconds'] * 1000:.2f} ms"),
        ("Final objective", f"{result['final_objective']:.4f}"),
    ]

    c.panel(
        "Optimisation summary",
        c.stat_grid_html(stats, columns=3),
        glass=False,
        delay=2,
    )

    c.spacer(1.0)

    changed = result["changed_features"]

    if changed:
        rows = []

        for f in changed:
            before = result["raw_before"][f]
            after = result["raw_after"][f]

            rows.append({
                "Feature": datasets.label(f),
                "Original": before,
                "Counterfactual": after,
                "Change": after - before,
            })

        table = pd.DataFrame(rows)

        c.card_open(
            "Feature changes",
            glass=False,
            note=(
                f"{len(changed)} of {len(datasets.NUMERIC_FEATURES)} numeric "
                "features moved."
            ),
        )

        st.dataframe(
            table.set_index("Feature"),
            use_container_width=True,
            column_config={
                col: st.column_config.NumberColumn(col, format="%.4f")
                for col in ["Original", "Counterfactual", "Change"]
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
            f'</div>'
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

    con = summary[summary["mode"] == "constrained"]

    items = []

    for _, row in con.iterrows():
        items.append((
            f"{row['source_risk']} → Low",
            f"{row['success_rate'] * 100:.0f}% of {int(row['n'])}",
        ))

    items.append((
        "Mean optimisation steps",
        f"{con['mean_steps'].mean():.1f}",
    ))

    items.append((
        "Mean runtime per case",
        f"{con['mean_runtime_seconds'].mean() * 1000:.1f} ms",
    ))

    c.panel(
        "Existing Phase 5 evidence",
        c.stat_grid_html(items, columns=4),
        note=(
            "Read from results/metrics/counterfactual_summary.csv. These are "
            "completed experiment results, not live output."
        ),
        delay=4,
    )

    st.markdown(
        c.source_tag(live=False),
        unsafe_allow_html=True,
    )
