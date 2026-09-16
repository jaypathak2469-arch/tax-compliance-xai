"""Reports — session history, per-assessment export, and drill-down.

Every export reads a structured record from services/records.py — the
canonical, session-scoped store that services/risk_assessment.py,
services/profiles.py and services/counterfactual.py all write to. Nothing
here calls the model or the counterfactual generator; it only formats
numbers that were already computed live on another page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import datasets, export_csv, export_pdf, pipeline, records, validation
from ui import components as c


def render() -> None:
    c.page_header(
        "Reports",
        "Export an assessment and its recourse result, or download the completed "
        "Phase 5 experiment files.",
    )

    rows = records.summary_rows()
    c.card_open("This session", glass=False,
                note="Assessments are kept for the current session only and are not "
                     "written to disk.")
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Record"), use_container_width=True)
    else:
        st.markdown(
            '<p class="section-note">No assessments yet. Score a taxpayer on the '
            'Assess or Profiles page and it will appear here.</p>',
            unsafe_allow_html=True,
        )
    c.card_close()
    c.spacer(1.0)

    if rows:
        _export_section(rows)
        c.spacer(1.2)
    else:
        c.placeholder("Assessment export and drill-down", "score a taxpayer first")
        c.spacer(1.2)

    contents = [
        "Taxpayer ID and every input value",
        "Predicted class and the three class probabilities",
        "Counterfactual result with the changed-feature table",
        "Constraint validation outcome, itemised",
        "L0, L1, L2, optimisation steps, runtime and final objective",
    ]
    bullets = "".join(
        f'<li style="margin-bottom:.4rem;color:var(--muted);line-height:1.6">{i}</li>'
        for i in contents
    )
    c.panel("What a report contains",
            f'<ul style="margin:0;padding-left:1.1rem;font-size:.86rem">{bullets}</ul>',
            delay=2)

    c.spacer(1.1)

    # Existing experiment files are downloadable now — they already exist.
    c.card_open("Completed Phase 5 experiment files", glass=False,
                note="Read straight from results/metrics. Downloading a copy never "
                     "modifies the originals.")
    files = [
        ("Counterfactual results (168 rows)", datasets.counterfactual_results,
         "counterfactual_results.csv"),
        ("Counterfactual summary", datasets.counterfactual_summary,
         "counterfactual_summary.csv"),
        ("Negative validation tests", datasets.negative_tests,
         "negative_validation_tests.csv"),
    ]
    for label, loader, filename in files:
        frame = loader()
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(
                f'<div style="padding-top:.45rem;font-size:.88rem">{label}'
                f'<span class="denominator"> · '
                f'{"not found" if frame.empty else f"{len(frame):,} rows"}</span></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.download_button(
                "Download", data=frame.to_csv(index=False) if not frame.empty else "",
                file_name=filename, mime="text/csv", disabled=frame.empty,
                use_container_width=True, key=f"dl_{filename}",
            )
    c.card_close()


def _export_section(rows: list[dict]) -> None:
    c.card_open("Export an assessment", glass=False,
                note="Choose a record from this session. The export contains only "
                     "values already computed on the Assess, Profiles or Recourse "
                     "page — nothing is recalculated here.")

    labels = [f"{r['Record']} · {r['Profile']} · {r['Predicted risk']}" for r in rows]
    record_ids = [r["Record"] for r in rows]
    choice = st.selectbox("Assessment", labels, label_visibility="collapsed")
    record_id = record_ids[labels.index(choice)]
    record = records.get(record_id)

    if record is None:
        c.notice("Record not found", "This assessment is no longer in session state.",
                 kind="stop")
        c.card_close()
        return

    a, b, _ = st.columns([1, 1, 2])
    with a:
        try:
            csv_text = export_csv.record_to_csv(record)
            st.download_button(
                "Export assessment CSV", data=csv_text,
                file_name=f"{record['profile_id']}_{record['record_id']}.csv",
                mime="text/csv", use_container_width=True, key=f"csv_{record_id}",
            )
        except Exception as exc:  # noqa: BLE001 - surface the real reason, no silent no-op
            st.button("Export assessment CSV", disabled=True, use_container_width=True,
                      help=f"CSV export failed: {exc}")
    with b:
        try:
            pdf_bytes = export_pdf.build_pdf(record)
            st.download_button(
                "Export assessment PDF", data=pdf_bytes,
                file_name=f"{record['profile_id']}_{record['record_id']}.pdf",
                mime="application/pdf", use_container_width=True, key=f"pdf_{record_id}",
            )
        except Exception as exc:  # noqa: BLE001
            st.button("Export assessment PDF", disabled=True, use_container_width=True,
                      help=f"PDF export failed: {exc}")

    c.card_close()
    c.spacer(1.0)
    _drilldown(record)


def _drilldown(record: dict) -> None:
    c.card_open(f"Record {record['record_id']} — {record['profile_id']}", glass=False,
                note=f"Source: {record['source']} · Assessed at "
                     f"{record['created_at'].replace('T', ' ')}")

    probs = {cls: p for cls, p in zip(pipeline.CLASSES, record["probs"])}
    st.markdown(c.probability_rows_html(probs, predicted=record["pred_class"]),
                unsafe_allow_html=True)

    with st.expander("Input values"):
        input_rows = [
            {"Feature": datasets.label(f), "Value": v if isinstance(v, str) else f"{float(v):,.4g}"}
            for f, v in record["raw_input"].items()
        ]
        st.dataframe(pd.DataFrame(input_rows).set_index("Feature"), use_container_width=True)

    cf = record.get("cf")
    if cf:
        st.markdown(c.source_tag(live=True), unsafe_allow_html=True)
        c.spacer(0.6)
        stats = [
            ("Target class", cf["target_class"]),
            ("Result class", cf["predicted_class"]),
            ("Status", cf["status"]),
            ("L0 (features changed)", str(cf["l0"])),
            ("L1 distance", f"{cf['l1']:.4f}"),
            ("L2 distance", f"{cf['l2']:.4f}"),
            ("Optimisation steps", str(cf["optimization_steps"])),
            ("Runtime", f"{cf['runtime_seconds'] * 1000:.2f} ms"),
        ]
        st.markdown(c.stat_grid_html(stats, columns=4), unsafe_allow_html=True)

        if cf["changed_features"]:
            with st.expander(f"Feature changes ({len(cf['changed_features'])})"):
                change_rows = []
                for f in cf["changed_features"]:
                    before, after = cf["raw_before"][f], cf["raw_after"][f]
                    change_rows.append({
                        "Feature": datasets.label(f), "Original": before,
                        "Counterfactual": after, "Change": after - before,
                    })
                st.dataframe(pd.DataFrame(change_rows).set_index("Feature"),
                            use_container_width=True)

        with st.expander("Constraint validation"):
            checks = validation.constraint_report(cf["feasible"], cf["issues"], cf["finite"])
            st.markdown(c.status_lines_html(checks), unsafe_allow_html=True)
    else:
        c.spacer(0.6)
        c.notice("No recourse generated for this record",
                 "This may already be a Low-risk prediction, or the Recourse page "
                 "was not used for it in this session.", kind="info")

    c.card_close()
