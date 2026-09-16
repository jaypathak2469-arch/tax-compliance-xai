"""Taxpayer profile explorer.

Searches the 4,946 profiles that entered the Phase 2 modelling population,
reassembled from the persisted split files.

"Assess risk" runs the profile through the live pipeline.
"Generate constrained counterfactual" sends a Medium/High live prediction
to the Recourse page for live constrained counterfactual generation.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from services import datasets, pipeline, predictor, records
from ui import components as c


def _fmt(feature: str, value) -> str:
    if feature == "tax_return_filed":
        return "Yes" if float(value) >= 0.5 else "No"

    if isinstance(value, str):
        return value

    v = float(value)

    if feature in {
        "annual_income",
        "outstanding_dues",
        "total_transaction_value",
        "average_transaction_value",
        "max_transaction_amount",
        "std_transaction_amount",
    }:
        return f"{v:,.2f}"

    if v.is_integer():
        return f"{int(v):,}"

    return f"{v:,.4f}"


def render() -> None:
    c.page_header(
        "Profile explorer",
        "Look up any taxpayer in the modelling population and inspect the exact "
        "values that feed the model.",
    )

    profiles = datasets.load_modelled_profiles()

    if profiles.empty:
        c.notice(
            "No profiles available",
            "The split files under data/processed were not found.",
            kind="stop",
        )
        return

    # -------------------------------------------------------------------------
    # PROFILE SELECTION
    # -------------------------------------------------------------------------

    ids = profiles["profile_id"].tolist()

    search, meta = st.columns([2, 1])

    with search:
        chosen = st.selectbox(
            f"Profile ID — {len(ids):,} in the modelling population",
            ids,
            index=ids.index("TAXP04291") if "TAXP04291" in ids else 0,
        )

    row = profiles[profiles["profile_id"] == chosen].iloc[0]

    with meta:
        st.markdown(
            f'<div style="padding-top:1.85rem">'
            f'{c.badge(str(row["split"]) + " split", "neutral")} '
            f'{c.badge(str(row["overall_risk"]) + " risk", row["overall_risk"].lower())}'
            f"</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # PROFILE DETAILS
    # -------------------------------------------------------------------------

    c.spacer(1.0)

    for section, features in datasets.FEATURE_GROUPS.items():
        items = [
            (datasets.label(f), _fmt(f, row[f]))
            for f in features
            if f in row.index
        ]

        c.panel(
            section,
            c.stat_grid_html(items, columns=4),
            glass=False,
            delay=2,
        )

        c.spacer(0.8)

    c.notice(
        "Dataset label, not a model prediction",
        f"{row['overall_risk']} is the recorded overall_risk value for {chosen} "
        "in the Phase 2 modelling frame. Use \"Assess risk\" below to see what "
        "the live model predicts for this same profile.",
        kind="info",
    )

    c.spacer(1.1)

    # -------------------------------------------------------------------------
    # ACTION BUTTONS
    #
    # We intentionally run the live prediction first.
    # Counterfactual availability is based on the LIVE model prediction,
    # not the dataset's recorded overall_risk label.
    # -------------------------------------------------------------------------

    a, b, _ = st.columns([1, 1.3, 2])

    with a:
        assess = st.button(
            "Assess risk",
            use_container_width=True,
        )

    with b:
        go_cf = st.button(
            "Generate constrained counterfactual",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # LIVE PREDICTION
    # -------------------------------------------------------------------------

    if assess or go_cf:

        raw = {
            f: row[f]
            for f in datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]
        }

        try:
            result = predictor.predict(raw)

        except pipeline.PipelineError as exc:
            c.spacer(0.9)

            c.notice(
                "Live prediction unavailable",
                str(exc),
                kind="stop",
            )

            return

        # ---------------------------------------------------------------------
        # SAVE ASSESSMENT RECORD
        # ---------------------------------------------------------------------

        record = records.record_assessment(
            profile_id=chosen,
            source="profile",
            raw_input=result["raw_input"],
            probs=result["probs"],
            pred_class=result["pred_class"],
            dataset_label=row["overall_risk"],
            split=row["split"],
        )

        st.session_state["latest_assessment"] = {
            "profile_id": chosen,
            "raw_input": result["raw_input"],
            "x0": result["x0"],
            "probs": result["probs"],
            "pred_class": result["pred_class"],
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "record_id": record["record_id"],
        }

        # ---------------------------------------------------------------------
        # COUNTERFACTUAL REQUEST
        #
        # Only Medium/High live predictions should proceed to recourse.
        # Low-risk taxpayers are already at the target class.
        # ---------------------------------------------------------------------

        if go_cf:

            if result["pred_class"] == "Low":

                c.spacer(0.9)

                c.notice(
                    "Recourse not applicable",
                    "The live model already predicts this taxpayer as Low risk. "
                    "A counterfactual toward Low is therefore not required.",
                    kind="info",
                )

            else:

                # Send the selected taxpayer to Recourse.
                st.session_state["recourse_prefill_profile"] = chosen

                st.session_state["nav_page_request"] = "Recourse"

                st.rerun()

        # ---------------------------------------------------------------------
        # SHOW LIVE MODEL PREDICTION
        # ---------------------------------------------------------------------

        c.spacer(0.9)

        probs = {
            cls: float(p)
            for cls, p in zip(
                pipeline.CLASSES,
                result["probs"],
            )
        }

        cls = result["pred_class"]

        badge_kind = {
            "Low": "low",
            "Medium": "medium",
            "High": "high",
        }[cls]

        c.card_open(delay=1)

        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'margin-bottom:.9rem">'
            f'<p class="section-title">Live model prediction</p>'
            f"{c.source_tag(live=True)}</div>"
            f'<span class="badge badge-{badge_kind}">{cls} risk</span>',
            unsafe_allow_html=True,
        )

        c.spacer(0.5)

        st.markdown(
            c.probability_rows_html(
                probs,
                predicted=cls,
            ),
            unsafe_allow_html=True,
        )

        c.card_close()

        # ---------------------------------------------------------------------
        # RECOURSE INFORMATION
        # ---------------------------------------------------------------------

        if cls != "Low":

            c.spacer(0.8)

            c.notice(
                "Recourse available",
                f"The live model predicts {cls} risk for {chosen}. "
                "A constrained counterfactual toward Low can be generated "
                "from the Recourse page.",
                kind="caution",
            )

        else:

            c.spacer(0.8)

            c.notice(
                "Recourse not applicable",
                "The live model already predicts Low risk for this taxpayer.",
                kind="info",
            )