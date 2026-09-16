"""Dashboard — population, model and recourse headlines.

Every figure on this page is read from an existing project file at run time. The
two population figures are shown side by side because they are different things:
5,000 taxpayers exist in the source data, 4,946 entered the Phase 2 modelling
population after the zero-transaction profiles were excluded.
"""
from __future__ import annotations

import streamlit as st

from services import artifacts, datasets
from ui import components as c


def render() -> None:
    c.page_header(
        "Risk and recourse overview",
        "Predicted tax-compliance risk across the modelled taxpayer population, "
        "with constrained counterfactual recourse toward the Low-risk class.",
    )

    pop = datasets.population_counts()
    reported = datasets.reported_metrics()
    p5 = datasets.phase5_metrics()

    # --- KPI row -----------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    source = pop.get("source")
    modelled = pop.get("modelled") or 0
    excluded = pop.get("excluded")

    with k1:
        c.kpi(
            "Taxpayers in source data",
            f"{source:,}" if source else "—",
            f"{modelled:,} entered modelling" if modelled else "modelling set unavailable",
            delay=1,
        )
    with k2:
        txn = pop.get("transactions")
        c.kpi("Financial transactions", f"{txn:,}" if txn else "—",
              "Aggregated to taxpayer level", delay=2)

    macro_f1 = None
    if reported:
        selected = next(
            (e for e in reported["phase4"]["experiments"] if e.get("selected")), None
        )
        macro_f1 = selected["macro_f1"] if selected else None
    with k3:
        c.kpi("MLP macro F1", f"{macro_f1 * 100:.2f}" if macro_f1 else "—",
              "Weighted cross-entropy, held-out test set",
              unit="%", accent="indigo", delay=3)

    success = None
    if p5:
        overall = [s for s in p5.get("success_rates", []) if s["mode"] == "constrained"]
        if overall:
            n = sum(s["n"] for s in overall)
            hits = sum(s["success_rate"] * s["n"] for s in overall)
            success = (hits / n, n) if n else None
    with k4:
        c.kpi(
            "Constrained recourse reached Low",
            f"{success[0] * 100:.0f}" if success else "—",
            f"{success[1]} held-out Medium and High cases" if success else "Phase 5 results unavailable",
            unit="%", accent="cyan", delay=4,
        )

    c.spacer(1.1)

    if source and excluded:
        c.notice(
            "Two population figures, two denominators",
            f"{source:,} taxpayer profiles exist in the integrated source data. "
            f"{excluded} of them have no transaction records and were excluded during "
            f"Phase 2, leaving {modelled:,} profiles in the modelling population. "
            "Percentages on this page use the modelling population unless stated otherwise.",
            kind="info",
        )
        c.spacer(1.1)

    # --- distribution + status --------------------------------------------
    left, right = st.columns([1.45, 1])

    with left:
        dist = datasets.risk_distribution("modelled")
        if dist.empty:
            c.placeholder("Risk distribution unavailable — split files not found", "Phase D")
        else:
            rows = dist.to_dict("records")
            denom = int(dist["denominator"].iloc[0])
            c.panel(
                "Risk distribution",
                c.risk_rows_html(rows, denom, "modelled taxpayer profiles"),
                note="Class labels from the Phase 2 modelling frame.",
                delay=2,
            )

    with right:
        checks = artifacts.artifact_checks() + artifacts.environment_checks()
        c.panel(
            "Artifacts and environment",
            c.status_lines_html(checks),
            note="Checked against the filesystem on every page load.",
            delay=3,
        )
        c.spacer(0.8)
        c.panel(
            "Runtime components",
            c.status_lines_html(artifacts.engine_checks()),
            note="Wired in later build phases.",
            delay=4,
        )

    c.spacer(1.2)

    # --- what the system does ---------------------------------------------
    steps = [
        ("Assess", "A taxpayer profile is standardised by the fitted preprocessor and "
                   "scored by the MLP into Low, Medium or High risk."),
        ("Explain", "For a Medium or High result, gradient optimisation searches for the "
                    "nearest input that the model would classify as Low."),
        ("Constrain", "The search is projected onto a feasibility set after every step: "
                      "age, prior defaults and transaction count cannot move; dues, late "
                      "filings and the anomaly ratio can only fall."),
        ("Validate", "An independent check re-tests the result against every constraint "
                     "and confirms the target class was actually reached."),
    ]
    cells = "".join(
        f"""<div style="flex:1;min-width:190px">
              <div class="eyebrow" style="color:var(--primary-soft)">{i}</div>
              <div style="font-weight:600;font-size:.95rem;margin:.3rem 0 .35rem">{name}</div>
              <div style="color:var(--muted);font-size:.83rem;line-height:1.6">{body}</div>
            </div>"""
        for i, (name, body) in enumerate(steps, 1)
    )
    c.panel(
        "How an assessment becomes a recourse path",
        f'<div style="display:flex;gap:1.6rem;flex-wrap:wrap">{cells}</div>',
        note="Each stage runs against the existing trained pipeline. The application "
             "does not train or refit anything.",
        delay=4,
    )

    c.spacer(1.0)
    c.notice(
        "Research prototype on synthetic data",
        "Counterfactual outputs are model-level recourse under the configured "
        "mathematical constraints. They are not regulatory, financial or tax advice.",
        kind="caution",
    )
