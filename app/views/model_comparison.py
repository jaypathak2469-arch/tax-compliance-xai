"""Model comparison — Phase 3 and Phase 4 reported results.

These numbers are transcribed from the project's progress presentation into
app/config/reported_metrics.json. The Phase 3/4 metric files and the trained
classical model artifacts are not present in the project, so the application
cannot and does not recompute them.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services import datasets
from ui import components as c
from ui import theme

METRIC_LABELS = {
    "accuracy": "Accuracy", "macro_f1": "Macro F1",
    "pr_auc": "PR-AUC", "high_f1": "High-risk F1",
}


def render() -> None:
    c.page_header(
        "Model comparison",
        "Held-out test performance for the three classical baselines and the "
        "weighted MLP that Phase 5 uses.",
    )

    reported = datasets.reported_metrics()
    if not reported:
        c.notice("Reported metrics unavailable",
                 "app/config/reported_metrics.json was not found.", kind="stop")
        return

    ev = reported["evaluation"]
    c.notice(
        reported["label"],
        f"Transcribed verbatim from the project progress presentation "
        f"(slides {', '.join(str(s) for s in reported['provenance']['slides'])}). "
        f"Evaluated on the {ev['n_records']}-record held-out test set: "
        f"Low {ev['class_counts']['Low']}, Medium {ev['class_counts']['Medium']}, "
        f"High {ev['class_counts']['High']}. "
        "The application does not recompute these values.",
        kind="info",
    )
    c.spacer(1.1)

    # --- combined table ----------------------------------------------------
    rows = [
        {"Model": m["name"], "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"],
         "PR-AUC": m["pr_auc"], "High-risk F1": m["high_f1"], "Family": "Classical"}
        for m in reported["phase3"]["models"]
    ]
    selected = next(e for e in reported["phase4"]["experiments"] if e["selected"])
    rows.append({
        "Model": "Weighted MLP", "Accuracy": None, "Macro F1": selected["macro_f1"],
        "PR-AUC": None, "High-risk F1": selected["high_f1"], "Family": "Neural",
    })
    table = pd.DataFrame(rows)

    c.card_open("Held-out test metrics", glass=False,
                note="Accuracy and PR-AUC were not reported for the MLP, so those "
                     "cells are left empty rather than filled in.")
    st.dataframe(
        table.set_index("Model"),
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.4f")
            for col in ["Accuracy", "Macro F1", "PR-AUC", "High-risk F1"]
        },
    )
    c.card_close()
    c.spacer(1.0)

    # --- grouped bars on the two metrics reported for every model ----------
    fig = go.Figure()
    for i, metric in enumerate(["Macro F1", "High-risk F1"]):
        sub = table.dropna(subset=[metric])
        fig.add_bar(
            name=metric, x=sub["Model"], y=sub[metric],
            marker_color=theme.SERIES[i], marker_line_width=0,
            text=[f"{v:.3f}" for v in sub[metric]], textposition="outside",
            textfont=dict(color=theme.MUTED, size=11),
        )
    fig.update_layout(barmode="group", yaxis_range=[0, 1.08],
                      yaxis_title="Score", bargap=0.32)
    theme.style_figure(fig, height=360, showlegend=True)

    c.card_open("Macro F1 and High-risk F1", glass=False,
                note="Only the two metrics reported for all four models are charted.")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    c.card_close()
    c.spacer(1.0)

    # --- MLP loss variants -------------------------------------------------
    p4 = reported["phase4"]
    variants = pd.DataFrame([
        {"Experiment": e["name"] + (" — selected" if e["selected"] else ""),
         "Test Macro F1": e["macro_f1"], "High F1": e["high_f1"],
         "High Precision": e["high_precision"], "High Recall": e["high_recall"]}
        for e in p4["experiments"]
    ])
    c.card_open("MLP loss settings", glass=False, note=p4["architecture"])
    st.dataframe(variants.set_index("Experiment"), use_container_width=True)
    c.card_close()
    c.spacer(1.0)

    notes = reported["phase3"]["notes"] + p4["notes"]
    bullets = "".join(
        f'<li style="margin-bottom:.45rem;color:var(--muted);line-height:1.65">{n}</li>'
        for n in notes
    )
    c.panel("Reported notes",
            f'<ul style="margin:0;padding-left:1.1rem;font-size:.86rem">{bullets}</ul>',
            delay=2)

    c.spacer(1.0)
    c.notice("Why the MLP is used for Phase 5", reported["comparison_note"], kind="info")
