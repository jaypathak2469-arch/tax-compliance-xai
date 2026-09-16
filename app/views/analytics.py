"""Analytics — charts built from the existing Phase 5 result files.

Everything here is read from results/metrics/. Nothing is recomputed by a model.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from services import datasets
from ui import components as c
from ui import theme


def _risk_tab() -> None:
    for scope, caption in (("modelled", "modelled taxpayer profiles"),
                           ("test", "held-out test records")):
        dist = datasets.risk_distribution(scope)
        if dist.empty:
            continue
        denom = int(dist["denominator"].iloc[0])
        c.panel(
            f"{'Modelling population' if scope == 'modelled' else 'Held-out test split'}",
            c.risk_rows_html(dist.to_dict("records"), denom, caption),
            glass=False, delay=2,
        )
        c.spacer(0.9)


def _performance_tab() -> None:
    reported = datasets.reported_metrics()
    if not reported:
        c.placeholder("Reported metrics unavailable", "Phase D")
        return
    c.notice(reported["label"],
             "See the Model comparison page for the full breakdown and provenance.",
             kind="info")
    c.spacer(0.9)
    models = reported["phase3"]["models"]
    fig = go.Figure()
    fig.add_bar(x=[m["name"] for m in models], y=[m["macro_f1"] for m in models],
                marker_color=theme.PRIMARY, marker_line_width=0, name="Macro F1")
    selected = next(e for e in reported["phase4"]["experiments"] if e["selected"])
    fig.add_bar(x=["Weighted MLP"], y=[selected["macro_f1"]],
                marker_color=theme.SECONDARY, marker_line_width=0, name="Macro F1")
    fig.update_layout(yaxis_range=[0, 1.05], yaxis_title="Macro F1", bargap=0.35)
    theme.style_figure(fig, height=340)
    c.card_open("Macro F1 across evaluated models", glass=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    c.card_close()


def _success_tab() -> None:
    summary = datasets.counterfactual_summary()
    if summary.empty:
        c.placeholder("counterfactual_summary.csv not found", "Phase D")
        return
    sub = summary[summary["mode"] != "overall"]
    fig = go.Figure()
    for i, mode in enumerate(["constrained", "unconstrained"]):
        m = sub[sub["mode"] == mode]
        fig.add_bar(name=f"{mode} — constraint valid", x=m["source_risk"], y=m["valid_rate"],
                    marker_color=theme.SERIES[i], marker_line_width=0)
    fig.update_layout(barmode="group", yaxis_range=[0, 1.08],
                      yaxis_title="Constraint validity rate", bargap=0.35)
    theme.style_figure(fig, height=340, showlegend=True)

    c.card_open("Target success and constraint validity", glass=False,
                note="Both modes reached the Low class in 100% of cases. They differ "
                     "entirely on feasibility: unconstrained results satisfy no "
                     "constraint check.")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(summary.set_index(["mode", "source_risk"]), use_container_width=True)
    c.card_close()


def _distance_tab() -> None:
    results = datasets.counterfactual_results()
    if results.empty:
        c.placeholder("counterfactual_results.csv not found", "Phase D")
        return
    c.notice("Optimisation-space measures",
             "L0, L1 and L2 are distances in the standardised feature space. They are "
             "not monetary amounts and not causal effect sizes.", kind="caution")
    c.spacer(0.9)
    for metric in ["l0", "l1", "l2"]:
        fig = go.Figure()
        for i, mode in enumerate(["constrained", "unconstrained"]):
            fig.add_histogram(x=results[results["mode"] == mode][metric], name=mode,
                              marker_color=theme.SERIES[i], opacity=0.62, nbinsx=14)
        fig.update_layout(barmode="overlay", xaxis_title=metric.upper(),
                          yaxis_title="Cases")
        theme.style_figure(fig, height=280, showlegend=True)
        c.card_open(f"{metric.upper()} distribution", glass=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        c.card_close()
        c.spacer(0.8)


def _validation_tab() -> None:
    neg = datasets.negative_tests()
    if neg.empty:
        c.placeholder("negative_validation_tests.csv not found", "Phase D")
        return
    grouped = neg.groupby("test")["rejected"].agg(["sum", "count"]).reset_index()
    rows = "".join(
        f"""<div class="status-line">
              <span class="s-dot {'s-ok' if r['sum'] == r['count'] else 's-fail'}"></span>
              <span class="status-name">{r['test'].replace('_', ' ')}</span>
              <span class="status-detail">{int(r['sum'])} of {int(r['count'])} rejected</span>
            </div>"""
        for _, r in grouped.iterrows()
    )
    c.panel("Negative validation tests", rows,
            note="Deliberately corrupted counterfactuals that the validator is "
                 "expected to reject.", glass=False, delay=2)


def _changed_tab() -> None:
    freq = datasets.changed_feature_frequency("constrained")
    if freq.empty:
        c.placeholder("Changed-feature data unavailable", "Phase D")
        return
    total = int(freq["total_cases"].iloc[0])
    c.notice("Not feature importance",
             "This counts how often the optimiser moved each feature across the "
             f"{total} constrained cases. It says nothing about how much a feature "
             "contributes to the model's prediction.", kind="caution")
    c.spacer(0.9)
    top = freq.head(18).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["cases"], y=[datasets.label(f) for f in top["feature"]],
        orientation="h", marker_color=theme.PRIMARY, marker_line_width=0,
    ))
    fig.update_layout(xaxis_title=f"Cases changed (of {total})")
    theme.style_figure(fig, height=520)
    c.card_open("Changed-feature frequency", glass=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    c.card_close()


def render() -> None:
    c.page_header(
        "Analytics",
        "Aggregate views over the completed Phase 5 experiment and the reported "
        "model results.",
    )
    tabs = st.tabs([
        "Risk distribution", "Model performance", "Counterfactual success",
        "Distance measures", "Constraint validation", "Changed features",
    ])
    for tab, fn in zip(tabs, [_risk_tab, _performance_tab, _success_tab,
                              _distance_tab, _validation_tab, _changed_tab]):
        with tab:
            c.spacer(0.6)
            fn()
