"""Audit Log page for TAX-XAI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services import audit


def render() -> None:
    st.title("Audit Log")
    st.caption("Transparent record of observable application actions in this local session and project.")

    audit.capture_observable_events()
    rows = audit.read_log()

    if not rows:
        st.info("No audit events have been recorded yet. Run an assessment or generate a counterfactual.")
        return

    df = pd.DataFrame(rows)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values("timestamp_utc", ascending=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Events", len(df))
    with c2:
        st.metric("Assessments", int((df["event"] == "Risk assessment").sum()))
    with c3:
        st.metric("Counterfactuals", int((df["event"] == "Counterfactual generated").sum()))

    st.divider()

    profile_options = ["All"] + sorted(
        [x for x in df["profile_id"].dropna().astype(str).unique() if x]
    )
    selected = st.selectbox("Filter by taxpayer", profile_options)
    if selected != "All":
        df = df[df["profile_id"].astype(str) == selected]

    event_options = ["All"] + sorted(df["event"].dropna().astype(str).unique().tolist())
    selected_event = st.selectbox("Filter by event", event_options)
    if selected_event != "All":
        df = df[df["event"].astype(str) == selected_event]

    st.dataframe(
        df[
            [
                "timestamp_utc",
                "event",
                "profile_id",
                "source",
                "model",
                "prediction",
                "details",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export audit log CSV",
        data=csv,
        file_name="tax_xai_audit_log.csv",
        mime="text/csv",
    )

    st.info(
        "Audit scope: local application events that the app can observe. "
        "This is not an identity/access-control audit trail and does not prove "
        "who performed an action. Sensitive credentials and secrets are not logged."
    )
