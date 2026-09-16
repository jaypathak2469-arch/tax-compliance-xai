"""TAX-XAI application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services import artifacts, paths  # noqa: E402
from services import audit

from ui import components as c  # noqa: E402
from views import (
    audit_log,  # noqa: E402
    about,
    analytics,
    batch_assessment,
    counterfactual,
    dashboard,
    data_quality,
    model_comparison,
    profiles,
    reports,
    risk_assessment,
    what_if,
)

st.set_page_config(
    page_title="TAX-XAI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": dashboard.render,
    "Assess": risk_assessment.render,
    "Batch Assessment": batch_assessment.render,
    "Data Quality": data_quality.render,
    "What-If Simulator": what_if.render,
    "Recourse": counterfactual.render,
    "Profiles": profiles.render,
    "Models": model_comparison.render,
    "Analytics": analytics.render,
    "Reports": reports.render,
    "Audit Log": audit_log.render,
    "Methodology": about.render,
}

BUILD_PHASE = "Phase F — live pipeline"
NAV_KEY = "nav_page"
NAV_REQUEST_KEY = "nav_page_request"


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """<div class="brand">
                 <div class="brand-mark"></div>
                 <div>
                   <div class="brand-name">TAX-XAI</div>
                   <div class="brand-sub">Constrained counterfactual explanation</div>
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-heading">Navigate</div>', unsafe_allow_html=True)
        st.session_state.setdefault(NAV_KEY, "Overview")
        requested = st.session_state.pop(NAV_REQUEST_KEY, None)
        if requested in PAGES:
            st.session_state[NAV_KEY] = requested
        st.radio("Navigate", list(PAGES), key=NAV_KEY, label_visibility="collapsed")

        failures = artifacts.blocking_failures()
        if failures:
            state, message = "s-fail", f"{len(failures)} artifact or dependency issue" + (
                "s" if len(failures) > 1 else ""
            )
        else:
            state, message = "s-ok", "All artifacts located"

        st.markdown(
            f"""<div class="sidebar-foot">
                  <div style="display:flex;align-items:center;gap:.55rem;
                       color:var(--muted);margin-bottom:.45rem">
                    <span class="s-dot {state}"></span>{message}
                  </div>
                  <div>{BUILD_PHASE}</div>
                  <div style="margin-top:.35rem;word-break:break-all">
                    {paths.PROJECT_ROOT.name}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
    return st.session_state[NAV_KEY]


def main() -> None:
    c.load_css(paths.STYLESHEET)
    st.session_state.setdefault("history", [])
    page = sidebar()
    failures = artifacts.blocking_failures()
    if failures:
        c.notice(
            "Some artifacts could not be found",
            " · ".join(f"{f.name}: {f.detail}" for f in failures),
            kind="stop",
        )
        c.spacer(1.0)
    try:
        PAGES[page]()
    except Exception as exc:
        st.error(f"The {page} page failed to render.")
        st.exception(exc)


if __name__ == "__main__":
    main()
