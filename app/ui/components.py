"""Presentation helpers.

Every function writes markup through st.markdown with unsafe_allow_html. None of
them compute anything — values arrive already resolved so that a component can
never quietly invent a number.
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from . import theme

_STATE_CLASS = {"ok": "s-ok", "warn": "s-warn", "fail": "s-fail", "pending": "s-pending"}


def load_css(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Stylesheet not found at {path}. The app runs unstyled.")


def page_header(title: str, lede: str = "") -> None:
    body = f'<p class="page-lede">{html.escape(lede)}</p>' if lede else ""
    st.markdown(
        f'<div class="enter"><h1 class="page-title">{html.escape(title)}</h1>{body}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)


def spacer(rem: float = 1.0) -> None:
    st.markdown(f"<div style='height:{rem}rem'></div>", unsafe_allow_html=True)


def kpi(label: str, value: str, foot: str = "", unit: str = "",
        accent: str | None = None, delay: int = 1) -> None:
    """A single KPI tile. `value` is rendered as given — no reformatting."""
    accent_cls = {"indigo": " kpi-accent", "cyan": " kpi-accent-cyan"}.get(accent or "", "")
    unit_html = f'<span class="unit">{html.escape(unit)}</span>' if unit else ""
    foot_html = f'<div class="kpi-foot">{html.escape(foot)}</div>' if foot else ""
    st.markdown(
        f"""<div class="glass interactive kpi enter d{delay}">
              <div class="kpi-label">{html.escape(label)}</div>
              <div class="kpi-value{accent_cls}">{html.escape(value)}{unit_html}</div>
              {foot_html}
            </div>""",
        unsafe_allow_html=True,
    )


def card_open(title: str = "", note: str = "", glass: bool = True,
              interactive: bool = False, delay: int | None = None) -> None:
    """Open a surface. Must be paired with card_close()."""
    cls = "glass" if glass else "solid"
    if interactive:
        cls += " interactive"
    if delay:
        cls += f" enter d{delay}"
    head = ""
    if title:
        note_html = f'<p class="section-note">{html.escape(note)}</p>' if note else ""
        head = (f'<div style="margin-bottom:.9rem">'
                f'<p class="section-title">{html.escape(title)}</p>{note_html}</div>')
    st.markdown(f'<div class="{cls}">{head}', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def panel(title: str, body_html: str, note: str = "", glass: bool = True,
          interactive: bool = False, delay: int | None = None) -> None:
    """A complete surface in one call, for markup built elsewhere."""
    cls = "glass" if glass else "solid"
    if interactive:
        cls += " interactive"
    if delay:
        cls += f" enter d{delay}"
    note_html = f'<p class="section-note">{html.escape(note)}</p>' if note else ""
    st.markdown(
        f"""<div class="{cls}">
              <div style="margin-bottom:.9rem">
                <p class="section-title">{html.escape(title)}</p>{note_html}
              </div>
              {body_html}
            </div>""",
        unsafe_allow_html=True,
    )


def risk_rows_html(rows: list[dict], denominator: int, denominator_label: str) -> str:
    """rows: [{"risk": "Low", "count": 4385, "share": 0.8866}, ...]"""
    out = []
    for r in rows:
        cls = theme.RISK_CLASS.get(r["risk"], "low")
        share = r["share"] * 100
        out.append(
            f"""<div class="risk-row">
                  <div style="flex:1">
                    <div class="risk-name">
                      <span class="risk-dot dot-{cls}"></span>{html.escape(r['risk'])} risk
                    </div>
                    <div class="bar-track">
                      <div class="bar-fill" style="--w:{share:.2f}%;
                           background:{theme.RISK_COLORS[r['risk']]}"></div>
                    </div>
                  </div>
                  <div class="risk-figs">
                    <div class="risk-share figure">{share:.2f}%</div>
                    <div class="risk-count figure">{r['count']:,} of {denominator:,}</div>
                  </div>
                </div>"""
        )
    out.append(f'<p class="denominator" style="margin-top:.7rem">'
               f'Denominator: {denominator:,} {html.escape(denominator_label)}</p>')
    return "".join(out)


def status_lines_html(checks) -> str:
    out = []
    for c in checks:
        dot = _STATE_CLASS.get(c.state, "s-pending")
        out.append(
            f"""<div class="status-line">
                  <span class="s-dot {dot}"></span>
                  <span class="status-name">{html.escape(c.name)}</span>
                  <span class="status-detail">{html.escape(c.detail)}</span>
                </div>"""
        )
    return "".join(out)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def notice(title: str, body: str, kind: str = "info") -> None:
    """kind: info | caution | stop"""
    st.markdown(
        f"""<div class="notice notice-{kind}">
              <div class="notice-title">{html.escape(title)}</div>
              <div class="notice-body">{html.escape(body)}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def stat_grid_html(items: list[tuple[str, str]], columns: int = 4) -> str:
    """Compact label/value grid for metric strips."""
    cells = "".join(
        f"""<div>
              <div class="kpi-label" style="font-size:.76rem">{html.escape(k)}</div>
              <div class="figure" style="font-size:1.12rem;font-weight:650;
                   letter-spacing:-.02em;margin-top:.15rem">{html.escape(v)}</div>
            </div>"""
        for k, v in items
    )
    return (f'<div style="display:grid;grid-template-columns:repeat({columns},1fr);'
            f'gap:1rem 1.4rem">{cells}</div>')


def probability_rows_html(probs: dict, predicted: str | None = None) -> str:
    """probs: {"Low": 0.12, "Medium": 0.80, "High": 0.08}"""
    out = []
    for risk in ("Low", "Medium", "High"):
        cls = theme.RISK_CLASS[risk]
        pct = probs.get(risk, 0.0) * 100
        marker = (f'<span class="badge badge-{cls}" style="margin-left:.5rem">'
                  f'predicted</span>') if risk == predicted else ""
        out.append(
            f"""<div class="risk-row">
                  <div style="flex:1">
                    <div class="risk-name">
                      <span class="risk-dot dot-{cls}"></span>{html.escape(risk)}{marker}
                    </div>
                    <div class="bar-track">
                      <div class="bar-fill" style="--w:{pct:.2f}%;
                           background:{theme.RISK_COLORS[risk]}"></div>
                    </div>
                  </div>
                  <div class="risk-figs"><div class="risk-share figure">{pct:.2f}%</div></div>
                </div>"""
        )
    return "".join(out)


def source_tag(live: bool) -> str:
    """Small inline tag distinguishing a live model call from a precomputed result."""
    if live:
        return ('<span class="badge badge-neutral" style="border-color:rgba(99,102,241,.4);'
                'color:var(--primary-soft)">live model output</span>')
    return '<span class="badge badge-neutral">precomputed phase 5 result</span>'


def placeholder(message: str, phase: str) -> None:
    """Honest empty state for a section that is not wired up yet."""
    st.markdown(
        f"""<div class="solid" style="border-style:dashed;text-align:center;padding:2.2rem 1.2rem">
              <p class="section-title" style="color:var(--muted)">{html.escape(message)}</p>
              <p class="section-note">Connects in {html.escape(phase)}.</p>
            </div>""",
        unsafe_allow_html=True,
    )
