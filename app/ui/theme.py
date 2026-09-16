"""Palette and chart theming.

The hex values mirror styles/main.css. Keep them in step: the CSS owns the page
chrome, this module owns anything drawn by Plotly.
"""
from __future__ import annotations

BG = "#0B1120"
SURFACE = "#111827"
SURFACE_2 = "#0F1729"
BORDER = "#1F2937"
PRIMARY = "#6366F1"
PRIMARY_SOFT = "#818CF8"
SECONDARY = "#06B6D4"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
FAINT = "#64748B"

LOW = "#10B981"
MEDIUM = "#F59E0B"
HIGH = "#EF4444"

RISK_COLORS = {"Low": LOW, "Medium": MEDIUM, "High": HIGH}
RISK_CLASS = {"Low": "low", "Medium": "medium", "High": "high"}

SERIES = [PRIMARY, SECONDARY, PRIMARY_SOFT, "#A78BFA", "#22D3EE", "#7DD3FC"]

FONT = "Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif"


def style_figure(fig, height: int = 320, showlegend: bool = False):
    """Apply the dark theme to a Plotly figure, in place."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=MUTED),
        margin=dict(l=8, r=8, t=28, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER,
                        font=dict(family=FONT, color=TEXT, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor="rgba(31,41,55,.8)", zerolinecolor="rgba(31,41,55,.9)",
                     linecolor=BORDER, tickfont=dict(color=FAINT))
    fig.update_yaxes(gridcolor="rgba(31,41,55,.8)", zerolinecolor="rgba(31,41,55,.9)",
                     linecolor=BORDER, tickfont=dict(color=FAINT))
    return fig
