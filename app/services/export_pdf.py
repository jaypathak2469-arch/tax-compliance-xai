"""Per-assessment PDF report.

Renders one structured record from ``services/records.py`` as a single PDF
using reportlab (already listed in requirements-app.txt). As with the CSV
export, nothing is computed here — every figure comes straight off the
record, itself a JSON-safe copy of a live ``predictor.predict()`` /
``explainer.generate()`` result. The report visually echoes the app's dark
FinTech palette (header band, risk colours) but keeps the body light for
print legibility.
"""
from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from . import datasets, validation

# Palette lifted from ui/theme.py, kept in step with the on-screen app.
INK = colors.HexColor("#0B1120")
PRIMARY = colors.HexColor("#6366F1")
MUTED = colors.HexColor("#475569")
FAINT = colors.HexColor("#94A3B8")
LOW = colors.HexColor("#059669")
MEDIUM = colors.HexColor("#B45309")
HIGH = colors.HexColor("#DC2626")
BORDER = colors.HexColor("#E2E8F0")
PANEL = colors.HexColor("#F8FAFC")

RISK_COLOR = {"Low": LOW, "Medium": MEDIUM, "High": HIGH}

# Same currency-vs-plain-numeric split the Assess and Profiles pages use, so a
# value reads the same way here as it does on screen (no scientific notation).
_CURRENCY_FEATURES = {
    "annual_income", "outstanding_dues", "total_transaction_value",
    "average_transaction_value", "max_transaction_amount", "std_transaction_amount",
}


def _fmt_value(feature: str, value) -> str:
    if isinstance(value, str):
        return value
    v = float(value)
    if feature in _CURRENCY_FEATURES:
        return f"{v:,.2f}"
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:,.4f}"

LIMITATION = (
    "These figures are the output of a trained model and a constrained optimisation "
    "search over that model's decision boundary. They are not regulatory, financial, "
    "or tax advice, and do not represent a guarantee of any real-world outcome."
)

_styles = getSampleStyleSheet()
_TITLE = ParagraphStyle("TXTitle", parent=_styles["Title"], textColor=colors.white,
                         fontSize=18, leading=22, spaceAfter=0)
_SUBTITLE = ParagraphStyle("TXSubtitle", parent=_styles["Normal"], textColor=colors.HexColor("#C7D2FE"),
                            fontSize=9.5, leading=13)
_H2 = ParagraphStyle("TXH2", parent=_styles["Heading2"], textColor=INK,
                      fontSize=12.5, spaceBefore=14, spaceAfter=6)
_BODY = ParagraphStyle("TXBody", parent=_styles["Normal"], textColor=INK, fontSize=9.5, leading=13)
_NOTE = ParagraphStyle("TXNote", parent=_styles["Normal"], textColor=MUTED, fontSize=8.3, leading=11.5)
_TAG = ParagraphStyle("TXTag", parent=_styles["Normal"], textColor=colors.HexColor("#4338CA"),
                       fontSize=8, leading=10)


def _header_table(record: dict) -> Table:
    cf = record.get("cf")
    cls = record["pred_class"]
    tag = "live model output"
    right = Paragraph(
        f'<font color="#C7D2FE">{tag}</font><br/>'
        f'<font color="white" size="13"><b>{cls} risk</b></font>',
        _SUBTITLE,
    )
    left = [
        Paragraph("TAX-XAI — Assessment report", _TITLE),
        Spacer(1, 3),
        Paragraph(
            f"Record {record['record_id']} &nbsp;·&nbsp; Profile {record['profile_id']} "
            f"&nbsp;·&nbsp; {record['created_at'].replace('T', ' ')}",
            _SUBTITLE,
        ),
    ]
    tbl = Table([[left, right]], colWidths=[120 * mm, 50 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (1, 0), (1, 0), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    return tbl


def _prob_table(probs: list[float], predicted: str | None = None) -> Table:
    rows = [["Class", "Probability", ""]]
    data_rows = []
    for cls, p in zip(("Low", "Medium", "High"), probs):
        marker = "  ← predicted" if cls == predicted else ""
        data_rows.append([cls, f"{p * 100:.2f}%", marker])
    rows += data_rows
    tbl = Table(rows, colWidths=[35 * mm, 30 * mm, 40 * mm])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, cls in enumerate(("Low", "Medium", "High"), start=1):
        style.append(("TEXTCOLOR", (0, i), (0, i), RISK_COLOR[cls]))
        style.append(("FONTNAME", (0, i), (1, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    return tbl


def _kv_table(items: list[tuple[str, str]], col_widths=(58 * mm, 45 * mm)) -> Table:
    tbl = Table(items, colWidths=list(col_widths))
    tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _input_table(raw_input: dict) -> Table:
    rows = [["Feature", "Value"]]
    for feature, value in raw_input.items():
        rows.append([datasets.label(feature), _fmt_value(feature, value)])
    tbl = Table(rows, colWidths=[95 * mm, 65 * mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _changes_table(cf: dict) -> Table:
    rows = [["Feature", "Original", "Counterfactual", "Change"]]
    for feature in cf["changed_features"]:
        before, after = cf["raw_before"][feature], cf["raw_after"][feature]
        change = after - before
        sign = "+" if change >= 0 else ""
        rows.append([
            datasets.label(feature), _fmt_value(feature, before), _fmt_value(feature, after),
            f"{sign}{_fmt_value(feature, change)}",
        ])
    tbl = Table(rows, colWidths=[70 * mm, 30 * mm, 33 * mm, 27 * mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return tbl


def _validation_table(cf: dict) -> Table:
    checks = validation.constraint_report(cf["feasible"], cf["issues"], cf["finite"])
    overall = validation.overall_valid(checks)
    rows = [["Check", "State", "Detail"]]
    for c in checks:
        rows.append([c.name, c.state.upper(), c.detail])
    tbl = Table(rows, colWidths=[45 * mm, 22 * mm, 93 * mm], repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for i, c in enumerate(checks, start=1):
        style.append(("TEXTCOLOR", (1, i), (1, i), LOW if c.state == "ok" else HIGH))
        style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    overall_p = Paragraph(
        f'Overall: <font color="{"#059669" if overall else "#DC2626"}"><b>'
        f'{"VALID" if overall else "INVALID"}</b></font>', _BODY,
    )
    return KeepTogether([tbl, Spacer(1, 4), overall_p])


def build_pdf(record: dict) -> bytes:
    """Render one structured assessment record to PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=0, bottomMargin=16 * mm, leftMargin=14 * mm, rightMargin=14 * mm,
        title=f"TAX-XAI assessment — {record['profile_id']}",
    )

    story = [_header_table(record), Spacer(1, 16)]

    meta_items = [("Source", record["source"]), ("Assessed at", record["created_at"].replace("T", " "))]
    if record.get("dataset_label"):
        meta_items.append(("Dataset label (overall_risk)", record["dataset_label"]))
    if record.get("split"):
        meta_items.append(("Dataset split", record["split"]))
    story.append(_kv_table(meta_items))

    story.append(Paragraph("Prediction", _H2))
    story.append(_prob_table(record["probs"], predicted=record["pred_class"]))

    story.append(Paragraph("Input values", _H2))
    story.append(_input_table(record["raw_input"]))

    cf = record.get("cf")
    if cf:
        story.append(Paragraph("Counterfactual recourse", _H2))
        story.append(_kv_table([
            ("Target class", cf["target_class"]),
            ("Result class", cf["predicted_class"]),
            ("Status", cf["status"]),
            ("Reached target", "Yes" if cf["success"] else "No"),
            ("Constrained search", "Yes" if cf["constrained"] else "No"),
        ]))
        story.append(Spacer(1, 6))
        story.append(_kv_table([
            ("L0 (features changed)", str(cf["l0"])),
            ("L1 distance", f"{cf['l1']:.4f}"),
            ("L2 distance", f"{cf['l2']:.4f}"),
            ("Optimisation steps", str(cf["optimization_steps"])),
            ("Runtime", f"{cf['runtime_seconds'] * 1000:.2f} ms"),
            ("Final objective", f"{cf['final_objective']:.4f}"),
        ]))

        story.append(Paragraph("Before &amp; after probability", _H2))
        before_after = Table(
            [["", "Low", "Medium", "High"],
             ["Before"] + [f"{p * 100:.2f}%" for p in cf["p0"]],
             ["After"] + [f"{p * 100:.2f}%" for p in cf["probs"]]],
            colWidths=[25 * mm, 35 * mm, 35 * mm, 35 * mm],
        )
        before_after.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(before_after)

        if cf["changed_features"]:
            story.append(Paragraph(
                f"Feature changes &nbsp;<font color=\"#94A3B8\" size=8>"
                f"{len(cf['changed_features'])} of {len(cf['raw_before'])} numeric features moved"
                f"</font>", _H2,
            ))
            story.append(_changes_table(cf))
        else:
            story.append(Paragraph("Feature changes", _H2))
            story.append(Paragraph("The optimiser did not move any feature.", _NOTE))

        story.append(Paragraph("Constraint validation", _H2))
        story.append(_validation_table(cf))
    else:
        story.append(Paragraph("Counterfactual recourse", _H2))
        story.append(Paragraph(
            "No counterfactual has been generated for this record. This profile may "
            "already be predicted Low risk, or recourse was not run in this session.",
            _NOTE,
        ))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=BORDER, thickness=0.6))
    story.append(Spacer(1, 6))
    story.append(Paragraph(LIMITATION, _NOTE))

    def _draw_header_bleed(canvas, _doc):
        # The header table above is placed with topMargin=0, but the very
        # first page still needs the dark band to reach the physical page
        # edge rather than stop at the frame's top inset.
        canvas.saveState()
        canvas.setFillColor(INK)
        canvas.rect(0, A4[1] - 0.1 * mm, A4[0], 0.1 * mm, fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_header_bleed)
    return buf.getvalue()
