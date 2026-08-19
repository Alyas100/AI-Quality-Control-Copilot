"""
Report Builder
================================================================================
Renders the Copilot's structured action plan (plus the underlying batch
payload) into a downloadable PDF report -- one of the Copilot's "actionable
commands" alongside the SMS draft and schedule-change suggestion.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

_RISK_HEX = {
    "SAFE": "#16a34a",
    "WARNING": "#b45309",
    "CRITICAL RISK": "#dc2626",
}
_RISK_COLORS = {k: colors.HexColor(v) for k, v in _RISK_HEX.items()}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="QCTitle", parent=styles["Title"], fontSize=20, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="QCSubtitle", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#71717a"), spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="QCSection", parent=styles["Heading2"], fontSize=12,
        spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#18181b"),
    ))
    styles.add(ParagraphStyle(
        name="QCBody", parent=styles["Normal"], fontSize=10, leading=14,
    ))
    return styles


def generate_pdf_report(payload: dict, action_plan: dict, used_live: bool) -> bytes:
    """Build the PDF report and return it as raw bytes, ready for a
    Streamlit download_button."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = _styles()
    story = []

    classification = action_plan.get("risk_classification", "WARNING")
    risk_color = _RISK_COLORS.get(classification, colors.HexColor("#b45309"))
    risk_hex = _RISK_HEX.get(classification, "#b45309")

    # ---- header
    story.append(Paragraph("FFA Copilot — Batch Action Plan", styles["QCTitle"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; "
        f"{'Live AI reasoning' if used_live else 'Offline rule-based estimate'} &middot; Mill Line 3",
        styles["QCSubtitle"],
    ))

    story.append(Table(
        [[Paragraph(f'<font color="{risk_hex}"><b>{classification}</b></font>', styles["QCBody"])]],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
            ("BOX", (0, 0), (-1, -1), 0.75, risk_color),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]),
        colWidths=[6.5 * inch],
    ))
    story.append(Spacer(1, 14))

    # ---- narrative
    story.append(Paragraph("Assessment", styles["QCSection"]))
    story.append(Paragraph(action_plan.get("narrative", "").replace("\n", "<br/>"), styles["QCBody"]))

    # ---- batch data table
    story.append(Paragraph("Batch Data", styles["QCSection"]))
    vision = payload["vision_analysis"]
    env = payload["environmental_conditions"]
    pred = payload["predictive_analysis"]
    data_rows = [
        ["Metric", "Value"],
        ["Ripeness category", vision["ripeness_category"]],
        ["Vision confidence", f"{vision['confidence'] * 100:.0f}%"],
        ["Harvest delay", f"{env['harvest_delay_hours']:.1f} hrs"],
        ["Storage temperature", f"{env['storage_temp_c']:.1f} °C"],
        ["Ambient humidity", f"{env['humidity_percent']:.1f}%"],
        ["Predicted FFA", f"{pred['predicted_ffa_percentage']:.2f}%"],
        ["Risk level", pred["risk_level"]],
    ]
    data_table = Table(data_rows, colWidths=[2.5 * inch, 4 * inch])
    data_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e4e7")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(data_table)

    # ---- SMS alert
    sms = action_plan.get("sms_alert", {})
    story.append(Paragraph("Dispatched Alert", styles["QCSection"]))
    if sms.get("send"):
        story.append(Paragraph(
            f"<b>To:</b> {sms.get('recipient', '')} &nbsp;&nbsp; "
            f"<b>Urgency:</b> {sms.get('urgency', '').upper()}",
            styles["QCBody"],
        ))
        story.append(Spacer(1, 4))
        story.append(Table(
            [[Paragraph(sms.get("message", ""), styles["QCBody"])]],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f5")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e4e7")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]),
            colWidths=[6.5 * inch],
        ))
    else:
        story.append(Paragraph("No SMS alert required for this batch.", styles["QCBody"]))

    # ---- schedule change
    sched = action_plan.get("schedule_change", {})
    story.append(Paragraph("Schedule Recommendation", styles["QCSection"]))
    if sched.get("recommended"):
        delay = sched.get("suggested_max_harvest_delay_hours")
        delay_text = f" Target ceiling: {delay:.0f} hrs for incoming trucks." if delay is not None else ""
        story.append(Paragraph(f"<b>{sched.get('action', '')}.</b>{delay_text}", styles["QCBody"]))
        story.append(Paragraph(sched.get("rationale", ""), styles["QCBody"]))
    else:
        story.append(Paragraph(sched.get("action", "No change needed") + ".", styles["QCBody"]))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e4e4e7")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated automatically by FFA Copilot. Verify against on-site lab results before acting on critical alerts.",
        styles["QCSubtitle"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()