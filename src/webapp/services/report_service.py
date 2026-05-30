from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _ensure_reports_dir(reports_dir: str) -> Path:
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def generate_pdf_report(result: Dict, reports_dir: str) -> str:
    """Generate a local PDF report from analysis results."""
    output_dir = _ensure_reports_dir(reports_dir)
    report_name = f"swing_report_{result['analysis_id']}.pdf"
    report_path = output_dir / report_name

    metrics_df = pd.DataFrame([result.get("metrics", {})])

    pdf = canvas.Canvas(str(report_path), pagesize=letter)
    width, height = letter
    y = height - 50

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "SwingSight AI Swing Report")
    y -= 25

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Analysis ID: {result.get('analysis_id')}")
    y -= 15
    pdf.drawString(50, y, f"Club Category: {result.get('final_club_category')}")
    y -= 20

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Metrics")
    y -= 15
    pdf.setFont("Helvetica", 10)

    for column, value in metrics_df.iloc[0].to_dict().items():
        pdf.drawString(50, y, f"- {column}: {value}")
        y -= 14
        if y < 70:
            pdf.showPage()
            y = height - 50

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Feedback")
    y -= 15
    pdf.setFont("Helvetica", 10)
    for item in result.get("feedback", []):
        pdf.drawString(50, y, f"- {item}")
        y -= 14
        if y < 70:
            pdf.showPage()
            y = height - 50

    pdf.save()
    return str(report_path)


def generate_word_report(result: Dict, reports_dir: str) -> str:
    """Generate a local Word report from analysis results."""
    output_dir = _ensure_reports_dir(reports_dir)
    report_name = f"swing_report_{result['analysis_id']}.docx"
    report_path = output_dir / report_name

    document = Document()
    document.add_heading("SwingSight AI Swing Report", level=1)
    document.add_paragraph(f"Analysis ID: {result.get('analysis_id')}")
    document.add_paragraph(f"Club Category: {result.get('final_club_category')}")

    document.add_heading("Metrics", level=2)
    metrics_df = pd.DataFrame([result.get("metrics", {})])
    for column, value in metrics_df.iloc[0].to_dict().items():
        document.add_paragraph(f"{column}: {value}")

    document.add_heading("Feedback", level=2)
    for item in result.get("feedback", []):
        document.add_paragraph(item, style="List Bullet")

    document.save(str(report_path))
    return str(report_path)
