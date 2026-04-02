# reporting.py - PDF report generation
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Preformatted, Image
)
from reportlab.lib.enums import TA_CENTER

logger = logging.getLogger(__name__)


def _create_styles():
    """Create reusable report styles."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=22, spaceAfter=20, alignment=TA_CENTER,
        textColor=colors.HexColor('#1a1a2e'))
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, spaceAfter=10, spaceBefore=15,
        textColor=colors.HexColor('#16213e'))
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'],
        fontSize=10, spaceAfter=5)
    code_style = ParagraphStyle(
        'Code', parent=styles['Code'],
        fontSize=8, fontName='Courier',
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10, rightIndent=10)
    return title_style, heading_style, normal_style, code_style


def _make_table(data, col_widths, header_color='#2196F3'):
    """Create styled table."""
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    return table


def generate_classification_report(results: Dict, config: Dict, output_path: str,
                                   plot_paths: Dict = None):
    """Generate PDF report for classification."""
    title_style, heading_style, normal_style, code_style = _create_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    elements = []
    
    # Title
    elements.append(Paragraph("AutoML Classification Report", title_style))
    elements.append(Spacer(1, 10))
    
    # Run Info
    elements.append(Paragraph(f"<b>Run ID:</b> {results.get('run_id', 'N/A')}", normal_style))
    elements.append(Paragraph(f"<b>Best Model:</b> {results.get('best_model_name', 'N/A')}", normal_style))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    
    if config.get('use_ensemble'):
        elements.append(Paragraph("<b>Ensemble:</b> Stacking Enabled", normal_style))
    if config.get('use_smote'):
        elements.append(Paragraph("<b>Class Imbalance:</b> SMOTE Applied", normal_style))
    if config.get('use_cv'):
        elements.append(Paragraph(f"<b>Cross-Validation:</b> {config.get('cv_folds', 5)}-Fold", normal_style))
    
    elements.append(Spacer(1, 15))
    
    # Leaderboard
    elements.append(Paragraph("Model Leaderboard", heading_style))
    metrics = results.get("models_metrics", [])
    table_data = [["Model", "Accuracy", "F1", "Precision", "Recall"]]
    for m in metrics:
        table_data.append([
            m['model_name'],
            f"{m['accuracy']:.4f}",
            f"{m['f1']:.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}"
        ])
    elements.append(_make_table(table_data, [2*inch, 1*inch, 1*inch, 1*inch, 1*inch], '#4CAF50'))
    elements.append(Spacer(1, 15))
    
    # CV Scores if available
    if results.get("cv_summary"):
        elements.append(Paragraph("Cross-Validation Summary", heading_style))
        cv_data = [["Model", "Mean Score", "Std Dev"]]
        for name, stats in results["cv_summary"].items():
            cv_data.append([name, f"{stats['mean']:.4f}", f"{stats['std']:.4f}"])
        elements.append(_make_table(cv_data, [2.5*inch, 1.5*inch, 1.5*inch]))
        elements.append(Spacer(1, 15))
    
    # Classification Report Text
    if results.get("classification_report_text"):
        elements.append(Paragraph("Classification Report", heading_style))
        elements.append(Preformatted(results["classification_report_text"], code_style))
    
    # Config summary
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Configuration", heading_style))
    for k, v in config.items():
        elements.append(Paragraph(f"<b>{k}:</b> {v}", normal_style))
    
    doc.build(elements)
    logger.info(f"Classification report saved: {output_path}")


def generate_regression_report(results: Dict, config: Dict, output_path: str,
                                plot_paths: Dict = None):
    """Generate PDF report for regression."""
    title_style, heading_style, normal_style, code_style = _create_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    elements = []
    
    elements.append(Paragraph("AutoML Regression Report", title_style))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph(f"<b>Run ID:</b> {results.get('run_id', 'N/A')}", normal_style))
    elements.append(Paragraph(f"<b>Best Model:</b> {results.get('best_model_name', 'N/A')}", normal_style))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("Model Leaderboard", heading_style))
    metrics = results.get("models_metrics", [])
    table_data = [["Model", "R²", "MAE", "RMSE"]]
    for m in metrics:
        table_data.append([m['model_name'], f"{m['r2']:.4f}", f"{m['mae']:.4f}", f"{m['rmse']:.4f}"])
    elements.append(_make_table(table_data, [2.5*inch, 1.2*inch, 1.2*inch, 1.2*inch]))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Configuration", heading_style))
    for k, v in config.items():
        elements.append(Paragraph(f"<b>{k}:</b> {v}", normal_style))
    
    doc.build(elements)
    logger.info(f"Regression report saved: {output_path}")

# [2026-02-21T09:00:00] Create reporting module for PDF/HTML experiment reports

# [2026-03-27T15:45:00] Add experiment comparison and leaderboard view

# [2026-02-21T09:00:00] Create reporting module for PDF/HTML experiment reports

# [2026-04-02T11:30:00] Improve model leaderboard sorting and filtering
