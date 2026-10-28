"""
ReportLab PDF Clinical Diagnostic & Ayurvedic Report Generator.
Generates structured medical diagnostic summaries including Grad-CAM visual findings.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_diagnostic_pdf(
    output_pdf_path: str,
    patient_identifier: str,
    prediction_data: Dict[str, Any],
    original_image_path: Optional[str] = None,
    gradcam_image_path: Optional[str] = None
) -> str:
    """
    Builds a professional clinical diagnostic report in PDF format.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        'DocHeader',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        alignment=1
    )
    
    subhead_style = ParagraphStyle(
        'DocSubHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4B5563'),
        alignment=1
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1F2937')
    )

    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    alert_style = ParagraphStyle(
        'Alert',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#DC2626')
    )

    story = []

    # Title & Header
    story.append(Paragraph("BURN INJURIES DIAGNOSTIC SYSTEM", header_style))
    story.append(Paragraph("Clinical Deep Learning & Ayurvedic Diagnostic Assessment", subhead_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1E3A8A'), spaceAfter=12))

    # Metadata Table
    case_id = f"BIDS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    meta_data = [
        [
            Paragraph(f"<b>Patient / User:</b> {patient_identifier}", body_style),
            Paragraph(f"<b>Case Ref:</b> {case_id}", body_style)
        ],
        [
            Paragraph(f"<b>Diagnostic Date:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}", body_style),
            Paragraph(f"<b>AI Model:</b> Deep CNN + Grad-CAM Interpretability", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Diagnostic Classification Result Box
    predicted_class = prediction_data.get("title", "Unknown")
    confidence = prediction_data.get("confidence", 0.0)
    severity = prediction_data.get("severity_level", "Unknown")
    is_emergency = prediction_data.get("is_emergency", False)

    status_color = colors.HexColor('#DC2626') if is_emergency else colors.HexColor('#2563EB')
    bg_color = colors.HexColor('#FEF2F2') if is_emergency else colors.HexColor('#EFF6FF')

    result_data = [
        [Paragraph(f"<b>DIAGNOSTIC OUTCOME:</b> {predicted_class.upper()}", ParagraphStyle('Res', parent=bold_body, fontSize=12, textColor=status_color))],
        [Paragraph(f"<b>Severity Level:</b> {severity} | <b>Model Confidence:</b> {confidence}%", body_style)],
        [Paragraph(f"<b>Dosha Involvement:</b> {prediction_data.get('dosha_involvement', 'N/A')}", body_style)]
    ]
    result_table = Table(result_data, colWidths=[540])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1.5, status_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 12))

    # Images (Original & Grad-CAM)
    img_cells = []
    if original_image_path and os.path.exists(original_image_path):
        try:
            img_cells.append([
                Paragraph("<b>Original Uploaded Skin Image</b>", body_style),
                RLImage(original_image_path, width=200, height=150)
            ])
        except Exception:
            pass

    if gradcam_image_path and os.path.exists(gradcam_image_path):
        try:
            img_cells.append([
                Paragraph("<b>Grad-CAM Explainability Heatmap</b>", body_style),
                RLImage(gradcam_image_path, width=200, height=150)
            ])
        except Exception:
            pass

    if len(img_cells) == 2:
        vis_table = Table([
            [img_cells[0][0], img_cells[1][0]],
            [img_cells[0][1], img_cells[1][1]]
        ], colWidths=[270, 270])
        vis_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(vis_table)
        story.append(Spacer(1, 10))

    # Clinical Presentation
    story.append(Paragraph("Clinical Presentation & Pathophysiology", section_title))
    story.append(Paragraph(prediction_data.get("clinical_presentation", "N/A"), body_style))
    story.append(Spacer(1, 8))

    # Immediate First Aid
    first_aid = prediction_data.get("immediate_first_aid", [])
    if first_aid:
        story.append(Paragraph("Immediate First-Aid Protocols", section_title))
        for step in first_aid:
            story.append(Paragraph(f"• {step}", body_style))
        story.append(Spacer(1, 8))

    # Ayurvedic Therapies
    therapies = prediction_data.get("ayurvedic_topical_therapies", [])
    if therapies:
        story.append(Paragraph("Ayurvedic & Classical Herbal Regimen", section_title))
        for t in therapies:
            story.append(Paragraph(f"<b>{t.get('remedy', '')}</b>", bold_body))
            story.append(Paragraph(f"<i>Method:</i> {t.get('method', '')}", body_style))
            story.append(Paragraph(f"<i>Therapeutic Rationale:</i> {t.get('benefit', '')}", body_style))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 4))

    # Safety Warnings & Dietary
    diet = prediction_data.get("dietary_guidelines", [])
    if diet:
        story.append(Paragraph("Dietary & Ahara-Vihara Guidelines", section_title))
        for d in diet:
            story.append(Paragraph(f"• {d}", body_style))
        story.append(Spacer(1, 6))

    precautions = prediction_data.get("safety_precautions", [])
    if precautions:
        story.append(Paragraph("Critical Safety Precautions & Red Flags", section_title))
        for p in precautions:
            story.append(Paragraph(f"⚠️ {p}", alert_style if is_emergency else body_style))
        story.append(Spacer(1, 8))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#9CA3AF'), spaceAfter=6, spaceBefore=6))
    disclaimer = (
        "<b>MEDICAL DISCLAIMER:</b> This report is generated by an automated Convolutional Neural Network (CNN) "
        "decision support system for triage and educational assessment. It does not substitute for an in-person clinical "
        "examination by a certified physician or burn care specialist. For severe burns, seek immediate emergency medical care."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle('Disc', parent=body_style, fontSize=7.5, leading=10, textColor=colors.HexColor('#6B7280'))))

    doc.build(story)
    return output_pdf_path
