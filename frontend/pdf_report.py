"""Generate PDF analysis report with images and metrics."""

import base64
import io
import tempfile
from datetime import datetime
from pathlib import Path

from fpdf import FPDF


def _save_b64_png(b64_value):
    if not b64_value:
        return None
    data = base64.b64decode(b64_value)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _chart_images(charts):
    if not charts:
        return []
    labels = [
        ("gradcam_panel", "Grad-CAM Panel"),
        ("probability_chart", "Probability Distribution"),
        ("radar_chart", "Emotion Radar"),
        ("donut_chart", "Top Emotion Share"),
        ("confidence_gauge", "Confidence Gauge"),
    ]
    items = []
    for key, label in labels:
        if charts.get(key):
            items.append((label, charts[key]))
    return items


def generate_pdf_report(result):
    """Build a PDF byte stream matching the on-screen analysis."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Facial Expression Recognition Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, "Model: Custom CNN | Dataset: FER2013", ln=True)
    pdf.ln(4)

    emotion = result.get("emotion", "Unknown")
    confidence = result.get("confidence_percent", 0)
    reliable = result.get("is_reliable", True)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Prediction Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Predicted emotion: {emotion}", ln=True)
    pdf.cell(0, 7, f"Confidence: {confidence}%", ln=True)
    pdf.cell(0, 7, f"Reliable identification: {'Yes' if reliable else 'No - retake photo'}", ln=True)
    pdf.cell(0, 7, f"Face detected: {'Yes' if result.get('face_detected') else 'No'}", ln=True)
    pdf.cell(0, 7, f"Processing time: {result.get('processing_ms', '-')} ms", ln=True)

    analysis = result.get("analysis") or {}
    if analysis.get("interpretation"):
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 5, analysis["interpretation"])
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "All Class Probabilities", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in result.get("scores", []):
        pdf.cell(0, 6, f"  {item['emotion']}: {item['percent']}%", ln=True)
    pdf.ln(3)

    if analysis.get("top3"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top 3 Predictions", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for i, item in enumerate(analysis["top3"], 1):
            pdf.cell(0, 6, f"  #{i} {item['emotion']} - {item['percent']}%", ln=True)
        pdf.ln(3)

    temp_files = []
    image_sections = [
        ("annotated_image", "Detected Face"),
        ("preprocessed_image", "Model Input (48x48)"),
        ("heatmap_image", "Grad-CAM Heatmap"),
        ("gradcam_image", "Grad-CAM Overlay"),
    ]

    for key, title in image_sections:
        path = _save_b64_png(result.get(key))
        if path:
            temp_files.append(path)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, title, ln=True)
            pdf.ln(2)
            pdf.image(path, w=120)

    for title, b64 in _chart_images(result.get("charts") or {}):
        path = _save_b64_png(b64)
        if path:
            temp_files.append(path)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, title, ln=True)
            pdf.ln(2)
            pdf.image(path, w=170)

    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytearray, bytes)) else out.encode("latin-1")

    for path in temp_files:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    return pdf_bytes
