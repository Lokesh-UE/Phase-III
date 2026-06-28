"""
Hugging Face Spaces entry point.
Upload a face image → emotion prediction + Grad-CAM + charts.
"""

import base64
import io
import os
import sys

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "frontend"))
from inference import EMOTIONS, predict_emotion  # noqa: E402


def analyze(image, use_face_detection):
    if image is None:
        return "Upload an image first.", None, None, None, None

    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    result = predict_emotion(pil_image, use_face_detection=use_face_detection)

    summary = (
        f"**{result['emoji']} {result['emotion']}** — "
        f"{result['confidence_percent']}% confidence\n\n"
        f"_{result['analysis']['interpretation']}_"
    )

    charts = result.get("charts") or {}
    gradcam_panel = _b64_to_pil(charts.get("gradcam_panel"))
    prob_chart = _b64_to_pil(charts.get("probability_chart"))
    radar = _b64_to_pil(charts.get("radar_chart"))
    annotated = _b64_to_pil(result.get("annotated_image"))

    return summary, annotated, gradcam_panel, prob_chart, radar


def _b64_to_pil(value):
    if not value:
        return None
    return Image.open(io.BytesIO(base64.b64decode(value)))


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(label="Upload face image", type="pil"),
        gr.Checkbox(label="Auto-detect face", value=True),
    ],
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Image(label="Detected face"),
        gr.Image(label="Grad-CAM panel"),
        gr.Image(label="Probability chart"),
        gr.Image(label="Radar chart"),
    ],
    title="Facial Expression Recognition",
    description=(
        "Custom CNN trained on **FER2013**. Classes: "
        + ", ".join(EMOTIONS)
        + ". Includes Grad-CAM and probability charts."
    ),
    examples=[],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
