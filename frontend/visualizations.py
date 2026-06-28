"""Matplotlib charts and Grad-CAM panels for the FER frontend."""

import base64
import io

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

EMOTION_COLORS = {
    "Angry": "#e74c3c",
    "Disgust": "#9b59b6",
    "Fear": "#3498db",
    "Happy": "#2ecc71",
    "Sad": "#f1c40f",
    "Surprise": "#e67e22",
    "Neutral": "#95a5a6",
}

CHART_STYLE = {
    "figure.facecolor": "#12151c",
    "axes.facecolor": "#171b24",
    "axes.edgecolor": "#2a3142",
    "axes.labelcolor": "#eef2ff",
    "text.color": "#eef2ff",
    "xtick.color": "#9aa4bf",
    "ytick.color": "#9aa4bf",
    "grid.color": "#2a3142",
    "font.family": "sans-serif",
}


def fig_to_base64(fig, dpi=120):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def compute_analysis(probabilities, emotions, predicted_idx):
    probs = np.asarray(probabilities, dtype=np.float64)
    sorted_idx = np.argsort(probs)[::-1]
    top_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1])

    confidence = float(probs[top_idx])
    second_confidence = float(probs[second_idx])
    margin = confidence - second_confidence

    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log(clipped)))
    max_entropy = float(np.log(len(emotions)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    if confidence >= 0.65 and margin >= 0.20:
        certainty = "High"
        certainty_note = "The model is strongly confident in this expression."
    elif confidence >= 0.40 and margin >= 0.10:
        certainty = "Medium"
        certainty_note = "The prediction is reasonable but some classes remain competitive."
    else:
        certainty = "Low"
        certainty_note = "The model is uncertain — the image may be ambiguous or low quality."

    interpretation = build_interpretation(emotions[top_idx], emotions[second_idx], confidence, margin)

    return {
        "second_emotion": emotions[second_idx],
        "second_confidence": second_confidence,
        "second_confidence_percent": round(second_confidence * 100, 1),
        "margin": margin,
        "margin_percent": round(margin * 100, 1),
        "entropy": round(entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "certainty": certainty,
        "certainty_note": certainty_note,
        "interpretation": interpretation,
        "top3": [
            {
                "emotion": emotions[i],
                "probability": float(probs[i]),
                "percent": round(float(probs[i]) * 100, 1),
            }
            for i in sorted_idx[:3]
        ],
    }


def build_interpretation(predicted, runner_up, confidence, margin):
  if predicted == "Happy":
    lead = "Positive affect with visible mouth/eye activation."
  elif predicted in ("Sad", "Fear"):
    lead = "Negative affect — model weighs brow, eye, and mouth tension."
  elif predicted == "Angry":
    lead = "Tense facial features around brows and mouth."
  elif predicted == "Surprise":
    lead = "Wide eyes and open mouth patterns detected."
  elif predicted == "Disgust":
    lead = "Nose/mouth region contributed strongly (hardest class in FER2013)."
  else:
    lead = "Balanced neutral expression with low extreme activation."

  if margin < 0.12:
    return f"{lead} Close runner-up: {runner_up} — consider a clearer frontal face."
  if confidence < 0.35:
    return f"{lead} Low confidence — lighting, angle, or crop may reduce accuracy."
  return f"{lead} Prediction is well separated from {runner_up}."


def plot_probability_bars(emotions, probabilities, predicted_emotion):
  with plt.rc_context(CHART_STYLE):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [EMOTION_COLORS[e] for e in emotions]
    y_pos = np.arange(len(emotions))
    percents = probabilities * 100

    bars = ax.barh(y_pos, percents, color=colors, edgecolor="#2a3142", height=0.65)
    predicted_idx = emotions.index(predicted_emotion)
    bars[predicted_idx].set_edgecolor("#34d399")
    bars[predicted_idx].set_linewidth(2.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Emotion Probability Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.35)

    for i, value in enumerate(percents):
      ax.text(min(value + 1.2, 97), i, f"{value:.1f}%", va="center", fontsize=9, color="#eef2ff")

    return fig_to_base64(fig)


def plot_radar_chart(emotions, probabilities, predicted_emotion):
  with plt.rc_context(CHART_STYLE):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False)
    values = np.concatenate([probabilities, probabilities[:1]])
    angles = np.concatenate([angles, angles[:1]])

    ax.plot(angles, values, color="#6c8cff", linewidth=2.2)
    ax.fill(angles, values, color="#6c8cff", alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Emotion Radar Profile", fontsize=13, fontweight="bold", pad=18)
    ax.grid(alpha=0.35)

    pred_idx = emotions.index(predicted_emotion)
    ax.scatter(angles[pred_idx], probabilities[pred_idx], color="#34d399", s=80, zorder=5)

    return fig_to_base64(fig)


def plot_donut_chart(emotions, probabilities, predicted_emotion):
  with plt.rc_context(CHART_STYLE):
    fig, ax = plt.subplots(figsize=(5, 5))
    sorted_pairs = sorted(zip(emotions, probabilities), key=lambda item: item[1], reverse=True)
    top = sorted_pairs[:3]
    other_sum = sum(prob for _, prob in sorted_pairs[3:])
    labels = [e for e, _ in top]
    sizes = [p for _, p in top]
    colors = [EMOTION_COLORS[e] for e in labels]

    if other_sum > 0.001:
      labels.append("Other")
      sizes.append(other_sum)
      colors.append("#4b5563")

    wedges, texts, autotexts = ax.pie(
      sizes,
      labels=labels,
      colors=colors,
      autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
      startangle=90,
      pctdistance=0.78,
      wedgeprops=dict(width=0.42, edgecolor="#12151c", linewidth=2),
      textprops={"color": "#eef2ff", "fontsize": 9},
    )
    for autotext in autotexts:
      autotext.set_color("#eef2ff")
      autotext.set_fontsize(8)

    ax.text(0, 0, f"{predicted_emotion}\n{probabilities[emotions.index(predicted_emotion)] * 100:.1f}%",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#34d399")
    ax.set_title("Top Emotion Share", fontsize=13, fontweight="bold", pad=12)

    return fig_to_base64(fig)


def plot_confidence_gauge(confidence):
  with plt.rc_context(CHART_STYLE):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="RdYlGn", extent=[0.05, 0.95, 0.42, 0.58])
    ax.plot([confidence], [0.5], marker="v", color="white", markersize=14, markeredgewidth=1.5, markeredgecolor="#12151c")
    ax.text(confidence, 0.28, f"{confidence * 100:.1f}%", ha="center", fontsize=14, fontweight="bold", color="#34d399")
    ax.text(0.05, 0.12, "Low", fontsize=8, color="#9aa4bf")
    ax.text(0.95, 0.12, "High", fontsize=8, color="#9aa4bf", ha="right")
    ax.set_title("Prediction Confidence", fontsize=13, fontweight="bold", pad=8)

    return fig_to_base64(fig)


def plot_gradcam_panel(original_rgb, gray_face, heatmap, overlay, predicted_emotion):
  """Four-panel Grad-CAM analysis figure."""
  with plt.rc_context(CHART_STYLE):
    fig = plt.figure(figsize=(12, 3.4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.08)

    panels = [
      (original_rgb, "Original upload", None),
      (gray_face, "Model input (48×48)", "gray"),
      (heatmap, "Grad-CAM heatmap", "jet"),
      (overlay, "Grad-CAM overlay", None),
    ]

    for idx, (image, title, cmap) in enumerate(panels):
      ax = fig.add_subplot(gs[0, idx])
      if cmap == "gray":
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
      elif cmap == "jet":
        ax.imshow(image, cmap="jet", vmin=0, vmax=1)
      else:
        display = image
        if display.dtype != np.uint8:
          display = np.clip(display, 0, 1)
          if display.max() <= 1:
            display = (display * 255).astype(np.uint8)
        ax.imshow(display)
      ax.set_title(title, fontsize=10, pad=6)
      ax.axis("off")

    fig.suptitle(f"Grad-CAM Analysis — Predicted: {predicted_emotion}", fontsize=13, fontweight="bold", y=1.02)

    legend_handles = [
      mpatches.Patch(color="#ff0000", label="High activation"),
      mpatches.Patch(color="#0000ff", label="Low activation"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    return fig_to_base64(fig, dpi=130)


def generate_all_charts(emotions, probabilities, predicted_emotion, original_rgb, gray_face, heatmap, overlay):
  probs = np.asarray(probabilities, dtype=np.float64)
  return {
    "probability_chart": plot_probability_bars(emotions, probs, predicted_emotion),
    "radar_chart": plot_radar_chart(emotions, probs, predicted_emotion),
    "donut_chart": plot_donut_chart(emotions, probs, predicted_emotion),
    "confidence_gauge": plot_confidence_gauge(float(probs[emotions.index(predicted_emotion)])),
    "gradcam_panel": plot_gradcam_panel(original_rgb, gray_face, heatmap, overlay, predicted_emotion),
  }
