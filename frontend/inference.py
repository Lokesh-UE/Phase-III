"""Inference helpers for the FER web frontend."""

import base64
import io
import time
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from visualizations import compute_analysis, generate_all_charts

IMG_SIZE = 48
DISPLAY_MAX = 320
# Only show a definitive label when these quality gates pass (reduces wrong detections)
MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.12
MIN_FACE_SIZE = 40
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJI = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}

ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    ROOT / "demo" / "CustomCNN_best.keras",
    ROOT / "CustomCNN_best.keras",
    ROOT / "checkpoints" / "custom_cnn_final.keras",
    ROOT / "checkpoints" / "CustomCNN_best.keras",
    Path(__file__).resolve().parent / "models" / "CustomCNN_best.keras",
]

_model = None
_face_cascade = None


def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def load_model():
    global _model
    if _model is not None:
        return _model

    model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Trained model not found. Place CustomCNN_best.keras in demo/ or checkpoints/."
        )

    _model = keras.models.load_model(str(model_path))
    return _model


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def pil_to_rgb(image):
    if isinstance(image, Image.Image):
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        return np.array(image)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if len(arr.shape) == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.shape[2] == 3 else arr


def resize_for_display(rgb_image, max_side=DISPLAY_MAX):
    height, width = rgb_image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(rgb_image, new_size, interpolation=cv2.INTER_AREA)
    return rgb_image


def detect_face(rgb_image):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=6, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    if len(faces) == 0:
        return None, None, 0.0

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_quality = min(1.0, (w * h) / (gray.shape[0] * gray.shape[1]) * 8)
    pad = int(0.12 * max(w, h))
    height, width = gray.shape
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    bbox = (x1, y1, x2, y2)
    face_gray = gray[y1:y2, x1:x2]
    return face_gray, bbox, face_quality


def draw_face_box(rgb_image, bbox):
    if bbox is None:
        return rgb_image.copy()
    annotated = rgb_image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (52, 211, 153), 2)
    cv2.putText(
        annotated, "Face", (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 211, 153), 1, cv2.LINE_AA,
    )
    return annotated


def preprocess_face(gray_face):
    resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    batch = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    return batch, normalized


def preprocess_upload(image, use_face_detection=True):
    rgb = pil_to_rgb(image)
    display_rgb = resize_for_display(rgb)
    face_quality = 0.0
    if use_face_detection:
        face_gray, bbox, face_quality = detect_face(rgb)
    else:
        face_gray, bbox = None, None

    if face_gray is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face_detected = False
        annotated = display_rgb
    else:
        gray = face_gray
        face_detected = True
        annotated = draw_face_box(display_rgb, _scale_bbox(bbox, rgb.shape, display_rgb.shape))

    batch, normalized = preprocess_face(gray)
    return batch, normalized, face_detected, annotated, display_rgb, face_quality


def _scale_bbox(bbox, src_shape, dst_shape):
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    sx = dst_w / src_w
    sy = dst_h / src_h
    x1, y1, x2, y2 = bbox
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def make_gradcam_heatmap(img_array, model, pred_index):
    last_conv_layer_name = None
    last_conv_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            last_conv_idx = i
    if last_conv_layer_name is None:
        return None

    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == last_conv_idx:
                conv_outputs = x
        predictions = x
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam(gray_face, heatmap):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_norm = np.clip(heatmap_resized, 0, 1)
    heatmap_color = cm.jet(heatmap_norm)[:, :, :3]
    face_rgb = np.stack([gray_face] * 3, axis=-1)
    blended = heatmap_color * 0.48 + face_rgb * 0.52
    return np.clip(blended, 0, 1), heatmap_norm


def array_to_base64(image_array, cmap=None):
    if cmap == "gray":
        pil_image = Image.fromarray(np.uint8(np.clip(image_array, 0, 1) * 255), mode="L")
    else:
        arr = image_array
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            if arr.max() <= 1:
                arr = (arr * 255).astype(np.uint8)
        pil_image = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def predict_probabilities(model, batch, accuracy_mode=False):
    """Single forward pass, or TTA average for higher accuracy."""
    pred = model.predict(batch, verbose=0)[0]
    if not accuracy_mode:
        return pred

    flipped = np.flip(batch, axis=2)
    pred_flip = model.predict(flipped, verbose=0)[0]
    return (pred + pred_flip) / 2.0


def assess_reliability(face_detected, confidence, margin, face_quality, accuracy_mode):
    threshold_conf = MIN_CONFIDENCE + (0.05 if accuracy_mode else 0.0)
    threshold_margin = MIN_MARGIN + (0.03 if accuracy_mode else 0.0)
    return bool(
        face_detected
        and confidence >= threshold_conf
        and margin >= threshold_margin
        and face_quality >= 0.02
    )


def build_scores(probabilities):
    scores = [
        {
            "emotion": emotion,
            "emoji": EMOTION_EMOJI[emotion],
            "probability": float(prob),
            "percent": round(float(prob) * 100, 1),
            "color": {
                "Angry": "#e74c3c", "Disgust": "#9b59b6", "Fear": "#3498db",
                "Happy": "#2ecc71", "Sad": "#f1c40f", "Surprise": "#e67e22", "Neutral": "#95a5a6",
            }[emotion],
        }
        for emotion, prob in zip(EMOTIONS, probabilities)
    ]
    scores.sort(key=lambda item: item["probability"], reverse=True)
    return scores


def predict_emotion(image, use_face_detection=True, accuracy_mode=False, include_charts=True):
    t0 = time.perf_counter()
    model = load_model()
    batch, gray_face, face_detected, annotated_rgb, _, face_quality = preprocess_upload(
        image, use_face_detection
    )

    probabilities = predict_probabilities(model, batch, accuracy_mode=accuracy_mode)
    predicted_idx = int(np.argmax(probabilities))
    raw_emotion = EMOTIONS[predicted_idx]
    confidence = float(probabilities[predicted_idx])
    second_idx = int(np.argsort(probabilities)[-2])
    margin = confidence - float(probabilities[second_idx])

    scores = build_scores(probabilities)
    analysis = compute_analysis(probabilities, EMOTIONS, predicted_idx)
    is_reliable = assess_reliability(face_detected, confidence, margin, face_quality, accuracy_mode)

    if is_reliable:
        predicted_emotion = raw_emotion
        emoji = EMOTION_EMOJI[raw_emotion]
        reliability_note = "High-confidence identification — result is reliable."
    else:
        predicted_emotion = "Uncertain"
        emoji = "❓"
        reliability_note = (
            "Could not confirm expression with high confidence. "
            "Upload a clear, frontal, well-lit face photo to avoid wrong detection."
        )
        analysis["interpretation"] = reliability_note
        analysis["certainty"] = "Low"
        analysis["certainty_note"] = reliability_note

    gradcam_b64 = None
    heatmap_b64 = None
    charts = {}
    chart_error = None
    gradcam_idx = predicted_idx if is_reliable else int(np.argmax(probabilities))

    if include_charts:
        try:
            heatmap = make_gradcam_heatmap(batch, model, gradcam_idx)
            if heatmap is not None:
                overlay, heatmap_norm = overlay_gradcam(gray_face, heatmap)
                gradcam_b64 = array_to_base64(overlay)
                heatmap_b64 = array_to_base64(heatmap_norm)
                try:
                    charts = generate_all_charts(
                        EMOTIONS,
                        probabilities,
                        raw_emotion,
                        annotated_rgb,
                        gray_face,
                        heatmap_norm,
                        overlay,
                    )
                except Exception as chart_exc:
                    chart_error = str(chart_exc)
        except Exception as gradcam_exc:
            chart_error = str(gradcam_exc)

    processing_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "emotion": predicted_emotion,
        "raw_emotion": raw_emotion,
        "emoji": emoji,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 1),
        "is_reliable": is_reliable,
        "reliability_note": reliability_note,
        "scores": scores,
        "analysis": analysis,
        "face_detected": face_detected,
        "face_quality": round(face_quality, 3),
        "annotated_image": array_to_base64(annotated_rgb),
        "preprocessed_image": array_to_base64(gray_face, cmap="gray"),
        "gradcam_image": gradcam_b64,
        "heatmap_image": heatmap_b64,
        "charts": charts,
        "chart_error": chart_error,
        "processing_ms": processing_ms,
        "accuracy_mode": accuracy_mode,
        "model_loaded": True,
    }
