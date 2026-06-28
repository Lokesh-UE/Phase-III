"""Flask web app for facial expression recognition."""

import os
import socket
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from inference import find_model_path, load_model, predict_emotion

APP_VERSION = "2.0"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


@app.after_request
def add_cors_headers(response):
    """Allow GitHub Pages frontend to call this API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/predict", methods=["OPTIONS"])
def predict_preflight():
    return "", 204


@app.route("/")
def index():
    model_path = find_model_path()
    return render_template(
        "index.html",
        model_ready=model_path is not None,
        model_path=str(model_path) if model_path else None,
    )


@app.route("/health")
def health():
    model_path = find_model_path()
    return jsonify(
        {
            "status": "ok",
            "version": APP_VERSION,
            "model_ready": model_path is not None,
            "model_path": str(model_path) if model_path else None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        image = Image.open(file.stream)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        use_face_detection = request.form.get("face_detection", "true").lower() != "false"
        result = predict_emotion(image, use_face_detection=use_face_detection)
        result["api_version"] = APP_VERSION
        return jsonify(result)
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file. Upload JPG, PNG, or WEBP."}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "model_loaded": False}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


def pick_port(preferred=5000, max_tries=10):
    for port in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return preferred


if __name__ == "__main__":
    if find_model_path():
        load_model()
        print(f"Model loaded from: {find_model_path()}")
    else:
        print("WARNING: Model not found. Add CustomCNN_best.keras to demo/ or checkpoints/.")

    preferred = int(os.environ.get("PORT", 5000))
    port = pick_port(preferred)
    if port != preferred:
        print(f"Port {preferred} is busy (old server still running?). Using http://127.0.0.1:{port}")
        print("Tip: stop the old server with Ctrl+C in its terminal, then restart.")
    else:
        print(f"Open http://127.0.0.1:{port}")

    app.run(host="0.0.0.0", port=port, debug=False)
