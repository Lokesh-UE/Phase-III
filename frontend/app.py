"""Flask web app for facial expression recognition."""

import io
import os
import socket

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

from inference import find_model_path, load_model, predict_emotion
from pdf_report import generate_pdf_report

APP_VERSION = "3.0"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
    return response


@app.route("/predict", methods=["OPTIONS"])
@app.route("/download-pdf", methods=["OPTIONS"])
def cors_preflight():
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


def _parse_image_upload():
    if "image" not in request.files:
        return None, jsonify({"error": "No image uploaded."}), 400
    file = request.files["image"]
    if not file.filename:
        return None, jsonify({"error": "Empty filename."}), 400
    image = Image.open(file.stream)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image, None, None


@app.route("/predict", methods=["POST"])
def predict():
    image, err_resp, err_code = _parse_image_upload()
    if err_resp:
        return err_resp, err_code

    try:
        use_face_detection = request.form.get("face_detection", "true").lower() != "false"
        accuracy_mode = request.form.get("accuracy_mode", "false").lower() == "true"
        result = predict_emotion(
            image,
            use_face_detection=use_face_detection,
            accuracy_mode=accuracy_mode,
        )
        result["api_version"] = APP_VERSION
        return jsonify(result)
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file. Upload JPG, PNG, or WEBP."}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "model_loaded": False}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No report data. Run analysis first."}), 400
    try:
        pdf_bytes = generate_pdf_report(data)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="facial_expression_report.pdf",
        )
    except Exception as exc:
        return jsonify({"error": f"PDF generation failed: {exc}"}), 500


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
    print(f"Open http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
