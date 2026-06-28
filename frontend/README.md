# Facial Expression Recognition — Web Frontend

Upload a face image and get an instant emotion prediction from the trained Custom CNN (FER2013).

## Features

- Drag-and-drop or click to upload images
- Automatic face detection and cropping
- **Full analysis dashboard** with:
  - Predicted emotion + certainty level (High / Medium / Low)
  - Runner-up, margin, entropy, and interpretation text
  - **Grad-CAM 4-panel figure** (original → input → heatmap → overlay)
  - **Probability bar chart** (all 7 emotions)
  - **Radar chart** emotion profile
  - **Donut chart** top emotion share
  - **Confidence gauge**
  - Visual pipeline and top-3 prediction cards

## Quick start

```bash
cd frontend
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser.

## Model file

The app looks for a trained model in this order:

1. `../demo/CustomCNN_best.keras`
2. `../checkpoints/custom_cnn_final.keras`
3. `../checkpoints/CustomCNN_best.keras`

If you trained the model with `run_project.py`, copy the best checkpoint into `demo/`:

```bash
copy ..\checkpoints\custom_cnn_final.keras ..\demo\CustomCNN_best.keras
```

## API

**POST** `/predict`

- Form field `image`: image file
- Form field `face_detection`: `true` or `false` (default `true`)

Returns JSON with predicted emotion, confidence, all class scores, and base64 images.

## Alternative: Gradio demo

A simpler Gradio version is also available in the `demo/` folder:

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Runs at http://localhost:7860
