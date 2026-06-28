# Facial Expression Recognition - Live Demo

## Overview

This is a **Gradio-based web demo** for the Facial Expression Recognition project. Upload any face image and the model will classify it into one of seven emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral**.

## Features

- Upload any face image (automatically resized to 48×48 grayscale)
- Get instant emotion prediction with confidence scores
- View **Grad-CAM heatmap** to see which facial regions the model focused on
- Compare predictions across all emotion classes

## Deploy on HuggingFace Spaces (Recommended)

### Step 1: Create a new Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Space name: `fer-demo` (or any name you prefer)
4. License: Apache 2.0
5. Select **Gradio** as the Space SDK
6. Click **"Create Space"**

### Step 2: Upload files
1. Clone the space locally:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/fer-demo
```

2. Copy these files into the cloned folder:
- `app.py`
- `requirements.txt`
- `CustomCNN_best.keras` (your trained model)

3. Push to HuggingFace:
```bash
cd fer-demo
git add .
git commit -m "Initial FER demo"
git push
```

4. Your demo will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/fer-demo`

### Step 3: Test the demo
- Upload a face image through the web interface
- The model predicts the emotion with confidence scores
- Grad-CAM highlights the regions the model used for its decision

## Local Development (Optional)

```bash
pip install -r requirements.txt
python app.py
```

The app will run at `http://localhost:7860`

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Custom CNN (3 Conv blocks + Dense) |
| Input size | 48 × 48 × 1 (grayscale) |
| Classes | 7 emotions |
| Test accuracy | 64.47% |
| Best class | Happy (F1 = 0.86) |
| Hardest class | Fear (F1 = 0.44) |

## Files

| File | Description |
|------|-------------|
| `app.py` | Gradio demo application |
| `requirements.txt` | Python dependencies |
| `CustomCNN_best.keras` | Trained model weights |

## Author

**Lokesh Chamakuri**  
University of Europe for Applied Sciences | ML Course 2026

- GitHub: [github.com/Lokesh-UE](https://github.com/Lokesh-UE)
- Kaggle: [kaggle.com/lokeshchamakuri](https://www.kaggle.com/lokeshchamakuri)
