---
title: Facial Expression Recognition
emoji: 🎭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Phase III — Facial Expression Recognition (Live Demo)

**Lokesh Chamakuri** · ML Project · FER2013

Upload a face image to predict one of **7 emotions** with **Grad-CAM** and probability charts.

## Live links

| Platform | URL |
|----------|-----|
| **This Space** | https://huggingface.co/spaces/Lokesh-UE/Phase-III |
| **GitHub Pages** | https://lokesh-ue.github.io/Phase-III/ |

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Flask UI: `cd frontend && pip install -r requirements.txt && python app.py`

## Model

Custom CNN · `demo/CustomCNN_best.keras` · 64.47% test accuracy on FER2013

## Phase II

https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition
