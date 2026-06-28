# Deploy on GitHub (no Hugging Face required)

Hugging Face is **optional**. You can host the interface on **GitHub Pages** and the ML backend on **Render** (free, deploys directly from your GitHub repo).

## How it works

| Part | Host | URL example |
|------|------|-------------|
| **UI** (HTML/CSS/JS) | GitHub Pages | `https://lokesh-ue.github.io/Phase-III/` |
| **API** (TensorFlow model) | Render (free) | `https://fer-phase-iii.onrender.com` |

GitHub Pages only serves static files — it cannot run Python or TensorFlow. The model runs on Render, which auto-deploys when you push to GitHub.

```
User → GitHub Pages (interface) → Render API (predictions) → results shown in browser
```

---

## Step 1 — Enable GitHub Pages

1. Open https://github.com/Lokesh-UE/Phase-III/settings/pages
2. **Build and deployment** → Source: **GitHub Actions**
3. Push the `docs/` folder to the `main` branch (already included in this repo)
4. Your UI will be live at: **https://lokesh-ue.github.io/Phase-III/**

---

## Step 2 — Deploy API on Render (free)

1. Go to https://render.com and sign in with **GitHub**
2. **New** → **Blueprint** → connect repo `Lokesh-UE/Phase-III`
3. Render reads `render.yaml` and creates the web service
4. Wait for deploy (~5–10 min first time)
5. Copy your Render URL, e.g. `https://fer-phase-iii.onrender.com`

---

## Step 3 — Connect UI to API

Edit `docs/static/js/config.js`:

```javascript
window.FER_API_URL = "https://fer-phase-iii.onrender.com";
```

Commit and push to GitHub. Pages will update in ~1 minute.

---

## Step 4 — Test

1. Open https://lokesh-ue.github.io/Phase-III/
2. Status pill should show **Model ready**
3. Upload a face image → **Run full analysis**

---

## Local development (no cloud)

```bash
cd frontend
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 — works without config.js (same-origin API).

---

## Alternatives to Render

| Service | Free tier | Notes |
|---------|-----------|-------|
| [Render](https://render.com) | Yes | Recommended, uses `render.yaml` |
| [Railway](https://railway.app) | Limited | Connect GitHub repo |
| [Fly.io](https://fly.io) | Limited | Docker deploy |

Hugging Face Spaces is also optional — use it only if you prefer their Gradio UI.

---

## Summary

- **Hugging Face:** not mandatory  
- **GitHub Pages:** hosts your interface at `*.github.io`  
- **Render/Railway:** runs the model (deployed from the same GitHub repo)  
- Everything stays tied to your **GitHub repository**
