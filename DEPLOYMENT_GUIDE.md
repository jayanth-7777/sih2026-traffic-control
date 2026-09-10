# Deploying to Streamlit Community Cloud (Free 24/7 Hosting)

This guide walks you through deploying your **City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking & Adaptive Traffic Control** prototype to **Streamlit Community Cloud** to get a permanent, free public URL (e.g., `https://sih2026-traffic-ai.streamlit.app`).

---

## 📋 Prerequisites
1. A free GitHub account ([github.com](https://github.com)).
2. A free Streamlit Community Cloud account ([share.streamlit.io](https://share.streamlit.io)), signed in with your GitHub account.

---

## 🚀 Step-by-Step Deployment Instructions

### Step 1: Create a New GitHub Repository
1. Open your browser and go to [github.com/new](https://github.com/new).
2. Set **Repository name**: e.g., `sih2026-adaptive-traffic-control`.
3. Set visibility to **Public** (required for free Streamlit Community Cloud hosting).
4. Click **Create repository**.

---

### Step 2: Push or Upload the Project Files to GitHub

#### Option A: Using Git (Terminal / Command Prompt)
If you have Git installed, open terminal in `c:\sih2026` and run:
```bash
git init
git add .
git commit -m "Deploy City-Wide ANPR Trajectory & Traffic AI prototype"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
```

#### Option B: Upload via GitHub Web Interface (No Git installation needed!)
1. In your new GitHub repository page, click the **"uploading an existing file"** link.
2. Drag and drop all files and folders from `c:\sih2026`:
   - `app.py`
   - `config.py`
   - `detector.py`
   - `trajectory_tracker.py`
   - `signal_engine.py`
   - `simulation.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/`
   - `models/`
   - `videos/`
3. Click **Commit changes**.

---

### Step 3: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click the **"Create app"** button (top right).
3. Select **"I already have an app"**.
4. Fill in the deployment form:
   - **Repository**: Select `<YOUR-USERNAME>/sih2026-adaptive-traffic-control`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Customize your URL (e.g. `sih2026-traffic-ai.streamlit.app`)
5. Click **"Deploy!"**

---

### Step 4: Live App Deployment
- Streamlit Cloud will automatically detect:
  - `requirements.txt` (installs PyTorch, Ultralytics, OpenCV Headless, Streamlit, etc.)
  - `packages.txt` (installs Debian system libraries `libgl1`, `libglib2.0-0`, `tesseract-ocr`)
  - `.streamlit/config.toml` (applies the dark glassmorphic theme and server settings)
- Within 2-3 minutes, your app will be live 24/7 with a permanent HTTPS link shareable with hackathon judges, mentors, and teammates!
