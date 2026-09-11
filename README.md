# ASL Hand Sign Recognition & Trainer 🤟

Real-time American Sign Language (ASL) recognition and gesture training powered by Google's **MediaPipe Tasks Vision** (`hand_landmarker.task`).

Supports dual execution modes:
1. **Interactive Web Application** (Streamlit Cloud & browser deployment via WebAssembly / WebGL)
2. **Native Desktop Window** (High-performance OpenCV HUD with local webcam)

---

## 🎯 Supported ASL Gestures

The trainer evaluates 10 foundational ASL letter signs with strict 2D anatomical normalization:

| Sign | Description |
|:----:|:------------|
| **A** | Fist with thumb resting along the side of the index finger |
| **B** | Four fingers straight up and flat together, thumb tucked across palm |
| **C** | All fingers curved forming a wide C-shape |
| **D** | Index finger straight up, thumb touching tips of middle, ring, and pinky |
| **I** | Pinky extended straight up, all other fingers folded in, thumb tucked |
| **L** | Index and thumb extended outward at 90 degrees |
| **O** | All fingertips curved touching thumb tip to form a circle |
| **V** | Index and middle fingers extended apart in a V-shape |
| **W** | Index, middle, and ring fingers extended upward |
| **Y** | Thumb and pinky extended wide outward, middle three fingers folded |

---

## 🚀 Deployment (Streamlit Community Cloud)

This repository is pre-configured for instant 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io):

1. Fork or push this repository to GitHub: `https://github.com/parthbhardwaj0906-code/ASL-Hand-Sign-Classifier`
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select your repository, set the branch to `main`, and the main file path to `main.py`.
4. Click **"Deploy"**!

### Deployment Configuration Included:
- `requirements.txt`: Streamlit and core vision packages.
- `packages.txt`: Linux Debian dependencies (`libgl1`, `libglib2.0-0`).
- `runtime.txt`: Standard Python 3.11 environment.

---

## 💻 Local Usage

### 1. Run as Web Application (Streamlit)
```bash
pip install -r requirements.txt
streamlit run main.py
```
Open your browser at `http://localhost:8501`. Allow camera access when prompted.

### 2. Run as Native Desktop Window (OpenCV)
```bash
pip install -r requirements.txt
python main.py
```
- Press **`q`** to quit.
- Press **`n`** for next sign.
- Press **`p`** for previous sign.
- Press **`Space`** to trigger test match.
- Click **`NEXT SIGN >`** directly on the on-screen HUD.

---

## ⚙️ Architecture & Features

- **MediaPipe Tasks Vision**: Uses the official `hand_landmarker.task` model with 0.35 detection, tracking, and presence confidence thresholds.
- **Decay Buffer**: 4-frame persistence memory prevents HUD flickering during fast hand movements or brief motion blur.
- **Score Smoothing**: 5-frame moving average score buffer for stable match confidence metrics.
- **Anatomical Scaling**: Normalized 2D palm-to-finger ratios ensure consistent recognition across varying distances and camera aspect ratios.
