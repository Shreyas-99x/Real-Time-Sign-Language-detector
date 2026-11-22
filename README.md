# SignSpeak 🤟 — Real-Time ASL Alphabet Communicator

**SignSpeak** is a real-time, browser-based assistive web application that translates American Sign Language (ASL) fingerspelling into text. Designed to assist mute and speech-impaired individuals, it runs entirely on-device in the browser using MediaPipe Hands and a zero-GC hybrid machine learning engine.

---

## ✨ Features

- **26 Pure English Alphabets (`A`–`Z`)**: Exclusively trained and tuned on single-handed ASL alphabet letters. Numbers and full-word sign hijacking have been removed for maximum precision.
- **Hybrid AI Engine (99.92% Accuracy)**:
  - **Canonical Orientation Normalization**: Dynamically aligns the hand coordinate system (wrist to middle MCP vertically upright) so gestures are recognized consistently regardless of camera angle or tilt.
  - **4-Layer MLP Neural Network** (`63 → 256 → 128 → 64 → 26`): Pre-trained on 26,000 augmented ASL samples with mirroring, yaw/pitch/roll rotations, and Gaussian jitter.
  - **Deterministic Geometric ASL Rule Engine**: Disambiguates difficult sign pairs (e.g., `G` vs `L`, `E` vs `S`, `U` vs `V`, `P` vs `H`, `X`, `R`).
- **Zero-GC & 60 FPS Performance**:
  - Pre-allocated typed arrays eliminate ~21,000 object allocations/second, preventing garbage collection stutter.
  - Batched 2D canvas drawing reduces skeleton and joint draw calls by 70%.
- **Repeat-Commit Prevention (Latch Protocol)**:
  - Automatically locks a captured letter until you relax your hand or switch gestures, preventing unwanted duplicate letters while holding a sign.
- **Fast & Responsive Timing**:
  - Dwell hold time: **550ms**
  - Recovery cooldown: **380ms**
  - Low-latency majority vote smoothing: **~65ms**
- **Sentence Builder & Autocomplete**:
  - Dynamic predictive word suggestions as you sign.
  - Action controls: `␣ Space`, `⌫ Back`, `⌫ Word`, `📋 Copy`, `🗑 Clear`, and `↵ New`.
  - Keyboard shortcuts (`Space`, `Backspace`, `Ctrl+Backspace`, `Ctrl+Enter`).
  - Emergency & common quick phrases (tap-to-insert).
  - Sound effect chime toggle (`🔊 Sound` / `🔇 Muted`) with persistent preferences.

---

## 📁 Project Structure

```
hasba/
├── web/
│   ├── index.html        # Main application interface
│   ├── style.css         # Responsive styling (Dark & Light themes)
│   ├── app.js            # Video loop, gesture latching, sentence builder
│   ├── classifier.js     # Hybrid normalizer, MLP forward pass, geometry engine
│   ├── words.js          # Dictionary & predictive prefix search
│   └── model/
│       ├── model.json    # MLP weights & architecture
│       ├── labels.json   # 26 alphabet classes (A-Z)
│       └── landmarks.json# Centroid reference coordinates
├── train/
│   ├── extract_and_train.py # Dataset processing & MLP training script
│   └── train_alphabets.py   # Landmark augmentation pipeline
├── serve.py              # Lightweight Python HTTP server
├── start.bat             # One-click desktop launcher
├── .gitignore            # Git exclusions (large dataset binaries, caches)
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Edge, Firefox, or Safari) with webcam access.
- Python 3.8+ (for local HTTP server).

### Running Locally

#### Option 1: One-Click Launcher (Windows)
Double-click **`start.bat`**. It will start the server and open your default browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

#### Option 2: Command Line
```powershell
# Navigate to directory
cd c:\Users\shrey\Desktop\hasba

# Start server
python serve.py
```
Open your browser to `http://127.0.0.1:8000`.

---

## 📖 How to Use

1. **Allow Camera Access**: Click "Start Camera" if prompted.
2. **Position Your Hand**: Place your hand inside the webcam frame. The cyan skeleton tracks your joints in real-time.
3. **Sign a Letter**: Form any ASL alphabet sign (`A`–`Z`).
4. **Hold for 0.5s**: Hold the sign steady; the green progress ring fills up and chimes when committed.
5. **Switch Gestures**: Change signs to continue spelling words, or tap autocomplete word suggestions.
6. **Mute/Unmute**: Use the `🔊 Sound` button in the header if you prefer silent signing.
