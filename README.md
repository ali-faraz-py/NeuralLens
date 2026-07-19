# 🔍 Neural Lens: High-Performance Image Recognition Engine

A full-stack Computer Vision web app built with **Python (FastAPI)** and **Next.js**. This tool utilizes **Deep Residual Learning** via the **ResNet50** architecture to transform raw visual data into structured mathematical insights with high probabilistic precision.

---

## 🚀 Live Demo
**Frontend:** [neurallens.vercel.app](https://neurallens.vercel.app)
**Backend API:** [neurallens-api.onrender.com](https://neurallens-api.onrender.com)

---

## 📺 Demo Preview
![NeuralLens Demo](assets/NeuralLens.gif)

---

## ✨ Features
- **Deep Residual Learning:** Leverages a 50-layer ResNet architecture to overcome the vanishing gradient problem, ensuring high-fidelity feature extraction.
- **Probabilistic Classification:** Identifies over **1,000 object categories** from the ImageNet dataset with a Top-5 confidence ranking system.
- **Viewfinder UI:** A custom "scanning" interface with animated corner brackets and a live scan-line effect while the model analyzes the image.
- **Real-Time Inference:** Optimized image preprocessing pipeline (224x224 RGB normalization) for fast classification via a dedicated API.
- **Confidence Readout:** Color-coded confidence tiers (high / medium / low) for both the top prediction and the full Top-5 breakdown.

---

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)
- **Frontend:** Next.js / React, Tailwind CSS
- **Deep Learning Engine:** TensorFlow / Keras
- **Architecture:** ResNet50 (Pre-trained on ImageNet)
- **Deployment:** Render (backend) + Vercel (frontend)

---

## 🚀 Installation & Local Setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Create a `.env.local` file in `frontend/` with:
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

## 📂 Project Structure
```text
neurallens/
├── backend/
│   ├── app/
│   │   ├── main.py       # FastAPI routes
│   │   └── predict.py    # ResNet50 model loading and inference engine
│   └── requirements.txt
├── frontend/
│   └── src/app/
│       ├── page.js       # Upload UI, viewfinder animation, results
│       └── globals.css   # Scan-line and pulse animations
├── assets/
│   └── NeuralLens.gif
└── README.md
```

---

## 🧠 Model Insights
The engine utilizes a **ResNet50 (Residual Network)**, a landmark architecture in Computer Vision.

* **The Architecture:** Unlike traditional sequential models, ResNet uses **shortcut connections** (identity mapping) to allow gradients to flow through deeper layers.

* **Input Transformation:** Images are mathematically resized to **224x224x3** and normalized using the specific mean/std-dev requirements of the ImageNet-trained weights.

* **The Output:** The model generates a Softmax probability distribution across 1,000 classes, which is then decoded into human-readable labels with associated confidence scores.

---

## Known Limitations
- Trained on **ImageNet**, which covers common objects, animals, and everyday items — it has no concept of "person" or "face" as categories, so photos of people may return unrelated object labels. This is expected behavior for an ImageNet-trained classifier, not a bug.
- The backend is hosted on Render's free tier, which spins down after inactivity. The first request after idle time may take 30-60 seconds to respond.

---

### 👤 Author
**Syed Ali Faraz** - [GitHub Profile](https://github.com/ali-faraz-py)

*If you found this project useful, please give the repository a ⭐!*