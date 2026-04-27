# 🔍 Neural Lens: High-Performance Image Recognition Engine

A professional-grade Computer Vision dashboard built with **Python** and **Streamlit**. This tool utilizes **Deep Residual Learning** via the **ResNet50** architecture to transform raw visual data into structured mathematical insights with high probabilistic precision.

---

## 🚀 Live Demo
**[Click here to try the Live App](https://neural-lens.streamlit.app/)**

---

## 📺 Demo Preview
![NeuralLens Demo](asessts\NeuralLens.gif)

---

## ✨ Features
- **Deep Residual Learning:** Leverages a 50-layer ResNet architecture to overcome the vanishing gradient problem, ensuring high-fidelity feature extraction.
- **Probabilistic Classification:** Identifies over **1,000 object categories** from the ImageNet dataset with a Top-5 confidence ranking system.
- **Real-Time Inference:** Optimized image preprocessing pipeline (224x224 RGB normalization) for near-instant classification.
- **Interactive Analytics:** Dynamic bar charts powered by **Plotly** to visualize the model's confidence distribution across different classes.
- **Technical UI:** Includes a comprehensive sidebar with model metadata, supported formats, and architecture specifications.

---

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Framework:** Streamlit (Web UI)
- **Deep Learning Engine:** TensorFlow / Keras
- **Architecture:** ResNet50 (Pre-trained on ImageNet)
- **Data Visualization:** Plotly Express
- **Image Processing:** Pillow & NumPy

---

## 🚀 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ali-faraz-py/NeuralLens](https://github.com/ali-faraz-py/NeuralLens)
   cd NeuralLens

2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt

3. **Run the application:**
   ```bash
    streamlit run app.py

---

## 📂 Project Structure
```text
neurallens/
├── app.py              # Streamlit Web UI and visualization logic
├── predict.py          # ResNet50 model loading and inference engine
├── requirements.txt    # Project dependencies (TensorFlow, Streamlit, etc.)
├── .gitattributes      # LFS tracking and GitHub language statistics
├── .gitignore          # Prevents tracking of cache and hidden files
└── explore.ipynb       # Research and benchmarking of various CV models
```

---

## 🧠 Model Insights
The engine utilizes a **ResNet50 (Residual Network)**, a landmark architecture in Computer Vision.

* **The Architecture:** Unlike traditional sequential models, ResNet uses **shortcut connections** (identity mapping) to allow gradients to flow through deeper layers.

* **Input Transformation:** Images are mathematically resized to **224x224x3** and normalized using the specific mean/std-dev requirements of the ImageNet-trained weights.

* **The Output:** The model generates a Softmax probability distribution across 1,000 classes, which is then decoded into human-readable labels with associated confidence scores.

---

### 👤 Author
**Syed Ali Faraz** - [GitHub Profile](https://github.com/ali-faraz-py)

*If you found this NLP pipeline useful, please give the repository a ⭐!*