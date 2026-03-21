# 🧠 Mental Health Emotion Analyser

> Detect emotions in text using fine-tuned DistilBERT — **92.60% accuracy** across 6 emotion categories

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://vaishnavi-emotion-analyser.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21F)](https://huggingface.co/vaishnavikongalla/mental-health-emotion-analyser)

---

## 🌐 Live App
**[vaishnavi-emotion-analyser.streamlit.app](https://vaishnavi-emotion-analyser.streamlit.app)**

---

## 📸 Screenshots

### Homepage
![Homepage](screenshots/Screenshot%202026-03-21%20112930.png)

### Single Emotion Prediction
![Prediction](screenshots/Screenshot%202026-03-21%20113103.png)

### All Emotion Scores
![Scores](screenshots/Screenshot%202026-03-21%20113140.png)

### Batch Analysis
![Batch](screenshots/Screenshot%202026-03-21%20113204.png)

---

## 📌 What This Project Does
Takes any text input and detects the emotion behind it using fine-tuned DistilBERT trained on 20,000 real-world tweets. Classifies into 6 emotions:

| Emotion | Emoji | Accuracy |
|---------|-------|----------|
| Sadness | 😢 | 97% |
| Joy | 😊 | 96% |
| Anger | 😠 | 91% |
| Fear | 😨 | 88% |
| Love | ❤️ | 78% |
| Surprise | 😲 | 73% |

**Overall Test Accuracy: 92.60%**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | DistilBERT (distilbert-base-uncased) |
| Framework | HuggingFace Transformers + PyTorch |
| Frontend | Streamlit |
| Dataset | Emotions Dataset (Kaggle) — 20,000 tweets |
| Model Hosting | HuggingFace Hub |
| Deployment | Streamlit Cloud |
| Training | Google Colab (T4 GPU) |

---

## 🚀 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/Vyshugithub-crypto/mental-health-emotion-analyser.git
cd mental-health-emotion-analyser

# Install dependencies
pip install streamlit transformers torch numpy

# Run the app
streamlit run app.py
```

Open in browser: `http://localhost:8501`

---

## 🏋️ Model Training

- Base Model: distilbert-base-uncased
- Dataset: 20,000 tweets (Kaggle)
- Epochs: 3 | Optimizer: AdamW | LR: 2e-5
- Platform: Google Colab T4 GPU
- Best Validation Accuracy: 93.65%
- Final Test Accuracy: 92.60%

---

## 👩‍💻 Built By
**Vaishnavi Kongalla**
- GitHub: [Vyshugithub-crypto](https://github.com/Vyshugithub-crypto)
- LinkedIn: [vaishnavi-kongalla](https://www.linkedin.com/in/vaishnavi-kongalla-9235192ab)
- Live App: [vaishnavi-emotion-analyser.streamlit.app](https://vaishnavi-emotion-analyser.streamlit.app)
