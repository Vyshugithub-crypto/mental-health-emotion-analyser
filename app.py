import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

st.set_page_config(
    page_title="Emotion Analyser",
    page_icon="🧠",
    layout="centered"
)

EMOTION_EMOJI = {
    "joy"      : "😊",
    "sadness"  : "😢",
    "anger"    : "😠",
    "fear"     : "😨",
    "love"     : "❤️",
    "surprise" : "😲"
}

EMOTION_COLOR = {
    "joy"      : "#FFD700",
    "sadness"  : "#6495ED",
    "anger"    : "#FF4500",
    "fear"     : "#9370DB",
    "love"     : "#FF69B4",
    "surprise" : "#00CED1"
}

@st.cache_resource
def load_model():
    model_path = "vaishnavikongalla/mental-health-emotion-analyser"
    tokenizer  = DistilBertTokenizer.from_pretrained(model_path)
    model      = DistilBertForSequenceClassification.from_pretrained(model_path)
    classes    = np.load("label_classes.npy", allow_pickle=True)
    model.eval()
    return tokenizer, model, classes

def predict_emotion(text, tokenizer, model, classes):
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        pred    = torch.argmax(probs).item()

    emotion    = classes[pred]
    confidence = probs[pred].item() * 100
    all_scores = {classes[i]: probs[i].item() * 100 for i in range(len(classes))}
    return emotion, confidence, all_scores

st.title("🧠 Mental Health Emotion Analyser")
st.markdown("*Detect emotions in text using fine-tuned DistilBERT — 92.60% accuracy*")
st.divider()

with st.spinner("Loading AI model..."):
    tokenizer, model, classes = load_model()
st.success("✅ Model loaded!")

tab1, tab2 = st.tabs(["Single Analysis", "Batch Analysis"])

with tab1:
    st.subheader("Analyse a single sentence")
    user_input = st.text_area(
        "Enter your text here:",
        placeholder="e.g. I am so happy today!",
        height=120
    )

    if st.button("🔍 Predict Emotion", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text!")
        else:
            with st.spinner("Analysing..."):
                emotion, confidence, all_scores = predict_emotion(
                    user_input, tokenizer, model, classes
                )

            emoji = EMOTION_EMOJI.get(emotion, "🤔")
            color = EMOTION_COLOR.get(emotion, "#888")

            st.markdown(f"""
            <div style='background:{color}22; border-left: 5px solid {color};
                        padding: 1rem; border-radius: 8px; margin: 1rem 0'>
                <h2 style='margin:0'>{emoji} {emotion.upper()}</h2>
                <p style='margin:0'>Confidence: <b>{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("All emotion scores:")
            for emo, score in sorted(all_scores.items(),
                                     key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(score / 100,
                        text=f"{EMOTION_EMOJI.get(emo,'')} {emo}")
                with col2:
                    st.write(f"{score:.1f}%")

with tab2:
    st.subheader("Analyse multiple sentences at once")
    batch_input = st.text_area(
        "Enter one sentence per line:",
        placeholder="I feel amazing today!\nI am so scared.\nThis makes me angry.",
        height=200
    )

    if st.button("🔍 Analyse All", type="primary"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one sentence!")
        else:
            results = []
            for line in lines:
                emotion, confidence, _ = predict_emotion(
                    line, tokenizer, model, classes
                )
                results.append({
                    "Text"      : line,
                    "Emotion"   : f"{EMOTION_EMOJI.get(emotion,'')} {emotion}",
                    "Confidence": f"{confidence:.1f}%"
                })

            import pandas as pd
            st.dataframe(pd.DataFrame(results), use_container_width=True)

st.divider()
st.markdown(
    "*Built by Vaishnavi Kongalla | DistilBERT | HuggingFace Transformers*"
)
