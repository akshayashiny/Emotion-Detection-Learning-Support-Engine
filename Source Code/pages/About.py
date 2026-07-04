import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About the Project")

st.markdown("""
# Emotion Detection & Learning Support Engine

## Overview

This AI-powered application helps students by detecting their emotions from learning-related text and providing personalized academic support.

## Features

- 😊 Emotion Detection using BiLSTM
- 🤖 Emotion Detection using BERT
- 🎓 Subject-wise Learning Support
- 🧠 AI Mentor using Gemini
- 📊 Analytics Dashboard
- 📜 Session History
- 📄 Downloadable Reports

## Technologies Used

- Streamlit
- TensorFlow
- PyTorch
- Hugging Face Transformers
- Google Gemini API
- Plotly
- Pandas

## Objective

To provide personalized, emotion-aware academic guidance and improve the learning experience for students.
""")