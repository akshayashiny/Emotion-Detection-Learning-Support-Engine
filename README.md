# 🧠 Emotion Detection & Learning Support Engine

An AI-powered web application that detects students' emotions from text and provides personalized learning guidance using deep learning models and Google Gemini AI.

---

# 📌 Project Overview

Students often experience emotions such as confusion, frustration, curiosity, or confidence while learning. This project analyzes a student's written learning problem, detects the underlying emotion using machine learning models, and provides personalized recommendations and AI-generated guidance.

The application combines deep learning with Generative AI to create an intelligent learning support system.

---

# 🎯 Objectives

- Detect student emotions from learning-related text.
- Compare emotion prediction using BiLSTM and BERT.
- Provide personalized learning recommendations.
- Generate empathetic AI guidance using Google Gemini.
- Maintain analysis history for continuous learning.
- Visualize analytics through an interactive dashboard.

---

# ✨ Features

- 🧠 Emotion Detection using BiLSTM
- 🤖 Emotion Detection using BERT
- 💬 AI Learning Guidance using Google Gemini
- 📚 Subject-wise Learning Support
- 📈 Emotion Confidence Score
- 📊 Analytics Dashboard
- 📜 Session History
- 📄 Downloadable Analysis Report
- 🌐 Interactive Streamlit Web Application

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Machine Learning

- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers

## AI

- Google Gemini API

## Web Framework

- Streamlit

## Data Processing

- NumPy
- Pandas
- Scikit-learn

## Visualization

- Plotly

---

# 📂 Project Structure

```
Emotion-Detection-Learning-Support-Engine/

│
├── Source Code/
│   ├── app.py
│   ├── emotion_detector.py
│   ├── gemini_service.py
│   ├── recommendation_engine.py
│   ├── utils.py
│   ├── config.py
│   ├── requirements.txt
│   │
│   ├── saved_models/
│   ├── history/
│   ├── pages/
│   └── uploads/
│
├── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/akshayashiny/Emotion-Detection-Learning-Support-Engine.git
```

## Navigate to the project

```bash
cd Emotion-Detection-Learning-Support-Engine
```

## Create a virtual environment

```bash
python -m venv venv
```

## Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Navigate to the Source Code folder:

```bash
cd "Source Code"
```

Run the application:

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# 🧠 Workflow

1. Enter the learning problem.
2. Select the subject.
3. Choose the emotion detection model (BiLSTM or BERT).
4. Click **Analyze Emotion**.
5. Detect emotion and confidence score.
6. Generate learning recommendations.
7. Generate AI guidance using Gemini.
8. Save the session to history.
9. Download the analysis report.

---

# 📊 Output

The application displays:

- Detected Emotion
- Confidence Score
- Selected Model
- Learning Recommendation
- AI Mentor Guidance
- Session History
- Analytics Dashboard
- Downloadable Report

---

# 🚀 Future Enhancements

- Voice-based emotion detection
- Multi-language support
- Speech emotion recognition
- Personalized learning roadmap
- Cloud deployment
- Database integration
- User authentication
- Mobile application support

---

## 👨‍💻 Developed By

### Team Members

- **Akshaya S**
- **Rithika Sundari S**
- **Pavan Eluri**

🎓 **B.Tech in Computer Science and Engineering**

---

# 📄 License

This project is developed for educational purposes.
