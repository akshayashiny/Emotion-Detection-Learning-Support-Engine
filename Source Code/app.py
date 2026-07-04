from utils import create_report
import streamlit as st
import os
import csv
from datetime import datetime

from emotion_detector import detect_emotion, detect_emotion_bert
from recommendation_engine import get_recommendation
from gemini_service import generate_response

# -----------------------
# Page Configuration
# -----------------------

st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="🧠",
    layout="wide"
)

# -----------------------
# Custom CSS
# -----------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.big-title{
    text-align:center;
    font-size:42px;
    color:white;
    font-weight:bold;
}

.sub{
    text-align:center;
    color:#BBBBBB;
    font-size:18px;
}

.metric-card{
    background:#1B263B;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    box-shadow:0px 5px 15px rgba(0,0,0,0.3);
}

.metric-title{
    font-size:18px;
    color:#B0B0B0;
}

.metric-value{
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    st.title("🧠 Emotion Detection")

    st.success("Learning Support Engine")

    st.write("---")

    st.write("✅ BiLSTM")
    st.write("✅ BERT")
    st.write("✅ Gemini AI")
    st.write("✅ Analytics")
    st.write("✅ Session History")
    st.write("✅ Report Download")

# -----------------------
# Header
# -----------------------

st.markdown(
    "<div class='big-title'>🧠 Emotion Detection & Learning Support Engine</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub'>AI Powered Student Emotion Detection using BiLSTM + BERT + Gemini</div>",
    unsafe_allow_html=True
)

st.write("")

# -----------------------
# Model Selection
# -----------------------

selected_model = st.selectbox(
    "Select Emotion Detection Model",
    ["BiLSTM","BERT"]
)

# -----------------------
# Subject
# -----------------------

field = st.selectbox(
    "Select Subject",
    [
        "Python",
        "Java",
        "Data Structures",
        "Algorithms",
        "Operating Systems",
        "DBMS",
        "Computer Networks",
        "Machine Learning",
        "Artificial Intelligence",
        "Cloud Computing",
        "Other"
    ]
)

# -----------------------
# Input
# -----------------------

user_text = st.text_area(
    "Describe your learning problem",
    height=200,
    placeholder="Example: I cannot understand recursion..."
)
# -----------------------
# Analyze
# -----------------------

if st.button("🚀 Analyze Emotion", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter your learning problem.")
        st.stop()

    with st.spinner("🔍 Detecting Emotion..."):

        # Emotion Detection
        if selected_model == "BiLSTM":
            emotion, confidence = detect_emotion(user_text)
        else:
            emotion, confidence = detect_emotion_bert(user_text)

        # Recommendation
        recommendation = get_recommendation(emotion)

        # Gemini AI
        ai_reply = generate_response(
            user_text,
            emotion,
            confidence,
            field
        )

        # -----------------------
        # Save Session History
        # -----------------------

        history_folder = "history"
        os.makedirs(history_folder, exist_ok=True)

        history_file = os.path.join(
            history_folder,
            "session_history.csv"
        )

        file_exists = os.path.isfile(history_file)

        with open(
            history_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "Model",
                    "Subject",
                    "Emotion",
                    "Confidence",
                    "User Input"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                selected_model,
                field,
                emotion,
                f"{confidence*100:.2f}%",
                user_text
            ])

    # -----------------------
    # Results
    # -----------------------

    st.success("✅ Analysis Completed Successfully!")

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🧠 Model</div>
            <div class="metric-value">{selected_model}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">😊 Emotion</div>
            <div class="metric-value">{emotion}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Confidence</div>
            <div class="metric-value">{confidence*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.progress(confidence)

    # -----------------------
    # Recommendation
    # -----------------------

    st.markdown("## 📘 Learning Recommendation")

    st.success(recommendation)

    # -----------------------
    # AI Mentor
    # -----------------------

    st.markdown("## 🤖 AI Mentor")

    with st.chat_message("assistant"):
        st.write(ai_reply)

    # -----------------------
    # Report
    # -----------------------

    report = create_report(
        subject=field,
        model=selected_model,
        emotion=emotion,
        confidence=confidence,
        recommendation=recommendation,
        ai_response=ai_reply
    )

    st.download_button(
        label="📄 Download Analysis Report",
        data=report,
        file_name="Emotion_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

# -----------------------
# Footer
# -----------------------

st.markdown("---")

st.caption(
    "🧠 Emotion Detection & Learning Support Engine | "
    "Built using Streamlit • BiLSTM • BERT • Gemini AI"
)