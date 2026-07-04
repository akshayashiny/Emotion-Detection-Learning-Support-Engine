import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Emotion Analytics Dashboard")

HISTORY_FILE = "history/session_history.csv"

if not os.path.exists(HISTORY_FILE):
    st.warning("No history found. Analyze some inputs first.")
    st.stop()

df = pd.read_csv(HISTORY_FILE)

# -----------------------
# Overview Metrics
# -----------------------

st.subheader("📌 Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Analyses", len(df))

with col2:
    st.metric("Unique Subjects", df["Subject"].nunique())

with col3:
    st.metric("Models Used", df["Model"].nunique())

st.divider()

# -----------------------
# Emotion Distribution
# -----------------------

st.subheader("😊 Emotion Distribution")

emotion_counts = df["Emotion"].value_counts()

fig1 = px.pie(
    values=emotion_counts.values,
    names=emotion_counts.index,
    title="Emotion Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------
# Subject Distribution
# -----------------------

st.subheader("📚 Subject Distribution")

subject_counts = df["Subject"].value_counts()

fig2 = px.bar(
    x=subject_counts.index,
    y=subject_counts.values,
    labels={
        "x": "Subject",
        "y": "Count"
    },
    title="Subject Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Model Usage
# -----------------------

st.subheader("🤖 Model Usage")

model_counts = df["Model"].value_counts()

fig3 = px.bar(
    x=model_counts.index,
    y=model_counts.values,
    color=model_counts.index,
    title="BiLSTM vs BERT Usage"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# Confidence Distribution
# -----------------------

st.subheader("🎯 Confidence Scores")

confidence = (
    df["Confidence"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float)
)

fig4 = px.histogram(
    x=confidence,
    nbins=10,
    title="Confidence Distribution"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------
# Recent History
# -----------------------

st.subheader("📋 Session History")

st.dataframe(
    df.sort_values("Timestamp", ascending=False),
    use_container_width=True
)

# -----------------------
# Download CSV
# -----------------------

st.download_button(
    "⬇️ Download Session History",
    data=df.to_csv(index=False),
    file_name="session_history.csv",
    mime="text/csv"
)