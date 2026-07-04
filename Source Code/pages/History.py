import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Session History",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Session History")

HISTORY_FILE = "history/session_history.csv"

if not os.path.exists(HISTORY_FILE):
    st.warning("No session history found.")
    st.stop()

df = pd.read_csv(HISTORY_FILE)

st.dataframe(
    df.sort_values("Timestamp", ascending=False),
    use_container_width=True
)

st.download_button(
    "Download History",
    df.to_csv(index=False),
    "session_history.csv",
    "text/csv"
)