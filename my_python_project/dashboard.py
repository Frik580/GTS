import streamlit as st
import sqlite3
import pandas as pd
import time

st.set_page_config(layout="wide", page_title="GTS 3.0 Terminal")

def load_data():
    # Используем context manager для автоматического закрытия соединения
    with sqlite3.connect("gts.db") as conn:
        return pd.read_sql("SELECT * FROM events ORDER BY timestamp DESC LIMIT 50", conn)

st.title("📊 GTS 3.0 LIVE Terminal")

refresh = st.button("🔄 Refresh")

df = load_data()

# вычисляем текущий режим
if not df.empty:
    avg_score = df["score"].head(10).mean()
else:
    avg_score = 0

col1, col2, col3 = st.columns(3)

col1.metric("Risk Score", round(avg_score, 2))

col2.metric("Regime",
    "🔴 RISK-OFF" if avg_score > 3 else
    "🟢 RISK-ON" if avg_score < -2 else
    "⚪ NEUTRAL"
)

col3.metric("Events", len(df))

st.subheader("🧠 Live Events Feed")
st.dataframe(df)

time.sleep(60)
st.rerun()