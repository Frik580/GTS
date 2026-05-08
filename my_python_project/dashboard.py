import streamlit as st
import pandas as pd
from db import get_db_connection

st.set_page_config(layout="wide", page_title="GTS 3.0 Terminal")

@st.cache_data(ttl=30)
def load_data():
    # Используем context manager для автоматического закрытия соединения
    with get_db_connection() as conn:
        return pd.read_sql("SELECT * FROM events ORDER BY timestamp DESC LIMIT 50", conn)

st.title("📊 GTS 3.0 LIVE Terminal")

st.button("🔄 Refresh")

df = load_data()

# вычисляем текущий режим
if not df.empty:
    avg_score = df["score"].head(10).mean()
    last_fng = df["fear_greed"].iloc[0] if "fear_greed" in df.columns else 50
else:
    avg_score = 0
    last_fng = 50

col1, col2, col3, col4 = st.columns(4)

col1.metric("Risk Score", round(avg_score, 2))
col4.metric("Fear & Greed", int(last_fng))

col2.metric("Regime",
    "🔴 RISK-OFF" if avg_score > 3 else
    "🟢 RISK-ON" if avg_score < -2 else
    "⚪ NEUTRAL"
)

col3.metric("Events", len(df))

st.subheader("🧠 Live Events Feed")
st.dataframe(df)

st.caption("Auto-refreshing every 60s...")