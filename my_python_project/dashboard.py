import asyncio
import streamlit as st
import pandas as pd
from db import get_db_connection

st.set_page_config(layout="wide", page_title="GTS 3.0 Terminal")

async def fetch_dashboard_data():
    async with get_db_connection() as conn:
        async with conn.execute("""
            SELECT e.title, e.score as news_score, e.event as type, e.is_black_swan, e.timestamp, 
                   p.target_asset, p.resolved, p.is_correct
            FROM events e
            LEFT JOIN predictions p ON e.id = p.event_id
            ORDER BY e.timestamp DESC LIMIT 50
        """) as cursor:
            rows = await cursor.fetchall()
            return pd.DataFrame([dict(r) for r in rows])

async def fetch_last_fng():
    async with get_db_connection() as conn:
        async with conn.execute("SELECT fear_greed FROM events WHERE fear_greed IS NOT NULL ORDER BY timestamp DESC LIMIT 1") as cursor:
            row = await cursor.fetchone()
            return row["fear_greed"] if row else 50

@st.cache_data(ttl=30)
def load_data():
    return asyncio.run(fetch_dashboard_data())

st.title("📊 GTS 3.0 LIVE Terminal")

st.button("🔄 Refresh")

df = load_data()

# вычисляем текущий режим
if not df.empty:
    avg_score = df["news_score"].head(10).mean()
    last_fng = asyncio.run(fetch_last_fng())
else:
    avg_score = 0
    last_fng = 50

col1, col2, col3, col4 = st.columns(4)

col1.metric("Risk Score", round(avg_score, 2))
col4.metric("Fear & Greed", int(last_fng))

col2.metric("Regime",
    "🔴 RISK-OFF" if avg_score > 2.5 else
    "🟢 RISK-ON" if avg_score < -1.5 else
    "⚪ NEUTRAL"
)

col3.metric("Events", len(df))

st.subheader("🧠 Live Events Feed")

st.dataframe(
    df,
    column_config={
        "resolved": st.column_config.SelectboxColumn(
            "Phase", 
            options={0: "🆕 New", 1: "⚡ Primary", 2: "✅ Done"}
        ),
        "is_correct": st.column_config.CheckboxColumn("Correct?"),
        "is_black_swan": st.column_config.CheckboxColumn("Swan 🦢"),
        "news_score": st.column_config.NumberColumn("Score", format="%.2f"),
        "timestamp": st.column_config.DatetimeColumn("Time", format="HH:mm:ss")
    },
    hide_index=True,
    use_container_width=True
)

st.caption("Auto-refreshing every 60s...")