import feedparser
import logging
import requests
import time
import os
import json
import google.generativeai as genai
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
from db import get_db_connection, init_db
from dotenv import load_dotenv

load_dotenv()
init_db()

# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("gts.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# =========================
# CONFIG
# =========================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def get_best_model():
    """
    Динамически выбирает лучшую доступную модель и проверяет поддержку JSON mode.
    """
    try:
        # Получаем список имен всех моделей, поддерживающих генерацию
        models_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Ищем модель 1.5 Flash по частичному совпадению имени
        flash_model = next((m for m in models_list if 'gemini-1.5-flash' in m), None)
        pro_model = next((m for m in models_list if 'gemini-pro' in m), None)

        if flash_model:
            logging.info(f"--- Выбрана модель: {flash_model} (JSON Mode ON) ---")
            return genai.GenerativeModel(flash_model), True
        elif pro_model:
            logging.info(f"--- Выбрана модель: {pro_model} (JSON Mode OFF) ---")
            return genai.GenerativeModel(pro_model), False
        else:
            raise Exception(f"Совместимые модели не найдены в списке: {models_list}")
    except Exception as e:
        if "API key was reported as leaked" in str(e):
            logging.critical("⚠️ КРИТИЧЕСКАЯ ОШИБКА: Ваш API-ключ заблокирован из-за утечки!")
            logging.critical("1. Создайте новый ключ: https://aistudio.google.com/app/apikey")
            logging.critical("2. Обновите GEMINI_API_KEY в файле .env")
            logging.critical("3. Добавьте .env в .gitignore")
        else:
            logging.error(f"Ошибка при выборе модели: {e}. Используем базовый fallback.")
        return genai.GenerativeModel('gemini-1.5-flash'), True

model, supports_json_mode = get_best_model()
logging.info(f"Текущая активная модель: {model.model_name}")

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=US+Iran",
    "https://news.google.com/rss/search?q=Hormuz",
    "https://www.google.com/alerts/feeds/08581651676967390390/13531311268805037202", 
    "https://www.google.com/alerts/feeds/08581651676967390390/9151711154181076810", 
    "https://www.google.com/alerts/feeds/08581651676967390390/9151711154181074984", 
    "https://www.google.com/alerts/feeds/08581651676967390390/17504039104683303894",
]

# Интервалы сканирования
CHECK_INTERVAL = 300  # Увеличим до 5 минут, чтобы не спамить запросами
COOLDOWN = 600        # Кулдаун уведомлений
LEARNING_INTERVAL = 3600 # Обучение раз в час
CLEANUP_INTERVAL = 86400 
RETENTION_DAYS = 14      # 30 дней — много для локальной БД, сократим до 14

# Динамическая задержка AI: 4 сек для Flash (15 RPM), 32 сек для Pro (2 RPM)
AI_DELAY = 4 if supports_json_mode else 32

# Параметры затухания: сделаем более агрессивным (0.95), так как баллы 
# в логах (450+) растут слишком быстро.
DECAY_FACTOR = 0.95 

# =========================
# STATE
# =========================

def init_state():
    scores = defaultdict(float)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Используем экспоненциальное затухание прямо в SQL (SQLite не имеет exp(), 
        # поэтому аппроксимируем через время или просто берем взвешенное среднее)
        # Здесь: чем старее запись, тем меньше её вклад в начальный score.
        cursor.execute("""
            SELECT entities, SUM(weighted_score) FROM (
                SELECT 
                    score * (1.0 - (julianday('now') - julianday(timestamp))) as weighted_score,
                    CASE WHEN oil != 'bearish' THEN 'OIL' ELSE 'GLOBAL' END as entities 
                FROM events WHERE timestamp > datetime('now', '-1 day')
            ) GROUP BY entities
        """)
        for key, val in cursor.fetchall():
            scores[key] = val
    return scores

event_scores = init_state()
event_last_sent = {}
learning_rate = 0.05

def load_weights():
    weights = {"US_IRAN": 2.5, "HORMUZ": 3.0, "OIL": 2.0, "GLOBAL": 1.0}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT event_key, weight FROM weights")
        for key, val in cursor.fetchall():
            weights[key] = val
    return weights

def save_weights():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for key, val in event_weights.items():
            cursor.execute("""
                INSERT INTO weights (event_key, weight) 
                VALUES (?, ?) 
                ON CONFLICT(event_key) DO UPDATE SET weight = excluded.weight
            """, (key, val))
        conn.commit()

event_weights = load_weights()
logging.info(f"--- Веса загружены: {event_weights} ---")

# =========================
# AI ENGINE
# =========================

def ai_analyze(text, max_retries=3):
    """
    Uses Gemini AI to perform deep sentiment analysis and NER.
    """
    prompt = f"""
    Analyze this financial news snippet: "{text}"
    Return ONLY a JSON object with this structure:
    {{
      "score": float (-10.0 to 10.0, where positive is risk-off/escalation, negative is risk-on/peace),
      "event_type": "military" | "economic" | "diplomatic" | "neutral",
      "entities": ["list of countries, companies or key regions"]
    }}
    Do not include any markdown formatting or explanations.
    """

    for attempt in range(max_retries):
        try:
            # Используем JSON Mode только если модель его поддерживает (1.5+)
            gen_config = {"response_mime_type": "application/json"} if supports_json_mode else {}
            
            response = model.generate_content(prompt, generation_config=gen_config)
            
            # Проверка, не заблокирован ли ответ фильтрами безопасности
            if not response.candidates or not response.candidates[0].content.parts:
                return 0.0, "neutral", []

            res_text = response.text.strip()
            
            # Надежный поиск границ JSON (на случай, если модель добавила текст)
            start = res_text.find('{')
            end = res_text.rfind('}') + 1

            if start == -1 or end == 0:
                return 0.0, "neutral", []

            data = json.loads(res_text[start:end])
            return float(data.get("score", 0)), data.get("event_type", "neutral"), data.get("entities", [])

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = (attempt + 1) * 30 # Увеличим время ожидания при ошибке
                logging.warning(f"⚠️ Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            logging.error(f"AI Analysis Error: {e}")
            break

    # Fallback logic
    if any(word in text.lower() for word in ["war", "strike", "attack"]):
        return 4.0, "military", ["Unknown"]
    return 0.0, "neutral", []

# =========================
# EVENT ENGINE
# =========================

def make_event_key(entities):
    if not entities:
        return "GLOBAL"

    if "Iran" in entities and "US" in entities:
        return "US_IRAN"

    if "Hormuz" in entities:
        return "HORMUZ"

    if "Oil" in entities:
        return "OIL"

    return "_".join(sorted(entities))

# =========================
# MARKET SIGNAL ENGINE
# =========================

def market_signals(score, event_key):
    mult = event_weights.get(event_key, 1.0)

    intensity = score * mult

    return {
        "nasdaq": "bearish" if intensity > 3 else "bullish" if intensity < -2 else "flat",
        "oil": "bullish" if intensity > 2 else "bearish",
        "soxs": "bullish" if intensity > 3 else "bearish",
        "vix": "bullish" if intensity > 2 else "flat"
    }

# =========================
# WEIGHT / IMPACT MODEL
# =========================

def get_weight(event_key):
    return event_weights.get(event_key, 1.0)


def predict_impact(score, event_key):
    return min(abs(score) * get_weight(event_key) * 12, 100)

# =========================
# SIGNAL ENGINE
# =========================

def generate_signal(prob, score):
    if prob > 70 and score > 0:
        return "🔴 HIGH RISK-OFF"
    elif prob > 40:
        return "🟠 MEDIUM RISK"
    elif score < 0:
        return "🟢 RISK-ON"
    else:
        return "⚪ NEUTRAL"

# =========================
# TELEGRAM
# =========================

def send_telegram(msg):
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=10
        )
        logging.info(f"TELEGRAM: {response.text}")
    except Exception as e:
        logging.error(f"Error sending telegram: {e}")

# =========================
# ANTI-SPAM
# =========================

def should_send(key):
    now = time.time()

    if key not in event_last_sent:
        event_last_sent[key] = now
        return True

    if now - event_last_sent[key] > COOLDOWN:
        event_last_sent[key] = now
        return True

    return False

# =========================
# LEARNING SYSTEM
# =========================

def update_weights(event_key, predicted, actual):
    error = actual - predicted

    event_weights[event_key] = event_weights.get(event_key, 1.0)
    event_weights[event_key] += learning_rate * error * 0.01

    # clamp
    event_weights[event_key] = max(0.5, min(5.0, event_weights[event_key]))


def get_market_data():
    """
    Fetches recent market data for key assets using yfinance.
    Returns a dictionary with percentage changes for relevant assets.
    """
    market_data = {}
    end_date = datetime.now()
    # Берем запас в 5 дней, чтобы гарантированно захватить торговые сессии (учитывая выходные)
    start_date = end_date - timedelta(days=5)

    tickers_to_fetch = {
        "^IXIC": "nasdaq_change", # NASDAQ Composite
        "CL=F": "oil_change",    # Crude Oil Futures
        "^VIX": "vix_change"     # CBOE Volatility Index
    }

    for ticker_symbol, data_key in tickers_to_fetch.items():
        try:
            # Download daily data, suppress progress bar
            data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1d", progress=False)
            if not data.empty and len(data) >= 2:
                # Берем две последние закрытые сессии
                # Используем .values.flatten(), чтобы избежать проблем с MultiIndex в новых версиях yf
                closes = data['Close'].values.flatten()
                yesterday_close = float(closes[-2])
                today_close = float(closes[-1])
                
                if yesterday_close != 0:
                    change = ((today_close - yesterday_close) / yesterday_close) * 100
                    market_data[data_key] = float(change)
            else:
                logging.warning(f"Warning: Not enough data for {ticker_symbol} to calculate daily change.")
        except Exception as e:
            logging.error(f"Error fetching data for {ticker_symbol}: {e}")

    return market_data


def learning_cycle():
    raw_market_data = get_market_data()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE resolved = 0")
        rows = cursor.fetchall()

    for row in rows:
        event_key = row[1]
        predicted = row[3]

        actual = 0 # Default actual move if no specific market data applies

        # Define how each event_key maps to actual market moves
        # Scaling factor: a 10% market move (e.g., VIX or Oil) corresponds to 100 impact score
        scaling_factor = 10

        if event_key == "OIL":
            if 'oil_change' in raw_market_data:
                actual = min(abs(raw_market_data['oil_change']) * scaling_factor, 100)
        elif event_key in ["US_IRAN", "HORMUZ", "GLOBAL"]:
            # For general risk events, prioritize VIX change, then inverted Nasdaq change
            if 'vix_change' in raw_market_data:
                actual = min(abs(raw_market_data['vix_change']) * scaling_factor, 100)
            elif 'nasdaq_change' in raw_market_data:
                # A drop in Nasdaq (negative change) means positive risk-off impact
                actual = min(abs(raw_market_data['nasdaq_change']) * scaling_factor, 100)

        update_weights(event_key, predicted, actual)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions
                SET resolved = 1, actual_move = ?
                WHERE id = ?
            """, (actual, row[0]))
            conn.commit()

    save_weights()

def cleanup_db():
    """
    Удаляет записи из БД, которые старше RETENTION_DAYS, чтобы предотвратить разрастание файла.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Удаляем старые события и прогнозы
            cursor.execute("DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')", (RETENTION_DAYS,))
            cursor.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-' || ? || ' days')", (RETENTION_DAYS,))
            conn.commit()  # Завершаем транзакцию после удаления
            
            # VACUUM пересобирает базу, освобождая место на диске
            old_isolation = conn.isolation_level
            conn.isolation_level = None  # Включаем autocommit для VACUUM
            conn.execute("VACUUM")
            conn.isolation_level = old_isolation
            
            logging.info(f"--- База данных оптимизирована: удалены данные старше {RETENTION_DAYS} дней ---")
    except Exception as e:
        logging.error(f"Ошибка при очистке БД: {e}")

# =========================
# MAIN LOOP
# =========================

last_learning_run = 0
last_cleanup_run = 0

while True:
    logging.info("GTS 4.0 scanning...")

    # Применяем затухание ко всем накопленным баллам перед новым сканированием
    for key in event_scores:
        event_scores[key] *= DECAY_FACTOR

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logging.error(f"Feed error {url}: {e}")
            continue

        # Ограничим количество новостей за один проход (max 5), 
        # чтобы не упираться в лимиты API Gemini Pro
        for entry in feed.entries[:5]:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM events WHERE link = ?", (entry.link,))
                if cursor.fetchone():
                    continue

            text = entry.title + " " + entry.get("summary", "")
            score, event_type, entities = ai_analyze(text)
            
            # Адаптивный Throttling
            logging.info(f"Waiting {AI_DELAY}s for next AI call...")
            time.sleep(AI_DELAY)

            event_key = make_event_key(entities)
            event_scores[event_key] += score
            market = market_signals(score, event_key)
            prob = predict_impact(event_scores[event_key], event_key)
            signal = generate_signal(prob, event_scores[event_key])

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (title, link, score, event, nasdaq, oil, soxs, vix)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.title, entry.link, score, event_type,
                    market["nasdaq"], market["oil"], market["soxs"], market["vix"]
                ))
                
                cursor.execute("""
                    INSERT INTO predictions (event_key, score, predicted_impact)
                    VALUES (?, ?, ?)
                """, (event_key, event_scores[event_key], prob))
                conn.commit()

            # TELEGRAM
            if should_send(event_key):

                msg = f"""
🧠 EVENT: {event_key}
🤖 Model: {model.model_name}

🚨 SIGNAL: {signal}

📊 Score: {event_scores[event_key]}
📈 Impact: {prob}%

📉 Nasdaq: {market['nasdaq']}
🛢 Oil: {market['oil']}
⚡ SOXS: {market['soxs']}
📊 VIX: {market['vix']}

📰 {entry.title}

🔗 {entry.link}
"""

                send_telegram(msg)

    # learning cycle
    current_time = time.time()
    if current_time - last_learning_run >= LEARNING_INTERVAL:
        learning_cycle()
        last_learning_run = current_time

    # Цикл очистки (запускается раз в сутки)
    if current_time - last_cleanup_run >= CLEANUP_INTERVAL:
        cleanup_db()
        last_cleanup_run = current_time

    time.sleep(CHECK_INTERVAL)