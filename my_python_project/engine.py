import feedparser
import logging
import re
import requests
import time
import os
import json
import google.generativeai as genai
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from db import get_db_connection, init_db
import config

init_db()

# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Инициализируем пул потоков для асинхронных задач (например, Telegram)
# 2-4 потока вполне достаточно для уведомлений
telegram_executor = ThreadPoolExecutor(max_workers=4)

# =========================
# CONFIG
# =========================

genai.configure(api_key=config.GEMINI_API_KEY)

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
AI_DELAY = config.AI_DELAY_JSON if supports_json_mode else config.AI_DELAY_NO_JSON
logging.info(f"Текущая активная модель: {model.model_name}")

# =========================
# STATE
# =========================

def init_state() -> Dict[str, float]:
    scores = defaultdict(float)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Восстанавливаем состояние из таблицы predictions, где уже есть event_key
        cursor.execute("""
            SELECT event_key, SUM(weighted_score) FROM (
                SELECT 
                    score * (1.0 - (julianday('now') - julianday(timestamp))) as weighted_score,
                    event_key
                FROM predictions WHERE timestamp > datetime('now', '-1 day')
            ) GROUP BY event_key
        """)
        for key, val in cursor.fetchall():
            # Ограничиваем начальное состояние порогом
            scores[key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, val))
    return scores

event_scores = init_state()
event_last_sent = {}
learning_rate = 0.05

def load_weights() -> Dict[str, float]:
    weights = {"US_IRAN": 2.5, "HORMUZ": 3.0, "OIL": 2.0, "GOLD": 1.5, "BTC": 1.2, "GLOBAL": 1.0}
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

def ai_analyze(text: str, max_retries: int = 3) -> Tuple[Optional[float], Optional[str], Optional[List[str]], str]:
    """
    Uses Gemini AI to perform deep sentiment analysis and NER.
    """
    prompt = f"""
    Analyze this financial news snippet: "{text}"
    Identify key entities. Use these standardized tags if applicable: "US", "Iran", "Hormuz", "Oil", "Gold", "Bitcoin".
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
                return None, None, None

            res_text = response.text.strip()
            
            # Надежный поиск границ JSON (на случай, если модель добавила текст)
            start = res_text.find('{')
            end = res_text.rfind('}') + 1

            if start == -1 or end == 0:
                return None, None, None

            data = json.loads(res_text[start:end])
            return float(data.get("score", 0)), data.get("event_type", "neutral"), data.get("entities", []), "AI"

        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "limit" in err_msg:
                wait_time = (attempt + 1) * 120 # Увеличиваем ожидание (2, 4, 6 минут)
                logging.warning(f"⚠️ Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            logging.error(f"AI Analysis Error: {e}")
            break

    # Fallback logic
    text_low = text.lower()
    found_entities = []
    # Используем регулярные выражения для поиска целых слов (\b)
    if re.search(r'\biran\b', text_low): found_entities.append("Iran")
    if re.search(r'\b(us|usa)\b', text_low): found_entities.append("US")
    if re.search(r'\bhormuz\b', text_low): found_entities.append("Hormuz")
    if re.search(r'\boil\b', text_low): found_entities.append("Oil")
    if re.search(r'\b(bitcoin|btc)\b', text_low): found_entities.append("Bitcoin")
    if re.search(r'\b(gold|xau)\b', text_low): found_entities.append("Gold")
    
    # Улучшенный скоринг в фоллбеке
    score = 4.0 if re.search(r'\b(war|strike|attack|conflict|escalation)\b', text_low) else 0.0
    return score, "neutral", found_entities, "Fallback"

# =========================
# EVENT ENGINE
# =========================

def make_event_key(entities: List[str]) -> str:
    if not entities or "Unknown" in entities:
        return "GLOBAL"

    # Приводим всё к нижнему регистру для поиска
    ents_low = [e.lower() for e in entities]
    ents_str = " ".join(ents_low)

    if "iran" in ents_str and ("us" in ents_str or "usa" in ents_str):
        return "US_IRAN"

    if "hormuz" in ents_str:
        return "HORMUZ"

    if "oil" in ents_str:
        return "OIL"

    if "gold" in ents_str or "xau" in ents_str:
        return "GOLD"

    if "bitcoin" in ents_str or "btc" in ents_str:
        return "BTC"

    return "_".join(sorted(list(set(entities)))) # Убираем дубликаты и сортируем

# =========================
# MARKET SIGNAL ENGINE
# =========================

def market_signals(score: float, event_key: str) -> Dict[str, str]:
    mult = event_weights.get(event_key, 1.0)

    intensity = score * mult

    return {
        "nasdaq": "bearish" if intensity > 3 else "bullish" if intensity < -2 else "flat",
        "oil": "bullish" if intensity > 2 else "bearish",
        "soxs": "bullish" if intensity > 3 else "bearish",
        "vix": "bullish" if intensity > 2 else "flat",
        "gold": "bullish" if intensity > 1.5 else "bearish" if intensity < -3 else "flat",
        "btc": "bearish" if intensity > 4 else "bullish" if intensity < -2 else "flat"
    }

# =========================
# WEIGHT / IMPACT MODEL
# =========================

def get_weight(event_key: str) -> float:
    return event_weights.get(event_key, 1.0)


def predict_impact(score: float, event_key: str) -> float:
    return min(abs(score) * get_weight(event_key) * 12, 100)

# =========================
# SIGNAL ENGINE
# =========================

def generate_signal(prob: float, score: float) -> str:
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

def send_telegram(msg: str):
    """Отправляет сообщение в Telegram асинхронно через ThreadPoolExecutor."""
    def _send():
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{config.BOT_TOKEN}/sendMessage",
                data={"chat_id": config.CHAT_ID, "text": msg},
                timeout=10
            )
            logging.info(f"TELEGRAM ASYNC: {response.status_code}")
        except Exception as e:
            logging.error(f"Error sending telegram in background thread: {e}")

    telegram_executor.submit(_send)

# =========================
# ANTI-SPAM
# =========================

def should_send(key: str) -> bool:
    now = time.time()

    if key not in event_last_sent:
        event_last_sent[key] = now
        return True

    if now - event_last_sent[key] > config.COOLDOWN:
        event_last_sent[key] = now
        return True

    return False

# =========================
# LEARNING SYSTEM
# =========================

def update_weights(event_key: str, predicted: float, actual: float):
    error = actual - predicted

    event_weights[event_key] = event_weights.get(event_key, 1.0)
    event_weights[event_key] += learning_rate * error * 0.01

    # clamp
    event_weights[event_key] = max(0.5, min(5.0, event_weights[event_key]))

def get_fear_greed_index() -> Tuple[Optional[float], Optional[str], float]:
    """
    Получает Fear & Greed Index. 
    Используем API alternative.me как надежный источник сентимента.
    """
    try:
        # Запрашиваем данные за 2 дня, чтобы вычислить изменение
        response = requests.get("https://api.alternative.me/fng/?limit=2", timeout=10)
        data = response.json()
        today_val = float(data['data'][0]['value'])
        yesterday_val = float(data['data'][1]['value'])
        label = data['data'][0]['value_classification']
        change = today_val - yesterday_val
        return today_val, label, change
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed: {e}")
        return None, None, 0

def get_market_data() -> Dict[str, Any]:
    """
    Fetches recent market data for key assets using yfinance.
    Returns a dictionary with percentage changes for relevant assets.
    """
    market_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    tickers_to_fetch = {
        "^IXIC": "nasdaq_change",
        "CL=F": "oil_change",
        "^VIX": "vix_change",
        "GC=F": "gold_change",
        "BTC-USD": "btc_change",
        "SOXS": "soxs_change"
    }

    try:
        # Оптимизация: загружаем все тикеры одним запросом
        all_data = yf.download(list(tickers_to_fetch.keys()), start=start_date, end=end_date, interval="1d", progress=False)
        
        for ticker_symbol, data_key in tickers_to_fetch.items():
            try:
                if ticker_symbol in all_data['Close'].columns:
                    ticker_data = all_data['Close'][ticker_symbol].dropna()
                    if len(ticker_data) >= 2:
                        yesterday_close = float(ticker_data.iloc[-2])
                        today_close = float(ticker_data.iloc[-1])
                        if yesterday_close != 0:
                            market_data[data_key] = ((today_close - yesterday_close) / yesterday_close) * 100
                else:
                    logging.warning(f"Ticker {ticker_symbol} missing in downloaded data")
            except Exception as e:
                logging.debug(f"Error processing {ticker_symbol}: {e}")
    except Exception as e:
        logging.error(f"Global yfinance error: {e}")

    # Добавляем Fear & Greed
    fng_val, fng_label, fng_change = get_fear_greed_index()
    if fng_val is not None:
        market_data['fng_val'] = fng_val
        market_data['fng_label'] = fng_label
        market_data['fng_change'] = fng_change

    return market_data


def learning_cycle():
    raw_market_data = get_market_data()
    if not raw_market_data:
        logging.warning("Skipping learning cycle: No market data available.")
        return

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Берем только последние 100 неразрешенных прогнозов, чтобы не перегружать цикл
        cursor.execute("SELECT * FROM predictions WHERE resolved = 0 ORDER BY timestamp DESC LIMIT 100")
        rows = cursor.fetchall()

        for row in rows:
            event_key = row['event_key']
            predicted = row['predicted_impact']
            actual = 0

            if event_key == "OIL" and 'oil_change' in raw_market_data:
                actual = min(abs(raw_market_data['oil_change']) * config.SCALING_FACTOR, 100)
            elif event_key == "GOLD" and 'gold_change' in raw_market_data:
                actual = min(abs(raw_market_data['gold_change']) * config.SCALING_FACTOR, 100)
            elif event_key == "BTC" and 'btc_change' in raw_market_data:
                actual = min(abs(raw_market_data['btc_change']) * (config.SCALING_FACTOR / 2), 100)
            elif event_key in ["US_IRAN", "HORMUZ", "GLOBAL"]:
                # Учитываем также изменение индекса страха (падение индекса = рост страха)
                fng_impact = abs(raw_market_data.get('fng_change', 0)) * 2
                if 'vix_change' in raw_market_data:
                    vix_impact = abs(raw_market_data['vix_change']) * config.SCALING_FACTOR
                    actual = min((vix_impact + fng_impact) / 2, 100)
                elif 'nasdaq_change' in raw_market_data:
                    actual = min(abs(raw_market_data['nasdaq_change']) * config.SCALING_FACTOR, 100)

            logging.info(f"Resolving prediction for {event_key}: Predicted {predicted:.2f}, Actual {actual:.2f}")
            update_weights(event_key, predicted, actual)

            cursor.execute("""
                UPDATE predictions
                SET resolved = 1, actual_move = ?
                WHERE id = ?
            """, (actual, row['id']))
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
            cursor.execute("DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
            cursor.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
            conn.commit()  # Завершаем транзакцию после удаления
            
            # VACUUM пересобирает базу, освобождая место на диске
            old_isolation = conn.isolation_level
            conn.isolation_level = None  # Включаем autocommit для VACUUM
            conn.execute("VACUUM")
            conn.isolation_level = old_isolation
            
            logging.info(f"--- База данных оптимизирована: удалены данные старше {config.RETENTION_DAYS} дней ---")
    except Exception as e:
        logging.error(f"Ошибка при очистке БД: {e}")

# =========================
# MAIN LOOP
# =========================

last_learning_run = 0 # Установка в 0 форсирует запуск обучения при первом проходе цикла
last_market_data_fetch = 0 # Для отслеживания времени последнего получения рыночных данных
last_cleanup_run = 0

while True:
    logging.info("GTS 4.0 scanning...")

    # Применяем затухание ко всем накопленным баллам перед новым сканированием
    for key in event_scores:
        event_scores[key] *= config.DECAY_FACTOR

    # Получаем актуальные рыночные данные один раз за цикл CHECK_INTERVAL
    current_market_data = get_market_data()
    btc_change_for_notification = current_market_data.get("btc_change", 0)
    gold_change_for_notification = current_market_data.get("gold_change", 0)
    soxs_change_for_notification = current_market_data.get("soxs_change", 0)
    fng_val = current_market_data.get("fng_val", 50)
    fng_label = current_market_data.get("fng_label", "Neutral")

    for url in config.RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logging.error(f"Feed error {url}: {e}")
            continue

        # Ограничиваем до 3 самых свежих новостей с каждого фида
        # Это экономит дневную квоту запросов (RPD)
        for entry in feed.entries[:3]:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM events WHERE link = ?", (entry.link,))
                if cursor.fetchone():
                    continue

            text = entry.title + " " + entry.get("summary", "")
            analysis = ai_analyze(text)
            
            if analysis[0] is None:
                logging.warning(f"Skipping entry due to AI failure: {entry.title[:50]}...")
                continue

            score, event_type, entities, source = analysis

            # Адаптивный Throttling
            logging.info(f"Waiting {AI_DELAY}s for next AI call...")
            time.sleep(AI_DELAY)

            event_key = make_event_key(entities)
            
            # Применяем инкремент с ограничением (Clamping)
            new_score = event_scores[event_key] + score
            event_scores[event_key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, new_score))
            
            market = market_signals(score, event_key)
            prob = predict_impact(event_scores[event_key], event_key)
            signal = generate_signal(prob, event_scores[event_key])

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (title, link, score, event, nasdaq, oil, soxs, gold, btc, vix, fear_greed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.title, entry.link, score, event_type,
                    market["nasdaq"], market["oil"], market["soxs"], market["gold"], market["btc"], market["vix"], fng_val
                ))
                
                cursor.execute("""
                    INSERT INTO predictions (event_key, score, predicted_impact)
                    VALUES (?, ?, ?)
                """, (event_key, event_scores[event_key], prob))
                conn.commit()

            # TELEGRAM
            # Дополнительная проверка для BTC: отправляем только при критических изменениях (>5%)
            if event_key == "BTC":
                if abs(btc_change_for_notification) < 5.0:
                    logging.info(f"Skipping BTC notification for '{entry.title}' - change ({btc_change_for_notification:.2f}%) not critical.")
                    continue # Пропускаем отправку уведомления для этой BTC новости

            if should_send(event_key):

                msg = f"""
🧠 EVENT: {event_key}
🤖 Model: {model.model_name} (Source: {source})

🚨 SIGNAL: {signal}

📊 Score: {event_scores[event_key]}
😨 Fear & Greed: {fng_val} ({fng_label})
📈 Impact: {prob}%

📉 Nasdaq: {market['nasdaq']}
🛢 Oil: {market['oil']}
⚡ SOXS: {market['soxs']} ({soxs_change_for_notification:+.2f}%)
✨ Gold: {market['gold']} ({gold_change_for_notification:+.2f}%)
₿ BTC: {market['btc']} ({btc_change_for_notification:+.2f}%)
📊 VIX: {market['vix']}

📰 {entry.title}

🔗 {entry.link}
"""

                send_telegram(msg)

    # learning cycle
    current_time = time.time()
    if current_time - last_learning_run >= config.LEARNING_INTERVAL:
        learning_cycle()
        last_learning_run = current_time

    # Цикл очистки (запускается раз в сутки)
    if current_time - last_cleanup_run >= config.CLEANUP_INTERVAL:
        cleanup_db()
        last_cleanup_run = current_time

    time.sleep(config.CHECK_INTERVAL)