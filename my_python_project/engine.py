import feedparser
import logging
import re
import time
import os
import json
import signal
import sys
import asyncio
import aiohttp
from google import genai
try:
    from transformers import pipeline
    # Используем легкую модель для сентимента (около 250МБ)
    local_sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Пул потоков для синхронных библиотек (feedparser, yfinance)
sync_executor = ThreadPoolExecutor(max_workers=5)

def shutdown_cleanup():
    """Выполняет очистку ресурсов при завершении работы."""
    logging.info("Закрытие соединений и остановка фоновых задач...")
    sync_executor.shutdown(wait=True) # Дожидаемся завершения всех задач в пуле
    logging.info("GTS 4.0 остановлен.")

# =========================
# CONFIG
# =========================

client = genai.Client(api_key=config.GEMINI_API_KEY)

def init_model_pool():
    """Инициализирует список доступных моделей Gemini для ротации при 429 ошибке."""
    pool = []
    try:
        all_models = list(client.models.list())
        models_list = [m.name for m in all_models if 'generateContent' in m.supported_actions]
        
        logging.info(f"Доступные модели в API: {len(models_list)}")

        # Маппинг семейств и их приоритетов (1 - высший)
        family_priority = {
            'gemini-3.1-flash': 1,
            'gemini-3.1-pro': 2,
            'gemini-3-flash': 3,
            'gemini-3-pro': 4,
            'gemini-2.5-flash': 5,
            'gemini-2.5-pro': 6,
            'gemini-2.0-flash': 7,
            'gemini-1.5-flash': 8,
            'gemini-1.5-pro': 9,
            'gemini-1.0-pro': 10,
            'gemini-flash-latest': 11,
            'gemini-pro-latest': 12
        }
        
        found_families = {} # family_name -> best_model_data

        for m_name in models_list:
            # Исключаем специализированные модели (аудио, видео, робототехника, встраивание),
            # которые не поддерживают JSON mode или не предназначены для анализа текста.
            if any(spec in m_name.lower() for spec in ['-tts', '-image', 'robotics', 'clip', 'embed']):
                continue

            for fam, priority in family_priority.items():
                # Ищем вхождение семейства в имя (например, 'gemini-1.5-flash' в 'models/gemini-1.5-flash-latest')
                if fam in m_name:
                    # Мы берем только самую "короткую" версию имени для каждого семейства 
                    # (обычно это базовая модель, а не специфический билд типа -001)
                    if fam not in found_families or len(m_name) < len(found_families[fam]['name']):
                        found_families[fam] = {
                            "name": m_name,
                            "priority": priority,
                            "supports_json": any(v in m_name for v in ["1.5", "2.0", "2.5", "3", "latest"])
                        }

        # Сортируем по приоритету и наполняем пул
        sorted_pool = sorted(found_families.values(), key=lambda x: x['priority'])
        for m_data in sorted_pool:
            pool.append({
                "name": m_data["name"],
                "supports_json": m_data["supports_json"]
            })
            logging.info(f"✅ Добавлена в пул ротации: {m_data['name']} (Приоритет {m_data['priority']})")

        if len(pool) < 2:
            logging.warning(f"⚠️ Мало семейств в пуле. Проверьте доступность 1.5 моделей. Доступные имена: {models_list}")

    except Exception as e:
        if "API key was reported as leaked" in str(e):
            logging.critical("⚠️ КРИТИЧЕСКАЯ ОШИБКА: Ваш API-ключ заблокирован из-за утечки!")
            logging.critical("1. Создайте новый ключ: https://aistudio.google.com/app/apikey")
            logging.critical("2. Обновите GEMINI_API_KEY в файле .env")
            logging.critical("3. Добавьте .env в .gitignore")
    
    if not pool:
        # Запасной вариант
        pool.append({
            "name": "models/gemini-1.5-flash",
            "supports_json": True
        })
    return pool

model_pool = init_model_pool()
current_model_idx = 0

def get_active_model():
    return model_pool[current_model_idx]

logging.info(f"Пул моделей готов: {[m['name'] for m in model_pool]}. Старт с: {get_active_model()['name']}")

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

def local_ai_analyze(text: str) -> Tuple[float, str, List[str], str]:
    """
    Бесплатная локальная замена Gemini с использованием Hugging Face.
    """
    try:
        # Анализ сентимента
        result = local_sentiment_pipe(text[:512])[0]
        # Преобразуем POSITIVE/NEGATIVE в шкалу от -10 до 10
        score = 5.0 if result['label'] == 'POSITIVE' else -5.0
        if result['score'] > 0.9: score *= 1.5 # Усиливаем уверенность

        # Простой NER на базе ключевых слов (как в фоллбеке, но интегрированный)
        found_entities = []
        text_low = text.lower()
        mapping = {"iran": "Iran", "us": "US", "usa": "US", "hormuz": "Hormuz", "oil": "Oil", "btc": "Bitcoin", "bitcoin": "Bitcoin", "gold": "Gold"}
        for key, val in mapping.items():
            if re.search(rf'\b{key}\b', text_low):
                if val not in found_entities: found_entities.append(val)
        
        # Определение типа (эвристика)
        event_type = "military" if any(x in text_low for x in ["war", "strike", "attack", "army"]) else "economic"
        
        return float(score), event_type, found_entities, "Local (HuggingFace)"
    except Exception as e:
        logging.error(f"Local AI Error: {e}")
        return 0.0, "neutral", [], "Error"

async def ai_analyze(text: str, max_retries: int = 3) -> Tuple[Optional[float], Optional[str], Optional[List[str]], str]:
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

    global current_model_idx

    for attempt in range(max_retries):
        model_tried_count = 0
        while model_tried_count < len(model_pool): # Внутренний цикл для перебора моделей в пуле
            try:
                active = model_pool[current_model_idx]
                # Используем JSON Mode только если текущая модель из пула его поддерживает
                gen_config = {"response_mime_type": "application/json"} if active["supports_json"] else {}
                
                response = await client.aio.models.generate_content(
                    model=active["name"],
                    contents=prompt,
                    config=gen_config
                )
                
                # Проверка, не заблокирован ли ответ фильтрами безопасности
                if not response.candidates or not response.candidates[0].content.parts:
                    # Это не 429, но и не успешный ответ. Считаем, что модель не справилась.
                    logging.warning(f"Модель {active['name']} заблокировала ответ по безопасности. Переключаюсь на следующую.")
                    model_tried_count += 1
                    current_model_idx = (current_model_idx + 1) % len(model_pool)
                    continue # Попробуем следующую модель в пуле немедленно

                res_text = response.text.strip()
                
                # Надежный поиск границ JSON (на случай, если модель добавила текст)
                start = res_text.find('{')
                end = res_text.rfind('}') + 1

                if start == -1 or end == 0:
                    logging.warning(f"Модель {active['name']} вернула невалидный JSON. Получено: {res_text[:100]}...")
                    # Это не 429, но и не успешный ответ. Считаем, что модель не справилась.
                    model_tried_count += 1
                    current_model_idx = (current_model_idx + 1) % len(model_pool)
                    continue # Попробуем следующую модель в пуле немедленно

                data = json.loads(res_text[start:end])
                return float(data.get("score", 0)), data.get("event_type", "neutral"), data.get("entities", []), active["name"]

            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "limit" in err_msg:
                    old_name = model_pool[current_model_idx]["name"]
                    current_model_idx = (current_model_idx + 1) % len(model_pool)
                    model_tried_count += 1
                    logging.warning(f"⚠️ Лимит {old_name} исчерпан. Переключаюсь на {model_pool[current_model_idx]['name']} (попытка {model_tried_count}/{len(model_pool)} в текущем цикле).")
                    if model_tried_count == len(model_pool):
                        break # Все модели в пуле исчерпали лимит, выходим из внутреннего цикла
                    continue # Пробуем следующую модель в пуле немедленно
                else:
                    # Другая ошибка (например, модель не поддерживает JSON mode). 
                    # Логируем, переключаемся на следующую модель и пробуем снова в этом же цикле.
                    logging.error(f"⚠️ Ошибка модели {active['name']}: {e}")
                    model_tried_count += 1
                    current_model_idx = (current_model_idx + 1) % len(model_pool)
                    continue # Пробуем следующую модель в пуле немедленно
        
        # Если мы дошли сюда, значит, ни одна модель не смогла успешно обработать запрос
        # в текущем цикле (либо все 429, либо другая ошибка).

                wait_time = (attempt + 1) * 120
                logging.warning(f"⚠️ Все модели в пуле исчерпали лимит или произошла другая ошибка. Повторная попытка через {wait_time}s... (Попытка {attempt+1}/{max_retries})")
                # Если это не первая попытка и всё еще 429, пробуем локальную модель вместо ожидания
                if attempt >= 1 and HAS_TRANSFORMERS:
                    logging.info("Switching to Local AI due to rate limits...")
                    return local_ai_analyze(text)
                await asyncio.sleep(wait_time)

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
    is_critical = re.search(r'\b(war|strike|attack|conflict|escalation|sanctions|emergency)\b', text_low)
    score = 4.0 if is_critical else 0.0

    # Если это не критично и сущности не найдены — лучше пропустить анализ, чем гадать
    if not found_entities and score == 0:
        return None, None, None, "No Relevance"
    
    return score, "neutral", found_entities, "Fallback (Regex)"

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

async def send_telegram(session: aiohttp.ClientSession, msg: str):
    """Отправляет сообщение в Telegram асинхронно."""
    try:
        async with session.post(
                f"https://api.telegram.org/bot{config.BOT_TOKEN}/sendMessage",
                data={"chat_id": config.CHAT_ID, "text": msg},
                timeout=10
        ) as response:
            logging.info(f"TELEGRAM ASYNC: {response.status}")
    except Exception as e:
        logging.error(f"Error sending telegram: {e}")

# =========================
# ANTI-SPAM
# =========================

def should_send(key: str, current_score: float) -> bool:
    now = time.time()

    # Если новость экстремально важная (например, score > 8), игнорируем кулдаун
    if abs(current_score) >= 8.0:
        return True

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

async def get_fear_greed_index(session: aiohttp.ClientSession) -> Tuple[Optional[float], Optional[str], float]:
    """
    Получает Fear & Greed Index. 
    Используем API alternative.me как надежный источник сентимента.
    """
    try:
        async with session.get("https://api.alternative.me/fng/?limit=2", timeout=10) as response:
            if response.status != 200:
                return None, None, 0
            data = await response.json()
        today_val = float(data['data'][0]['value'])
        yesterday_val = float(data['data'][1]['value'])
        label = data['data'][0]['value_classification']
        change = today_val - yesterday_val
        return today_val, label, change
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed: {e}")
        return None, None, 0

async def get_market_data(session: aiohttp.ClientSession) -> Dict[str, Any]:
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
        # yfinance синхронный, запускаем в экзекуторе
        loop = asyncio.get_event_loop()
        all_data = await loop.run_in_executor(
            sync_executor, 
            lambda: yf.download(list(tickers_to_fetch.keys()), start=start_date, end=end_date, interval="1d", progress=False)
        )
        
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
    fng_val, fng_label, fng_change = await get_fear_greed_index(session)
    if fng_val is not None:
        market_data['fng_val'] = fng_val
        market_data['fng_label'] = fng_label
        market_data['fng_change'] = fng_change

    return market_data


async def learning_cycle(session: aiohttp.ClientSession):
    raw_market_data = await get_market_data(session)
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

async def main():
    last_learning_run = 0
    last_cleanup_run = 0
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        while True:
            logging.info("GTS 4.0 scanning...")

            for key in event_scores:
                event_scores[key] *= config.DECAY_FACTOR

            current_market_data = await get_market_data(session)
            btc_change = current_market_data.get("btc_change", 0)
            gold_change = current_market_data.get("gold_change", 0)
            soxs_change = current_market_data.get("soxs_change", 0)
            fng_val = current_market_data.get("fng_val", 50)
            fng_label = current_market_data.get("fng_label", "Neutral")

            for url in config.RSS_FEEDS:
                try:
                    # feedparser блокирующий, запускаем в экзекуторе
                    feed = await loop.run_in_executor(sync_executor, lambda: feedparser.parse(url))
                except Exception as e:
                    logging.error(f"Feed error {url}: {e}")
                    continue

                for entry in feed.entries[:3]:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM events WHERE link = ?", (entry.link,))
                        if cursor.fetchone():
                            continue

                    text = entry.title + " " + entry.get("summary", "")
                    analysis = await ai_analyze(text)
                    
                    if analysis[0] is None:
                        continue

                    score, event_type, entities, source = analysis
                    
                    # Динамическая задержка: 15с для Flash (JSON), 60с для Pro (No JSON)
                    current_delay = config.AI_DELAY_JSON if get_active_model()["supports_json"] else config.AI_DELAY_NO_JSON
                    await asyncio.sleep(current_delay)

                    event_key = make_event_key(entities)
                    event_scores[event_key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, event_scores[event_key] + score))
                    
                    market = market_signals(score, event_key)
                    prob = predict_impact(event_scores[event_key], event_key)
                    sig_type = generate_signal(prob, event_scores[event_key])

                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO events (title, link, score, event, nasdaq, oil, soxs, gold, btc, vix, fear_greed)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (entry.title, entry.link, score, event_type, market["nasdaq"], market["oil"], market["soxs"], market["gold"], market["btc"], market["vix"], fng_val))
                        cursor.execute("INSERT INTO predictions (event_key, score, predicted_impact) VALUES (?, ?, ?)", (event_key, event_scores[event_key], prob))
                        conn.commit()

                    if should_send(event_key, score):
                        if event_key == "BTC" and abs(btc_change) < 5.0:
                            continue

                        msg = f"""
🧠 EVENT: {event_key}
🤖 Model: {source}
� SIGNAL: {sig_type}
 Score: {event_scores[event_key]}
😨 Fear & Greed: {fng_val} ({fng_label})
📈 Impact: {prob}%
📉 Nasdaq: {market['nasdaq']}
🛢 Oil: {market['oil']}
⚡ SOXS: {market['soxs']} ({soxs_change:+.2f}%)
✨ Gold: {market['gold']} ({gold_change:+.2f}%)
₿ BTC: {market['btc']} ({btc_change:+.2f}%)
📊 VIX: {market['vix']}
📰 {entry.title}
🔗 {entry.link}
"""
                        await send_telegram(session, msg)

            current_time = time.time()
            if current_time - last_learning_run >= config.LEARNING_INTERVAL:
                await learning_cycle(session)
                last_learning_run = current_time

            if current_time - last_cleanup_run >= config.CLEANUP_INTERVAL:
                cleanup_db()
                last_cleanup_run = current_time

            await asyncio.sleep(config.CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Получен сигнал завершения (Ctrl+C или системный).")
    except Exception as e:
        logging.error(f"Непредвиденная ошибка в основном цикле: {e}")
    finally:
        shutdown_cleanup()