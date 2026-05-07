import feedparser
import logging
import re
import time
import json
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
from datetime import datetime
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
learning_rate = config.LEARNING_RATE
# Карта связей: какой ключ к какому активу привязан для обучения
event_asset_map = {}

def load_weights() -> Dict[str, float]:
    # Создаем базовые веса из TRACKED_KEYWORDS, преобразуя ключи в формат EVENT_KEY (например, "US Iran" -> "US_IRAN")
    weights = {}
    for k, v in config.TRACKED_KEYWORDS.items():
        # Сортируем части ключа (например, "US Iran" -> "IRAN_US"), чтобы они всегда совпадали с выходом make_event_key
    for k, info in config.TRACKED_KEYWORDS.items():
        weight, target_asset = info if isinstance(info, tuple) else (info, "global")
        
        key_parts = sorted(k.upper().replace(" ", "_").split("_"))
        weights["_".join(key_parts)] = v
        canonical_key = "_".join(key_parts)
        
        weights[canonical_key] = weight
        event_asset_map[canonical_key] = target_asset
    
    # Синхронизация специфичных имен (если в конфиге Bitcoin, а в логике BTC)
    if "BITCOIN" in weights: weights["BTC"] = weights.pop("BITCOIN")
    if "GLOBAL" not in weights: weights["GLOBAL"] = 1.0

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT event_key, weight FROM weights")
        for key, val in cursor.fetchall():
            weights[key] = val
    return weights

def load_system_settings() -> float:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'")
        row = cursor.fetchone()
        if row:
            logging.info(f"✅ IMPACT_MULTIPLIER загружен из БД: {row[0]}")
            return row[0]
        logging.info(f"⚠️ Настройки в БД не найдены, использую default из config: {config.IMPACT_MULTIPLIER}")
        return config.IMPACT_MULTIPLIER

global_impact_multiplier = load_system_settings()

def save_state():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Сохраняем веса
        for key, val in event_weights.items():
            cursor.execute("""
                INSERT INTO weights (event_key, weight) 
                VALUES (?, ?) 
                ON CONFLICT(event_key) DO UPDATE SET weight = excluded.weight
            """, (key, val))
        
        # Сохраняем глобальный множитель
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('impact_multiplier', ?)", (global_impact_multiplier,))
        conn.commit()

event_weights = load_weights()
save_state()  # Автоматическая синхронизация конфига с базой данных при старте
logging.info(f"--- Веса загружены: {event_weights} ---")
logging.info(f"--- Текущий IMPACT_MULTIPLIER: {global_impact_multiplier:.2f} ---")

# =========================
# AI ENGINE
# =========================

def _get_fallback_entity_search_map() -> Dict[str, str]:
    """
    Generates a mapping from lowercase search terms (words/phrases) to
    canonical entity names (as expected by make_event_key) for fallback.
    """
    search_map = {}
    for phrase in config.TRACKED_KEYWORDS.keys():
        # Add the full phrase as a search term, mapping to itself
        search_map[phrase.lower()] = phrase

        # Split multi-word phrases into individual canonical entities if relevant
        words = phrase.split()
        if len(words) > 1:
            if phrase == "US Iran": # Special case for US Iran
                search_map["us"] = "US"
                search_map["usa"] = "US"
                search_map["iran"] = "Iran"
            # For other multi-word phrases, we generally want the full phrase as an entity
        
        # Add common aliases for single-word entities
        if phrase.lower() == "bitcoin":
            search_map["btc"] = "Bitcoin"
        if phrase.lower() == "gold":
            search_map["xau"] = "Gold"
        if phrase.lower() == "oil":
            search_map["cl=f"] = "Oil" # Futures symbol

    return search_map

fallback_entity_map = _get_fallback_entity_search_map()

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
        for key, val in fallback_entity_map.items():
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
    # Формируем строку с тегами из конфига для подсказки нейросети
    tags_hint = ", ".join([f'"{k}"' for k in config.TRACKED_KEYWORDS.keys()])
    prompt = f"""
    Analyze this financial news snippet: "{text}"
    Identify key entities. Use these standardized tags if applicable: {tags_hint}.
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
    for search_term, canonical_name in fallback_entity_map.items():
        if re.search(rf'\b{re.escape(search_term)}\b', text_low):
        if re.search(r'\b' + re.escape(search_term) + r'\b', text_low):
            if canonical_name not in found_entities:
                found_entities.append(canonical_name)
    
    # Улучшенный скоринг в фоллбеке
    is_critical = re.search(r'\b(war|strike|attack|conflict|escalation|sanctions|emergency)\b', text_low)
    score = 4.0 if is_critical else 0.0

    # Если это не критично и сущности не найдены — лучше пропустить анализ, чем гадать
    if not found_entities and score == 0:
        return None, None, None, "No Relevance"
    
    return score, "neutral", found_entities, "Fallback (Regex)"

async def run_global_research():
    """
    Просит ИИ проанализировать глобальные рынки и предложить ключевые слова,
    которые окажут наибольшее влияние на отслеживаемые активы.
    """
    assets = ["nasdaq", "oil", "soxs", "vix", "gold", "btc", "hbm"]
    prompt = f"""
    As a senior macro strategist, identify the top 15 global entities, geopolitical triggers, or economic factors 
    that will most significantly impact these assets over the next 30 days: {assets}.
    
    Return ONLY a JSON list of objects:
    [
      {{
        "keyword": "Entity Name (e.g., TSMC, Strait of Hormuz, US Jobless Claims)",
        "asset": "target asset from the list",
        "impact_direction": "bullish/bearish",
        "reasoning": "Short professional explanation of the correlation"
      }}
    ]
    """
    
    logging.info("--- Starting Global AI Research ---")
    try:
        active = get_active_model()
        gen_config = {"response_mime_type": "application/json"} if active["supports_json"] else {}
        
        response = await client.aio.models.generate_content(
            model=active["name"],
            contents=prompt,
            config=gen_config
        )
        
        res_text = response.text.strip()
        start = res_text.find('[')
        end = res_text.rfind(']') + 1
        suggestions = json.loads(res_text[start:end])

        with get_db_connection() as conn:
            cursor = conn.cursor()
            for s in suggestions:
                cursor.execute("""
                    INSERT INTO ai_global_suggestions (keyword, asset, impact_direction, reasoning)
                    VALUES (?, ?, ?, ?)
                """, (s['keyword'], s['asset'], s['impact_direction'], s['reasoning']))
            conn.commit()
        logging.info(f"✅ AI Global Research finished. Found {len(suggestions)} new suggestions.")
    except Exception as e:
        logging.error(f"Error during global research: {e}")

# =========================
# EVENT ENGINE
# =========================

def make_event_key(entities: List[str]) -> str:
    if not entities or "Unknown" in entities:
        return "GLOBAL"

    # Список известных макро-сущностей для нормализации
    canonical_map = {
        "USA": "US", "UNITED STATES": "US",
        "IRAN": "IRAN",
        "ISRAEL": "ISRAEL",
        "CHINA": "CHINA",
        "RUSSIA": "RUSSIA",
        "FED": "FED", "FEDERAL RESERVE": "FED",
        "BITCOIN": "BTC", "BTC": "BTC",
        "GOLD": "GOLD", "XAU": "GOLD",
        "OIL": "OIL", "CRUDE": "OIL",
        "NVDA": "NVIDIA", "NVIDIA": "NVIDIA"
    }

    # Автоматически добавляем все отслеживаемые слова из конфига в мапу нормализации
    for kw in config.TRACKED_KEYWORDS.keys():
        kw_up = kw.upper().replace(" ", "_")
        if kw_up not in canonical_map:
            canonical_map[kw_up] = kw_up

    # 1. Нормализация и фильтрация
    normalized = []
    for ent in entities:
        ent_up = ent.upper().replace(" ", "_")
        # Проверяем по мапе синонимов
        found_canonical = False
        for syn, canonical in canonical_map.items():
            if syn in ent_up:
                normalized.append(canonical)
                found_canonical = True
                break
        if not found_canonical:
            normalized.append(ent_up)

    # 2. Удаляем дубликаты после нормализации
    unique_ents = sorted(list(set(normalized)))

    # 3. ОБОБЩЕНИЕ: 
    # Если в списке есть ключевые слова из TRACKED_KEYWORDS, оставляем только их.
    # Если нет — берем максимум 2 первые сущности, чтобы не плодить длинные ключи.
    tracked_upper = [k.upper().replace(" ", "_") for k in config.TRACKED_KEYWORDS.keys()]
    
    # Проверяем в обе стороны: либо тема входит в сущность (NVIDIA -> NVIDIA_CORP), 
    # либо сущность является частью темы (US -> US_IRAN)
    matches = [e for e in unique_ents if any(t in e or e in t for t in tracked_upper)]
    
    result_ents = matches if matches else unique_ents[:2]

    return "_".join(result_ents)

# =========================
# MARKET SIGNAL ENGINE
# =========================

def market_signals(score: float, event_key: str) -> Dict[str, str]:
    mult = event_weights.get(event_key, 1.0)

    intensity = score * mult

    return {
        "nasdaq": "bearish" if intensity > config.SIGNAL_THRESHOLD_HIGH else "bullish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "oil": "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "bearish",
        "soxs": "bullish" if intensity > config.SIGNAL_THRESHOLD_HIGH else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "vix": "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "flat",
        "gold": "bullish" if intensity > config.SIGNAL_THRESHOLD_LOW else "bearish" if intensity < -config.SIGNAL_THRESHOLD_HIGH else "flat",
        "btc": "bearish" if intensity > config.SIGNAL_THRESHOLD_BTC else "bullish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "hbm": "bullish" if intensity < -config.SIGNAL_THRESHOLD_MED else "bearish" if intensity > config.SIGNAL_THRESHOLD_MED else "flat"
    }

# =========================
# WEIGHT / IMPACT MODEL
# =========================

def get_weight(event_key: str) -> float:
    """
    Возвращает вес для ключа. Если точного ключа нет, 
    ищет максимальный вес среди отдельных сущностей в этом ключе.
    """
    if event_key in event_weights:
        return event_weights[event_key]
    
    # Если ключа нет (например, это длинная комбинация), 
    # проверяем веса отдельных компонентов (MIRA, OPENAI и т.д.)
    parts = event_key.split('_')
    if len(parts) > 1:
        sub_weights = [event_weights.get(p, 1.0) for p in parts]
        return max(sub_weights) # Возвращаем самое сильное влияние из известных
        
    return 1.0

def predict_impact(score: float, event_key: str) -> float:
    return min(abs(score) * get_weight(event_key) * global_impact_multiplier, 100)

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
    global global_impact_multiplier
    error = actual - predicted
    adjustment = learning_rate * error * 0.01

    # Обновляем основной ключ
    event_weights[event_key] = max(0.5, min(5.0, event_weights.get(event_key, 1.0) + adjustment))

    # Обучаем глобальный множитель:
    # Если ошибка положительная (actual > predicted), множитель должен расти.
    # Используем меньший шаг для стабильности (0.005)
    global_impact_multiplier = max(1.0, min(10.0, global_impact_multiplier + (learning_rate * error * 0.005)))

    # Атомарное обучение: обновляем веса каждой отдельной сущности в ключе
    # Это позволяет системе выучить, что "OPENAI" важен, даже если он встретился в новом контексте
    parts = event_key.split('_')
    if len(parts) > 1:
        for part in parts:
            if len(part) > 2: # Игнорируем слишком короткие сущности
                event_weights[part] = max(0.5, min(5.0, event_weights.get(part, 1.0) + adjustment))

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

    tickers_to_fetch = {
        "^IXIC": "nasdaq_change",
        "CL=F": "oil_change",
        "^VIX": "vix_change",
        "GC=F": "gold_change",
        "BTC-USD": "btc_change",
        "SOXS": "soxs_change",
        "SMH": "hbm_change"
    }

    try:
        # yfinance синхронный, запускаем в экзекуторе
        loop = asyncio.get_event_loop()
        all_data = await loop.run_in_executor(
            sync_executor, 
            lambda: yf.download(list(tickers_to_fetch.keys()), period="1wk", interval="1h", progress=False)
        )
        
        lookback = config.MARKET_LOOKBACK_HOURS

        for ticker_symbol, data_key in tickers_to_fetch.items():
            try:
                if ticker_symbol in all_data['Close'].columns:
                    ticker_data = all_data['Close'][ticker_symbol].dropna()
                    # Нам нужно как минимум lookback + 1 свечей, чтобы вычислить разницу
                    if len(ticker_data) > lookback:
                        current_price = float(ticker_data.iloc[-1])
                        past_price = float(ticker_data.iloc[-(lookback + 1)])
                        if past_price != 0:
                            market_data[data_key] = ((current_price - past_price) / past_price) * 100
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
            score = row['score']
            actual = 0
            raw_change = 0
            
            # Определение корреляции: 
            # -1: Положительный score (риск) = падение цены (Nasdaq, BTC)
            #  1: Положительный score (риск) = рост цены (Gold, Oil, VIX, SOXS)
            correlation = -1 if event_key in ["NASDAQ", "BTC", "NVIDIA", "OPENAI", "GLOBAL"] else 1

            if event_key == "OIL" and 'oil_change' in raw_market_data:
            # Берем целевой актив из нашей новой карты (по умолчанию global)
            target = event_asset_map.get(event_key, "global")
            
            # Логика выбора данных для обучения на основе привязанного актива
            if target == "oil" and 'oil_change' in raw_market_data:
                raw_change = raw_market_data['oil_change']
            elif event_key == "GOLD" and 'gold_change' in raw_market_data:
                correlation = 1
            elif target == "gold" and 'gold_change' in raw_market_data:
                raw_change = raw_market_data['gold_change']
            elif event_key == "BTC" and 'btc_change' in raw_market_data:
                correlation = 1
            elif target == "btc" and 'btc_change' in raw_market_data:
                raw_change = raw_market_data['btc_change']
                correlation = -1 # Для BTC риск обычно означает падение
            elif event_key in ["NVIDIA", "OPENAI"]:
                correlation = -1
            elif target == "hbm":
                hbm_c = raw_market_data.get('hbm_change', 0)
                soxs_c = raw_market_data.get('soxs_change', 0) / -3 # Инвертируем SOXS для сопоставления с ростом сектора
                raw_change = hbm_c if abs(hbm_c) > abs(soxs_c) else soxs_c
                correlation = -1 # Рост риска = падение чипмейкеров
            elif event_key in ["US_IRAN", "HORMUZ", "GLOBAL"]:
                correlation = -1
            elif target == "nasdaq":
                raw_change = raw_market_data.get('nasdaq_change', 0)
                correlation = -1
            else: # target == "global" или "vix"
                vix_c = raw_market_data.get('vix_change', 0)
                nasdaq_c = raw_market_data.get('nasdaq_change', 0)
                if vix_c != 0:
                    raw_change = vix_c
                    correlation = 1 # Риск = Рост VIX
                else:
                    raw_change = nasdaq_c
                    correlation = -1 # Риск = Падение Nasdaq

            # Пропускаем обучение, если движение цены ниже порога (рынок закрыт или шум).
            if abs(raw_change) < config.LEARNING_THRESHOLD:
                continue

            actual = min(abs(raw_change) * config.SCALING_FACTOR, 100)
            
            # Проверка совпадения направления: (Score * Change * Correlation) > 0
            is_correct = 1 if (score * raw_change * correlation) > 0 else 0

            status_icon = "✅" if is_correct else "❌"
            logging.info(f"Result for {event_key}: {status_icon} (Score: {score:.1f}, Change: {raw_change:.2f}%)")
            update_weights(event_key, predicted, actual)

            cursor.execute("""
                UPDATE predictions
                SET resolved = 1, actual_move = ?, is_correct = ?
                WHERE id = ?
            """, (actual, is_correct, row['id']))
        conn.commit()

    save_state()
    logging.info(f"System settings saved. New IMPACT_MULTIPLIER: {global_impact_multiplier:.2f}")

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
    last_research_run = 0
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        # Запускаем цикл обучения сразу при старте, чтобы обработать старые записи
        logging.info("Первичный запуск цикла обучения...")
        await learning_cycle(session)

        while True:
            logging.info("GTS 4.0 scanning...")

            for key in event_scores:
                event_scores[key] *= config.DECAY_FACTOR

            current_market_data = await get_market_data(session)
            btc_change = current_market_data.get("btc_change", 0)
            gold_change = current_market_data.get("gold_change", 0)
            hbm_change = current_market_data.get("hbm_change", 0)
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
                    
                    # Для сигналов используем кумулятивный балл (состояние рынка)
                    # Для вероятности (Impact) теперь тоже, но с исправленным множителем 4.0
                    market = market_signals(event_scores[event_key], event_key)
                    prob = predict_impact(event_scores[event_key], event_key) 
                    sig_type = generate_signal(prob, event_scores[event_key])

                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO events (title, link, score, event, nasdaq, oil, hbm, soxs, gold, btc, vix, fear_greed)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (entry.title, entry.link, score, event_type, market["nasdaq"], market["oil"], market["hbm"], market["soxs"], market["gold"], market["btc"], market["vix"], fng_val))
                        cursor.execute("INSERT INTO predictions (event_key, score, predicted_impact) VALUES (?, ?, ?)", (event_key, event_scores[event_key], prob))
                        conn.commit()

                    if should_send(event_key, score):
                        if event_key == "BTC" and abs(btc_change) < config.BTC_MIN_VOLATILITY_FOR_ALERT:
                            continue

                        msg = f"""
🧠 EVENT: {event_key}
🤖 Model: {source}
� SIGNAL: {sig_type}
 Score: {event_scores[event_key]:.2f}
😨 Fear & Greed: {fng_val} ({fng_label})
📈 Impact: {prob:+.2f}%
📉 Nasdaq: {market['nasdaq']}
🧊 HBM Index: {market['hbm']} ({hbm_change:+.2f}%)
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

            if current_time - last_research_run >= config.RESEARCH_INTERVAL:
                await run_global_research()
                last_research_run = current_time

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