import feedparser
import logging
import re
import sqlite3
import time
import json
import asyncio
import aiohttp
from google import genai
from difflib import SequenceMatcher
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
            # 'gemini-3.1-pro': 2,
            'gemini-3-flash': 3,
            # 'gemini-3-pro': 4,
            'gemini-2.5-flash': 5,
            # 'gemini-2.5-pro': 6,
            # 'gemini-2.0-flash': 7,
            # 'gemini-1.5-flash': 8,
            # 'gemini-1.5-pro': 9,
            # 'gemini-1.0-pro': 10,
            # 'gemini-flash-latest': 11,
            # 'gemini-pro-latest': 12
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
                "supports_json": m_data["supports_json"],
                "provider": "gemini"
            })
            logging.info(f"✅ Добавлена в пул ротации: {m_data['name']} (Приоритет {m_data['priority']})")

        # Добавляем бесплатные модели из OpenRouter для отказоустойчивости
        if config.OPENROUTER_API_KEY:
            or_models = [
                {"name": "nvidia/nemotron-3-super-120b-a12b:free", "supports_json": True, "provider": "openrouter"},
                # {"name": "deepseek/deepseek-r1:free", "supports_json": True, "provider": "openrouter"},
                # {"name": "google/gemma-3-27b-it:free", "supports_json": True, "provider": "openrouter"},
                {"name": "openai/gpt-oss-120b:free", "supports_json": True, "provider": "openrouter"},
                {"name": "google/gemini-2.0-flash-lite-preview-02-05:free", "supports_json": True, "provider": "openrouter"},
                # {"name": "qwen/qwen-2.5-72b-instruct:free", "supports_json": True, "provider": "openrouter"},
                # {"name": "mistralai/mistral-7b-instruct:free", "supports_json": True, "provider": "openrouter"},
            ]
            for m in or_models:
                pool.append(m)
                logging.info(f"✅ Добавлена в пул ротации (OpenRouter): {m['name']}")

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
            "supports_json": True,
            "provider": "gemini"
        })
    return pool

model_pool = init_model_pool()
current_model_idx = 0
# Локи для защиты глобального состояния
state_lock = asyncio.Lock()
scores_lock = asyncio.Lock()
model_lock = asyncio.Lock()
db_lock = asyncio.Lock()

def get_active_model():
    return model_pool[current_model_idx]

logging.info(f"Пул моделей готов: {[m['name'] for m in model_pool]}. Старт с: {get_active_model()['name']}")

# =========================
# STATE
# =========================

# Храним время последнего обновления (затухания) для каждого ключа
event_last_update = {}

async def apply_decay(key: str, is_market_active: bool) -> float:
    """
    Применяет затухание к баллу ключа на основе времени, прошедшего с последнего обновления.
    Формула: Score = Score * (DecayFactor ^ (DeltaTime / Interval))
    """
    if key not in event_scores or event_scores[key] == 0:
        event_last_update[key] = time.time()
        return 0.0

    now = time.time()
    last_update = event_last_update.get(key, now)
    delta_t = now - last_update
    
    decay_factor = config.DECAY_FACTOR if is_market_active else config.NIGHT_DECAY_FACTOR
    # Интервал, за который балл должен уменьшиться на decay_factor (из конфига это 180с)
    intervals_passed = delta_t / config.CHECK_INTERVAL
    
    async with scores_lock:
        event_scores[key] *= (decay_factor ** intervals_passed)
    event_last_update[key] = now
    
    return event_scores[key]

# Инициализируем словарь до вызова функции init_state
event_last_sent = {}
# Глобальные наборы для предотвращения дублирования в памяти (race condition)
processed_urls = set()
processed_titles = [] # Используем список для хранения последних заголовков (для нечеткого поиска)
processed_slugs = {} # slug -> timestamp для семантической дедупликации

def init_state() -> Dict[str, float]:
    scores = defaultdict(float)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Восстанавливаем состояние из таблицы predictions, где уже есть event_key
        # Группируем по timestamp, чтобы не суммировать дубликаты от разных активов одной новости
        # Убираем расчет затухания из SQL, так как apply_decay сделает это точнее при старте
        cursor.execute("""
            SELECT event_key, SUM(raw_score) FROM (
                SELECT 
                    MAX(score) as raw_score,
                    event_key, timestamp
                FROM predictions WHERE timestamp > datetime('now', '-1 day')
                GROUP BY event_key, timestamp
            ) GROUP BY event_key
        """)
        for key, val in cursor.fetchall():
            # Ограничиваем начальное состояние порогом
            scores[key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, val))

        # Также инициализируем event_last_sent из последних прогнозов, чтобы избежать дублей при рестарте
        cursor.execute("""
            SELECT event_key, MAX(timestamp) FROM predictions GROUP BY event_key
        """)
        for key, ts_str in cursor.fetchall():
            # Преобразуем строку времени SQLite в unix timestamp
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            event_last_update[key] = dt.timestamp()
            event_last_sent[key] = dt.timestamp()
            
        # Загружаем историю для дедупликации
        cursor.execute("SELECT link, title, slug, timestamp FROM events WHERE timestamp > datetime('now', '-1 day')")
        for row in cursor.fetchall():
            processed_urls.add(row['link'])
            processed_titles.append(row['title'])
            if row['slug']:
                dt = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                processed_slugs[row['slug']] = dt.timestamp()

    return scores

event_scores = init_state()
learning_rate = config.LEARNING_RATE
# Карта связей: какой ключ к какому активу привязан для обучения
event_asset_map = {}

def load_weights() -> Dict[str, float]:
    # Создаем базовые веса из TRACKED_KEYWORDS, преобразуя ключи в формат EVENT_KEY (например, "US Iran" -> "US_IRAN")
    weights = {}
    for k, info in config.TRACKED_KEYWORDS.items():
        if isinstance(info, tuple):
            weight = info[0]
            target_assets = info[1] if len(info) > 1 else ["global"]
        else:
            weight = info
            target_assets = ["global"]
        
        if not isinstance(target_assets, list):
            target_assets = [target_assets]
        
        key_parts = sorted(k.upper().replace(" ", "_").split("_"))
        canonical_key = "_".join(key_parts)
        
        weights[canonical_key] = weight
        event_asset_map[canonical_key] = target_assets
    
    # Синхронизация специфичных имен (если в конфиге Bitcoin, а в логике BTC)
    if "BITCOIN" in weights:
        if "BTC" not in weights: # Только если BTC еще не определен явно
            weights["BTC"] = weights.pop("BITCOIN")
        else: # Если оба существуют, оставляем BTC и удаляем BITCOIN
            weights.pop("BITCOIN")
        if "BITCOIN" in event_asset_map:
            event_asset_map["BTC"] = event_asset_map.pop("BITCOIN")

    # Гарантируем наличие ключа GLOBAL
    if "GLOBAL" not in weights: weights["GLOBAL"] = 1.0
    if "GLOBAL" not in event_asset_map: event_asset_map["GLOBAL"] = ["global"]

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
        if "memory" in phrase.lower():
            search_map["hbm"] = "HBM"

    return search_map

fallback_entity_map = _get_fallback_entity_search_map()

async def ai_analyze(text: str, session: Optional[aiohttp.ClientSession] = None, max_retries: int = 3) -> Tuple[Optional[float], Optional[str], Optional[List[str]], Optional[str], bool, str]:
    """
    Uses Gemini AI to perform deep sentiment analysis and NER.
    """
    # Формируем строку с тегами из конфига для подсказки нейросети
    tags_hint = ", ".join([f'"{k}"' for k in config.TRACKED_KEYWORDS.keys()])
    prompt = f"""
    Analyze this financial news snippet: "{text}"
    Identify key entities. Use these standardized tags if applicable: {tags_hint}.
    IMPORTANT: Distinguish between actual assets/companies (e.g., "Gold" as commodity, "Nvidia" as company) and descriptive terms or adjectives (e.g., "gold visa", "oil paintings"). Do not tag an asset if it's used as an adjective or metaphor.
    Identify the core unique event being reported.
    Return ONLY a JSON object with this structure:
    {{
      "score": float (-10.0 to 10.0). CRITICAL: Use Finance Risk Scale:
               NEGATIVE SCORE (-10 to -1) = GOOD news for markets (Growth, Profits, Rate cuts, Peace).
               POSITIVE SCORE (1 to 10) = BAD news for markets (War, Inflation, Defaults, Rate hikes).
               Example: Alphabet winning in AI is a NEGATIVE score (around -3.0).
               Range 0.0 to 1.5 is for Neutral/Routine news.
      "event_type": "military" | "economic" | "diplomatic" | "neutral" | "tech",
      "entities": ["list of countries, companies or key regions"],
      "slug": "short_snake_case_event_id (2-4 words). Use the same slug for different articles reporting the same core event (e.g., 'korea_ai_tax_impact').",
      "is_black_swan": boolean (True ONLY for extreme, unpredictable, market-shaking rarities like 9/11, sudden wars, or total structural collapses)
    }}
    Do not include any markdown formatting or explanations.
    """

    for attempt in range(max_retries):
        model_tried_count = 0
        while model_tried_count < len(model_pool): # Внутренний цикл для перебора моделей в пуле
            async with model_lock:
                global current_model_idx
                active_idx = current_model_idx
            
            try:
                active = model_pool[active_idx]
                res_text = ""

                if active.get("provider") == "openrouter":
                    payload = {
                        "model": active["name"],
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    if active["supports_json"]:
                        payload["response_format"] = {"type": "json_object"}
                    
                    s = session if session else aiohttp.ClientSession()
                    try:
                        async with s.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                                "HTTP-Referer": "https://gts-project.io",
                                "X-Title": "GTS 4.0",
                                "Content-Type": "application/json"
                            },
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60, connect=15)
                        ) as resp:
                            if resp.status != 200:
                                raise Exception(f"OpenRouter Error {resp.status}")
                            res_json = await resp.json()
                            res_text = (res_json.get('choices', [{}])[0].get('message', {}).get('content') or "").strip()
                    finally:
                        if not session: await s.close()
                else:
                    # Gemini logic
                    gen_config = {"response_mime_type": "application/json"} if active["supports_json"] else {}
                    response = await client.aio.models.generate_content(
                        model=active["name"],
                        contents=prompt,
                        config=gen_config
                    )
                
                # Проверка, не заблокирован ли ответ фильтрами безопасности
                if active.get("provider") == "gemini" and (not response.candidates or not response.candidates[0].content.parts):
                    # Это не 429, но и не успешный ответ. Считаем, что модель не справилась.
                    logging.warning(f"Модель {active['name']} заблокировала ответ по безопасности. Переключаюсь на следующую.")
                    model_tried_count += 1
                    async with model_lock:
                        current_model_idx = (current_model_idx + 1) % len(model_pool)
                    continue # Попробуем следующую модель в пуле немедленно

                if active.get("provider") == "gemini":
                    res_text = (response.text or "").strip()
                
                # Надежный поиск границ JSON (на случай, если модель добавила текст)
                start = res_text.find('{')
                end = res_text.rfind('}') + 1

                if start == -1 or end == 0:
                    logging.warning(f"Модель {active['name']} вернула невалидный JSON. Получено: {res_text[:100]}...")
                    # Это не 429, но и не успешный ответ. Считаем, что модель не справилась.
                    model_tried_count += 1
                    async with model_lock:
                        current_model_idx = (current_model_idx + 1) % len(model_pool)
                    continue # Попробуем следующую модель в пуле немедленно

                data = json.loads(res_text[start:end])
                return float(data.get("score", 0)), data.get("event_type", "neutral"), data.get("entities", []), data.get("slug"), bool(data.get("is_black_swan", False)), active["name"]

            except Exception as e:
                err_msg = str(e).lower()
                # Обработка 404 (модель не найдена) и 429 (лимиты/таймауты)
                if any(x in err_msg for x in ["429", "404", "quota", "limit", "timeout"]):
                    old_name = model_pool[active_idx]["name"]
                    async with model_lock:
                        current_model_idx = (current_model_idx + 1) % len(model_pool)
                    model_tried_count += 1
                    logging.warning(f"⚠️ Модель {old_name} временно недоступна ({err_msg}). Переключаюсь на {model_pool[current_model_idx]['name']}...")
                    if model_tried_count == len(model_pool):
                        break # Все модели в пуле исчерпали лимит, выходим из внутреннего цикла
                    continue # Пробуем следующую модель в пуле немедленно
                else:
                    # Другая ошибка (например, модель не поддерживает JSON mode). 
                    # Логируем, переключаемся на следующую модель и пробуем снова в этом же цикле.
                    logging.error(f"⚠️ Ошибка модели {active['name']}: {e}")
                    model_tried_count += 1
                    async with model_lock:
                        current_model_idx = (current_model_idx + 1) % len(model_pool)
                    continue # Пробуем следующую модель в пуле немедленно
        
        # Если весь пул моделей исчерпан (все вернули 429 или ошибки)
        wait_time = (attempt + 1) * 60
        logging.warning(f"⚠️ Все модели в пуле ({len(model_pool)}) временно недоступны. Повтор через {wait_time}s...")
        
        await asyncio.sleep(wait_time)

    # Fallback logic
    text_low = text.lower()
    found_entities = []
    for search_term, canonical_name in fallback_entity_map.items():
        if re.search(r'\b' + re.escape(search_term) + r'\b', text_low):
            if canonical_name not in found_entities:
                found_entities.append(canonical_name)
    
    # Улучшенный скоринг в фоллбеке
    is_critical = re.search(r'\b(war|strike|attack|conflict|escalation|sanctions|emergency)\b', text_low)
    score = 4.0 if is_critical else 0.0

    # Если это не критично и сущности не найдены — лучше пропустить анализ, чем гадать
    if not found_entities and score == 0:
        return None, None, None, None, False, "No Relevance"
    
    slug = "_".join([e.lower() for e in found_entities[:2]]) if found_entities else "general_market"
    return score, "neutral", found_entities, slug, False, "Fallback (Regex)"

# =========================
# EVENT ENGINE
# =========================

def make_event_key(entities: List[str]) -> str:
    # Очистка входного списка от пустых значений, None и заглушек
    valid_entities = [str(e).strip() for e in (entities or []) if e and str(e).strip() != "Unknown"]

    if not valid_entities:
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
        "HBM": "HBM", "HIGH_BANDWIDTH_MEMORY": "HBM",
        "NVDA": "NVIDIA", "NVIDIA": "NVIDIA",
        "DONALD_TRUMP": "TRUMP", "MAGA": "TRUMP"
    }

    # Автоматически добавляем все отслеживаемые слова из конфига в мапу нормализации
    for kw in config.TRACKED_KEYWORDS.keys():
        kw_up = kw.upper().replace(" ", "_")
        if kw_up not in canonical_map:
            canonical_map[kw_up] = kw_up

    # 1. Нормализация и фильтрация
    normalized = []
    for ent in valid_entities:
        ent_up = ent.upper().replace(" ", "_")
        # Проверяем по мапе синонимов
        found_canonical = False
        for syn, canonical in canonical_map.items():
            # Требуем точного совпадения для всех отслеживаемых сущностей,
            # чтобы избежать ложных срабатываний (например, "gold visa" не превращалась в GOLD)
            if syn == ent_up:
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
    matches = []
    for e in unique_ents:
        # Теперь требуем только точного совпадения сущности с отслеживаемым тегом,
        # полагаясь на то, что ИИ уже получил список правильных тегов в prompt.
        matched_tag = next((t for t in tracked_upper if t == e), e)
        matches.append(matched_tag)
    
    # 4. СТРОГОЕ ОГРАНИЧЕНИЕ ПО СЛОВАМ (MAX_ENTITY_PARTS)
    # Разбиваем все найденные теги на отдельные слова, чтобы лимит работал по словам.
    all_words = []
    for m in (matches if matches else unique_ents):
        all_words.extend(m.split('_'))
    
    # Убираем дубликаты и принудительно сортируем слова алфавитно для консистентности ключей
    final_parts = sorted(list(set(all_words)))
    result_ents = final_parts[:config.MAX_ENTITY_PARTS]

    if not result_ents:
        return "GLOBAL"

    return "_".join(result_ents).strip().upper()

# =========================
# MARKET SIGNAL ENGINE
# =========================

def market_signals(score: float, event_key: str) -> Dict[str, str]:
    intensity = score # Вес уже применен при накоплении в event_scores

    return {
        "nasdaq": "bearish" if intensity > config.SIGNAL_THRESHOLD_HIGH else "bullish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "sp500": "bearish" if intensity > config.SIGNAL_THRESHOLD_HIGH else "bullish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "oil": "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "soxs": "bullish" if intensity > config.SIGNAL_THRESHOLD_HIGH else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "vix": "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "gold": "bullish" if intensity > config.SIGNAL_THRESHOLD_LOW else "bearish" if intensity < -config.SIGNAL_THRESHOLD_HIGH else "flat",
        "btc": "bearish" if intensity > config.SIGNAL_THRESHOLD_BTC else "bullish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat"
    }

# =========================
# WEIGHT / IMPACT MODEL
# =========================

async def get_weight(event_key: str) -> float:
    """
    Возвращает вес для ключа. Если точного ключа нет, 
    ищет максимальный вес среди отдельных сущностей в этом ключе.
    """
    if event_key in event_weights:
        return event_weights[event_key]
    
    # Проверяем, не является ли весь ключ (или его часть) известным тегом из конфига
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
    # Вес уже применен в event_scores, здесь используем только глобальный множитель
    return min(abs(score) * global_impact_multiplier, 100)

# =========================
# SIGNAL ENGINE
# =========================

def generate_signal(prob: float, score: float) -> str:
    if score > 0:  # Медвежий сценарий (Risk-Off)
        if prob > 70: return "🔴 HIGH RISK-OFF"
        if prob > 40: return "🟠 MEDIUM RISK"
        return "🟡 CAUTION"
    elif score < 0:  # Бычий сценарий (Risk-On)
        if prob > 70: return "🚀 STRONG RISK-ON"
        return "🟢 RISK-ON"
    
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

def should_send(key: str, current_score: float, is_black_swan: bool = False) -> bool:
    now = time.time()

    # Если новость экстремально важная (например, score > 8), игнорируем кулдаун
    if abs(current_score) >= 8.0 or is_black_swan:
        # Но даже для важных новостей даем 2 минуты, чтобы не слать дубли от разных агентств
        if key in event_last_sent and (now - event_last_sent[key] < 120):
            logging.info(f"High-score spam prevention for {key}")
            return False
        event_last_sent[key] = now
        return True

    if key not in event_last_sent:
        event_last_sent[key] = now
        return True

    if now - event_last_sent[key] > config.COOLDOWN:
        event_last_sent[key] = now
        return True

    # Очистка старых записей из памяти (простой механизм prune)
    if len(event_last_sent) > 1000:
        cutoff = now - (config.COOLDOWN * 2)
        keys_to_del = [k for k, v in event_last_sent.items() if v < cutoff]
        for k in keys_to_del: del event_last_sent[k]

    return False

# =========================
# LEARNING SYSTEM
# =========================

async def update_weights(event_key: str, error: float):
    """Обновляет веса событий на основе ошибки прогноза."""
    adjustment = learning_rate * error * 0.01

    # Обновляем основной ключ
    event_weights[event_key] = max(0.5, min(5.0, event_weights.get(event_key, 1.0) + adjustment))
    logging.info(f"📈 Weight adjustment for {event_key}: {adjustment:+.4f} (New weight: {event_weights[event_key]:.2f})")

    # Атомарное обучение (опционально): обновляем части ключа, только если это не части одного имени/названия.
    # Чтобы избежать дробления имен (как KEVIN_O'LEARY), мы можем отключить этот блок или сделать его строже.
    parts = event_key.split('_')
    if len(parts) > 1 and len(parts) <= config.MAX_ENTITY_PARTS:
        for part in parts:
            # Обновляем только если эта сущность уже известна системе как самостоятельная
            if len(part) > 2 and part in event_weights:
                event_weights[part] = max(0.5, min(5.0, event_weights.get(part, 1.0) + adjustment))

def calibrate_multiplier(avg_error: float):
    """Корректирует глобальный множитель влияния на основе средней ошибки всей выборки."""
    global global_impact_multiplier
    old_mult = global_impact_multiplier
    # Используем меньший шаг для стабильности (0.005)
    global_impact_multiplier = max(1.0, min(10.0, old_mult + (learning_rate * avg_error * 0.005)))
    if abs(global_impact_multiplier - old_mult) > 0.0001:
        logging.info(f"⚙️ Calibration: IMPACT_MULTIPLIER adjusted {old_mult:.2f} -> {global_impact_multiplier:.2f} (avg_err: {avg_error:.1f})")

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
    except aiohttp.ClientConnectorError:
        logging.error("Fear & Greed API: Connection failed. Check your DNS/Internet.")
        return None, None, 0
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
        "^GSPC": "sp500_change",
        "CL=F": "oil_change",
        "^VIX": "vix_change",
        "GLD": "gold_change",
        "BTC-USD": "btc_change",
        "SOXS": "soxs_change",
        "SMH": "smh_change",
        "MU": "mu_change"
    }

    stale_map = {}
    last_bar_time = 0
    try:
        # yfinance синхронный, запускаем в экзекуторе
        loop = asyncio.get_event_loop()
        all_data = await loop.run_in_executor(
            sync_executor, 
            lambda: yf.download(list(tickers_to_fetch.keys()), period="1wk", interval="1h", progress=False)
        )
        
        if all_data.empty or 'Close' not in all_data.columns:
            logging.error("Yahoo Finance returned no data. Check internet connection and system clock.")
            return {}

        lookback = config.MARKET_LOOKBACK_HOURS

        # Кэшируем доступ к ценам закрытия для оптимизации
        close_prices = all_data['Close']

        for ticker_symbol, data_key in tickers_to_fetch.items():
            try:
                # Извлекаем данные для конкретного тикера, если они есть в ответе
                if ticker_symbol in close_prices:
                    ticker_data = close_prices[ticker_symbol].dropna()
                    # Нам нужно как минимум lookback + 1 свечей, чтобы вычислить разницу
                    if len(ticker_data) > lookback:
                        current_price = float(ticker_data.iloc[-1])
                        
                        # Проверяем свежесть данных для конкретного тикера (4 часа)
                        current_bar_time = ticker_data.index[-1].timestamp()
                        is_ticker_stale = (time.time() - current_bar_time) > (4 * 3600)
                        stale_map[data_key] = is_ticker_stale
                        
                        last_bar_time = max(last_bar_time, current_bar_time)
                        
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
    else:
        logging.warning("Proceeding without Fear & Greed data due to fetch error.")

    # Общий флаг для режима затухания (если хоть что-то активно, например BTC)
    market_data['is_stale'] = (time.time() - last_bar_time) > (4 * 3600) if last_bar_time > 0 else True
    market_data['stale_map'] = stale_map

    return market_data

def count_eligible_predictions() -> int:
    """Возвращает количество новостей, готовых к обучению (старше MARKET_LOOKBACK_HOURS)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE resolved = 0 AND timestamp < datetime('now', '-' || ? || ' hours')", (config.MARKET_LOOKBACK_HOURS,))
        return cursor.fetchone()[0]


async def learning_cycle(session: aiohttp.ClientSession):
    raw_market_data = await get_market_data(session)
    if not raw_market_data:
        logging.warning("Skipping learning cycle: No market data available.")
        return

    loop = asyncio.get_event_loop()
    async with db_lock:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Берем только неразрешенные прогнозы, созданные более MARKET_LOOKBACK_HOURS назад.
        cursor = conn.cursor()
        # Берем только неразрешенные прогнозы, созданные более MARKET_LOOKBACK_HOURS назад.
        # Это исключает преждевременную оценку новостей, которые только что вышли.
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE resolved = 0 AND timestamp < datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC LIMIT 100
        """, (config.MARKET_LOOKBACK_HOURS,))
        rows = cursor.fetchall()
        logging.info(f"🧠 Начало цикла обучения. Найдено кандидатов для обработки: {len(rows)}")

        updates_by_key = defaultdict(list) # Для агрегации обновлений весов
        all_errors = [] # Для калибровки глобального множителя
        stale_map = raw_market_data.get('stale_map', {})

        for row in rows:
            event_key = row['event_key']
            predicted = row['predicted_impact']
            score = row['score']
            target = row['target_asset'] if row['target_asset'] else "global"
            
            actual = 0
            raw_change = 0
            correlation = 0
            data_key = ""

            # Обучение на основе конкретного актива, указанного в прогнозе
            if target == "oil" and 'oil_change' in raw_market_data:
                data_key = 'oil_change'
                raw_change = raw_market_data['oil_change']
                correlation = 1
            elif target == "gold" and 'gold_change' in raw_market_data:
                data_key = 'gold_change'
                raw_change = raw_market_data['gold_change']
                correlation = 1
            elif target == "btc" and 'btc_change' in raw_market_data:
                data_key = 'btc_change'
                raw_change = raw_market_data['btc_change']
                correlation = -1
            elif target == "nasdaq" and 'nasdaq_change' in raw_market_data:
                data_key = 'nasdaq_change'
                raw_change = raw_market_data['nasdaq_change']
                correlation = -1
            elif target == "sp500" and 'sp500_change' in raw_market_data:
                data_key = 'sp500_change'
                raw_change = raw_market_data['sp500_change']
                correlation = -1
            elif target == "soxs" and 'soxs_change' in raw_market_data:
                data_key = 'soxs_change'
                raw_change = raw_market_data['soxs_change']
                correlation = 1
            elif target == "vix" and 'vix_change' in raw_market_data:
                data_key = 'vix_change'
                raw_change = raw_market_data['vix_change']
                correlation = 1
            else: # "global" logic
                vix_c = raw_market_data.get('vix_change', 0)
                nasdaq_c = raw_market_data.get('nasdaq_change', 0)
                if vix_c != 0:
                    data_key = 'vix_change'
                    raw_change = vix_c
                    correlation = 1 # Риск = Рост VIX
                else:
                    data_key = 'nasdaq_change'
                    raw_change = nasdaq_c
                    correlation = -1 # Риск = Падение Nasdaq

            # Пропускаем, если данные по этому конкретному активу устарели (рынок закрыт)
            if not data_key or stale_map.get(data_key, True):
                continue

            # Пропускаем обучение, если движение цены ниже порога (рынок закрыт или шум).
            if abs(raw_change) < config.LEARNING_THRESHOLD:
                continue

            # Если новость нейтральная (ниже порога NEUTRAL_SCORE_THRESHOLD), помечаем как resolved,
            # но не считаем это ошибкой прогноза и не логируем в результат обучения.
            if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
                cursor.execute("UPDATE predictions SET resolved = 1 WHERE id = ?", (row['id'],))
                logging.debug(f"Learning: Skipping low-score event {event_key} (Score {score:.1f} < Threshold {config.NEUTRAL_SCORE_THRESHOLD})")
                continue

            scaling = config.ASSET_SCALING_FACTORS.get(target, config.ASSET_SCALING_FACTORS["global"])
            actual = min(abs(raw_change) * scaling, 100)
            
            # Проверка совпадения направления: (Score * Change * Correlation) > 0
            is_correct = 1 if (score * raw_change * correlation) > 0 else 0
            status_icon = "✅" if is_correct else "❌"
            
            direction_desc = "MATCH" if is_correct else "CONTRARY"
            logging.info(f"Learning: {event_key} | {target} | {status_icon} | {direction_desc} | Score: {score:.1f} | Mkt: {raw_change:+.2f}%")
            
            # Накапливаем данные для агрегированного обучения (защита от переобучения)
            error = actual - predicted
            updates_by_key[event_key].append(error)
            all_errors.append(error)

            cursor.execute("""
                UPDATE predictions
                SET resolved = 1, actual_move = ?, is_correct = ?
                WHERE id = ?
            """, (actual, is_correct, row['id']))

        # 1. Агрегированное обновление весов (защита от "двойного" обучения на пачке новостей)
        for e_key, errors in updates_by_key.items():
            avg_err = sum(errors) / len(errors)
            await update_weights(e_key, avg_err)

        # 2. Калибровка глобального множителя (один раз за цикл на основе всей выборки)
        if all_errors:
            calibrate_multiplier(sum(all_errors) / len(all_errors))
        conn.commit()

    save_state()
    logging.info(f"System settings saved. New IMPACT_MULTIPLIER: {global_impact_multiplier:.2f}")

async def cleanup_db():
    """
    Удаляет записи из БД, которые старше RETENTION_DAYS, чтобы предотвратить разрастание файла.
    Также удаляет ключи из таблицы весов, значение которых ниже MIN_WEIGHT_THRESHOLD.
    """
    async with db_lock:
        try:
        global event_weights
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Получаем список каноничных ключей из конфига, которые НЕЛЬЗЯ удалять
            tracked_keys = []
            for k in config.TRACKED_KEYWORDS.keys():
                key_parts = sorted(k.upper().replace(" ", "_").split("_"))
                tracked_keys.append("_".join(key_parts))
            placeholders = ', '.join(['?'] * len(tracked_keys))

            # Удаляем старые события и прогнозы
            cursor.execute("DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
            cursor.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
            
            # 1. Удаляем ключи с критически низким весом
            cursor.execute("DELETE FROM weights WHERE weight <= ?", (config.MIN_WEIGHT_THRESHOLD,))
            
            # 2. Удаляем "забытые" ключи, которых нет в последних прогнозах и нет в TRACKED_KEYWORDS
            cursor.execute(f"""
                DELETE FROM weights 
                WHERE event_key NOT IN (SELECT DISTINCT event_key FROM predictions)
                AND event_key NOT IN ({placeholders})
            """, tracked_keys)
            
            deleted_weights = cursor.rowcount
            
            conn.commit()  # Завершаем транзакцию после удаления
        
        # VACUUM должен выполняться вне транзакции
        with get_db_connection() as conn:
            conn.execute("PRAGMA journal_mode=DELETE") # Отключаем WAL для VACUUM
            conn.execute("VACUUM")
            conn.execute("PRAGMA journal_mode=WAL") # Возвращаем WAL
            
            # Обновляем веса в оперативной памяти после очистки БД
            event_weights = load_weights()
            
            logging.info(f"--- База данных оптимизирована: удалены данные старше {config.RETENTION_DAYS} дней "
                         f"и {deleted_weights} ключей с весом < {config.MIN_WEIGHT_THRESHOLD} ---")
    except Exception as e:
        logging.error(f"Ошибка при очистке БД: {e}")

# =========================
# MAIN LOOP
# =========================

def clean_title(title: str) -> str:
    """Удаляет мусор из заголовка (названия источников, лишние знаки)."""
    # Удаляем источники в конце: "Title - Reuters" или "Title | CNBC"
    cleaned = re.sub(r'\s+[-|]\s+.*$', '', title)
    # Удаляем пунктуацию и нормализуем пробелы для лучшего нечеткого сравнения
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    return " ".join(cleaned.lower().split())

def is_fuzzy_duplicate(new_title: str, existing_titles: List[str], threshold: float) -> bool:
    """Проверяет заголовок на схожесть с уже существующими в кэше."""
    if not new_title:
        return False
    
    new_clean = clean_title(new_title)
    for title in existing_titles:
        # Сравниваем очищенные версии
        if SequenceMatcher(None, new_clean, clean_title(title)).ratio() > threshold:
            return True
    return False

async def process_single_feed(url: str, session: aiohttp.ClientSession, loop: asyncio.AbstractEventLoop, fng_val: float, fng_label: str, market_context: Dict[str, Any], is_market_active: bool):
    """Обрабатывает одну RSS ленту."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    raw_data = None
    for attempt in range(2): # 2 попытки на случай временного сбоя
        try:
            async with session.get(url, headers=headers, timeout=20) as response:
                if response.status == 200:
                    raw_data = await response.read()
                    break
                elif response.status == 429:
                    logging.warning(f"Rate limited by Google News for {url}")
                    await asyncio.sleep(5)
                else:
                    logging.debug(f"Feed {url} returned status {response.status}")
        except Exception as e:
            if attempt == 1:
                error_str = str(e)
                if "getaddrinfo" in error_str:
                    logging.error(f"🌐 DNS/Connection Error for {url}: Check internet or DNS settings.")
                else:
                    logging.error(f"Feed error {url}: {e}")
                return
            await asyncio.sleep(2)

    if not raw_data:
        return

    feed = await loop.run_in_executor(sync_executor, lambda: feedparser.parse(raw_data))

    # Адаптивное количество записей RSS для обработки
    max_entries_to_process = config.RSS_MAX_ENTRIES if is_market_active else config.RSS_MAX_ENTRIES_INACTIVE
    
    for entry in feed.entries[:max_entries_to_process]:
        # Адаптивное окно возраста новости:
        # Если рынок активен — окно узкое (6ч), чтобы реагировать на свежее.
        # Если рынок закрыт — расширяем окно до 72ч (уикенд), чтобы собрать контекст к открытию.
        max_age_h = config.MAX_NEWS_AGE_HOURS if is_market_active else 72
        
        published = entry.get('published_parsed')
        if published:
            pub_time = time.mktime(published)
            if (time.time() - pub_time) > (max_age_h * 3600):
                continue

        original_title = entry.title
        cleaned_t = clean_title(original_title)

        async with state_lock:
            # 1. Проверка по URL (мгновенно из памяти)
            if entry.link in processed_urls:
                continue
            
            # 2. Нечеткая проверка заголовка (предотвращает дубли с разным текстом)
            if is_fuzzy_duplicate(entry.title, processed_titles, config.DUPLICATE_TITLE_THRESHOLD):
                logging.debug(f"Обнаружен нечеткий дубликат заголовка: {entry.title}")
                processed_urls.add(entry.link)
                continue
            
            # Резервируем новость в памяти ПЕРЕД запуском AI
            processed_urls.add(entry.link)
            processed_titles.append(entry.title)
            if len(processed_titles) > 1000: # Увеличим кэш до 1000 для надежности
                processed_titles.pop(0)

        # 3. Проверка в БД за последние 3 часа
        async with db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT title FROM events WHERE timestamp > datetime('now', '-3 hours') ORDER BY timestamp DESC LIMIT 20")
                db_titles = [row['title'] for row in cursor.fetchall()]
                if is_fuzzy_duplicate(original_title, db_titles, config.DUPLICATE_TITLE_THRESHOLD):
                    continue

        text = entry.title + " " + entry.get("summary", "")
        analysis = await ai_analyze(text, session=session)
        
        if analysis[0] is None:
            continue

        score, event_type, entities, slug, is_black_swan, source = analysis

        # 4. Семантическая проверка дубликатов по slug (AI-generated)
        # Если этот 'смысл' новости уже встречался недавно, пропускаем
        async with state_lock:
            if slug and slug in processed_slugs:
                if time.time() - processed_slugs[slug] < config.MAX_NEWS_AGE_HOURS * 3600:
                    logging.info(f"Семантический дубликат пропущен: {slug} | {entry.title}")
                    continue
            
            if slug:
                processed_slugs[slug] = time.time()
                if len(processed_slugs) > 1000: # Прунинг кэша
                    first_key = next(iter(processed_slugs))
                    del processed_slugs[first_key]

        # Определение рейтинга доверия источнику новости
        # Google News RSS обычно указывает источник в конце заголовка через дефис или в поле source
        news_source = entry.get('source', {}).get('title', '').lower()
        if not news_source and ' - ' in entry.title:
            news_source = entry.title.split(' - ')[-1].lower()
            
        trust_factor = config.DEFAULT_TRUST_SCORE
        for s_key, s_weight in config.SOURCE_TRUST_LEVELS.items():
            if s_key.lower() in news_source:
                trust_factor = s_weight
                break
        
        # Применяем коэффициент доверия к баллу новости
        score *= trust_factor
        
        # Дополнительное снижение веса для нефинансовых типов событий
        # Применяем decay ко всем нейтральным новостям, а не только к тем, что выше порога
        if event_type in ["neutral", "diplomatic", "tech"] and abs(score) > 0:
            # Если новость не экономическая, снижаем её влияние на накопленный балл сильнее
            score *= (config.NON_FINANCIAL_SCORE_DECAY_FACTOR * 0.5)
        
        current_delay = config.AI_DELAY_JSON if get_active_model()["supports_json"] else config.AI_DELAY_NO_JSON
        await asyncio.sleep(current_delay)

        event_key = make_event_key(entities)

        # Применяем затухание к существующему баллу перед добавлением новой новости
        await apply_decay(event_key, is_market_active)

        # Фильтр значимости: теперь в базу попадают только новости с баллом >= NEUTRAL_SCORE_THRESHOLD (2.5)
        if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
            logging.info(f"Skipping news for {event_key}: Score {score:.2f} is below threshold {config.NEUTRAL_SCORE_THRESHOLD}")
            continue

        # МЕХАНИЗМ ПОЛНОГО РАЗВОРОТА (PIVOT): Если новость сильная и против тренда — забываем историю
        async with scores_lock:
            if event_scores[event_key] != 0 and (event_scores[event_key] * score) < 0:
                if abs(score) >= config.PIVOT_THRESHOLD:
                    logging.info(f"💥 FULL PIVOT for {event_key}: Resetting accumulated score ({event_scores[event_key]:.2f}) due to high-impact opposite news ({score:+.2f})")
                    event_scores[event_key] = 0

            # Применяем вес ключа и накапливаем балл.
            weight = await get_weight(event_key)
            weighted_score = score * weight
            event_scores[event_key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, event_scores[event_key] + weighted_score))

        market = market_signals(event_scores[event_key], event_key)
        prob = predict_impact(event_scores[event_key], event_key) 
        sig_type = generate_signal(prob, event_scores[event_key])

        # Улучшенный поиск активов: проверяем event_key и его части на соответствие event_asset_map
        target_assets_set = set()

        # 1. Прямое совпадение event_key с ключом в event_asset_map
        if event_key in event_asset_map:
            target_assets_set.update(event_asset_map[event_key])

        # 2. Поиск по частям event_key, но только если сущностей немного
        # Ограничение через MAX_ENTITY_PARTS делает систему строже, исключая случайные связи
        parts = event_key.split('_')
        if len(parts) <= config.MAX_ENTITY_PARTS:
            for part in parts:
                if part in event_asset_map:
                    target_assets_set.update(event_asset_map[part])

        # 3. Поиск, если event_key является подстрокой или содержит ключ из event_asset_map
        # (например, event_key="IRAN", а в event_asset_map есть "IRAN_US")
        for tracked_key, assets in event_asset_map.items():
            # Проверяем, является ли event_key подстрокой tracked_key или наоборот
            if tracked_key != event_key and (event_key in tracked_key or tracked_key in event_key):
                target_assets_set.update(assets)

        # Гарантируем наличие хотя бы одного актива и исключаем пустые значения/None
        target_assets = [a for a in target_assets_set if a]
        if not target_assets:
            target_assets = ["global"]

        # Проверяем анти-спам ДО записи в базу, чтобы не плодить дубли
        can_send_alert = should_send(event_key, score, is_black_swan)

        async with db_lock:
            try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                # Сохраняем событие (link UNIQUE защитит от полных дублей)
                cursor.execute("""
                    INSERT INTO events (title, link, score, event, nasdaq, sp500, oil, soxs, gold, btc, vix, fear_greed, slug, is_black_swan)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (entry.title, entry.link, score, event_type, market["nasdaq"], market["sp500"], market["oil"], market["soxs"], market["gold"], market["btc"], market["vix"], fng_val, slug, 1 if is_black_swan else 0))
                
                for asset_name in target_assets:
                    cursor.execute("""
                        INSERT INTO predictions (event_key, score, predicted_impact, target_asset, resolved) 
                        VALUES (?, ?, ?, ?, 0)
                    """, (event_key, event_scores[event_key], prob, str(asset_name)))
                conn.commit()
        except sqlite3.IntegrityError:
            logging.info(f"Новость уже обработана другой лентой (URL duplicate): {entry.title}")
            continue

        # Отправляем уведомление, если прошли все фильтры и кулдаун
        if can_send_alert:
            if event_key == "BTC" and abs(market_context.get("btc_change", 0)) < config.BTC_MIN_VOLATILITY_FOR_ALERT:
                continue
            
            # Собираем прогнозы и текущие изменения по целевым активам
            forecast_details = []

            # 1. Сначала всегда добавляем глобальный рынок (Market) первым и без процентов
            if any(a.lower() == "global" for a in target_assets):
                forecast_details.append(f"🌍 MARKET: {sig_type}")

            # 2. Добавляем остальные активы
            for asset in target_assets:
                a_key = asset.lower()
                if a_key == "global":
                    continue
                
                change = market_context.get(f"{a_key}_change", 0.0)
                signal = market.get(a_key, "flat").upper()
                icon = "🟢" if "BULLISH" in signal else "🔴" if "BEARISH" in signal else "⚪"
                forecast_details.append(f"{icon} {a_key.upper()}: {signal} ({change:+.2f}%)")
            
            forecast_str = "\n".join(forecast_details)

            # Проверка на дивергенцию (расхождение настроения новости и общего тренда)
            divergence_tag = ""
            # Если итоговый скор очень низкий (Risk-On), а новость пришла с высоким плюсом (Risk-Off)
            if event_scores[event_key] < -5 and score > 1.5:
                divergence_tag = "⚠️ COUNTER-TREND NEWS\n"
            elif event_scores[event_key] > 5 and score < -1.5:
                divergence_tag = "⚠️ COUNTER-TREND NEWS\n"

            black_swan_header = "🦢🦢🦢 BLACK SWAN EVENT 🦢🦢🦢\n" if is_black_swan else ""

            msg = (
                f"{black_swan_header}"
                f"🧠 EVENT: {event_key}\n"
                f"🤖 Model: {source}\n"
                f"{divergence_tag}"
                f"Score: {event_scores[event_key]:.2f} (News: {score:+.2f}) | Impact: {prob:+.2f}%\n"
                f"-------------------\n"
                f"{forecast_str}\n"
                f"-------------------\n"
                f"📰 {entry.title}\n"
                f"🔗 {entry.link}"
            )
            await send_telegram(session, msg)


async def main():
    last_learning_run = 0
    last_cleanup_run = 0
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        # Запускаем цикл обучения сразу при старте, чтобы обработать старые записи
        logging.info("Первичный запуск цикла обучения...")
        await learning_cycle(session)
        last_learning_run = time.time()

        while True:
            eligible_count = count_eligible_predictions()
            time_to_next = max(0, config.LEARNING_INTERVAL - (time.time() - last_learning_run))
            minutes_left = int(time_to_next // 60)
            
            logging.info(f"📡 GTS 4.0 scanning... [До обучения: {minutes_left} мин | Готово новостей: {eligible_count}]")

            current_market_data = await get_market_data(session)
            is_market_active = not current_market_data.get('is_stale', True)
            
            if not is_market_active:
                logging.info("🌙 Night mode: Using slower decay factor to preserve sentiment.")

            # Принудительно обновляем затухание для всех ключей в начале цикла,
            # чтобы Dashboard видел актуальные "остывшие" значения.
            for key in list(event_scores.keys()):
                await apply_decay(key, is_market_active)

            fng_val = current_market_data.get("fng_val", 50)
            fng_label = current_market_data.get("fng_label", "Neutral")

            # Запускаем обработку лент с небольшой задержкой (staggered start),
            # чтобы избежать одновременного удара по серверам Google.
            for url in config.RSS_FEEDS:
                asyncio.create_task(process_single_feed(url, session, loop, fng_val, fng_label, current_market_data, is_market_active))
                await asyncio.sleep(0.5) # Пауза 500мс между запросами к разным лентам

            current_time = time.time()
            if current_time - last_learning_run >= config.LEARNING_INTERVAL:
                await learning_cycle(session)
                last_learning_run = current_time

            if current_time - last_cleanup_run >= config.CLEANUP_INTERVAL:
                await cleanup_db()
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