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
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict, Counter, OrderedDict
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

# Полная блокировка сообщения про AFC через глобальный фильтр
# logging.getLogger().addFilter(lambda record: "AFC is enabled" not in record.getMessage())

# logging.getLogger("google").setLevel(logging.WARNING)
# logging.getLogger("absl").setLevel(logging.WARNING)

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

class ModelRotator:
    """Атомарная ротация моделей для AI-вызовов."""
    def __init__(self, pool):
        self.pool = pool
        self._idx = 0
        self._lock = asyncio.Lock()

    def get_active(self) -> Dict:
        return self.pool[self._idx]

    async def rotate(self) -> Dict:
        async with self._lock:
            self._idx = (self._idx + 1) % len(self.pool)
            return self.get_active()

class GTSStateManager:
    """Инкапсуляция всего состояния GTS: баллы, веса, дедупликация."""
    def __init__(self):
        self.scores = defaultdict(float)
        self.last_update = {}
        self.urls = OrderedDict()  # LRU кэш для URL
        self.titles = OrderedDict() # LRU кэш для заголовков (нечеткий поиск)
        self.slugs = OrderedDict()  # LRU кэш для AI-тегов событий
        self.metrics = Counter()
        self.ai_timings = []
        self.weights = {}
        self.last_sent = {}
        self.multiplier = config.IMPACT_MULTIPLIER
        self.asset_map = {}
        self.lock = asyncio.Lock()
        self.db_lock = asyncio.Lock()
        self.learning_rate = config.LEARNING_RATE

    def init_from_db(self):
        """Загрузка начального состояния из БД."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Загрузка множителя
            cursor.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'")
            row = cursor.fetchone()
            if row: self.multiplier = row[0]
            logging.info(f"✅ IMPACT_MULTIPLIER загружен из БД: {self.multiplier:.2f}")

            # 2. Базовые веса из конфига + БД
            self._load_config_weights()
            cursor.execute("SELECT event_key, weight FROM weights")
            for key, val in cursor.fetchall():
                self.weights[key] = val
            logging.info(f"--- Веса загружены: {self.weights} ---")

            # 3. Восстановление баллов и времени обновлений
            cursor.execute("""
                SELECT event_key, SUM(raw_score), MAX(timestamp) FROM (
                    SELECT MAX(score) as raw_score, event_key, timestamp
                    FROM predictions WHERE timestamp > datetime('now', '-1 day')
                    GROUP BY event_key, timestamp
                ) GROUP BY event_key
            """)
            for key, val, ts_str in cursor.fetchall():
                self.scores[key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, val))
                dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                self.last_update[key] = dt.timestamp()
                self.last_sent[key] = dt.timestamp()

            # 4. История для дедупликации
            cursor.execute("SELECT link, title, slug, timestamp FROM events WHERE timestamp > datetime('now', '-1 day')")
            for row in cursor.fetchall():
                self.urls[row['link']] = True
                self.titles[row['title']] = True
                if row['slug']:
                    dt = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    self.slugs[row['slug']] = dt.timestamp()
            self._prune_caches()

    def _load_config_weights(self):
        """Парсинг весов из TRACKED_KEYWORDS."""
        # Очищаем текущие веса и карту активов перед загрузкой из конфига
        self.weights.clear()
        self.asset_map.clear()

        for k, info in config.TRACKED_KEYWORDS.items():
            weight = info[0] if isinstance(info, tuple) else info
            target_assets = info[1] if isinstance(info, tuple) and len(info) > 1 else ["global"]
            
            key_parts = sorted(k.upper().replace(" ", "_").split("_"))
            canonical_key = "_".join(key_parts)
            
            self.weights[canonical_key] = weight
            self.asset_map[canonical_key] = target_assets
        
        if "BITCOIN" in self.weights: 
            self.weights["BTC"] = self.weights.pop("BITCOIN")
            if "BITCOIN" in self.asset_map: self.asset_map["BTC"] = self.asset_map.pop("BITCOIN")
        
        if "GLOBAL" not in self.weights: self.weights["GLOBAL"] = 1.0
        if "GLOBAL" not in self.asset_map: self.asset_map["GLOBAL"] = ["global"]

    async def apply_decay(self, key: str, is_market_active: bool) -> float:
        async with self.lock:
            if key not in self.scores or self.scores[key] == 0:
                self.last_update[key] = time.time()
                return 0.0
            now = time.time()
            last_upd = self.last_update.get(key, now)
            delta = now - last_upd
            decay = config.DECAY_FACTOR if is_market_active else config.NIGHT_DECAY_FACTOR
            intervals = delta / config.CHECK_INTERVAL
            self.scores[key] *= (decay ** intervals)
            self.last_update[key] = now
            return self.scores[key]

    async def update_score(self, key: str, score: float, is_market_active: bool):
        """Атомарное обновление балла с учетом PIVOT и затухания."""
        await self.apply_decay(key, is_market_active)
        async with self.lock:
            # Pivot logic
            if self.scores[key] != 0 and (self.scores[key] * score) < 0:
                if abs(score) >= config.PIVOT_THRESHOLD:
                    logging.info(f"💥 PIVOT for {key}")
                    self.scores[key] = 0
            
            weight = await self.get_weight(key)
            self.scores[key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, self.scores[key] + (score * weight)))

    async def get_weight(self, event_key: str) -> float:
        if event_key in self.weights: return self.weights[event_key]
        parts = event_key.split('_')
        if len(parts) > 1:
            return max([self.weights.get(p, 1.0) for p in parts])
        return 1.0

    async def save_to_db(self):
        async with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for key, val in self.weights.items():
                    cursor.execute("INSERT OR REPLACE INTO weights (event_key, weight) VALUES (?, ?)", (key, val))
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('impact_multiplier', ?)", (self.multiplier,))
                conn.commit()

    def _prune_caches(self):
        """Ограничение размера LRU кэшей."""
        while len(self.urls) > 2000: self.urls.popitem(last=False)
        while len(self.titles) > 1000: self.titles.popitem(last=False)
        while len(self.slugs) > 1000: self.slugs.popitem(last=False)

    def is_url_processed(self, url: str) -> bool:
        if url in self.urls:
            self.urls.move_to_end(url)
            return True
        return False

    def add_url(self, url: str, title: str):
        self.urls[url] = True
        self.urls.move_to_end(url)
        self.titles[title] = True
        self.titles.move_to_end(title)
        self._prune_caches()

    def log_metrics(self):
        """Периодический вывод статистики в лог."""
        avg_ai_time = sum(self.ai_timings) / len(self.ai_timings) if self.ai_timings else 0
        logging.info("--- [GTS METRICS REPORT] ---")
        logging.info(f"📊 News: {self.metrics['news_sent_telegram']} sent / {self.metrics['news_received']} received")
        logging.info(f"🛡️ Filters: URL={self.metrics['news_duplicate_url']}, Fuzzy={self.metrics['news_duplicate_fuzzy']}, Slug={self.metrics['news_duplicate_slug']}, LowScore={self.metrics['news_low_score']}")
        logging.info(f"🧠 AI: Avg Time {avg_ai_time:.2f}s, Requests {self.metrics['ai_requests']}")
        
        err_429 = {k: v for k, v in self.metrics.items() if k.startswith("429_")}
        if err_429: logging.info(f"⚠️ Rate Limits (429): {err_429}")
        self.ai_timings = self.ai_timings[-100:] # Храним только последние 100 замеров

    async def get_db_titles(self, hours: int = 3) -> List[str]:
        async with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT title FROM events WHERE timestamp > datetime('now', '-{hours} hours') ORDER BY timestamp DESC LIMIT 20")
                return [row['title'] for row in cursor.fetchall()]

state = GTSStateManager()
state.init_from_db()
model_rotator = ModelRotator(init_model_pool())
news_queue = asyncio.Queue()

logging.info(f"Пул моделей готов: {[m['name'] for m in model_rotator.pool]}. Старт с: {model_rotator.get_active()['name']}")
logging.info(f"--- Текущий IMPACT_MULTIPLIER: {state.multiplier:.2f} ---")

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

async def ai_analyze(text: str, rotator: ModelRotator, session: Optional[aiohttp.ClientSession] = None, max_retries: int = 3) -> Tuple[Optional[float], Optional[str], Optional[List[str]], Optional[str], bool, str]:
    """
    Uses Gemini AI to perform deep sentiment analysis and NER.
    """
    start_time = time.time()
    state.metrics["ai_requests"] += 1
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
        pool_size = len(rotator.pool)
        while model_tried_count < pool_size:
            try:
                active = rotator.get_active()
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
                    await rotator.rotate()
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
                    await rotator.rotate()
                    continue # Попробуем следующую модель в пуле немедленно

                data = json.loads(res_text[start:end])
                
                duration = time.time() - start_time
                state.ai_timings.append(duration)
                return float(data.get("score", 0)), data.get("event_type", "neutral"), data.get("entities", []), data.get("slug"), bool(data.get("is_black_swan", False)), active["name"]

            except Exception as e:
                err_msg = str(e).lower()
                # Обработка 404 (модель не найдена) и 429 (лимиты/таймауты)
                if any(x in err_msg for x in ["429", "404", "quota", "limit", "timeout"]):
                    state.metrics[f"429_{active['name']}"] += 1
                    old_name = rotator.get_active()["name"] # Получаем имя текущей модели до ротации
                    new_model = await rotator.rotate() # Ротируем и получаем новую модель
                    model_tried_count += 1
                    logging.warning(f"⚠️ Модель {old_name} недоступна ({err_msg}). Переключаюсь на {new_model['name']}...")
                    if model_tried_count == pool_size: # Если все модели в пуле исчерпали лимит
                        break # Все модели в пуле исчерпали лимит, выходим из внутреннего цикла
                    continue # Пробуем следующую модель в пуле немедленно
                else:
                    # Другая ошибка (например, модель не поддерживает JSON mode). 
                    # Логируем, переключаемся на следующую модель и пробуем снова в этом же цикле.
                    logging.error(f"⚠️ Ошибка модели {active['name']}: {e}")
                    model_tried_count += 1
                    await rotator.rotate()
                    continue # Пробуем следующую модель в пуле немедленно
        
        # Если весь пул моделей исчерпан (все вернули 429 или ошибки)
        wait_time = (attempt + 1) * 60 # Увеличиваем время ожидания с каждой попыткой
        logging.warning(f"⚠️ Все модели в пуле ({pool_size}) временно недоступны. Повтор через {wait_time}s...")
        
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

def predict_impact(score: float, state: GTSStateManager) -> float:
    # Вес уже применен в event_scores, здесь используем только глобальный множитель
    return min(abs(score) * state.multiplier, 100)

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

def should_send(key: str, current_score: float, state: GTSStateManager, is_black_swan: bool = False) -> bool:
    now = time.time()

    # Если новость экстремально важная (например, score > 8), игнорируем кулдаун
    if abs(current_score) >= 8.0 or is_black_swan:
        # Но даже для важных новостей даем 2 минуты, чтобы не слать дубли от разных агентств
        if key in state.last_sent and (now - state.last_sent[key] < 120):
            logging.info(f"High-score spam prevention for {key}")
            return False # Используем state.last_sent
        state.last_sent[key] = now
        return True

    if key not in state.last_sent:
        state.last_sent[key] = now
        return True

    if now - state.last_sent[key] > config.COOLDOWN:
        state.last_sent[key] = now
        return True

    # Очистка старых записей из памяти (простой механизм prune)
    if len(state.last_sent) > 1000:
        cutoff = now - (config.COOLDOWN * 2)
        keys_to_del = [k for k, v in state.last_sent.items() if v < cutoff]
        for k in keys_to_del: del state.last_sent[k]

    return False

# =========================
# LEARNING SYSTEM
# =========================

async def update_weights(event_key: str, error: float, state: GTSStateManager):
    """Обновляет веса событий на основе ошибки прогноза."""
    adjustment = state.learning_rate * error * 0.01

    # Обновляем основной ключ
    state.weights[event_key] = max(0.5, min(5.0, state.weights.get(event_key, 1.0) + adjustment))
    logging.info(f"📈 Weight for {event_key}: {state.weights[event_key]:.2f}")

    # Атомарное обучение (опционально): обновляем части ключа, только если это не части одного имени/названия.
    parts = event_key.split('_')
    if len(parts) > 1 and len(parts) <= config.MAX_ENTITY_PARTS:
        for part in parts:
            if len(part) > 2 and part in state.weights:
                state.weights[part] = max(0.5, min(5.0, state.weights.get(part, 1.0) + adjustment))

def calibrate_multiplier(avg_error: float, state: GTSStateManager):
    """Корректирует глобальный множитель влияния на основе средней ошибки всей выборки."""
    old_mult = state.multiplier
    # Используем меньший шаг для стабильности (0.005)
    state.multiplier = max(1.0, min(10.0, old_mult + (state.learning_rate * avg_error * 0.005)))
    if abs(state.multiplier - old_mult) > 0.0001:
        logging.info(f"⚙️ Multiplier: {old_mult:.2f} -> {state.multiplier:.2f}")

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
            lambda: yf.download(list(tickers_to_fetch.keys()), period="1wk", interval="15m", progress=False)
        )
        
        if all_data.empty or 'Close' not in all_data.columns:
            logging.error("Yahoo Finance returned no data. Check internet connection and system clock.")
            return {}

        lookback = config.MARKET_LOOKBACK_HOURS

        # Кэшируем доступ к ценам закрытия для оптимизации
        close_prices = all_data['Close']

        market_data['price_history'] = close_prices # Передаем историю цен для обучения

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


async def learning_cycle(session: aiohttp.ClientSession, state: GTSStateManager):
    raw_market_data = await get_market_data(session)
    if not raw_market_data or 'price_history' not in raw_market_data:
        logging.warning("Skipping learning cycle: No market data available.")
        return

    price_history = raw_market_data['price_history']
    # Маппинг внутренних имен активов на тикеры yfinance
    asset_ticker_map = {
        "nasdaq": "^IXIC",
        "sp500": "^GSPC",
        "oil": "CL=F",
        "vix": "^VIX",
        "gold": "GLD",
        "btc": "BTC-USD",
        "soxs": "SOXS"
    }

    async with state.db_lock:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE resolved < 2 
                ORDER BY timestamp DESC LIMIT 200
            """)
            rows = cursor.fetchall()
            logging.info(f"🧠 Начало цикла обучения. Найдено кандидатов для обработки: {len(rows)}")

            updates_by_key = defaultdict(list) # Для агрегации обновлений весов
            all_errors = [] # Для калибровки глобального множителя
            stale_map = raw_market_data.get('stale_map', {})

            for row in rows:
                event_key = row['event_key']
                event_type = row['event_type'] if row['event_type'] else 'neutral'
                is_black_swan = row['is_black_swan'] if 'is_black_swan' in row.keys() else 0
                predicted = row['predicted_impact']
                score = row['score']
                target = row['target_asset'] if row['target_asset'] else "global"

                # Определяем окна на основе конфига
                conf = config.EVENT_TYPE_LOOKBACK.get(event_type, {"primary": 1, "secondary": 4})
                p_win = conf["primary"]
                s_win = conf["secondary"]
                
                if is_black_swan:
                    s_win = max(s_win, config.BLACK_SWAN_LOOKBACK_HOURS)

                # Приводим к UTC для сравнения с индексами yfinance
                prediction_time = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=None)
                age_hours = (datetime.now(timezone.utc).replace(tzinfo=None) - prediction_time).total_seconds() / 3600
                
                target_lookback = 0
                new_resolved_status = row['resolved']

                # Фаза 1: Первичная реакция (Primary)
                if row['resolved'] == 0:
                    if age_hours >= p_win:
                        target_lookback = p_win
                        new_resolved_status = 1
                    else: continue
                # Фаза 2: Закрепление тренда (Secondary)
                elif row['resolved'] == 1:
                    if age_hours >= s_win:
                        target_lookback = s_win
                        new_resolved_status = 2
                    else: continue

                actual = 0
                raw_change = 0
                correlation = 0
                
                target_ticker = asset_ticker_map.get(target.lower())
                if target.lower() == "global":
                    target_ticker = "^VIX"
                    correlation = 1

                if target_ticker and target_ticker in price_history:
                    # Фильтруем серию по времени, чтобы найти цену в момент новости и через N часов
                    ts = price_history[target_ticker].dropna()
                    ts.index = ts.index.tz_localize(None) # Убираем TZ для сравнения
                    
                    try:
                        # Цена в момент новости (ближайшая доступная)
                        price_at_news = ts.asof(prediction_time)
                        # Цена через target_lookback часов после новости
                        target_time = prediction_time + timedelta(hours=target_lookback)
                        price_after_lookback = ts.asof(target_time)
                        
                        if price_at_news and price_after_lookback and price_at_news != 0:
                            raw_change = ((price_after_lookback - price_at_news) / price_at_news) * 100
                            if not correlation:
                                if target.lower() in ["oil", "vix", "soxs", "gold"]:
                                    correlation = 1
                                else:
                                    correlation = -1
                        else: continue
                    except Exception: continue
                else: continue

                dynamic_threshold = config.LEARNING_THRESHOLD * (1 + (target_lookback / 10))
                if abs(raw_change) < dynamic_threshold:
                    continue

                if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
                    cursor.execute("UPDATE predictions SET resolved = 2 WHERE id = ?", (row['id'],))
                    logging.debug(f"Learning: Skipping low-score event {event_key} (Score {score:.1f} < Threshold {config.NEUTRAL_SCORE_THRESHOLD})")
                    continue

                scaling = config.ASSET_SCALING_FACTORS.get(target, config.ASSET_SCALING_FACTORS["global"])
                actual = min(abs(raw_change) * scaling, 100)
                
                is_correct = 1 if (score * raw_change * correlation) > 0 else 0
                error = actual - predicted

                if new_resolved_status == 1:
                    # Фаза 1: Быстрая калибровка множителя и фильтрация RAM-баллов
                    all_errors.append(error)
                    if is_correct == 0 and abs(score) > config.NEUTRAL_SCORE_THRESHOLD:
                        async with state.lock:
                            state.scores[event_key] *= 0.5 # Штраф за ошибку в Primary окне
                else:
                    # Фаза 2: Уточнение веса конкретного события (Long-term)
                    updates_by_key[event_key].append(error)

                cursor.execute("""
                    UPDATE predictions
                    SET resolved = ?, actual_move = ?, is_correct = ?
                    WHERE id = ?
                """, (new_resolved_status, actual, is_correct, row['id']))

            # 1. Агрегированное обновление весов (защита от "двойного" обучения на пачке новостей)
            for e_key, errors in updates_by_key.items():
                avg_err = sum(errors) / len(errors)
                await update_weights(e_key, avg_err, state) # Передаем state

            # 2. Калибровка глобального множителя (один раз за цикл на основе всей выборки)
            if all_errors:
                calibrate_multiplier(sum(all_errors) / len(all_errors), state) # Передаем state
            conn.commit()

    await state.save_to_db() # Сохраняем состояние через state manager
    logging.info(f"System settings saved. New IMPACT_MULTIPLIER: {state.multiplier:.2f}") # Используем state.multiplier

async def cleanup_db(state: GTSStateManager): # Принимаем state
    """
    Удаляет записи из БД, которые старше RETENTION_DAYS, чтобы предотвратить разрастание файла.
    Также удаляет ключи из таблицы весов, значение которых ниже MIN_WEIGHT_THRESHOLD.
    """
    async with state.db_lock:
        try:
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
                
                # Обновляем веса
                state._load_config_weights()
                
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

async def process_single_feed(url: str, session: aiohttp.ClientSession, loop: asyncio.AbstractEventLoop, market_data: Dict[str, Any]):
    """Обрабатывает одну RSS ленту."""
    is_market_active = not market_data.get('is_stale', True)
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
        state.metrics["news_received"] += 1
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

        async with state.lock: # Используем лок из state manager
            if state.is_url_processed(entry.link):
                state.metrics["news_duplicate_url"] += 1
                continue
            if is_fuzzy_duplicate(entry.title, list(state.titles.keys()), config.DUPLICATE_TITLE_THRESHOLD):
                state.metrics["news_duplicate_fuzzy"] += 1
                state.add_url(entry.link, entry.title)
                continue
            state.add_url(entry.link, entry.title)

        # Проверка в БД
        db_titles = await state.get_db_titles()
        if is_fuzzy_duplicate(original_title, db_titles, config.DUPLICATE_TITLE_THRESHOLD):
            state.metrics["news_duplicate_fuzzy"] += 1
            continue # Пропускаем, если дубликат, и идем к следующей новости

        # Ставим в очередь для AI анализа
        await news_queue.put((entry, market_data))

async def news_worker(worker_id: int, session: aiohttp.ClientSession, state: GTSStateManager, rotator: ModelRotator):
    """Воркер для обработки новостей из очереди."""
    logging.info(f"Worker {worker_id} started.")
    while True:
        entry, market_data = await news_queue.get()
        try:
            await process_queued_news(entry, market_data, session, state, rotator)
        except Exception as e:
            logging.error(f"Worker {worker_id} error: {e}")
        finally:
            news_queue.task_done()

async def process_queued_news(entry: Any, market_data: Dict, session: aiohttp.ClientSession, state: GTSStateManager, rotator: ModelRotator):
    """AI анализ и скоринг новости из очереди."""
    is_market_active = not market_data.get('is_stale', True)
    fng_val = market_data.get("fng_val", 50)
    
    try:
        text = entry.title + " " + entry.get("summary", "")
        analysis = await ai_analyze(text, rotator, session=session)
        if analysis[0] is None: return
        score, event_type, entities, slug, is_black_swan, source = analysis

        # Семантическая проверка дубликатов по slug (AI-generated)
        async with state.lock: # Используем лок из state manager
            if slug and slug in state.slugs:
                if time.time() - state.slugs[slug] < config.MAX_NEWS_AGE_HOURS * 3600:
                    state.metrics["news_duplicate_slug"] += 1
                    return
            if slug:
                state.slugs[slug] = time.time()
                state.slugs.move_to_end(slug)
                while len(state.slugs) > 1000: state.slugs.popitem(last=False)

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
        
        current_delay = config.AI_DELAY_JSON if rotator.get_active()["supports_json"] else config.AI_DELAY_NO_JSON
        await asyncio.sleep(current_delay)

        event_key = make_event_key(entities)

        # Обновляем скор, включая затухание и PIVOT логику
        await state.update_score(event_key, score, is_market_active)

        # Фильтр значимости: теперь в базу попадают только новости с баллом >= NEUTRAL_SCORE_THRESHOLD
        if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
            state.metrics["news_low_score"] += 1
            logging.info(f"Skipping news for {event_key}: Score {score:.2f} is below threshold {config.NEUTRAL_SCORE_THRESHOLD}")
            return # Используем return вместо continue, так как это функция

        market = market_signals(state.scores[event_key], event_key) # Используем state.scores
        prob = predict_impact(state.scores[event_key], state) # Используем state.scores и state
        sig_type = generate_signal(prob, state.scores[event_key]) # Используем state.scores

        # Улучшенный поиск активов: проверяем event_key и его части на соответствие event_asset_map
        target_assets_set = set()

        # 1. Прямое совпадение event_key с ключом в event_asset_map
        if event_key in state.asset_map: # Используем state.asset_map
            target_assets_set.update(state.asset_map[event_key])

        # 2. Поиск по частям event_key, но только если сущностей немного
        # Ограничение через MAX_ENTITY_PARTS делает систему строже, исключая случайные связи
        parts = event_key.split('_')
        if len(parts) <= config.MAX_ENTITY_PARTS:
            for part in parts:
                if part in state.asset_map: # Используем state.asset_map
                    target_assets_set.update(state.asset_map[part])

        # 3. Поиск, если event_key является подстрокой или содержит ключ из event_asset_map
        # (например, event_key="IRAN", а в event_asset_map есть "IRAN_US")
        for tracked_key, assets in state.asset_map.items(): # Используем state.asset_map
            # Проверяем, является ли event_key подстрокой tracked_key или наоборот
            if tracked_key != event_key and (event_key in tracked_key or tracked_key in event_key):
                target_assets_set.update(assets)

        # Гарантируем наличие хотя бы одного актива и исключаем пустые значения/None
        target_assets = [a for a in target_assets_set if a]
        if not target_assets: target_assets = ["global"]

        # Проверяем анти-спам ДО записи в базу, чтобы не плодить дубли
        can_send_alert = should_send(event_key, score, state, is_black_swan) # Передаем state

        async with state.db_lock: # Используем лок из state manager
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
                            INSERT INTO predictions (event_key, score, predicted_impact, target_asset, resolved, event_type, is_black_swan)
                            VALUES (?, ?, ?, ?, 0, ?, ?)
                        """, (event_key, state.scores[event_key], prob, str(asset_name), event_type, 1 if is_black_swan else 0))
                    conn.commit()
            except sqlite3.IntegrityError:
                logging.info(f"Новость уже обработана другой лентой (URL duplicate): {entry.title}") # Используем return вместо continue
                return

        # Отправляем уведомление, если прошли все фильтры и кулдаун
        if can_send_alert:
            if event_key == "BTC" and abs(market_data.get("btc_change", 0)) < config.BTC_MIN_VOLATILITY_FOR_ALERT: # Используем market_data
                return # Используем return вместо continue
            
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
                
                change = market_data.get(f"{a_key}_change", 0.0) # Используем market_data
                signal = market.get(a_key, "flat").upper()
                icon = "🟢" if "BULLISH" in signal else "🔴" if "BEARISH" in signal else "⚪"
                forecast_details.append(f"{icon} {a_key.upper()}: {signal} ({change:+.2f}%)")
            
            forecast_str = "\n".join(forecast_details)

            # Проверка на дивергенцию (расхождение настроения новости и общего тренда)
            divergence_tag = ""
            # Если итоговый скор очень низкий (Risk-On), а новость пришла с высоким плюсом (Risk-Off)
            if state.scores[event_key] < -5 and score > 1.5: # Используем state.scores
                divergence_tag = "⚠️ COUNTER-TREND NEWS\n"
            elif state.scores[event_key] > 5 and score < -1.5: # Используем state.scores
                divergence_tag = "⚠️ COUNTER-TREND NEWS\n"

            black_swan_header = "🦢🦢🦢 BLACK SWAN EVENT 🦢🦢🦢\n" if is_black_swan else ""

            msg = (
                f"{black_swan_header}"
                f"🧠 EVENT: {event_key}\n"
                f"🤖 Model: {source}\n"
                f"{divergence_tag}"
                f"Score: {state.scores[event_key]:.2f} (News: {score:+.2f}) | Impact: {prob:+.2f}%\n" # Используем state.scores
                f"-------------------\n"
                f"{forecast_str}\n"
                f"-------------------\n"
                f"📰 {entry.title}\n"
                f"🔗 {entry.link}"
            )
            state.metrics["news_sent_telegram"] += 1
            await send_telegram(session, msg)
    except Exception as e: # Добавлена обработка ошибок для воркера
        logging.error(f"Error processing news in queue: {e}")


async def main():
    last_learning_run = 0
    last_cleanup_run = 0
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        workers = []
        try:
            # Запуск воркеров
            workers = [asyncio.create_task(news_worker(i, session, state, model_rotator)) for i in range(2)]

            # Первичный запуск цикла обучения, чтобы обработать старые записи
            logging.info("Первичный запуск цикла обучения...")
            await learning_cycle(session, state)
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

                for key in list(state.scores.keys()):
                    await state.apply_decay(key, is_market_active)

                for url in config.RSS_FEEDS:
                    asyncio.create_task(process_single_feed(url, session, loop, current_market_data))
                    await asyncio.sleep(0.5) # Пауза 500мс между запросами к разным лентам

                current_time = time.time()
                if current_time - last_learning_run >= config.LEARNING_INTERVAL:
                    await learning_cycle(session, state)
                    last_learning_run = current_time
                if current_time - last_cleanup_run >= config.CLEANUP_INTERVAL:
                    await cleanup_db(state)
                    last_cleanup_run = current_time

                state.log_metrics()
                await asyncio.sleep(config.CHECK_INTERVAL)
        except asyncio.CancelledError:
            logging.info("Основной цикл остановлен (CancelledError).")
        finally:
            # Явная остановка фоновых воркеров
            for w in workers:
                w.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            logging.info("Все фоновые задачи завершены.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("GTS 4.0: Работа завершена пользователем (Ctrl+C).")
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        # Гарантированный запуск очистки ресурсов при любом завершении (Ctrl+C, ошибка, системный сигнал)
        shutdown_cleanup()