import feedparser
import logging
from logging.handlers import RotatingFileHandler
import re
import sqlite3
import time
import json
import asyncio
import aiohttp
import math
import html
import calendar
from google import genai
import pandas as pd
from urllib.parse import urlparse
from difflib import SequenceMatcher
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict, Counter, OrderedDict
from db import get_db_connection, init_db
import config

init_db()

START_TIME = time.time()

# Глобальный маппинг активов на тикеры
ASSET_TICKER_MAP = {
    "nasdaq": "^IXIC",
    "sp500": "^GSPC",
    "acwi": "ACWI",
    "tip": "TIP",
    "oil": "CL=F",
    "vix": "^VIX",
    "gold": "GLD",
    "btc": "BTC-USD",
    "soxs": "SOXS",
    "soxx": "SOXX",
    "global": "GLOBAL_REGIME"
}

# =========================
# LOGGING CONFIG
# =========================

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        RotatingFileHandler(
            config.LOG_FILE, 
            encoding='utf-8', 
            maxBytes=10*1024*1024, # 10 МБ
            backupCount=2          # Хранить 2 старых архива
        ),
        logging.StreamHandler()
    ]
)

# Блокировка технических сообщений Google API (AFC is enabled)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger().addFilter(lambda record: "AFC is enabled" not in record.getMessage())

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
            'gemini-2.5-flash-lite': 6,
            # 'gemini-2.0-flash': 7,
            'gemini-3.5-flash': 8,
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
                {"name": "openai/gpt-oss-120b:free", "supports_json": True, "provider": "openrouter"},
                {"name": "deepseek/deepseek-v4-flash:free", "supports_json": True, "provider": "openrouter"},
                {"name": "poolside/laguna-m.1:free", "supports_json": True, "provider": "openrouter"}
                # {"name": "openrouter/free", "supports_json": False, "provider": "openrouter"}
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
        self.scores = defaultdict(float)  # Key: (event_key, asset)
        self.last_update = {}            # Key: (event_key, asset)
        self.urls = OrderedDict()  # LRU кэш для URL
        self.titles = OrderedDict() # LRU кэш для заголовков (нечеткий поиск)
        self.embeddings = OrderedDict() # LRU кэш для векторных эмбеддингов
        self.slugs = OrderedDict()  # LRU кэш для AI-тегов событий
        self.narrative_counts = defaultdict(int) # Счетчик повторений для Narrative Tracking
        self.metrics = Counter()
        self.ai_timings = []
        self.weights = {}                # Key: (event_key, asset)
        self.source_performance = {} # Накопленный WinRate источников
        self.last_sent = {}
        self.last_price_alert = {}       # Key: asset_name, Value: timestamp
        self.multiplier = config.IMPACT_MULTIPLIER
        self.asset_multipliers = {}      # Key: asset_name
        self.hourly_summary_news = []    # New list to store news for hourly summary
        self.asset_map = {}
        self.score_lock = asyncio.Lock()
        self.weight_lock = asyncio.Lock()
        self.cache_lock = asyncio.Lock()
        self.narrative_lock = asyncio.Lock()
        self.db_lock = asyncio.Lock()
        self.gemini_limiter = asyncio.Semaphore(config.GEMINI_CONCURRENCY)
        self.openrouter_limiter = asyncio.Semaphore(config.OPENROUTER_CONCURRENCY)
        self.last_market_data_time = 0
        self.market_data_status = "initializing"
        self.market_data_timings = []
        self.last_ai_call = 0
        self.learning_rate = config.LEARNING_RATE

    async def init_from_db(self):
        """Загрузка начального состояния из БД."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Загрузка множителя
            cursor.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'")
            row = cursor.fetchone()
            if row: self.multiplier = row[0]
            logging.info(f"✅ IMPACT_MULTIPLIER загружен из БД: {self.multiplier:.2f}")

            # 2. Базовые веса из конфига + БД
            await self._load_config_weights()
            cursor.execute("SELECT event_key, target_asset, weight FROM weights")
            for key, asset, val in cursor.fetchall():
                self.weights[(key, asset)] = val
            
            # 2.5 Загрузка перформанса источников
            cursor.execute("SELECT source_domain, correct_count, total_resolved FROM source_stats WHERE total_resolved > 5")
            for domain, correct, total in cursor.fetchall():
                if domain:
                    winrate = correct / total
                    self.source_performance[domain.lower()] = winrate
            
            # 2.6 Загрузка множителей активов
            cursor.execute("SELECT target_asset, multiplier FROM asset_stats")
            for asset, mult in cursor.fetchall():
                if asset:
                    self.asset_multipliers[asset.lower()] = mult
                
            logging.info(f"--- Веса загружены: {self.weights} ---")

            # 3. Восстановление баллов и времени обновлений
            # Загружаем все индивидуальные прогнозы, чтобы применить затухание к каждому
            cursor.execute("""
                SELECT event_key, target_asset, score, timestamp 
                FROM predictions 
                WHERE timestamp > datetime('now', '-' || ? || ' day')
            """, (config.RAM_SCORE_LOOKBACK_DAYS,))
            
            now_ts = time.time()
            decay_ref = config.DECAY_REFERENCE_SECONDS
            # Вне рынка используем Night Decay для исторической загрузки (более консервативно)
            decay_factor = config.NIGHT_DECAY_FACTOR 

            for row in cursor.fetchall():
                dt = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                age_seconds = now_ts - dt.timestamp()
                
                # Рассчитываем, сколько осталось от балла спустя время
                decayed_val = row['score'] * (decay_factor ** (age_seconds / decay_ref))
                
                composite_key = (row['event_key'], row['target_asset'])
                self.scores[composite_key] += decayed_val
                
                # Сохраняем метку самого свежего обновления
                if dt.timestamp() > self.last_update.get(composite_key, 0):
                    self.last_update[composite_key] = dt.timestamp()
                    self.last_sent[composite_key] = dt.timestamp()

            # Клэмпинг после агрегации
            for k in list(self.scores.keys()):
                self.scores[k] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, self.scores[k]))

            # 4. История для дедупликации
            cursor.execute("SELECT link, title, slug, timestamp FROM events WHERE timestamp > datetime('now', '-' || ? || ' day')", (config.RAM_SCORE_LOOKBACK_DAYS,))
            temp_slug_list = []
            for row in cursor.fetchall():
                self.urls[row['link']] = True
                self.titles[row['title']] = True
                if row['slug']:
                    dt = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    self.slugs[row['slug']] = dt.timestamp()
                    temp_slug_list.append((row['slug'], dt.timestamp()))
            
            # Восстанавливаем счетчики нарративов (Heat) за последние SLUG_DUPLICATE_HOURS
            cutoff_narrative = time.time() - (config.SLUG_DUPLICATE_HOURS * 3600)
            for s_key, s_ts in temp_slug_list:
                if s_ts > cutoff_narrative:
                    self.narrative_counts[s_key] += 1

            # 5. Векторные эмбеддинги для семантической дедупликации (за последние 3 дня)
            cursor.execute("SELECT title, vector, timestamp FROM embeddings WHERE timestamp > datetime('now', '-' || ? || ' days')", (config.RAM_EMBEDDING_LOOKBACK_DAYS,))
            for row in cursor.fetchall():
                try:
                    dt = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    self.embeddings[row['title']] = (json.loads(row['vector']), dt.timestamp())
                except: continue
            self._prune_caches()

    def add_news_for_summary(self, news_data: Dict):
        self.hourly_summary_news.append(news_data)

    def clear_hourly_summary_news(self):
        self.hourly_summary_news.clear()

    async def _load_config_weights(self):
        """Парсинг весов из TRACKED_KEYWORDS."""
        async with self.weight_lock:
            # Очищаем текущие веса и карту активов перед загрузкой из конфига
            self.weights.clear()
            self.asset_map.clear()

            for k, info in config.TRACKED_KEYWORDS.items():
                weight = info[0] if isinstance(info, tuple) else info
                target_assets = info[1] if isinstance(info, tuple) and len(info) > 1 else ["global"]
                
                key_parts = sorted(k.upper().replace(" ", "_").split("_"))
                canonical_key = "_".join(key_parts)
                
                self.asset_map[canonical_key] = target_assets
                for asset in target_assets:
                    self.weights[(canonical_key, asset)] = weight
            
            # Коррекция BTC
            if ("BITCOIN", "btc") in self.weights:
                self.weights[("BTC", "btc")] = self.weights.pop(("BITCOIN", "btc"))

            if ("global", "global") not in self.weights: self.weights[("global", "global")] = 1.0
            if "global" not in self.asset_map: self.asset_map["global"] = ["global"]

    def _internal_decay(self, composite_key: tuple, is_market_active: bool, now: float):
        """Внутренний расчет затухания без блокировки."""
        if composite_key not in self.scores or self.scores[composite_key] == 0:
            self.last_update[composite_key] = now
            return

        last_upd = self.last_update.get(composite_key, now)
        delta = now - last_upd
        if delta <= 0: return

        decay = config.DECAY_FACTOR if is_market_active else config.NIGHT_DECAY_FACTOR
        # Пересчитываем затухание на основе реального времени (в минутах), а не интервалов сканирования
        self.scores[composite_key] *= (decay ** (delta / config.DECAY_REFERENCE_SECONDS))
        self.last_update[composite_key] = now

    async def apply_decay(self, composite_key: tuple, is_market_active: bool) -> float:
        async with self.score_lock:
            self._internal_decay(composite_key, is_market_active, time.time())
            return self.scores[composite_key]

    async def update_score(self, event_key: str, asset: str, score: float, is_market_active: bool):
        """Атомарное обновление балла с учетом PIVOT и затухания."""
        async with self.score_lock:
            composite_key = (event_key, asset)
            now = time.time()
            self._internal_decay(composite_key, is_market_active, now)
            # Pivot logic
            if self.scores[composite_key] != 0 and (self.scores[composite_key] * score) < 0:
                if abs(score) >= config.PIVOT_THRESHOLD:
                    logging.info(f"💥 PIVOT for {event_key} on {asset}")
                    self.scores[composite_key] = 0
            
            weight = self.get_weight(event_key, asset)
            self.scores[composite_key] = max(-config.MAX_SCORE_THRESHOLD, min(config.MAX_SCORE_THRESHOLD, self.scores[composite_key] + (score * weight)))

    def get_weight(self, event_key: str, asset: str) -> float:
        if (event_key, asset) in self.weights: 
            return self.weights[(event_key, asset)]
        # Fallback to global if asset-specific weight not found
        if (event_key, "global") in self.weights:
            return self.weights[(event_key, "global")]
            
        parts = event_key.split('_')
        if len(parts) > 1:
            # Try finding weights for individual parts for the specific asset
            return max([self.get_weight(p, asset) for p in parts])
        return 1.0

    async def save_to_db(self):
        async with self.db_lock:
            # Делаем снимки данных под соответствующими локами, чтобы не блокировать DB-транзакцию
            async with self.weight_lock:
                weights_snapshot = list(self.weights.items())
            
            async with self.score_lock:
                multipliers_snapshot = list(self.asset_multipliers.items())
                global_mult = self.multiplier

            with get_db_connection() as conn:
                cursor = conn.cursor()
                for (key, asset), val in weights_snapshot:
                    cursor.execute("INSERT OR REPLACE INTO weights (event_key, target_asset, weight) VALUES (?, ?, ?)", 
                                 (key, asset, val))
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('impact_multiplier', ?)", (global_mult,))
                for asset, mult in multipliers_snapshot:
                    cursor.execute("UPDATE asset_stats SET multiplier = ? WHERE target_asset = ?", (mult, asset))
                conn.commit()

    def _prune_caches(self):
        """Ограничение размера LRU кэшей."""
        now = time.time()
        while len(self.urls) > 2000: self.urls.popitem(last=False)
        while len(self.titles) > 1000: self.titles.popitem(last=False)
        
        # Очистка слагов и связанных счетчиков нарративов
        while len(self.slugs) > 1000:
            slug_key, _ = self.slugs.popitem(last=False)
            self.narrative_counts.pop(slug_key, None)
            
        # Очистка затухших нарративов по времени (старше окна из конфига)
        cutoff = now - (config.SLUG_DUPLICATE_HOURS * 3600)
        expired_slugs = [k for k, ts in self.slugs.items() if ts < cutoff]
        for k in expired_slugs:
            self.slugs.pop(k, None)
            self.narrative_counts.pop(k, None)

        # Очистка эмбеддингов по количеству и по времени (TTL)
        # Используем RAM_EMBEDDING_LOOKBACK_DAYS для синхронизации с БД
        max_emb_age_sec = config.RAM_EMBEDDING_LOOKBACK_DAYS * 86400
        
        while self.embeddings:
            # Проверяем самый старый элемент (первый в OrderedDict)
            first_title = next(iter(self.embeddings))
            _, ts = self.embeddings[first_title]
            
            if len(self.embeddings) > 500 or (now - ts) > max_emb_age_sec:
                self.embeddings.popitem(last=False)
            else:
                break

        # Очистка "мертвых" баллов (decayed to near-zero)
        # Удаляем ключи, которые затухли ниже порога значимости
        expired_scores = [k for k, v in self.scores.items() if abs(v) < 0.001]
        for k in expired_scores:
            self.scores.pop(k, None)
            self.last_update.pop(k, None)
            self.last_sent.pop(k, None)

    def is_url_processed(self, url: str) -> bool:
        if url in self.urls:
            self.urls.move_to_end(url)
            return True
        return False

    def add_url(self, url: str, title: str, embedding: Optional[List[float]] = None):
        self.urls[url] = True
        self.urls.move_to_end(url)
        self.titles[title] = True
        self.titles.move_to_end(title)
        if embedding:
            self.embeddings[title] = (embedding, time.time())
            try:
                with get_db_connection() as conn:
                    conn.execute("INSERT OR REPLACE INTO embeddings (title, vector) VALUES (?, ?)", 
                                 (title, json.dumps(embedding)))
                    conn.commit()
            except Exception as e:
                logging.debug(f"DB Embedding save skip: {e}")
        self._prune_caches()

    def log_metrics(self):
        """Периодический вывод статистики в лог."""
        avg_ai_time = sum(self.ai_timings) / len(self.ai_timings) if self.ai_timings else 0
        avg_market_time = sum(self.market_data_timings) / len(self.market_data_timings) if self.market_data_timings else 0
        last_sync = datetime.fromtimestamp(self.last_market_data_time).strftime('%H:%M:%S') if self.last_market_data_time else "Never"

        logging.info("--- [GTS METRICS REPORT] ---")
        logging.info(f"📊 News: {self.metrics['news_sent_telegram']} sent / {self.metrics['news_received']} received")
        logging.info(f"🛡️ Filters: Source={self.metrics['news_source_filtered']}, URL={self.metrics['news_duplicate_url']}, Fuzzy={self.metrics['news_duplicate_fuzzy']}, Semantic={self.metrics['news_duplicate_semantic']}, Slug={self.metrics['news_duplicate_slug']}, LowScore={self.metrics['news_low_score']}")
        logging.info(f"🧠 AI: Avg Time {avg_ai_time:.2f}s, Requests {self.metrics['ai_requests']}")
        logging.info(f"📈 Market: Provider={self.market_data_status}, LastSync={last_sync}, AvgTime={avg_market_time:.2f}s")
        logging.info(f"🩺 Health: Queue={news_queue.qsize()}, RAM_Scores={len(self.scores)}, Uptime={round((time.time() - START_TIME)/3600, 2)}h")
        
        # Narrative Tracking Debug
        active_narratives = {k: v for k, v in self.narrative_counts.items() if v > 0}
        if active_narratives:
            top_narratives = dict(sorted(active_narratives.items(), key=lambda x: x[1], reverse=True)[:5])
            logging.info(f"🔥 Active Narratives (top 5): {top_narratives}")

        err_429 = {k: v for k, v in self.metrics.items() if k.startswith("429_")}
        if err_429: logging.info(f"⚠️ Rate Limits (429): {err_429}")
        self.ai_timings = self.ai_timings[-100:] # Храним только последние 100 замеров
        self.market_data_timings = self.market_data_timings[-50:]

    async def get_db_titles(self, hours: int = 3) -> List[str]:
        async with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT title FROM events WHERE timestamp > datetime('now', '-{hours} hours') ORDER BY timestamp DESC LIMIT 100")
                return [row['title'] for row in cursor.fetchall()]

state = GTSStateManager()
model_rotator = ModelRotator(init_model_pool())
news_queue = asyncio.Queue(maxsize=100)

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

async def get_embedding(text: str, rotator: ModelRotator, state: GTSStateManager, session: Optional[aiohttp.ClientSession] = None) -> Optional[List[float]]:
    """Получает векторное представление текста через API с автоматическим фоллбеком."""
    embedding = None

    # 1. Попытка через основной EMBEDDING_MODEL (Gemini)
    async with state.gemini_limiter:
        try:
            res = await client.aio.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=text
            )
            if res and res.embeddings:
                embedding = res.embeddings[0].values
                logging.debug(f"💎 Эмбеддинг получен через основную модель ({config.EMBEDDING_MODEL})")
                return embedding
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                logging.warning(f"⚠️ Лимит основной модели эмбеддингов (429). Переключаюсь на OpenRouter...")
            else:
                logging.warning(f"⚠️ Ошибка основного эмбеддинга ({config.EMBEDDING_MODEL}): {e}")
    
    # 2. Попытка через OPENROUTER_EMBEDDING_MODEL (активируется только при неудаче основной)
    if embedding is None and config.OPENROUTER_API_KEY:
        or_model = config.OPENROUTER_EMBEDDING_MODEL

        s = session if session else aiohttp.ClientSession()
        async with state.openrouter_limiter:
            try:
                async with s.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                    json={
                        "model": or_model,
                        "input": text
                    },
                    timeout=15
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data and len(data['data']) > 0:
                            logging.info(f"✅ Эмбеддинг успешно получен через OpenRouter (Model: {or_model})")
                            return data['data'][0]['embedding']
                    else:
                        logging.warning(f"⚠️ OpenRouter Embeddings Error: {resp.status}")
            except Exception as e:
                logging.warning(f"⚠️ Ошибка эмбеддинга OpenRouter: {e}")
            finally:
                if not session: await s.close()
            
    return embedding

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

async def ai_analyze(text: str, rotator: ModelRotator, pub_time: str = "Unknown", session: Optional[aiohttp.ClientSession] = None, max_retries: int = 3) -> Tuple[Optional[float], Optional[str], Optional[List[str]], Optional[str], bool, str, float]:
    """
    Uses Gemini AI to perform deep sentiment analysis and NER.
    """
    start_time = time.time()
    state.metrics["ai_requests"] += 1
    # Формируем строку с тегами из конфига для подсказки нейросети
    tags_hint = ", ".join([f'"{k}"' for k in config.TRACKED_KEYWORDS.keys()])
    current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    prompt = f"""
    Current Time: {current_time_utc} UTC
    Article Published At: {pub_time}
    Analyze this financial news snippet: "{text}"
    Identify key entities. Prioritize these tags: {tags_hint}, but also detect NEW EMERGENT market narratives (e.g. specific tech like 'HBM', 'Inference', or geopolitical events).
    IMPORTANT: Distinguish between actual assets/companies (e.g., "Gold" as commodity, "Nvidia" as company) and descriptive terms or adjectives (e.g., "gold visa", "oil paintings"). Do not tag an asset if it's used as an adjective or metaphor.
    Identify the core unique event being reported. Determine if this news is a FRESH MARKET CATALYST or a LATE RECAP of a past event.
    SLUG RULES: Use the most specific geographic or entity-based ID. If multiple news report the same strike/event, they MUST have the same slug. For example, use 'iran_strike_us_base' instead of 'middle_east_tensions'.
    Return ONLY a JSON object with this exact structure (no markdown):
    {{
      "primary_asset": "the most impacted asset from the list, or 'global'",
      "score": number,
      "event_type": "military" | "economic" | "diplomatic" | "neutral" | "tech",
      "entities": ["list of countries, companies or key regions"],
      "slug": "STORY_ID. Unique identifier for this specific event. Be consistent across different reports of the same physical event.",
      "is_black_swan": boolean,
      "confidence": number,
      "summary": "Краткий пересказ новости на русском языке (3-4 предложения).",
      "title_ru": "Заголовок новости на русском языке."
    }}

    Scoring Rules: 
    - POSITIVE Score (1.0 to 10.0) = BAD/RISK-OFF news (War, Inflation, Rate Hikes). A positive score predicts that VIX, SOXS, Gold, and Oil will RISE, while Equities will FALL.
    - NEGATIVE Score (-10.0 to -1.0) = GOOD/RISK-ON news (Peace, Rate Cuts, Growth). A negative score predicts that Equities (Nasdaq/SP500) will RISE, while VIX will FALL.
    - Score MUST be 0.0 if the news is a RECAP, SUMMARY, or reports events that ALREADY occurred.
    - Only assign significant scores (abs > 2.0) to BRAND NEW, UNEXPECTED developments.
    """

    for attempt in range(max_retries):
        model_tried_count = 0
        pool_size = len(rotator.pool)
        while model_tried_count < pool_size:
            active = rotator.get_active()
            provider = active.get("provider", "gemini")
            limiter = state.gemini_limiter if provider == "gemini" else state.openrouter_limiter
            
            try:
                async with limiter:
                    # Задержка нужна только для Gemini (free tier), OpenRouter сам управляет очередью
                    if provider == "gemini":
                        delay = config.AI_DELAY_JSON if active.get("supports_json") else config.AI_DELAY_NO_JSON
                        wait_time = max(0, delay - (time.time() - state.last_ai_call))
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)
                        state.last_ai_call = time.time()
                    
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
                        response = await asyncio.wait_for(client.aio.models.generate_content(
                            model=active["name"],
                            contents=prompt,
                            config=gen_config
                        ), timeout=60)
                    
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

                    # Убираем возможные артефакты markdown, если модель их добавила
                    # (даже если промпт просит "ONLY JSON")
                    json_str = res_text[start:end].replace('```json', '').replace('```', '')

                    data = json.loads(json_str)
                    
                    # --- ВАЛИДАЦИЯ ОТВЕТА ---
                    res_slug = data.get("slug")
                    res_type = data.get("event_type")
                    raw_score = float(data.get("score", 0))
                    raw_conf = float(data.get("confidence", 0))

                    # 1. Проверка на наличие обязательных полей (Slug и Event Type)
                    if not res_slug or not res_type:
                        logging.warning(f"⚠️ Модель {active['name']} вернула неполный JSON (отсутствует slug или event_type).")
                        model_tried_count += 1
                        await rotator.rotate()
                        continue

                    # 2. Проверка на "пустой" смысл (Score 0 при высокой уверенности)
                    if raw_score == 0 and raw_conf > 0.5:
                        logging.debug(f"AI Analysis: Нейтральное событие/рекап (Score 0, Conf {raw_conf:.2f}). Пропуск.")
                        return None, None, None, None, False, active["name"], raw_conf, "", ""
                    
                    duration = time.time() - start_time
                    state.ai_timings.append(duration)
                    
                    # Нормализация уверенности: если модель вернула 75.0 вместо 0.75
                    confidence = max(0.0, min(1.0, raw_conf if raw_conf <= 1.0 else raw_conf / 100.0))
                    
                    clamped_raw_score = max(-10.0, min(10.0, raw_score)) # Clamp raw score to expected range [-10, 10]
                    return clamped_raw_score, res_type, data.get("entities", []), res_slug, bool(data.get("is_black_swan", False)), active["name"], confidence, data.get("summary", ""), data.get("title_ru", "")

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
         return None, None, None, None, False, "No Relevance", 0.0, "", ""
    
    slug = "_".join([e.lower() for e in found_entities[:2]]) if found_entities else "general_market"
    return score, "neutral", found_entities, slug, False, "Fallback (Regex)", 0.5, "", ""

# =========================
# EVENT ENGINE
# =========================

def make_event_key(entities: List[str], slug: Optional[str] = None) -> str:
    # Очистка входного списка от пустых значений, None и заглушек
    valid_entities = [
        str(e).strip() for e in (entities or []) 
        if e is not None and str(e).strip().lower() not in ("unknown", "none", "null", "")
    ]

    # Если есть Slug от ИИ, он является лучшим кандидатом на Event Key, 
    # так как ИИ обучен соблюдать консистентность в рамках одного сюжета.
    if slug:
        return slug.strip().upper()

    if not valid_entities:
        return "global"

    # Автоматически добавляем все отслеживаемые слова из конфига в мапу нормализации
    canonical_map = config.ENTITY_CANONICAL_MAP.copy()
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

    # 3. ПОИСК ПЕРЕСЕЧЕНИЙ С TRACKED_KEYWORDS
    # Если сущность содержит в себе или является частью ключа из конфига, 
    # мы принудительно используем ключ из конфига для консолидации.
    consolidated = []
    tracked_keys = [k.upper().replace(" ", "_") for k in config.TRACKED_KEYWORDS.keys()]
    
    for ent in unique_ents:
        found_match = False
        for t_key in tracked_keys:
            # Если "NVIDIA" входит в "NVIDIA_CORP" или наоборот
            if t_key in ent or ent in t_key:
                consolidated.append(t_key)
                found_match = True
                break
        if not found_match:
            consolidated.append(ent)

    # 4. ФОРМИРОВАНИЕ ИТОГОВОГО КЛЮЧА
    # Убираем дубликаты и берем первые MAX_ENTITY_PARTS
    final_parts = sorted(list(set(consolidated)))
    
    # Если после консолидации у нас есть части, которые соответствуют TRACKED_KEYWORDS,
    # отдаем им приоритет перед случайными сущностями.
    priority_parts = [p for p in final_parts if p in tracked_keys]
    if priority_parts:
        return "_".join(priority_parts[:config.MAX_ENTITY_PARTS]).upper()

    return "_".join(final_parts[:config.MAX_ENTITY_PARTS]).upper()

# =========================
# MARKET SIGNAL ENGINE
# =========================

def market_signals(score: float) -> Dict[str, str]:
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

def predict_impact(score: float, multiplier: float, realized_vol: float) -> float:
    """Рассчитывает процент влияния на основе волатильности (Dynamic Volatility Scaling)."""
    return min(abs(score) * multiplier * realized_vol, 100.0)

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

async def send_telegram(session: aiohttp.ClientSession, msg: str, max_length: int = 4000):
    """Отправляет сообщение в Telegram асинхронно."""
    try:
        # Разделяем сообщение на части, если оно слишком длинное
        msg_parts = [msg[i:i + max_length] for i in range(0, len(msg), max_length)]
        
        for part in msg_parts:
            async with session.post(
                    f"https://api.telegram.org/bot{config.BOT_TOKEN}/sendMessage",
                    data={"chat_id": config.CHAT_ID, "text": part, "parse_mode": "HTML"},
                    timeout=10
            ) as response:
                if response.status != 200:
                    logging.error(f"Telegram API error: {response.status} - {await response.text()}")
                else:
                    logging.info(f"TELEGRAM ASYNC: {response.status}")
            await asyncio.sleep(0.5) # Небольшая пауза между частями, чтобы не превысить лимит Telegram

    except Exception as e:
        logging.error(f"Error sending telegram: {e}")

# =========================
# ANTI-SPAM
# =========================

def should_send(key: str, current_score: float, state: GTSStateManager, is_black_swan: bool = False) -> bool:
    now = time.time()

    # Если новость экстремально важная (например, score > 8), игнорируем кулдаун
    if abs(current_score) >= config.BLACK_SWAN_SCORE_THRESHOLD or is_black_swan:
        # Увеличиваем кулдаун до 10 минут (как основной COOLDOWN), чтобы не спамить перепечатками
        # Для критических новостей разрешаем повтор раз в 5 минут, если это реально важно
        if key in state.last_sent and (now - state.last_sent[key] < 300):
             return False
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

async def update_weights(event_key: str, asset: str, error: float, state: GTSStateManager, is_correct: bool = True):
    """Обновляет веса событий на основе ошибки прогноза."""
    async with state.weight_lock:
        # Асимметричное обучение: если направление неверное, учимся быстрее (штрафуем сильнее)
        lr_multiplier = 1.0 if is_correct else config.ASYMMETRIC_LR_FACTOR
        
        # Обучение на основе ошибки прогноза амплитуды и множителя направления
        adjustment = state.learning_rate * error * lr_multiplier

        composite_key = (event_key, asset)
        # Основной ключ получает 100% корректировки
        old_w = state.weights.get(composite_key, 1.0)
        state.weights[composite_key] = max(0.5, min(5.0, old_w + adjustment))
        logging.info(f"📈 Weight for {event_key} ({asset}): {state.weights[composite_key]:.2f}")

        # Атомарное обучение: обновляем части ключа (например, US и IRAN по отдельности)
        # Но с меньшим коэффициентом (например, 50% от основного шага), чтобы не размывать точность
        parts = event_key.split('_')
        if len(parts) > 1 and len(parts) <= config.MAX_ENTITY_PARTS:
            for part in parts:
                p_key = (part, asset)
                if len(part) > 2 and p_key in state.weights:
                    part_old_w = state.weights.get(p_key, 1.0)
                    state.weights[p_key] = max(0.5, min(5.0, part_old_w + (adjustment * 0.5)))

def calibrate_multiplier(avg_error: float, state: GTSStateManager, asset: Optional[str] = None):
    """Корректирует множитель влияния (глобальный или для конкретного актива)."""
    if asset:
        asset_low = asset.lower()
        old_mult = state.asset_multipliers.get(asset_low, state.multiplier)
        # Адаптация чувствительности к сигме
        new_mult = max(0.01, min(2.0, old_mult + (state.learning_rate * avg_error)))
        state.asset_multipliers[asset_low] = new_mult
        if abs(new_mult - old_mult) > 0.0001:
            logging.info(f"⚙️ Multiplier ({asset_low}): {old_mult:.2f} -> {new_mult:.2f}")
    else:
        old_mult = state.multiplier
        # Глобальная калибровка чувствительности к сигме
        state.multiplier = max(0.01, min(2.0, old_mult + (state.learning_rate * avg_error)))
        if abs(state.multiplier - old_mult) > 0.0001:
            logging.info(f"⚙️ Multiplier (GLOBAL): {old_mult:.2f} -> {state.multiplier:.2f}")

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

async def fetch_async_prices(session: aiohttp.ClientSession, tickers: List[str]) -> Tuple[pd.DataFrame, str]:
    """Асинхронная загрузка цен через профессиональный API (TwelveData) с фоллбеком."""
    if config.MARKET_DATA_PROVIDER == "twelvedata" and config.MARKET_DATA_API_KEY:
        # Маппинг тикеров под формат TwelveData
        td_map = {
            "^IXIC": "IXIC", "^GSPC": "SPX", "CL=F": "WTI/USD", 
            "BTC-USD": "BTC/USD", "^VIX": "VIX", "^MOVE": "MOVE", "DX-Y.NYB": "DXY"
        }
        symbols = ",".join([td_map.get(t, t) for t in tickers])
        url = f"https://api.twelvedata.com/time_series?symbol={symbols}&interval=15min&outputsize=500&apikey={config.MARKET_DATA_API_KEY}"
        
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    logging.debug("Цены успешно получены через TwelveData")
                    data = await resp.json()
                    combined = {}
                    for sym, res in data.items():
                        if 'values' in res:
                            df = pd.DataFrame(res['values'])
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df = df.set_index('datetime')['close'].astype(float)
                            # Возвращаем тикер к формату бота
                            inv_map = {v: k for k, v in td_map.items()}
                            combined[inv_map.get(sym, sym)] = df
                    return pd.DataFrame(combined).sort_index(), "twelvedata"
                else:
                    raise Exception(f"API returned status {resp.status}")
        except Exception as e:
            logging.error(f"TwelveData Error: {e}. Falling back to yfinance...")

    # --- FALLBACK TO YFINANCE (Non-Production) ---
    logging.debug("Запрос цен через yfinance (fallback/default)")
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        sync_executor, 
        lambda: yf.download(tickers, period="5d", interval="15m", progress=False)['Close']
    )
    return df, "yfinance"

async def get_market_data(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """
    Fetches recent market data for key assets using yfinance.
    Returns a dictionary with percentage changes for relevant assets.
    """
    market_data = {}
    start_time = time.time()

    tickers_to_fetch = {
        "^IXIC": "nasdaq_change",
        "^GSPC": "sp500_change",
        "ACWI": "acwi_change",
        "TIP": "tip_change",
        "CL=F": "oil_change",
        "^VIX": "vix_change",
        "GLD": "gold_change",
        "BTC-USD": "btc_change",
        "SOXS": "soxs_change",
        "SOXX": "soxx_change",
        "SMH": "smh_change",
        "MU": "mu_change",
        "^MOVE": "move_change",
        "DX-Y.NYB": "dxy_change",
        "HYG": "hyg_change",
        "^TNX": "tnx_yield",
        "^IRX": "irx_yield"
    }

    stale_map = {}
    last_bar_time = 0
    try:
        # Заменяем блокирующий вызов на новый асинхронный метод
        close_prices, provider_name = await fetch_async_prices(session, list(tickers_to_fetch.keys()))
        
        if close_prices.empty:
            logging.error("Market Data fetch failed. Check API keys and provider status.")
            return {}

        duration = time.time() - start_time
        state.market_data_timings.append(duration)
        logging.info(f"✅ Market data fetched from {provider_name.upper()} in {duration:.2f}s")

        market_data['active_provider'] = provider_name
        lookback = config.MARKET_LOOKBACK_HOURS

        # --- РАСЧЕТ COMPOSITE GLOBAL REGIME ---
        try:
            # 1. Расчет кривой доходности (Growth proxy)
            if "^TNX" in close_prices.columns and "^IRX" in close_prices.columns:
                close_prices['YIELD_CURVE'] = close_prices["^TNX"] - close_prices["^IRX"]
            
            # 2. Нормализация доходностей для индекса стресса
            # Мы считаем изменения: рост VIX, MOVE, DXY = +стресс; рост HYG, Curve = -стресс
            returns = close_prices.pct_change().fillna(0)
            
            w = config.GLOBAL_REGIME_WEIGHTS
            stress_returns = (
                returns.get('^VIX', 0) * w['vix'] +
                returns.get('^MOVE', 0) * w['move'] +
                returns.get('DX-Y.NYB', 0) * w['dxy'] +
                (returns.get('HYG', 0) * -1.0) * w['hyg'] +
                (returns.get('YIELD_CURVE', 0) * -1.0) * w['growth']
            )
            
            # Создаем синтетический индекс "Global Stress", стартующий со 100
            global_regime_index = (1 + stress_returns).cumprod() * 100
            close_prices['GLOBAL_REGIME'] = global_regime_index
            
            # Рассчитываем итоговое изменение для алертов
            bars_lookback = config.MARKET_LOOKBACK_HOURS * 4
            if len(global_regime_index) > bars_lookback:
                curr_regime = global_regime_index.iloc[-1]
                past_regime = global_regime_index.iloc[-(bars_lookback + 1)]
                market_data['global_change'] = ((curr_regime - past_regime) / past_regime) * 100
            else:
                market_data['global_change'] = 0.0
        except Exception as e:
            logging.error(f"Error calculating Composite Global Regime: {e}")

        market_data['price_history'] = close_prices # Передаем историю цен для обучения

        # Рассчитываем количество свечей (баров) исходя из 15-минутного интервала
        # 1 час = 4 свечи по 15 минут
        bars_lookback = config.MARKET_LOOKBACK_HOURS * 4

        for ticker_symbol, data_key in tickers_to_fetch.items():
            try:
                # Извлекаем данные для конкретного тикера, если они есть в ответе
                if ticker_symbol in close_prices.columns:
                    ticker_data = close_prices[ticker_symbol].dropna()
                    # Нам нужно как минимум bars_lookback + 1 свечей
                    if len(ticker_data) > bars_lookback:
                        current_price = float(ticker_data.iloc[-1])
                        
                        # Проверяем свежесть данных для конкретного тикера (4 часа)
                        current_bar_time = ticker_data.index[-1].timestamp()
                        is_ticker_stale = (time.time() - current_bar_time) > (4 * 3600)
                        stale_map[data_key] = is_ticker_stale
                        
                        last_bar_time = max(last_bar_time, current_bar_time)
                        
                        # Сравниваем с ценой N часов назад (с учетом интервала свечей)
                        past_price = float(ticker_data.iloc[-(bars_lookback + 1)])
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
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) as phase1,
                SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as phase2
            FROM predictions 
            WHERE resolved < 2 AND timestamp < datetime('now', '-' || ? || ' hours')
        """, (config.MARKET_LOOKBACK_HOURS,))
        row = cursor.fetchone()
        return (row[0] or 0) + (row[1] or 0)

async def learning_cycle(session: aiohttp.ClientSession, state: GTSStateManager, raw_market_data: Optional[Dict] = None):
    if not raw_market_data:
        raw_market_data = await get_market_data(session)
        
    if not raw_market_data or 'price_history' not in raw_market_data:
        logging.warning("Skipping learning cycle: No market data available.")
        return

    price_history = raw_market_data['price_history']
    
    def get_ewma_beta(target_rets, bench_rets):
        """Расчет беты через EWMA для адаптивности к режиму."""
        if len(target_rets) < 10: return 1.0
        # RiskMetrics lambda = 0.94 -> alpha = 1 - 0.94
        alpha = 1 - config.EWMA_LAMBDA
        cov = target_rets.ewm(alpha=alpha).cov(bench_rets).iloc[-1]
        var = bench_rets.ewm(alpha=alpha).var().iloc[-1]
        beta = cov / var if var > 0 else 1.0
        return max(-config.BETA_CLIP, min(config.BETA_CLIP, beta))

    def calculate_expected_move(target_key, bench_cfg, prediction_time, b_move_raw, price_history):
        try:
            target_ticker = ASSET_TICKER_MAP.get(target_key)
            # Получаем доходности для расчета беты (окно перед новостью)
            hist_end = prediction_time.replace(tzinfo=None)
            hist_start = hist_end - timedelta(days=2) 

            if bench_cfg["type"] == "leveraged":
                return b_move_raw * bench_cfg["factor"]
            
            if bench_cfg["type"] == "multi_factor":
                # BTC: 0.7 * Nasdaq_Beta * Nasdaq_Return + (-0.3 * DXY_Beta * DXY_Return)
                expected = 0.0
                # В данной версии упрощаем до взвешенного движения бенчмарков
                # В идеале здесь нужен запуск регрессии
                return b_move_raw * bench_cfg["weights"][0] 

            bench_ticker = bench_cfg["primary"]
            if target_ticker in price_history and bench_ticker in price_history:
                t_rets = price_history[target_ticker].loc[hist_start:hist_end].pct_change().dropna()
                b_rets = price_history[bench_ticker].loc[hist_start:hist_end].pct_change().dropna()
                beta = get_ewma_beta(t_rets, b_rets)
                return b_move_raw * beta
            
            return b_move_raw * bench_cfg.get("factor", 1.0)
        except: return 0.0

    async with state.db_lock:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # JOIN с таблицей events гарантирует, что прогноз привязан к реальному событию из ленты
            cursor.execute("""
                SELECT p.*, e.link as source_link 
                FROM predictions p
                JOIN events e ON p.timestamp = e.timestamp
                WHERE p.resolved < 2 
                ORDER BY timestamp ASC LIMIT 1000
            """)
            rows = cursor.fetchall()
            logging.info(f"🧠 Начало цикла обучения. Найдено кандидатов для обработки: {len(rows)}")

            updates_by_key = defaultdict(list) # Для агрегации обновлений весов
            all_errors = [] # Для калибровки глобального множителя
            errors_by_asset = defaultdict(list) # Для калибровки множителей активов
            stale_map = raw_market_data.get('stale_map', {})
            
            batch_updates = []
            processed_source_links = set() # Чтобы не обновлять источник дважды за одну новость

            for row in rows:
                event_key = row['event_key']
                event_type = row['event_type'] if row['event_type'] else 'neutral'
                is_black_swan = row['is_black_swan'] if 'is_black_swan' in row.keys() else 0
                predicted = row['predicted_impact']
                score = row['score']
                target = row['target_asset'] if row['target_asset'] else "global"

                # 1. Базовая защита от некорректных данных
                if not event_key or event_key == "UNKNOWN":
                    batch_updates.append((2, 0, 0, row['id']))
                    continue

                # 2. Проверка белого списка (только если ONLY_SPECIFIC_SOURCES = True)
                if config.ONLY_SPECIFIC_SOURCES:
                    source_domain = row['source_domain']
                    if source_domain:
                        is_allowed = any(domain in source_domain.lower() for domain in config.SPECIFIC_SOURCES_LIST)
                        if not is_allowed:
                            logging.debug(f"Learning: Skipping {event_key} - source '{source_domain}' not in whitelist anymore")
                            # Помечаем как разрешенное (resolved=2), но не обновляем веса
                            batch_updates.append((2, 0, 0, row['id']))
                            continue

                # Определяем окна на основе конфига
                conf = config.EVENT_TYPE_LOOKBACK.get(event_type, {"primary": 1, "secondary": 4})
                p_win = conf["primary"]
                s_win = conf["secondary"]
                
                if is_black_swan:
                    s_win = max(s_win, config.BLACK_SWAN_LOOKBACK_HOURS)

                # Приводим к UTC для сравнения с индексами yfinance
                prediction_time = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - prediction_time).total_seconds() / 3600
                
                target_lookback = 0
                new_resolved_status = row['resolved']

                # Порог для принудительного удаления из очереди (window + запас 12ч)
                force_resolve_age = s_win + 12

                # Фаза 1: Первичная реакция (Primary)
                if row['resolved'] == 0:
                    if age_hours >= p_win:
                        target_lookback = p_win
                        new_resolved_status = 1
                    else: 
                        # Если новость слишком старая, но так и не прошла Фазу 1
                        if age_hours > force_resolve_age:
                            logging.info(f"🗑️ Force resolve (Phase 1 Expired): {event_key}")
                            batch_updates.append((2, 0, 0, row['id']))
                        continue
                # Фаза 2: Закрепление тренда (Secondary)
                elif row['resolved'] == 1:
                    if age_hours >= s_win:
                        target_lookback = s_win
                        new_resolved_status = 2
                    else: continue

                actual = 0
                raw_change = 0
                correlation = 0
                
                target_ticker = ASSET_TICKER_MAP.get(target.lower())
                if target_ticker and target_ticker in price_history.columns:
                    # Фильтруем серию по времени, чтобы найти цену в момент новости и через N часов
                    ts = price_history[target_ticker].dropna()
                    if ts.index.tz is not None:
                        ts = ts.tz_convert('UTC').tz_localize(None)
                    
                    if ts.empty or prediction_time.replace(tzinfo=None) < ts.index[0]:
                        continue

                    try:
                        prediction_time_naive = prediction_time.replace(tzinfo=None)
                        target_time = prediction_time_naive + timedelta(hours=target_lookback)

                        # Ищем ПЕРВУЮ доступную цену после события (метод backfill)
                        # Это позволяет поймать ГЭП открытия после выходных/ночи
                        idx_at = ts.index.get_indexer([prediction_time_naive], method='backfill')[0]
                        idx_after = ts.index.get_indexer([target_time], method='backfill')[0]

                        if idx_at != -1 and idx_after != -1:
                            # Если индексы одинаковые, значит рынок всё еще закрыт (нет новых свечей)
                            # Мы НЕ помечаем как resolved, а ждем следующего цикла
                            if idx_at == idx_after:
                                # Если это выходной для обычных активов — просто ждем
                                if target.lower() not in ['btc', 'crypto'] and prediction_time.weekday() >= 5:
                                    continue
                                continue
                                
                            p_at = float(ts.iloc[idx_at])
                            p_after = float(ts.iloc[idx_after])
                            
                            if p_at != 0:
                                raw_change = ((p_after - p_at) / p_at) * 100
                                
                                # --- РАСЧЕТ ALPHA (ABNORMAL RETURN) ---
                                alpha_move = raw_change
                                b_cfg = config.ASSET_BENCHMARK_CONFIG.get(target.lower())
                                
                                if b_cfg:
                                    bench_key = b_cfg["primary"]
                                    b_ts = price_history[bench_key].dropna().tz_localize(None)
                                    # Берем те же точки времени для бенчмарка
                                    idx_b_at = b_ts.index.get_indexer([prediction_time_naive], method='backfill')[0]
                                    idx_b_after = b_ts.index.get_indexer([target_time], method='backfill')[0]
                                    
                                    if idx_b_at != -1 and idx_b_after != -1:
                                        b_at = float(b_ts.iloc[idx_b_at])
                                        b_after = float(b_ts.iloc[idx_b_after])
                                        if b_at != 0:
                                            b_move = ((b_after - b_at) / b_at) * 100
                                            expected = calculate_expected_move(target.lower(), b_cfg, prediction_time, b_move, price_history)
                                            
                                            # Расчет реализованной волатильности для Z-нормализации
                                            vol_window = config.VOLATILITY_WINDOW
                                            asset_rets = price_history[target_ticker].pct_change().tail(vol_window)
                                            realized_vol = asset_rets.std() * 100 # В процентах
                                            
                                            alpha_raw = raw_change - expected
                                            # Z-Alpha: нормализуем избыточную доходность по волатильности
                                            z_alpha = alpha_raw / realized_vol if realized_vol > 0.01 else alpha_raw
                                            
                                            alpha_move = z_alpha
                                            logging.info(f"📐 Z-ALPHA [{target}]: Raw {raw_change:+.2f}% (Exp {expected:+.2f}%) | Vol: {realized_vol:.2f}% | Z: {z_alpha:+.2f}")

                                if not correlation:
                                    if target.lower() in ["oil", "vix", "soxs", "gold", "global"]:
                                        correlation = 1
                                    else:
                                        correlation = -1
                                
                                # Для обучения используем очищенную Alpha вместо сырого изменения
                                raw_change = alpha_move
                        else:
                            
                             # Цены отсутствуют в истории - проверяем на "протухание"
                            if age_hours > force_resolve_age:
                                logging.info(f"🗑️ Force resolve (No Price Data): {event_key}")
                                # Не закрываем в 0, если это выходные для фондового рынка
                                if target.lower() not in ['btc', 'crypto'] and datetime.now(timezone.utc).weekday() >= 5:
                                    continue 
                                batch_updates.append((2, 0, -1, row['id']))
                            continue
                    except Exception as e:
                        if age_hours > force_resolve_age:
                            logging.info(f"🗑️ Force resolve (Error): {event_key} - {e}")
                            batch_updates.append((2, 0, -1, row['id']))
                        continue
                else: 
                    if age_hours > force_resolve_age:
                        logging.info(f"🗑️ Force resolve (No Ticker): {event_key}")
                        batch_updates.append((2, 0, -1, row['id']))
                    continue

                dynamic_threshold = config.LEARNING_THRESHOLD * (1 + (target_lookback / 10))
                
                # Даже если изменение цены маленькое, мы ДОЛЖНЫ обновить статус resolved в БД,
                # иначе эти новости будут копиться в бэклоге бесконечно.
                if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
                    batch_updates.append((2, 0, 0, row['id']))
                    logging.debug(f"Learning: Skipping low-score event {event_key} (Score {score:.1f} < Threshold {config.NEUTRAL_SCORE_THRESHOLD})")
                    continue

                # Если фактическое движение рынка ниже динамического порога,
                # мы обычно не учимся на нем, так как это считается рыночным шумом.
                # Однако, если оценка ИИ очень сильна (уровень Black Swan),
                # мы хотим учиться на ЛЮБОМ движении, каким бы малым оно ни было,
                # чтобы точно откалибровать влияние таких критических событий.
                if abs(raw_change) < dynamic_threshold and abs(score) < config.BLACK_SWAN_SCORE_THRESHOLD:
                    batch_updates.append((new_resolved_status, 0, 0, row['id']))
                    continue

                # Если мы дошли сюда, значит движение достаточно значимо ИЛИ score был очень сильным.
                # actual теперь является количеством стандартных отклонений (Z-Score)
                actual = min(abs(raw_change), 10.0)
                
                is_correct = 1 if (score * (row['actual_move'] if row['actual_move'] else raw_change) * correlation) > 0 else 0
                # Если направление неверное, ошибка становится значительно более "негативной",
                # что заставляет систему агрессивнее снижать множители и веса.
                effective_error = (actual - predicted) if is_correct else -(actual + predicted)
                error = effective_error

                if new_resolved_status == 1:
                    # Фаза 1: Быстрая калибровка множителя и фильтрация RAM-баллов
                    all_errors.append(error)
                    errors_by_asset[target.lower()].append(error)
                    if not is_correct and abs(score) > config.NEUTRAL_SCORE_THRESHOLD:
                        async with state.score_lock:
                            # Штрафуем балл в памяти сильнее при неверном направлении
                            state.scores[(event_key, target)] *= 0.3 

                    # Накопление статистики по источникам (только при первом переходе в 'resolved')
                    if row['resolved'] == 0:
                        source_link = row['source_link']
                        if source_link not in processed_source_links:
                            source_domain = row['source_domain'] if row['source_domain'] else "unknown"
                            # Учитываем ложные срабатывания (is_correct=0) как больший вклад в ошибку
                            cursor.execute("""
                                INSERT INTO source_stats (source_domain, total_resolved, correct_count, sum_error, sum_confidence)
                                VALUES (?, 1, ?, ?, ?)
                                ON CONFLICT(source_domain) DO UPDATE SET
                                    total_resolved = total_resolved + 1,
                                    correct_count = correct_count + EXCLUDED.correct_count,
                                    sum_error = sum_error + EXCLUDED.sum_error,
                                    sum_confidence = sum_confidence + EXCLUDED.sum_confidence
                            """, (source_domain.lower(), is_correct, abs(error), row['confidence']))
                            processed_source_links.add(source_link)

                        # NEW: Update asset_stats
                        if target: # Ensure target_asset is not empty
                            cursor.execute("""
                                INSERT INTO asset_stats (target_asset, total_resolved, correct_count, sum_error)
                                VALUES (?, 1, ?, ?)
                                ON CONFLICT(target_asset) DO UPDATE SET
                                    total_resolved = total_resolved + 1,
                                    correct_count = correct_count + EXCLUDED.correct_count,
                                    sum_error = sum_error + EXCLUDED.sum_error
                            """, (target, is_correct, abs(error)))

                else:
                    # Фаза 2: Уточнение веса конкретного события (Long-term)
                    updates_by_key[(event_key, target)].append((error, is_correct))

                batch_updates.append((new_resolved_status, actual, is_correct, row['id']))

            if batch_updates:
                cursor.executemany("""
                    UPDATE predictions SET resolved = ?, actual_move = ?, is_correct = ? WHERE id = ?
                """, batch_updates)
                logging.info(f"✅ Пакетное обновление завершено: {len(batch_updates)} записей.")

            # 1. Агрегированное обновление весов (защита от "двойного" обучения на пачке новостей)
            for (e_key, asset), data_list in updates_by_key.items():
                avg_err = sum(d[0] for d in data_list) / len(data_list)
                mostly_correct = sum(1 for d in data_list if d[1]) / len(data_list) > 0.5
                await update_weights(e_key, asset, avg_err, state, is_correct=mostly_correct)

            # 2. Калибровка множителей по активам
            for asset, errors in errors_by_asset.items():
                avg_asset_err = sum(errors) / len(errors)
                calibrate_multiplier(avg_asset_err, state, asset=asset)
                
                # Автоматический сброс при низком WinRate
                cursor.execute("SELECT total_resolved, correct_count FROM asset_stats WHERE target_asset = ?", (asset,))
                stats = cursor.fetchone()
                if stats and stats['total_resolved'] >= config.MIN_SAMPLE_SIZE_FOR_RESET:
                    wr = (stats['correct_count'] / stats['total_resolved']) * 100
                    if wr < config.MIN_WINRATE_BEFORE_RESET:
                        state.asset_multipliers[asset] = config.IMPACT_MULTIPLIER
                        logging.warning(f"📉 WinRate актива {asset.upper()} ({wr:.1f}%) ниже порога. Множитель сброшен до {config.IMPACT_MULTIPLIER}")

            # 2. Калибровка глобального множителя (один раз за цикл на основе всей выборки)
            if all_errors:
                calibrate_multiplier(sum(all_errors) / len(all_errors), state) # Передаем state
            conn.commit()

    await state.save_to_db() # Сохраняем состояние через state manager
    logging.info(f"System settings saved. New IMPACT_MULTIPLIER: {state.multiplier:.2f}") # Используем state.multiplier

def _execute_vacuum():
    """Вспомогательная функция для выполнения VACUUM в отдельном потоке."""
    try:
        with get_db_connection() as conn:
            conn.execute("PRAGMA journal_mode=DELETE") # Отключаем WAL для VACUUM
            conn.execute("VACUUM")
            conn.execute("PRAGMA journal_mode=WAL") # Возвращаем WAL
            logging.info("📦 База данных сжата (VACUUM завершен)")
    except Exception as e:
        logging.error(f"VACUUM error: {e}")

async def cleanup_db(state: GTSStateManager):
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
                cursor.execute("DELETE FROM embeddings WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.EMBEDDING_RETENTION_DAYS,))
                
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
            
            # VACUUM должен выполняться вне транзакции. 
            # Используем run_in_executor, чтобы не блокировать событийный цикл.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(sync_executor, _execute_vacuum)

            # Полная синхронизация RAM с БД после очистки (восстановление обученных весов)
            await state.init_from_db()
            logging.info(f"--- База данных оптимизирована: удалены данные старше {config.RETENTION_DAYS} дней "
                         f"и {deleted_weights} ключей с весом < {config.MIN_WEIGHT_THRESHOLD} ---")
        except Exception as e:
            logging.error(f"Ошибка при очистке БД: {e}")

# =========================
# MAIN LOOP
# =========================

def clean_title(title: str) -> str:
    """Удаляет мусор из заголовка (названия источников, лишние знаки)."""
    # Удаляем источники в конце: "Title - Reuters", "Title | CNBC", "Title : Source"
    cleaned = re.sub(r'\s+[-|:]\s+.*$', '', title)
    # Удаляем префиксы "Breaking:", "Update:"
    cleaned = re.sub(r'(?i)^(breaking|update|exclusive|just in):\s*', '', cleaned)
    # Список стоп-слов, которые часто меняются в заголовках об одном событии
    stop_words = {'reports', 'hit', 'triggers', 'massive', 'says', 'amid', 'following', 'after', 'due', 'warns', 'shows'}
    # Удаляем пунктуацию
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    # Очищаем от стоп-слов и нормализуем пробелы
    words = [w for w in cleaned.lower().split() if w not in stop_words]
    return " ".join(words)

def is_fuzzy_duplicate(new_title: str, existing_titles: List[str], threshold: float) -> bool:
    """Проверяет заголовок на схожесть с уже существующими в кэше."""
    if not new_title:
        return False
    
    new_clean = clean_title(new_title)
    for title in existing_titles:
        # Сравниваем очищенные версии
        ratio = SequenceMatcher(None, new_clean, clean_title(title)).ratio()
        if ratio > threshold:
            logging.info(f"🚫 Fuzzy duplicate ({ratio:.2f}): '{new_title}' ≈ '{title}'")
            return True
    return False

async def is_semantic_duplicate(new_title: str, new_embedding: List[float], state: GTSStateManager) -> bool:
    """Сравнивает эмбеддинг новой новости с кэшем эмбеддингов."""
    if not new_embedding:
        return False
    
    now = time.time()
    async with state.cache_lock:
        for title, (cached_emb, cached_ts) in state.embeddings.items():
            # Защита от сравнения векторов разной длины (разных моделей)
            if len(new_embedding) != len(cached_emb):
                continue
            similarity = cosine_similarity(new_embedding, cached_emb)
            if similarity > config.SEMANTIC_DUPLICATE_THRESHOLD:
                # Hybrid check: similarity AND time gap
                # Если время между новостями большое (> WINDOW), мы пропускаем как возможное развитие сюжета
                time_gap_h = (now - cached_ts) / 3600
                
                if time_gap_h < config.SEMANTIC_DEDUPLICATION_WINDOW:
                    logging.info(f"🚫 Semantic duplicate ({similarity:.2f}, gap {time_gap_h:.1f}h): '{new_title}'")
                    return True
                else:
                    # Это позволяет системе ловить цепочки эскалации (например, через 12 часов после угрозы последовал удар)
                    logging.info(
                        f"📈 Escalation/Update? High similarity ({similarity:.2f}) "
                        f"but time gap is large ({time_gap_h:.1f}h). Allowing analysis."
                    )
    return False

class SocialEntry(dict):
    """Вспомогательный класс для доступа к словарю через точку (как у объектов feedparser)."""
    def __getattr__(self, name): return self.get(name)

async def fetch_stocktwits(session: aiohttp.ClientSession, symbol: str, market_data: Dict[str, Any]):
    """Специализированный загрузчик для StockTwits (JSON API)."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                for msg in data.get('messages', []):
                    # Приводим JSON StockTwits к формату, похожему на entry из feedparser
                    entry = SocialEntry({
                        'title': msg.get('body'),
                        'link': f"https://stocktwits.com/message/{msg.get('id')}",
                        'published_parsed': time.strptime(msg.get('created_at'), '%Y-%m-%dT%H:%M:%SZ'),
                        'source': {'title': 'StockTwits'},
                        'summary': msg.get('body')
                    })
                    # Отправляем в общую логику фильтрации
                    await process_social_entry(entry, market_data)
    except Exception as e:
        logging.debug(f"StockTwits fetch error for {symbol}: {e}")

async def process_social_entry(entry: Any, market_data: Dict[str, Any]):
    """Общая точка входа для социальных постов после базовой нормализации."""
    await news_queue.put((entry, market_data))

async def process_single_feed(url: str, session: aiohttp.ClientSession, loop: asyncio.AbstractEventLoop, market_data: Dict[str, Any]):
    """Обрабатывает одну RSS ленту."""
    is_market_active = not market_data.get('is_stale', True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    raw_data = None
    for attempt in range(2): # 2 попытки на случай временного сбоя
        try:
            proxy = config.HTTP_PROXY if config.USE_PROXY else None
            async with session.get(
                url, 
                headers=headers, 
                timeout=20, 
                proxy=proxy
            ) as response:
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

    # Адаптивное окно возраста новости:
    # Если рынок активен — окно узкое, если закрыт — используем лимит для неактивного времени.
    max_age_h = config.MAX_NEWS_AGE_HOURS if is_market_active else config.MAX_NEWS_AGE_HOURS_INACTIVE
    
    # 1. Фильтруем всю ленту от старых записей ПЕРЕД обработкой
    fresh_entries = []
    current_utc_ts = datetime.now(timezone.utc).timestamp()
    for entry in feed.entries:
        # Проверяем и дату публикации, и дату обновления (иногда есть только одно из них)
        pub_struct = entry.get('published_parsed') or entry.get('updated_parsed')
        if pub_struct:
            # Используем calendar.timegm для корректного преобразования UTC struct_time
            pub_timestamp = calendar.timegm(pub_struct)
            age_h = (current_utc_ts - pub_timestamp) / 3600
            if age_h <= max_age_h:
                fresh_entries.append(entry)
            else:
                # Переводим в DEBUG, чтобы не спамить в консоль
                logging.debug(f"Skipping old news: '{entry.title}' (Age: {age_h:.1f}h, Max: {max_age_h}h)")
        # Если даты публикации нет, мы её не добавляем (более строгий подход к качеству данных)

    # 2. Сортируем по времени (самые свежие — первые) и берем в пределах лимита
    fresh_entries.sort(
        key=lambda x: calendar.timegm(x.get('published_parsed') or x.get('updated_parsed') or time.gmtime(0)), 
        reverse=True
    )
    max_entries_to_process = config.RSS_MAX_ENTRIES if is_market_active else config.RSS_MAX_ENTRIES_INACTIVE
    
    processed_count = 0
    for entry in fresh_entries:
        state.metrics["news_received"] += 1
        original_title = entry.title

        # Извлекаем название источника для фильтрации (из метаданных или заголовка)
        src_name = entry.get('source', {}).get('title', '').lower()
        if not src_name and ' - ' in entry.title:
            src_name = entry.title.split(' - ')[-1].lower()

        # Если включен режим фильтрации (ONLY_SPECIFIC_SOURCES)
        if config.ONLY_SPECIFIC_SOURCES:
            # Проверяем, есть ли разрешенный домен в ссылке или в названии источника
            link_match = any(re.search(r'(?:^|[./])' + re.escape(domain) + r'(?:[./]|$)', entry.link.lower()) for domain in config.SPECIFIC_SOURCES_LIST)
            
            # src_match: Точное совпадение имени источника с элементом списка или его частями (без частичных вхождений)
            src_match = False
            if src_name:
                normalized_feed_src_for_word_match = src_name.lower()
                for allowed_domain in config.SPECIFIC_SOURCES_LIST:
                    domain_parts = allowed_domain.lower().split('.')
                    root_domain = domain_parts[0]
                    
                    # Используем границы слов \b, чтобы "x" не матчило "fox"
                    # Также обрабатываем специфические маппинги для крупных СМИ
                    if re.search(r'\b' + re.escape(root_domain) + r'\b', normalized_feed_src_for_word_match) or \
                       (root_domain == "wsj" and "wall street journal" in normalized_feed_src_for_word_match) or \
                       (root_domain == "x" and "twitter" in normalized_feed_src_for_word_match) or \
                       (root_domain == "ft" and "financial times" in normalized_feed_src_for_word_match):
                        src_match = True
                        break
            
            # Если это ссылка от агрегатора (Google/Yahoo), требуем, чтобы именно ИСТОЧНИК (src_match) был в белом списке
            if "google.com" in entry.link.lower() or "yahoo.com" in entry.link.lower():
                if not src_match:
                    logging.debug(f"Filter: Source '{src_name}' not in whitelist for {entry.link}")
                    state.metrics["news_source_filtered"] += 1
                    continue
            elif not link_match:
                logging.info(f"🛡️ Filter Blocked: {entry.link} (Add this domain to SPECIFIC_SOURCES_LIST)")
                state.metrics["news_source_filtered"] += 1
                continue

        # Теперь ограничиваем количество новостей, которые ПРОШЛИ фильтр по белому списку
        processed_count += 1
        if processed_count > max_entries_to_process:
            logging.debug(f"Reached max_entries_to_process ({max_entries_to_process}) for feed {url}")
            break

        # 1. Быстрая проверка на дубликаты (URL и Fuzzy)
        use_semantic = config.USE_EMBEDDINGS and (config.GEMINI_API_KEY or config.OPENROUTER_API_KEY)
        fuzzy_threshold = config.DUPLICATE_TITLE_THRESHOLD if use_semantic else config.FALLBACK_DUPLICATE_THRESHOLD

        async with state.cache_lock:
            if state.is_url_processed(entry.link):
                state.metrics["news_duplicate_url"] += 1
                continue

            # Проверка по заголовкам из кэша (быстрая)
            if is_fuzzy_duplicate(entry.title, list(state.titles.keys()), fuzzy_threshold):
                state.metrics["news_duplicate_fuzzy"] += 1
                state.add_url(entry.link, entry.title)
                continue

        # Проверка в БД (перед тяжелым AI запросом эмбеддинга для экономии API)
        db_titles = await state.get_db_titles()
        if is_fuzzy_duplicate(original_title, db_titles, fuzzy_threshold):
            state.metrics["news_duplicate_fuzzy"] += 1
            continue

        # 2. Получение эмбеддинга (БЕЗ блокировки, так как это сетевой запрос)
        new_embedding = None
        if use_semantic:
            new_embedding = await get_embedding(entry.title, model_rotator, state, session=session)
            
            # 3. Семантическая проверка (is_semantic_duplicate сам управляет своей блокировкой)
            if new_embedding and await is_semantic_duplicate(entry.title, new_embedding, state):
                state.metrics["news_duplicate_semantic"] += 1
                async with state.cache_lock:
                    state.add_url(entry.link, entry.title, embedding=new_embedding)
                continue

        # 4. Финальное добавление в кэш (если новость прошла все фильтры)
        async with state.cache_lock:
            state.add_url(entry.link, entry.title, embedding=new_embedding)

        # Ставим в очередь для AI анализа
        # Реализация вытесняющей очереди (Sliding Window)
        # Если очередь полна, удаляем старейший элемент перед вставкой нового
        try:
            news_queue.put_nowait((entry, market_data))
        except asyncio.QueueFull:
            try:
                # Извлекаем и сразу помечаем как "завершенный" (отброшенный) элемент
                news_queue.get_nowait()
                news_queue.task_done() 
                
                # Теперь место гарантированно есть (с учетом GIL и асинхронности)
                news_queue.put_nowait((entry, market_data))
                logging.warning("⚠️ Очередь переполнена: старая новость вытеснена для сохранения свежести данных.")
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                # В случае экстремальной гонки данных просто пропускаем текущую новость
                pass

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
    
    # Извлекаем время публикации из RSS (published или updated)
    pub_time = entry.get('published') or entry.get('updated') or "Unknown Date"

    try:
        text_for_ai = entry.title + " " + entry.get("summary", "") # Use a different variable name to avoid confusion
        analysis = await ai_analyze(text_for_ai, rotator, pub_time=pub_time, session=session)
        if analysis[0] is None: return
        score, event_type, entities, slug, is_black_swan, model_name, confidence, ai_summary, title_ru = analysis

        # Сброс флага Black Swan, если score новости недостаточно велик (защита от галлюцинаций ИИ)
        if is_black_swan and abs(score) < config.BLACK_SWAN_SCORE_THRESHOLD:
            logging.info(f"🦢 Понижение статуса Black Swan для {slug}: индивидуальный score {score:.2f} < {config.BLACK_SWAN_SCORE_THRESHOLD}")
            is_black_swan = False

        narrative_multiplier = 1.0

        # Фильтр по уровню уверенности
        if confidence < config.CONFIDENCE_THRESHOLD:
            logging.info(f"Skipping news '{entry.title}': Confidence {confidence:.2f} is below threshold {config.CONFIDENCE_THRESHOLD}")
            return

        # Slug Logic (Duplicates & Narrative Tracking)
        normalized_slug = slug.strip().lower() if slug else None
        async with state.narrative_lock: # Используем лок из state manager
            if normalized_slug:
                if normalized_slug in state.slugs:
                    delta_sec = time.time() - state.slugs[normalized_slug]
                    
                    # Spam prevention: if news arrives too fast (< 15 min), it's a duplicate
                    if delta_sec < 900:
                        logging.info(f"🐌 Slug spam prevention: {normalized_slug}")
                        state.metrics["news_duplicate_slug"] += 1
                        return
                    
                    # Narrative Boost logic (if enabled)
                    if delta_sec < config.SLUG_DUPLICATE_HOURS * 3600:
                        state.narrative_counts[normalized_slug] += 1
                        boost = state.narrative_counts[normalized_slug] * config.NARRATIVE_BOOST_PER_HIT
                        narrative_multiplier = min(config.NARRATIVE_MAX_MULTIPLIER, 1.0 + boost)
                        logging.info(f"📈 Narrative Boost for {normalized_slug}: x{narrative_multiplier:.2f} (Hit #{state.narrative_counts[normalized_slug]})")
                    else:
                        # Если прошло слишком много времени, нарратив "остыл", начинаем заново
                        state.narrative_counts[normalized_slug] = 1
                
                state.slugs[normalized_slug] = time.time()
                state.slugs.move_to_end(normalized_slug)
                
                while len(state.slugs) > 1000: state.slugs.popitem(last=False)

        # Определение рейтинга доверия источнику новости
        # Google News RSS обычно указывает источник в конце заголовка через дефис или в поле source
        source_meta = entry.get('source', {}).get('title', '').lower()
        # Если метаданных нет, пробуем достать из заголовка или домена ссылки
        source_title = source_meta
        if not source_title and ' - ' in entry.title:
            source_title = entry.title.split(' - ')[-1].lower()
        if not source_title:
            source_title = urlparse(entry.link).netloc.lower().replace("www.", "")

        # Комбинированный Trust Factor: Статика + Динамика (из БД)
        base_trust = config.DEFAULT_TRUST_SCORE
        for s_key, s_weight in config.SOURCE_TRUST_LEVELS.items():
            if s_key.lower() in source_title:
                base_trust = s_weight
                break
        
        # Если по источнику есть статистика, корректируем базовый траст
        dynamic_adj = 1.0
        s_low = source_title.lower()
        if s_low in state.source_performance:
            wr = state.source_performance[s_low]
            if wr > 0.65: dynamic_adj = 1.2  # Повышаем вес надежным
            elif wr < 0.45: dynamic_adj = 0.7 # Снижаем вес часто ошибающимся
        
        trust_factor = base_trust * dynamic_adj
        
        # Применяем коэффициент доверия источника и уверенность модели
        score *= (trust_factor * confidence)
        
        # ПРИМЕНЯЕМ NARRATIVE BOOST
        score *= narrative_multiplier
        
        event_key = make_event_key(entities, slug=slug)

        # Улучшенный поиск активов
        target_assets_set = set()

        # 1. Прямое совпадение event_key с ключом в event_asset_map
        if event_key in state.asset_map: # Используем state.asset_map
            target_assets_set.update(state.asset_map[event_key])

        # 2. Поиск по частям event_key, но только если сущностей немного
        # Ограничение через MAX_ENTITY_PARTS делает систему строже, исключая случайные связи
        parts = event_key.split('_')
        event_parts_set = set(parts)
        
        if len(parts) <= config.MAX_ENTITY_PARTS:
            for part in parts:
                if part in state.asset_map: # Используем state.asset_map
                    target_assets_set.update(state.asset_map[part])

        # 3. Поиск по точному вхождению слов (частей ключа) для исключения вложений типа GOLD в GOLDMAN
        for tracked_key, assets in state.asset_map.items(): # Используем state.asset_map
            tracked_parts_set = set(tracked_key.split('_'))
            # Если все части отслеживаемого ключа есть в событии или наоборот
            if tracked_key != event_key and (tracked_parts_set.issubset(event_parts_set) or event_parts_set.issubset(tracked_parts_set)):
                target_assets_set.update(assets)

        # Гарантируем наличие "global" для корректного расчета общего скора и сигналов в Telegram
        target_assets = [a for a in target_assets_set if a]
        if "global" not in target_assets:
            target_assets.append("global")

        # --- DYNAMIC NARRATIVE DISCOVERY ---
        if config.NARRATIVE_AUTO_DISCOVERY and normalized_slug:
            if state.narrative_counts[normalized_slug] >= config.MIN_NARRATIVE_STREAK:
                async with state.db_lock:
                    with get_db_connection() as conn:
                        for asset in target_assets:
                            weight_key = (normalized_slug.upper(), asset.lower())
                            if weight_key not in state.weights:
                                state.weights[weight_key] = 1.0
                                conn.execute("INSERT OR IGNORE INTO weights (event_key, target_asset, weight) VALUES (?, ?, 1.0)",
                                           (weight_key[0], weight_key[1]))
                                logging.info(f"🆕 DISCOVERED NARRATIVE: {weight_key[0]} now tracked for {asset}")

        # Обновляем баллы для каждого целевого актива
        for asset_name in target_assets:
            await state.update_score(event_key, asset_name, score, is_market_active)

        # Фильтр значимости: теперь в базу попадают только новости с баллом >= NEUTRAL_SCORE_THRESHOLD
        if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
            state.metrics["news_low_score"] += 1
            logging.info(f"Skipping news for {event_key}: Score {score:.2f} is below threshold {config.NEUTRAL_SCORE_THRESHOLD}")
            return # Используем return вместо continue, так как это функция

        # Находим самый волатильный актив (с максимальным накопленным баллом) для заголовка
        top_asset = "global"
        top_score = state.scores.get((event_key, "global"), 0.0)
        
        for asset_name in target_assets:
            current_a_score = state.scores.get((event_key, asset_name), 0.0)
            if abs(current_a_score) > abs(top_score):
                top_score = current_a_score
                top_asset = asset_name

        ref_score = top_score
        
        # Динамический расчет волатильности для сигналов
        price_history = market_data.get('price_history')
        
        top_asset_low = top_asset.lower()
        top_ticker = ASSET_TICKER_MAP.get(top_asset_low)
        top_vol = 1.0
        if price_history is not None and top_ticker and top_ticker in price_history.columns:
            top_vol = price_history[top_ticker].pct_change().tail(config.VOLATILITY_WINDOW).std() * 100
            if pd.isna(top_vol) or top_vol < 0.01: top_vol = 1.0
            
        top_multiplier = state.asset_multipliers.get(top_asset_low, state.multiplier)
        prob = predict_impact(ref_score, top_multiplier, top_vol)
        
        market = market_signals(ref_score)
        sig_type = generate_signal(prob, ref_score)

        # Проверяем анти-спам ДО записи в базу, чтобы не плодить дубли
        can_send_alert = should_send(event_key, score, state, is_black_swan) 

        async with state.db_lock: # Используем лок из state manager
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    # Сохраняем событие (link UNIQUE защитит от полных дублей)
                    cursor.execute("""
                        INSERT INTO events (title, link, score, event, nasdaq, sp500, oil, soxs, gold, btc, vix, fear_greed, slug, is_black_swan, summary, title_ru)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (entry.title, entry.link, score, event_type, market["nasdaq"], market["sp500"], market["oil"], market["soxs"], market["gold"], market["btc"], market["vix"], fng_val, slug, 1 if is_black_swan else 0, ai_summary, title_ru))
                    
                    for asset_name in target_assets:
                        a_low = asset_name.lower()
                        a_ticker = ASSET_TICKER_MAP.get(a_low)
                        a_vol = 1.0
                        if price_history is not None and a_ticker and a_ticker in price_history.columns:
                            a_vol = price_history[a_ticker].pct_change().tail(config.VOLATILITY_WINDOW).std() * 100
                            if pd.isna(a_vol) or a_vol < 0.01: a_vol = 1.0
                        
                        a_mult = state.asset_multipliers.get(a_low, state.multiplier)
                        a_score = state.scores.get((event_key, a_low), 0.0)
                        a_prob = predict_impact(a_score, a_mult, a_vol)

                        cursor.execute("""
                            INSERT INTO predictions (event_key, score, predicted_impact, target_asset, resolved, event_type, is_black_swan, confidence, source_domain, model_name)
                            VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                        """, (event_key, a_score, a_prob, a_low, event_type, 1 if is_black_swan else 0, confidence, source_title, model_name))
                    conn.commit()

                if config.ENABLE_HOURLY_REPORT:
                    # Добавляем новость в список для часового отчета (только один раз для события)
                    state.add_news_for_summary({
                        "event_key": event_key,
                        "score": score,
                        "impact": prob, 
                        "title": title_ru or entry.title,
                        "summary": ai_summary,
                        "link": entry.link,
                        "source": source_title
                    })

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
                g_chg = market_data.get("global_change", 0.0)
                forecast_details.append(f"🌍 MARKET: {sig_type} (Stress: {g_chg:+.2f}%)")

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
            if top_score < -5 and score > 1.5:
                divergence_tag = "<b>⚠️ COUNTER-TREND NEWS</b>\n"
            elif top_score > 5 and score < -1.5:
                divergence_tag = "<b>⚠️ COUNTER-TREND NEWS</b>\n"

            black_swan_header = "<b>🦢🦢🦢 BLACK SWAN EVENT 🦢🦢🦢</b>\n" if is_black_swan else ""
            narrative_tag = f"<b>🔥 NARRATIVE x{narrative_multiplier:.2f}</b>\n" if narrative_multiplier > 1.0 else ""
            summary_part = f"📝 <b>Summary:</b> {html.escape(ai_summary)}\n\n" if ai_summary else ""

            msg = (
                f"{black_swan_header}"
                f"🧠 <b>EVENT:</b> {html.escape(event_key)}\n"
                f"🤖 <b>Model:</b> {html.escape(model_name)} (Conf: {confidence:.2f})\n"
                f"📢 <b>Source:</b> {html.escape(source_title.upper())}\n"
                f"{divergence_tag}"
                f"{narrative_tag}"
                f"<b>Score ({top_asset.upper()}):</b> {top_score:.2f} (News: {score:+.2f}) | <b>Impact:</b> {prob:+.2f}%\n"
                f"-------------------\n"
                f"{forecast_str}\n"
                f"-------------------\n"
                f"{summary_part}"
                f"📰 <a href='{html.escape(entry.link)}'>{html.escape(title_ru or entry.title)}</a>"
            )
            state.metrics["news_sent_telegram"] += 1
            await send_telegram(session, msg)
    except Exception as e: # Добавлена обработка ошибок для воркера
        logging.error(f"Error processing news in queue: {e}")


async def send_hourly_summary(session: aiohttp.ClientSession, state: GTSStateManager):
    """Формирует и отправляет ежечасный отчет по новостям в Telegram."""
    news_to_report = state.hourly_summary_news.copy()
    
    # Если память пуста (после перезагрузки), пробуем восстановить из БД
    if not news_to_report:
        try:
            async with state.db_lock:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT event_key, score, predicted_impact as impact, title_ru as title, summary, link, source_domain as source
                        FROM predictions p
                        JOIN events e ON p.timestamp = e.timestamp
                        WHERE p.timestamp >= datetime('now', '-1 hour')
                        GROUP BY e.link ORDER BY e.timestamp DESC
                    """)
                    news_to_report = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Error recovering hourly summary: {e}")
            return

    if not news_to_report:
        return

    summary_msg_parts = [
        "<b>--- GTS HOURLY NEWS SUMMARY ---</b>",
        f"Отчет за час до {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]

    # Анализ точности моделей за последние 6 часов (чтобы захватить недавно разрешенные прогнозы)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_name, COUNT(*) as total, SUM(is_correct) as correct
                FROM predictions 
                WHERE resolved >= 1 AND timestamp >= datetime('now', '-6 hours')
                GROUP BY model_name
            """)
            rows = cursor.fetchall()
            
            if rows:
                summary_msg_parts.append("\n<b>🏆 MODEL ACCURACY (Last 6h):</b>")
                best_model = None
                max_wr = -1.0
                
                for row in rows:
                    name = (row['model_name'] or "Unknown").split('/')[-1]
                    total = row['total']
                    correct = row['correct'] or 0
                    wr = (correct / total * 100) if total > 0 else 0
                    summary_msg_parts.append(f"🤖 <b>{html.escape(name)}:</b> {wr:.1f}% ({correct}/{total})")
                    if wr > max_wr:
                        max_wr = wr
                        best_model = name
                summary_msg_parts.append("-------------------")
    except Exception as e:
        logging.error(f"Error calculating model stats for summary: {e}")

    for news_item in news_to_report:
        summary_msg_parts.append(f"\n🧠 <b>EVENT:</b> {html.escape(news_item['event_key'])} | <b>Score:</b> {news_item['score']:.2f} | <b>Impact:</b> {news_item['impact']:.2f}%")
        summary_msg_parts.append(f"📢 <b>Source:</b> {html.escape(news_item.get('source', 'Unknown').upper())}")
        summary_msg_parts.append(f"📰 <a href='{html.escape(news_item['link'])}'>{html.escape(news_item['title'])}</a>")
        if news_item['summary']:
            summary_msg_parts.append(f"📝 <b>Summary:</b> {html.escape(news_item['summary'])}")
        summary_msg_parts.append(f"-------------------")
    
    final_summary_msg = "\n".join(summary_msg_parts)
    
    logging.info("Отправка ежечасного отчета по новостям в Telegram.")
    await send_telegram(session, final_summary_msg)
    state.clear_hourly_summary_news()

async def get_healthcheck() -> Dict[str, Any]:
    """Возвращает текущие показатели здоровья системы."""
    return {
        "status": "ok",
        "queue_size": news_queue.qsize(),
        "scores_in_ram": len(state.scores),
        "uptime_seconds": int(time.time() - START_TIME),
        "ai_requests": state.metrics["ai_requests"],
        "active_model": model_rotator.get_active()["name"],
        "market_data_provider": state.market_data_status,
        "last_market_sync": datetime.fromtimestamp(state.last_market_data_time).strftime('%H:%M:%S') if state.last_market_data_time else "Never"
    }

async def main():
    last_learning_run = 0
    last_cleanup_run = 0
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        workers = []
        try:
            # Инициализация состояния из БД в асинхронном контексте
            await state.init_from_db()
            logging.info(f"🚀 Поставщик рыночных данных: {config.MARKET_DATA_PROVIDER.upper()}")
            
            # Запуск воркеров на основе конфигурации (1 для Free Gemini, 2+ для платных тарифов)
            workers = [asyncio.create_task(news_worker(i, session, state, model_rotator)) for i in range(config.NUM_WORKERS)]

            # Первичный запуск цикла обучения, чтобы обработать старые записи
            logging.info("Первичный запуск цикла обучения...")
            await learning_cycle(session, state)
            last_learning_run = time.time()
            
            logging.info("Первичная очистка базы данных...")
            await cleanup_db(state)
            last_cleanup_run = time.time()
            last_summary_run = time.time() # Инициализируем время последнего отчета

            while True:
                eligible_count = count_eligible_predictions()
                time_to_next = max(0, config.LEARNING_INTERVAL - (time.time() - last_learning_run))
                minutes_left = int(time_to_next // 60)
                
                logging.info(f"📡 GTS 4.0 scanning... [До обучения: {minutes_left} мин | Готово новостей: {eligible_count}]")

                current_market_data = await get_market_data(session)
                if current_market_data:
                    state.last_market_data_time = time.time()
                    state.market_data_status = current_market_data.get('active_provider', 'unknown')
                else:
                    state.market_data_status = "FAILED"

                is_market_active = not current_market_data.get('is_stale', True)
                
                if not is_market_active:
                    logging.info("🌙 Night mode: Using slower decay factor to preserve sentiment.")

                # Мониторинг резких движений цен (Нефть)
                if is_market_active:
                    oil_move = current_market_data.get("oil_change", 0.0)
                    if abs(oil_move) >= config.OIL_SHARP_MOVE_THRESHOLD:
                        now = time.time()
                        if now - state.last_price_alert.get("oil", 0) > config.COOLDOWN:
                            direction = "🚀 РОСТ" if oil_move > 0 else "🔻 ПАДЕНИЕ"
                            alert_msg = (
                                f"⚠️ <b>РЕЗКОЕ ДВИЖЕНИЕ ЦЕНЫ: OIL</b>\n"
                                f"Направление: {direction} <b>{oil_move:+.2f}%</b>\n"
                                f"Окно мониторинга: ~{config.MARKET_LOOKBACK_HOURS}ч"
                            )
                            await send_telegram(session, alert_msg)
                            state.last_price_alert["oil"] = now

                for key in list(state.scores.keys()):
                    await state.apply_decay(key, is_market_active)

                for url in config.RSS_FEEDS:
                    asyncio.create_task(process_single_feed(url, session, loop, current_market_data))
                    await asyncio.sleep(0.5) # Пауза 500мс между запросами к разным лентам
                
                # Дополнительно опрашиваем StockTwits для ключевых тикеров
                if config.SOCIAL_SEARCH_ENABLED:
                    for asset in ["NVDA", "BTC", "TSLA", "AMD"]:
                        asyncio.create_task(fetch_stocktwits(session, asset, current_market_data))
                        await asyncio.sleep(1)

                current_time = time.time()
                if current_time - last_learning_run >= config.LEARNING_INTERVAL:
                    await learning_cycle(session, state, raw_market_data=current_market_data)
                    last_learning_run = current_time
                if current_time - last_cleanup_run >= config.CLEANUP_INTERVAL:
                    await cleanup_db(state)
                    last_cleanup_run = current_time
                if config.ENABLE_HOURLY_REPORT and current_time - last_summary_run >= config.HOURLY_SUMMARY_INTERVAL:
                    await send_hourly_summary(session, state)
                    last_summary_run = current_time

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