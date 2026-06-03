import time
import json
import logging
import asyncio
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from db import get_db_connection
import config

class MetricsManager:
    def __init__(self):
        self.metrics = Counter()
        self.ai_timings = []
        self.market_data_timings = []
        self.start_time = time.time()

    def log_report(self, q_size: int, scores_count: int, market_status: str, last_sync: str):
        avg_ai = sum(self.ai_timings) / len(self.ai_timings) if self.ai_timings else 0
        avg_market = sum(self.market_data_timings) / len(self.market_data_timings) if self.market_data_timings else 0
        
        if q_size > 80:
            logging.warning(f"⚠️ High load detected: News queue is {q_size}/500. AI workers might be too slow.")

        logging.info("--- [GTS METRICS REPORT] ---")
        logging.info(f"📊 News: {self.metrics['news_sent_telegram']} sent / {self.metrics['news_received']} received")
        logging.info(f"🛡️ Filters: Source={self.metrics['news_source_filtered']}, URL={self.metrics['news_duplicate_url']}, Fuzzy={self.metrics['news_duplicate_fuzzy']}, Semantic={self.metrics['news_duplicate_semantic']}, Slug={self.metrics['news_duplicate_slug']}, LowScore={self.metrics['news_low_score']}")
        logging.info(f"🧠 AI: Avg Time {avg_ai:.2f}s, Requests {self.metrics['ai_requests']}")
        logging.info(f"📈 Market: Provider={market_status}, LastSync={last_sync}, AvgTime={avg_market:.2f}s")
        logging.info(f"🩺 Health: Queue={q_size}, RAM_Scores={scores_count}, Uptime={round((time.time() - self.start_time)/3600, 2)}h")

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __iadd__(self, other):
        # Это позволит делать state.metrics += Counter(...) если нужно
        self.metrics.update(other)
        return self

    def get(self, key, default=0):
        return self.metrics.get(key, default)

class MarketStateManager:
    def __init__(self):
        self.last_market_data_time = 0
        self.market_data_status = "initializing"

class CacheManager:
    def __init__(self):
        self.urls = OrderedDict()
        self.titles = OrderedDict()
        self.embeddings = OrderedDict()
        self.slugs = OrderedDict()
        self.narrative_counts = defaultdict(int)
        self.cache_lock = asyncio.Lock()

    def is_url_processed(self, url: str) -> bool:
        if url in self.urls:
            self.urls.move_to_end(url)
            return True
        return False

    def add_url(self, url: str, title: str, embedding: Optional[List[float]] = None):
        self.urls[url] = True
        self.titles[title] = True
        if embedding:
            self.embeddings[title] = (embedding, time.time())
        self._prune_caches()

    def _prune_caches(self):
        now = time.time()
        while len(self.urls) > 2000: self.urls.popitem(last=False)
        while len(self.titles) > 1000: self.titles.popitem(last=False)
        while len(self.embeddings) > 1000: self.embeddings.popitem(last=False)
        
        cutoff = now - (config.SLUG_DUPLICATE_HOURS * 3600)
        expired_slugs = [k for k, ts in self.slugs.items() if ts < cutoff]
        for k in expired_slugs:
            self.slugs.pop(k, None)
            self.narrative_counts.pop(k, None)

class LearningManager:
    def __init__(self):
        self.weights = {}
        self.asset_map = {}
        self.source_performance = {}
        self.weight_lock = asyncio.Lock()

    async def load_config_weights(self):
        async with self.weight_lock:
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

    def get_weight(self, event_key: str, asset: str) -> float:
        if (event_key, asset) in self.weights: return self.weights[(event_key, asset)]
        if (event_key, "global") in self.weights: return self.weights[(event_key, "global")]
        return 1.0

class ScoreManager:
    def __init__(self, learning_mgr: LearningManager):
        self.scores = defaultdict(float)
        self.last_update = {}
        self.last_sent = {}
        self.last_sent_score = {}
        self.multiplier = config.IMPACT_MULTIPLIER
        self.asset_multipliers = {}
        self.learning = learning_mgr
        self.score_lock = asyncio.Lock()

    async def update_score(self, event_key: str, asset: str, score: float, is_market_active: bool):
        async with self.score_lock:
            composite_key = (event_key, asset)
            now = time.time()
            self._apply_decay_internal(composite_key, is_market_active, now)
            
            weight = self.learning.get_weight(event_key, asset)
            self.scores[composite_key] = max(-config.MAX_SCORE_THRESHOLD, 
                                            min(config.MAX_SCORE_THRESHOLD, 
                                                self.scores[composite_key] + (score * weight)))

    def _apply_decay_internal(self, composite_key: tuple, is_market_active: bool, now: float):
        if composite_key not in self.scores or self.scores[composite_key] == 0:
            self.last_update[composite_key] = now
            return
        last_upd = self.last_update.get(composite_key, now)
        delta = now - last_upd
        decay = config.DECAY_FACTOR if is_market_active else config.NIGHT_DECAY_FACTOR
        self.scores[composite_key] *= (decay ** (delta / config.DECAY_REFERENCE_SECONDS))
        self.last_update[composite_key] = now

    def __getitem__(self, key):
        return self.scores[key]

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __len__(self):
        return len(self.scores)

    def keys(self):
        return self.scores.keys()

    async def apply_decay(self, composite_key: tuple, is_market_active: bool):
        async with self.score_lock:
            self._apply_decay_internal(composite_key, is_market_active, time.time())

    def get(self, key, default=0.0):
        return self.scores.get(key, default)

class GTSStateManager:
    """Фасад для доступа к специализированным менеджерам состояния."""
    def __init__(self):
        self.metrics = MetricsManager()
        self.market = MarketStateManager()
        self.cache = CacheManager()
        self.learning = LearningManager()
        self.scores = ScoreManager(self.learning)
        
        self.db_lock = asyncio.Lock()
        self.gemini_limiter = asyncio.Semaphore(config.GEMINI_CONCURRENCY)
        self.openrouter_limiter = asyncio.Semaphore(config.OPENROUTER_CONCURRENCY)
        self.deepseek_limiter = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)
        self.hourly_summary_news = []
        self.last_price_alert = {}
        self.learning_rate = config.LEARNING_RATE

    def __getitem__(self, key):
        # Позволяет обращаться к метрикам напрямую через state["key"]
        return self.metrics[key]

    def __setitem__(self, key, value):
        # Позволяет устанавливать метрики напрямую через state["key"] = val
        self.metrics[key] = value

    @property
    def last_market_data_time(self):
        return self.market.last_market_data_time

    @last_market_data_time.setter
    def last_market_data_time(self, value):
        self.market.last_market_data_time = value

    @property
    def market_data_status(self):
        return self.market.market_data_status

    @market_data_status.setter
    def market_data_status(self, value):
        self.market.market_data_status = value

    async def apply_decay(self, key, is_market_active):
        await self.scores.apply_decay(key, is_market_active)

    @property
    def multiplier(self):
        return self.scores.multiplier

    @property
    def weights(self):
        return self.learning.weights

    @property
    def weights_dict(self):
        return self.learning.weights

    @weights_dict.setter
    def weights_dict(self, value):
        self.learning.weights = value

    @multiplier.setter
    def multiplier(self, value):
        self.scores.multiplier = value

    @property
    def asset_multipliers(self):
        return self.scores.asset_multipliers

    @property
    def cache_lock(self):
        return self.cache.cache_lock

    @property
    def asset_map(self):
        return self.learning.asset_map

    @property
    def narrative_counts(self):
        return self.cache.narrative_counts

    @property
    def slugs(self):
        return self.cache.slugs

    @property
    def source_performance(self):
        return self.learning.source_performance

    @property
    def last_sent(self):
        return self.scores.last_sent

    @property
    def last_sent_score(self):
        return self.scores.last_sent_score

    def log_metrics(self, q_size: int = 0):
        """Метод для обратной совместимости с вызовом в engine.py"""
        self.metrics.log_report(
            q_size=q_size, 
            scores_count=len(self.scores),
            market_status=self.market.market_data_status,
            last_sync=datetime.fromtimestamp(self.market.last_market_data_time).strftime('%H:%M:%S') 
            if self.market.last_market_data_time else "Never"
        )

    def add_url(self, url: str, title: str, embedding: Optional[List[float]] = None):
        self.cache.add_url(url, title, embedding)

    def add_news_for_summary(self, news_data: Dict):
        self.hourly_summary_news.append(news_data)

    def clear_hourly_summary_news(self):
        self.hourly_summary_news.clear()

    async def get_db_titles(self, hours: int = 3) -> List[str]:
        async with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT title FROM events WHERE timestamp > datetime('now', '-{hours} hours') ORDER BY timestamp DESC LIMIT 100")
                return [row['title'] for row in cursor.fetchall()]

    async def init_from_db(self):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Загрузка настроек
            cursor.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'")
            row = cursor.fetchone()
            if row: self.scores.multiplier = row[0]

            await self.learning.load_config_weights()
            # Загрузка весов из БД
            cursor.execute("SELECT event_key, target_asset, weight FROM weights")
            for key, asset, val in cursor.fetchall():
                self.learning.weights[(key, asset)] = val
            
            # Загрузка истории баллов (упрощено для краткости)
            cursor.execute("SELECT event_key, target_asset, score, timestamp FROM predictions WHERE timestamp > datetime('now', '-7 day')")
            # ... логика восстановления баллов ...
            logging.info("✅ Состояние успешно инициализировано из БД")

    async def save_to_db(self):
        async with self.db_lock:
            with get_db_connection() as conn:
                # Сохранение весов и множителей
                for (key, asset), val in self.learning.weights.items():
                    conn.execute("INSERT OR REPLACE INTO weights (event_key, target_asset, weight) VALUES (?, ?, ?)", (key, asset, val))
                conn.commit()