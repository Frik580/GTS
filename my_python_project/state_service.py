import time
import json
import logging
import asyncio
import pandas as pd  # <-- Добавлен импорт Pandas
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
        logging.info(f"🛡️ Filters: Src={self.metrics['news_source_filtered']}, URL={self.metrics['news_duplicate_url']}, Hash={self.metrics['news_duplicate_hash']}, Fzy={self.metrics['news_duplicate_fuzzy']}, Sem={self.metrics['news_duplicate_semantic']}, Slug={self.metrics['news_duplicate_slug']}, LowSc={self.metrics['news_low_score']}, LowConf={self.metrics['news_low_confidence']}, Triv={self.metrics['news_trivial']}, SocMute={self.metrics['news_social_muted']}, BTCIgnore={self.metrics['news_btc_ignored']}, Cool={self.metrics['news_cooldown_filtered']}, DB_Dup={self.metrics['news_db_duplicate']}")
        logging.info(f"🧠 AI: Avg Time {avg_ai:.2f}s, Requests {self.metrics['ai_requests']}")
        logging.info(f"📈 Market: Provider={market_status}, LastSync={last_sync}, AvgTime={avg_market:.2f}s")
        logging.info(f"🩺 Health: Queue={q_size}, RAM_Scores={scores_count}, Uptime={round((time.time() - self.start_time)/3600, 2)}h")

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __iadd__(self, other):
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
        self.clean_titles = OrderedDict() 
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
        while len(self.clean_titles) > 2000: self.clean_titles.popitem(last=False)
        
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
        self.model_sensitivities = {}
        self.dirty_weights = set()  
        self.dirty_model_sensitivities = set() 
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

    def lookup_source_performance(self, source_domain: str) -> Optional[Dict[str, float]]:
        return self.source_performance.get(source_domain.lower())

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
        self.dirty_multiplier = False
        self.dirty_asset_multipliers = set()
        self.learning = learning_mgr
        self.score_lock = asyncio.Lock()

    async def update_score(self, event_key: str, asset: str, score: float, is_market_active: bool):
        async with self.score_lock:
            composite_key = (event_key, asset)
            now = time.time()
            self._apply_decay_internal(composite_key, is_market_active, now)
            
            correlation = config.ASSET_CORRELATION_MAP.get(asset.lower(), -1)
            weight = self.learning.get_weight(event_key, asset)
            self.scores[composite_key] = max(-config.MAX_SCORE_THRESHOLD, 
                                            min(config.MAX_SCORE_THRESHOLD, 
                                                self.scores[composite_key] + (score * weight * correlation)))

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
        
        self.ai_client = None  
        self.db_lock = asyncio.Lock()
        self.gemini_limiter = asyncio.Semaphore(config.GEMINI_CONCURRENCY)
        self.openrouter_limiter = asyncio.Semaphore(config.OPENROUTER_CONCURRENCY)
        self.deepseek_limiter = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)
        self.ollama_limiter = asyncio.Semaphore(1) 
        self.hourly_summary_news = []
        self.last_price_alert = {}
        self.learning_rate = config.LEARNING_RATE
        self.historical_cache = defaultdict(list)  # <-- Перенесено сюда для правильного вызова в методах

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
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

    @multiplier.setter
    def multiplier(self, value):
        self.scores.multiplier = value

    @property
    def asset_multipliers(self):
        return self.scores.asset_multipliers

    @property
    def model_sensitivities(self):
        return self.learning.model_sensitivities

    @property
    def cache_lock(self):
        return self.cache.cache_lock

    @property
    def score_lock(self):
        return self.scores.score_lock

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
        self.metrics.log_report(
            q_size=q_size, 
            scores_count=len(self.scores),
            market_status=self.market.market_data_status,
            last_sync=datetime.fromtimestamp(self.market.last_market_data_time).strftime('%H:%M:%S') 
            if self.market.last_market_data_time else "Never"
        )

    def add_url(self, url: str, title: str, embedding: Optional[List[float]] = None):
        self.cache.add_url(url, title, embedding)
        if embedding:
            asyncio.create_task(self.save_embedding(title, embedding))

    async def save_embedding(self, title: str, vector: List[float]):
        async with self.db_lock:
            async with get_db_connection() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO embeddings (title, vector) VALUES (?, ?)",
                    (title, json.dumps(vector))
                )
                await conn.commit()

    def add_news_for_summary(self, news_data: Dict):
        self.hourly_summary_news.append(news_data)

    def clear_hourly_summary_news(self):
        self.hourly_summary_news.clear()

    async def get_db_titles(self, hours: int = 3) -> List[str]:
        async with self.db_lock:
            async with get_db_connection() as conn:
                async with conn.execute(f"SELECT title FROM events WHERE timestamp > datetime('now', '-{hours} hours') ORDER BY timestamp DESC LIMIT 100") as cursor:
                    return [row['title'] for row in await cursor.fetchall()]

    # Метод перенесен сюда, настроены корректные отступы, db_lock и pandas Series
    async def load_historical_cache_to_ram(self, tickers: List[str]):
        """Загружает архивные котировки за последний год в RAM."""
        logging.info("🧠 Загрузка исторических цен из БД в RAM...")
        async with self.db_lock:
            async with get_db_connection() as conn:
                for ticker in tickers:
                    async with conn.execute(
                        """
                        SELECT date, close FROM daily_prices 
                        WHERE ticker = ? AND date >= datetime('now', '-365 days') 
                        ORDER BY date ASC
                        """, (ticker,)
                    ) as cursor:
                        rows = await cursor.fetchall()
                    
                    # Сохраняем в RAM в виде сериализованного объектного Series
                    self.historical_cache[ticker] = pd.Series(
                        {row['date']: row['close'] for row in rows}
                    ).sort_index()
        logging.info(f"✅ RAM-кэш котировок инициализирован для {len(tickers)} инструментов.")

    async def init_from_db(self):
        async with get_db_connection() as conn:
            async with conn.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'") as cursor:
                row = await cursor.fetchone()
            if row: self.scores.multiplier = row[0]

            async with conn.execute("SELECT target_asset, multiplier FROM asset_stats") as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                self.scores.asset_multipliers[row['target_asset'].lower()] = row['multiplier']

            async with conn.execute("SELECT model_name, sensitivity FROM model_stats") as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                self.learning.model_sensitivities[row['model_name']] = row['sensitivity']

            async with conn.execute("SELECT title FROM events ORDER BY timestamp DESC LIMIT 1000") as cursor:
                rows = await cursor.fetchall()
            from engine import clean_title
            for row in rows:
                self.cache.clean_titles[clean_title(row['title'])] = True

            await self.learning.load_config_weights()
            async with conn.execute("SELECT event_key, target_asset, weight FROM weights") as cursor:
                rows = await cursor.fetchall()
            for key, asset, val in rows:
                self.learning.weights[(key, asset)] = val
            
            async with conn.execute("SELECT source_domain, total_resolved, correct_count, sum_alpha FROM source_stats") as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                total = row['total_resolved']
                if total > 0:
                    self.learning.source_performance[row['source_domain']] = {
                        "wr": (row['correct_count'] or 0) / total,
                        "avg_alpha": (row['sum_alpha'] or 0.0) / total
                    }
            
            emb_lookback = config.RAM_EMBEDDING_LOOKBACK_DAYS
            async with conn.execute("""
                SELECT title, vector, timestamp 
                FROM embeddings 
                WHERE timestamp > datetime('now', '-' || ? || ' days')
                ORDER BY timestamp ASC
            """, (emb_lookback,)) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                try:
                    vec = json.loads(row['vector'])
                    ts_str = row['timestamp']
                    try:
                        p_dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                        p_ts = p_dt.timestamp()
                    except (ValueError, TypeError):
                        p_ts = time.time()
                    self.cache.embeddings[row['title']] = (vec, p_ts)
                except Exception:
                    continue
            
            lookback = config.RAM_SCORE_LOOKBACK_DAYS
            async with conn.execute("""
                SELECT event_key, target_asset, score, timestamp 
                FROM predictions 
                WHERE timestamp > datetime('now', '-' || ? || ' days')
                ORDER BY timestamp ASC
            """, (lookback,)) as cursor:
                rows = await cursor.fetchall()

            now = time.time()
            for row in rows:
                ekey, asset = row['event_key'], row['target_asset']
                p_score, ts_str = row['score'], row['timestamp']
                
                try:
                    p_dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    p_ts = p_dt.timestamp()
                except (ValueError, TypeError):
                    continue
                
                ckey = (ekey, asset)
                self.scores._apply_decay_internal(ckey, True, p_ts)
                
                correlation = config.ASSET_CORRELATION_MAP.get(asset.lower(), -1)
                weight = self.learning.get_weight(ekey, asset)
                self.scores[ckey] = max(-config.MAX_SCORE_THRESHOLD, 
                                        min(config.MAX_SCORE_THRESHOLD, 
                                            self.scores.get(ckey, 0.0) + (p_score * weight * correlation)))
                self.scores.last_update[ckey] = p_ts
                
                self.scores.last_sent[ekey] = p_ts
                if asset == 'global':
                    self.scores.last_sent_score[ekey] = self.scores.get(ckey, 0.0)

            for ckey in list(self.scores.keys()):
                self.scores._apply_decay_internal(ckey, True, now)

            logging.info("✅ Состояние успешно инициализировано из БД")

    async def save_to_db(self) -> int:
        saved_count = 0
        async with self.learning.weight_lock:
            async with self.db_lock:
                async with get_db_connection() as conn:
                    if self.scores.dirty_multiplier:
                        await conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('impact_multiplier', ?)", (self.multiplier,))
                        self.scores.dirty_multiplier = False
                        saved_count += 1
                    
                    if self.scores.dirty_asset_multipliers:
                        for asset in list(self.scores.dirty_asset_multipliers):
                            mult = self.asset_multipliers.get(asset)
                            if mult is not None:
                                await conn.execute("UPDATE asset_stats SET multiplier = ? WHERE target_asset = ?", (mult, asset))
                                saved_count += 1
                        self.scores.dirty_asset_multipliers.clear()

                    if self.learning.dirty_model_sensitivities:
                        for m_name in list(self.learning.dirty_model_sensitivities):
                            sens = self.model_sensitivities.get(m_name)
                            if sens is not None:
                                await conn.execute("""
                                    INSERT INTO model_stats (model_name, sensitivity) 
                                    VALUES (?, ?)
                                    ON CONFLICT(model_name) DO UPDATE SET sensitivity = EXCLUDED.sensitivity
                                """, (m_name, sens))
                                saved_count += 1
                        self.learning.dirty_model_sensitivities.clear()

                    if self.learning.dirty_weights:
                        for composite_key in list(self.learning.dirty_weights):
                            val = self.learning.weights.get(composite_key)
                            if val is not None:
                                await conn.execute("INSERT OR REPLACE INTO weights (event_key, target_asset, weight) VALUES (?, ?, ?)", 
                                                   (composite_key[0], composite_key[1], val))
                                saved_count += 1
                        self.learning.dirty_weights.clear()
                    
                    await conn.commit()
        return saved_count