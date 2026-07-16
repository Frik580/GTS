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

    # === МЕТОДЫ СБОРА ИНДИКАТОРОВ ДЛЯ SOXS v5.0 ===

    async def get_last_capex_signals(self, days: int = 15) -> Dict[str, Tuple[int, Optional[str], Optional[str], Optional[str]]]:
        """
        Извлекает последние подтвержденные ИИ сигналы capex за последние N дней.
        Возвращает словарь с кортежем (сигнал, event_key, timestamp, link).
        """
        signals = {ticker: (0, None, None, None) for ticker in ["MSFT", "META", "AMZN", "GOOGL"]}
        async with self.db_lock:
            async with get_db_connection() as conn:
                # Фильтруем события по тикерам и наличию сигналов
                async with conn.execute("""
                    SELECT p.event_key, p.capex_signal, p.timestamp, e.link
                    FROM predictions p
                    JOIN events e ON p.event_id = e.id
                    WHERE p.capex_signal IS NOT NULL AND p.capex_signal != 0
                    AND p.timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY p.timestamp DESC
                """, (days,)) as cursor:
                    rows = await cursor.fetchall()
                    
                    # Пытаемся сопоставить сущности
                    for row in rows:
                        ekey = row['event_key'].upper()
                        for comp in signals.keys():
                            if comp in ekey:
                                # Сохраняем самый свежий ненулевой сигнал для каждой компании
                                if signals[comp][0] == 0:
                                    signals[comp] = (row['capex_signal'], row['event_key'], row['timestamp'], row['link'])
        return signals

    async def get_last_guidance_signals(self, days: int = 15) -> Dict[str, Tuple[int, Optional[str], Optional[str], Optional[str]]]:
        """
        Извлекает последние подтвержденные ИИ сигналы guidance производителей чипов.
        Возвращает словарь с кортежем (сигнал, event_key, timestamp, link).
        """
        signals = {ticker: (0, None, None, None) for ticker in ["NVDA", "AVGO", "AMD", "MU"]}
        async with self.db_lock:
            async with get_db_connection() as conn:
                async with conn.execute("""
                    SELECT p.event_key, p.guidance_signal, p.timestamp, e.link
                    FROM predictions p
                    JOIN events e ON p.event_id = e.id
                    WHERE p.guidance_signal IS NOT NULL AND p.guidance_signal != 0
                    AND p.timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY p.timestamp DESC
                """, (days,)) as cursor:
                    rows = await cursor.fetchall()
                    
                    for row in rows:
                        ekey = row['event_key'].upper()
                        for comp in signals.keys():
                            if comp in ekey or (comp == "AVGO" and "BROADCOM" in ekey):
                                if signals[comp][0] == 0: # Если сигнал для этой компании еще не найден
                                    signals[comp] = (row['guidance_signal'], row['event_key'], row['timestamp'], row['link'])
        return signals

    async def get_divergence_metrics_10d(self) -> Tuple[int, List[str]]:
        """
        Оценивает силу дивергенции (Price confirmation) по 10-балльной шкале.
        Считает количество подтвержденных сильных ИИ-событий (новости с score > 4.0),
        которые привели к отрицательному движению цены или отсутствию реакции рынка.
        """
        async with self.db_lock:
            async with get_db_connection() as conn:
                async with conn.execute("""
                    SELECT event_key FROM predictions
                    WHERE score > 4.0 AND is_correct = 0 AND target_asset = 'soxs'
                    AND timestamp >= datetime('now', '-10 days') GROUP BY event_key
                """) as cursor:
                    rows = await cursor.fetchall()
                    keys = [row['event_key'] for row in rows] if rows else []
                    # Масштабируем до диапазона 0..10
                    return min(10, len(keys) * 2), keys

    _market_caps_cache = {}
    _last_caps_fetch = 0

    async def get_rotation_ranking(self) -> Tuple[int, float, float, float]:
        """
        Leadership Fatigue Indicator v2.0 (Ротация во второй эшелон):
        Сравнивает взвешенную по капитализации доходность "лидеров" (Tier 1)
        и "преследователей" (Tier 2) за последние 10 торговых дней.
        """
        now = time.time()
        all_rotation_tickers = ["NVDA", "AVGO", "ASML", "AMD", "MU", "INTC", "QCOM"]

        # Проверяем, есть ли данные в кэше. Если нет, принудительно обновляем.
        missing_caps = any(t not in self._market_caps_cache for t in all_rotation_tickers)
        missing_price_tickers = [t for t in all_rotation_tickers if self.historical_cache.get(t) is None or self.historical_cache[t].empty]
        missing_prices = bool(missing_price_tickers)

        force_update = missing_caps or missing_prices

        # Кэшируем капитализацию на 24 часа, чтобы не делать лишних запросов
        if force_update or (now - self._last_caps_fetch > 86400):
            try:
                # Обновляем исторические цены, если их нет
                if missing_prices: # Используем флаг, а не список
                    logging.info(f"⏳ Обновление кэша цен для Rotation Indicator (отсутствуют: {missing_price_tickers})...")
                    await self.load_historical_cache_to_ram(missing_price_tickers)

                import yfinance as yf
                temp_caps = {}
                logging.info("⏳ Обновление кэша капитализации для Rotation Indicator...")

                for ticker in all_rotation_tickers:
                    try:
                        # Устанавливаем таймаут для каждого запроса, чтобы избежать зависаний
                        t = yf.Ticker(ticker)
                        cap = t.info.get('marketCap')
                        if cap:
                            temp_caps[ticker] = cap
                    except Exception as e:
                        logging.warning(f"⚠️ Не удалось получить marketCap для {ticker}: {str(e)[:100]}")

                if len(temp_caps) >= len(all_rotation_tickers) // 2: # Убедимся, что получили хотя бы половину данных
                    self._market_caps_cache = temp_caps
                    self._last_caps_fetch = now
                    logging.info(f"✅ Кэш капитализации для Rotation Indicator обновлен: {self._market_caps_cache}")
            except Exception as e:
                logging.error(f"⚠️ Не удалось обновить кэш капитализации для Rotation Indicator: {e}")

        try:
            tier1 = {"NVDA": 0, "AVGO": 0, "ASML": 0}
            tier2 = {"AMD": 0, "MU": 0, "INTC": 0, "QCOM": 0}

            def get_weighted_return(basket: dict) -> float:
                total_cap = 0
                weighted_return = 0
                
                valid_tickers_in_basket = [t for t in basket.keys() if self.historical_cache.get(t) is not None and not self.historical_cache[t].empty and t in self._market_caps_cache]
                if not valid_tickers_in_basket:
                    # Логируем, почему корзина пуста
                    missing_p = [t for t in basket.keys() if self.historical_cache.get(t) is None or self.historical_cache[t].empty]
                    missing_c = [t for t in basket.keys() if t not in self._market_caps_cache]
                    logging.warning(f"Rotation: пустая корзина. Нет цен для: {missing_p}. Нет капитализации для: {missing_c}")
                    return 0.0

                for ticker in valid_tickers_in_basket:
                    total_cap += self._market_caps_cache[ticker]

                if total_cap == 0: return 0.0

                for ticker in valid_tickers_in_basket:
                    weight = self._market_caps_cache[ticker] / total_cap
                    price_series = self.historical_cache[ticker]
                    if len(price_series) > 10:
                        ret = price_series.pct_change(10).iloc[-1]
                        if pd.notna(ret):
                            weighted_return += ret * weight
                return weighted_return

            ret_tier1 = get_weighted_return(tier1)
            ret_tier2 = get_weighted_return(tier2)

            if ret_tier1 == 0 and ret_tier2 == 0: return 3, 0.0, 0.0, 0.0 # Фоллбек, если нет данных

            spread = ret_tier2 - ret_tier1
            if spread <= 0:
                score = max(0, int(3 + spread * 50)) # Увеличиваем чувствительность к отставанию
            else:
                score = min(10, int(3 + spread * 100)) # И к опережению
            return score, spread, ret_tier1, ret_tier2
        except Exception as e:
            logging.warning(f"Ошибка в get_rotation_ranking: {e}")
            return 3, 0.0, 0.0, 0.0

    async def save_quant_decision(self, bear_prob: float, target_pos: float, capex: float, guidance: float, triggers: List[str]):
        """Сохраняет исторический слепок решения в SQLite."""
        async with self.db_lock:
            async with get_db_connection() as conn:
                await conn.execute("""
                    INSERT INTO quant_decisions (bear_probability, target_position, capex_score, guidance_score, active_triggers)
                    VALUES (?, ?, ?, ?, ?)
                """, (bear_prob, target_pos, capex, guidance, ", ".join(triggers)))
                await conn.commit()

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
                        rows = await cursor.fetchall() # <-- Здесь rows пустой, если в БД нет данных

                    # Если в БД нет данных, принудительно загружаем из yfinance
                    if not rows:
                        logging.warning(f"Данные для {ticker} не найдены в локальной БД. Загрузка из yfinance...")
                        try:
                            import yfinance as yf
                            # Загружаем дневные данные за последний год
                            downloaded_data = yf.download(ticker, period="1y", interval="1d", progress=False)
                            
                            data = pd.Series(dtype=float)

                            if not downloaded_data.empty:
                                if isinstance(downloaded_data, pd.Series):
                                    data = downloaded_data
                                elif 'Close' in downloaded_data.columns:
                                    # .squeeze() handles the case where yfinance returns a DataFrame
                                    # with a MultiIndex, resulting in df['Close'] being a
                                    # single-column DataFrame instead of a Series.
                                    close_data = downloaded_data['Close'].squeeze()
                                    if isinstance(close_data, pd.Series):
                                        data = close_data

                            if not data.empty:
                                # Сохраняем в БД для будущего использования
                                records_to_save = []
                                for dt, val in data.items():
                                    if pd.notna(val):
                                        date_str = dt.strftime('%Y-%m-%d')
                                        records_to_save.append((ticker, date_str, float(val)))
                                
                                if records_to_save:
                                    await conn.executemany(
                                        "INSERT OR REPLACE INTO daily_prices (ticker, date, close) VALUES (?, ?, ?)",
                                        records_to_save
                                    )
                                    await conn.commit()
                                    logging.info(f"✅ Сохранено {len(records_to_save)} записей для {ticker} в локальную БД.")
                                
                                # Заполняем RAM-кэш свежими данными
                                self.historical_cache[ticker] = data.sort_index()
                            else:
                                self.historical_cache[ticker] = pd.Series(dtype=float) # Создаем пустую серию, чтобы избежать повторных попыток
                        except Exception as e:
                            logging.error(f"❌ Ошибка при загрузке {ticker} из yfinance: {e}")
                    else:
                        # Если данные есть в БД, используем их
                        self.historical_cache[ticker] = pd.Series({row['date']: row['close'] for row in rows}).sort_index()

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