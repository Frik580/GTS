import feedparser
import logging
from logging.handlers import RotatingFileHandler
import re
import time
import json
import asyncio
import aiohttp
import aiosqlite
import math
import html
import calendar
from google import genai
import numpy as np
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

# Новые модули
from state_service import GTSStateManager
from ai_processor import ai_analyze_batch, get_embedding, is_semantic_duplicate
from model_factory import init_model_pool, ModelRotator


START_TIME = time.time()

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

async def metrics_reporter_task(state: GTSStateManager):
    """Отдельный фоновый процесс для вывода статистики раз в минуту."""
    while True:
        try:
            state.log_metrics(news_queue.qsize())
        except Exception as e:
            logging.error(f"Error in metrics reporter: {e}")
        await asyncio.sleep(60) # Выводим отчет ровно раз в минуту

def shutdown_cleanup():
    """Выполняет очистку ресурсов при завершении работы."""
    logging.info("Закрытие соединений и остановка фоновых задач...")
    sync_executor.shutdown(wait=True) # Дожидаемся завершения всех задач в пуле
    logging.info("GTS 4.0 остановлен.")

# =========================
# CONFIG
# =========================

state = GTSStateManager()
model_rotator = ModelRotator(init_model_pool())
news_queue = asyncio.Queue(maxsize=500) # Увеличиваем буфер для сглаживания всплесков новостей

logging.info(f"Пул моделей готов: {[m['name'] for m in model_rotator.pool]}. Старт с: {model_rotator.get_active()['name']}")
logging.info(f"--- Текущий IMPACT_MULTIPLIER: {state.multiplier:.2f} ---")

# =========================
# AI ENGINE
# =========================

SOURCE_NAME_DOMAIN_MAP = {
    "reuters": "reuters.com",
    "bloomberg": "bloomberg.com",
    "bloomberg.com": "bloomberg.com",
    "financial times": "ft.com",
    "ft": "ft.com",
    "wsj": "wsj.com",
    "wall street journal": "wsj.com",
    "benzinga": "benzinga.com",
    "digitimes": "digitimes.com",
    "tom's hardware": "tomshardware.com",
    "tomshardware": "tomshardware.com",
    "the register": "theregister.com",
    "investor's business daily": "investors.com",
}

def normalize_source_domain(value: str) -> str:
    """Return a stable source domain from a URL or known RSS source name."""
    if not value:
        return ""

    raw = str(value).strip().lower()
    if raw in SOURCE_NAME_DOMAIN_MAP:
        return SOURCE_NAME_DOMAIN_MAP[raw]

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    domain = (parsed.netloc or parsed.path.split("/")[0]).lower()
    domain = domain.replace("www.", "").strip()

    if domain in SOURCE_NAME_DOMAIN_MAP:
        return SOURCE_NAME_DOMAIN_MAP[domain]
    if "." not in domain:
        return ""
    return domain

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
        # Ограничиваем количество частей слага, чтобы избежать слишком длинных ключей
        # Нормализуем разделители и принудительно удаляем любые нелатинские символы (кириллицу и т.д.)
        raw_normalized = slug.strip().upper().replace("-", "_").replace(".", "_").replace(" ", "_")
        normalized_slug = re.sub(r'[^A-Z0-9_]', '', raw_normalized)
        
        slug_parts = [p for p in normalized_slug.split("_") if p]
        # Нормализация: убираем 'S' в конце для объединения STRIKES/STRIKE
        norm_parts = [p[:-1] if p.endswith('S') and len(p) > 3 else p for p in slug_parts[:config.MAX_ENTITY_PARTS]]
        return "_".join(norm_parts)

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

def market_signals(intensity: float) -> Dict[str, str]:
    """
    Определяет сигналы. Теперь intensity — это Bullish Intensity актива.
    Positive = Bullish, Negative = Bearish.
    """
    return {
        "nasdaq": "bearish" if intensity > config.SIGNAL_THRESHOLD_MED else "bullish" if intensity < -config.SIGNAL_THRESHOLD_HIGH else "flat",
        "sp500":  "bearish" if intensity > config.SIGNAL_THRESHOLD_MED else "bullish" if intensity < -config.SIGNAL_THRESHOLD_HIGH else "flat",
        "oil":    "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "soxs":   "bullish" if intensity > config.SIGNAL_THRESHOLD_HIGH else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "vix":    "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "bearish" if intensity < -config.SIGNAL_THRESHOLD_MED else "flat",
        "gold":   "bullish" if intensity > config.SIGNAL_THRESHOLD_LOW else "bearish" if intensity < -config.SIGNAL_THRESHOLD_HIGH else "flat",
        "btc":    "bullish" if intensity > config.SIGNAL_THRESHOLD_MED else "bearish" if intensity < -config.SIGNAL_THRESHOLD_BTC else "flat"
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

def generate_signal(prob: float, intensity: float) -> str:
    if intensity > 0:  # Положительный Score от ИИ = Risk-Off (Напряженность растет)
        if prob > 70: return "🔴 HIGH RISK-OFF"
        if prob > 40: return "🟠 MEDIUM RISK"
        return "🟡 CAUTION"
    elif intensity < 0:  # Отрицательный Score от ИИ = Risk-On (Позитив для рынков)
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

def should_send(key: str, news_score: float, current_total_score: float, state: GTSStateManager, is_black_swan: bool = False) -> bool:
    now = time.time()

    # Если сама новость — "Черный лебедь" или очень сильная, разрешаем внеплановый алерт
    if abs(news_score) >= config.BLACK_SWAN_SCORE_THRESHOLD or is_black_swan:
        # Увеличиваем кулдаун до 10 минут (как основной COOLDOWN), чтобы не спамить перепечатками
        # Для критических новостей разрешаем повтор раз в 5 минут, если это реально важно
        if key in state.last_sent and (now - state.last_sent[key] < 300):
             return False
        state.last_sent[key] = now
        return True

    if key not in state.last_sent:
        state.last_sent[key] = now
        state.last_sent_score[key] = current_total_score
        return True

    if now - state.last_sent[key] > config.COOLDOWN:
        # Проверяем, добавила ли новая новость значимый вес к сюжету
        prev_score = state.last_sent_score.get(key, 0)
        score_delta = abs(current_total_score - prev_score)
        
        # Если балл вырос меньше чем на 2.0 (порог нейтральности), это просто перепечатка
        if score_delta < 2.0:
            logging.info(f"🔇 Suppression: Similar news for {key} (delta {score_delta:.2f} < 2.0)")
            return False

        state.last_sent[key] = now
        state.last_sent_score[key] = current_total_score
        return True

    return False

# =========================
# LEARNING SYSTEM
# =========================

async def update_weights(event_key: str, asset: str, error: float, state: GTSStateManager, is_correct: bool = True):
    """Обновляет веса событий на основе ошибки прогноза."""
    async with state.learning.weight_lock:
        # Асимметричное обучение: если направление неверное, учимся быстрее (штрафуем сильнее)
        lr_multiplier = 1.0 if is_correct else config.ASYMMETRIC_LR_FACTOR
        
        # Обучение на основе ошибки прогноза амплитуды и множителя направления
        adjustment = config.LEARNING_RATE * error * lr_multiplier

        composite_key = (event_key, asset)
        # Основной ключ получает 100% корректировки
        old_w = state.learning.weights.get(composite_key, 1.0)
        state.learning.weights[composite_key] = max(0.5, min(5.0, old_w + adjustment))
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
            logging.info(f"⚙️ Multiplier ({asset_low}): {old_mult:.2f} -> {new_mult:.2f} (avg_err: {avg_error:+.2f})")
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
        state.metrics.market_data_timings.append(duration)
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

async def count_eligible_predictions() -> int:
    """Возвращает количество новостей, готовых к обучению."""
    async with get_db_connection() as conn:
        async with conn.execute("""
            SELECT 
                SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) as phase1,
                SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as phase2
            FROM predictions 
            WHERE resolved < 2 AND timestamp < datetime('now', '-' || ? || ' hours')
        """, (config.MARKET_LOOKBACK_HOURS,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return (row[0] or 0) + (row[1] or 0)
            return 0

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
            target_ticker = config.ASSET_TICKER_MAP.get(target_key)
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
        async with get_db_connection() as conn:
            # JOIN с таблицей events гарантирует, что прогноз привязан к реальному событию из ленты
            async with conn.execute("""
                SELECT p.*, e.link as source_link 
                FROM predictions p
                JOIN events e ON p.event_id = e.id
                WHERE p.resolved < 2 
                ORDER BY p.timestamp ASC LIMIT 1000
            """) as cursor:
                rows = await cursor.fetchall()
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
                    batch_updates.append((2, 0, 0, 0, row['id']))
                    continue

                # 2. Проверка белого списка (только если ONLY_SPECIFIC_SOURCES = True)
                if config.ONLY_SPECIFIC_SOURCES:
                    source_domain = row['source_domain']
                    if source_domain:
                        is_allowed = any(domain in source_domain.lower() for domain in config.SPECIFIC_SOURCES_LIST)
                        if not is_allowed:
                            logging.debug(f"Learning: Skipping {event_key} - source '{source_domain}' not in whitelist anymore")
                            # Помечаем как разрешенное (resolved=2), но не обновляем веса
                            batch_updates.append((2, 0, 0, 0, row['id']))
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
                # Увеличиваем порог для фондового рынка, чтобы пережить выходные (48ч + запас)
                force_resolve_age = s_win + (72 if target.lower() not in ['btc', 'crypto'] else 12)

                # Фаза 1: Первичная реакция (Primary)
                if row['resolved'] == 0:
                    if age_hours >= p_win:
                        target_lookback = p_win
                        new_resolved_status = 1
                    else: 
                        # Если новость слишком старая, но так и не прошла Фазу 1
                        if age_hours > force_resolve_age:
                            if target.lower() not in ['btc', 'crypto'] and (datetime.now(timezone.utc).weekday() >= 5 or raw_market_data.get('is_stale', True)):
                                continue
                            logging.info(f"🗑️ Force resolve (Phase 1 Expired): {event_key}")
                        batch_updates.append((2, 0, 0, 0, row['id']))
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
                
                target_ticker = config.ASSET_TICKER_MAP.get(target.lower())
                if target_ticker and target_ticker in price_history.columns:
                    try:
                        # Фильтруем серию по времени, чтобы найти цену в момент новости и через N часов
                        ts = price_history[target_ticker].dropna()
                        if ts.index.tz is not None:
                            ts = ts.tz_convert('UTC').tz_localize(None)
                        
                        if ts.empty or prediction_time.replace(tzinfo=None) < ts.index[0]:
                            continue

                        prediction_time_naive = prediction_time.replace(tzinfo=None)
                        idx_at = ts.index.get_indexer([prediction_time_naive], method='backfill')[0]
                        if idx_at == -1:
                            continue

                        # Смещаем целевое время: отсчитываем окно влияния от первой доступной торговой свечи.
                        # Это позволяет поймать движение ПОСЛЕ открытия рынка (гэпа).
                        actual_start_time = ts.index[idx_at]
                        shifted_target_time = actual_start_time + timedelta(hours=target_lookback)

                        # Ждем, пока рынок "продышится" достаточное время после открытия
                        if ts.index[-1] < shifted_target_time:
                            continue

                        idx_after = ts.index.get_indexer([shifted_target_time], method='backfill')[0]

                        if idx_at != -1 and idx_after != -1 and idx_at != idx_after:
                                
                            p_at = float(ts.iloc[idx_at])
                            p_after = float(ts.iloc[idx_after])
                            
                            if p_at != 0:
                                raw_change_pct = ((p_after - p_at) / p_at) * 100
                                
                                # 1. Расчет реализованной волатильности для Z-нормализации
                                asset_data = price_history[target_ticker].dropna()
                                asset_rets = asset_data.pct_change().dropna().tail(config.VOLATILITY_WINDOW)
                                realized_vol_raw = asset_rets.std() * 100
                                if pd.isna(realized_vol_raw) or realized_vol_raw == 0:
                                    realized_vol_raw = 1.0
                                vol_floor = config.GLOBAL_Z_ALPHA_VOL_FLOOR if target.lower() == "global" else config.Z_ALPHA_VOL_FLOOR
                                realized_vol = max(realized_vol_raw, vol_floor)

                                # 2. Расчет ожидаемого движения (Benchmark)
                                expected_pct = 0.0
                                b_cfg = config.ASSET_BENCHMARK_CONFIG.get(target.lower())
                                if b_cfg:
                                    bench_key = b_cfg["primary"]
                                    if bench_key in price_history.columns:
                                        b_ts = price_history[bench_key].dropna().tz_localize(None)
                                        idx_b_at = b_ts.index.get_indexer([prediction_time_naive], method='backfill')[0]
                                        # Используем смещенное время, чтобы сопоставить движение бенчмарка с активом
                                        idx_b_after = b_ts.index.get_indexer([shifted_target_time], method='backfill')[0]
                                        if idx_b_at != -1 and idx_b_after != -1:
                                            b_at, b_after = float(b_ts.iloc[idx_b_at]), float(b_ts.iloc[idx_b_after])
                                            if b_at != 0:
                                                b_move = ((b_after - b_at) / b_at) * 100
                                                expected_pct = calculate_expected_move(target.lower(), b_cfg, prediction_time, b_move, price_history)

                                # 3. Z-Alpha: нормализуем избыточную доходность по волатильности
                                z_alpha = (raw_change_pct - expected_pct) / realized_vol
                                
                                correlation = config.ASSET_CORRELATION_MAP.get(target.lower(), -1)
                                # Для обучения и записи в БД используем Z-Alpha (сигмы)
                                raw_change = z_alpha
                                logging.info(f"📐 Z-ALPHA [{target}]: Raw {raw_change_pct:+.2f}% (Exp {expected_pct:+.2f}%) | Vol: {realized_vol_raw:.2f}% | Z: {z_alpha:+.2f}")
                        else:
                            
                            # Цены отсутствуют в истории - проверяем на "протухание"
                            if age_hours > force_resolve_age:
                                # Не закрываем в 0, если это выходные для фондового рынка
                                if target.lower() not in ['btc', 'crypto'] and (datetime.now(timezone.utc).weekday() >= 5 or raw_market_data.get('is_stale', True)):
                                    continue 
                                logging.info(f"🗑️ Force resolve (No Price Data): {event_key}")
                            batch_updates.append((2, 0, 0, 0, row['id']))
                            continue
                    except Exception as e:
                        if age_hours > force_resolve_age:
                            logging.info(f"🗑️ Force resolve (Error): {event_key} - {e}")
                            batch_updates.append((2, 0, 0, 0, row['id']))
                        continue
                else: 
                    if age_hours > force_resolve_age:
                        logging.info(f"🗑️ Force resolve (No Ticker): {event_key}")
                        batch_updates.append((2, 0, 0, 0, row['id']))
                    continue

                dynamic_threshold = config.LEARNING_THRESHOLD * (1 + (target_lookback / 10))
                
                # Даже если изменение цены маленькое, мы ДОЛЖНЫ обновить статус resolved в БД,
                # иначе эти новости будут копиться в бэклоге бесконечно.
                if abs(score) < config.NEUTRAL_SCORE_THRESHOLD:
                    batch_updates.append((2, 0, 0, 0, row['id']))
                    logging.debug(f"Learning: Skipping low-score event {event_key} (Score {score:.1f} < Threshold {config.NEUTRAL_SCORE_THRESHOLD})")
                    continue

                # Если фактическое движение рынка ниже динамического порога,
                # мы обычно не учимся на нем, так как это считается рыночным шумом.
                # Однако, если оценка ИИ очень сильна (уровень Black Swan),
                # мы хотим учиться на ЛЮБОМ движении, каким бы малым оно ни было,
                # чтобы точно откалибровать влияние таких критических событий.
                if abs(raw_change) < dynamic_threshold and abs(score) < config.BLACK_SWAN_SCORE_THRESHOLD:
                    batch_updates.append((new_resolved_status, 0, 0, 0, row['id']))
                    continue

                # Если мы дошли сюда, значит движение достаточно значимо ИЛИ score был очень сильным.
                # actual теперь является количеством стандартных отклонений (Z-Score)
                actual = min(abs(raw_change), 10.0)

                # FIX: Используем знаковое raw_change для определения корректности направления.
                # row['actual_move'] нельзя использовать, так как там хранится модуль (abs) из предыдущей фазы,
                # что приводит к потере информации о направлении движения рынка.
                is_correct = 1 if (score * raw_change * correlation) > 0 else 0

                # Если направление неверное, ошибка становится значительно более "негативной",
                # что заставляет систему агрессивнее снижать множители и веса.
                effective_error = (actual - predicted) if is_correct else -(actual + predicted)
                error = effective_error

                # КАЛИБРОВКА MSF: Подстраиваем чувствительность модели
                model_sens = state.model_sensitivities.get(row['model_name'], 1.0)
                state.model_sensitivities[row['model_name']] = max(0.1, min(5.0, model_sens + (state.learning_rate * error * 0.5)))
                
                # Обновляем счетчик resolved для модели в базе
                await conn.execute("""
                    INSERT INTO model_stats (model_name, total_resolved, sensitivity) 
                    VALUES (?, 1, ?) 
                    ON CONFLICT(model_name) DO UPDATE SET 
                        total_resolved = total_resolved + 1,
                        sensitivity = EXCLUDED.sensitivity
                """, (row['model_name'], state.model_sensitivities.get(row['model_name'], 1.0)))

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
                            if source_domain:
                                # Учитываем корреляцию: для источника важно, угадал ли он направление конкретного актива
                                directional_alpha = (raw_change * correlation) if score > 0 else -(raw_change * correlation)
                                # Учитываем ложные срабатывания (is_correct=0) как больший вклад в ошибку
                                await conn.execute("""
                                    INSERT INTO source_stats (source_domain, total_resolved, correct_count, sum_error, sum_confidence, sum_alpha, sum_alpha_sq)
                                    VALUES (?, 1, ?, ?, ?, ?, ?)
                                    ON CONFLICT(source_domain) DO UPDATE SET
                                        total_resolved = total_resolved + 1,
                                        correct_count = correct_count + EXCLUDED.correct_count,
                                        sum_error = sum_error + EXCLUDED.sum_error,
                                        sum_confidence = sum_confidence + EXCLUDED.sum_confidence,
                                        sum_alpha = sum_alpha + EXCLUDED.sum_alpha,
                                        sum_alpha_sq = sum_alpha_sq + EXCLUDED.sum_alpha_sq
                                """, (source_domain.lower(), is_correct, abs(error), row['confidence'], directional_alpha, directional_alpha**2))
                                processed_source_links.add(source_link)

                        # NEW: Update asset_stats
                        if target: # Ensure target_asset is not empty
                            await conn.execute("""
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

                batch_updates.append((new_resolved_status, actual, is_correct, raw_change, row['id']))

            if batch_updates:
                await conn.executemany("""
                    UPDATE predictions SET resolved = ?, actual_move = ?, is_correct = ?, signed_alpha = ? WHERE id = ?
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
                async with conn.execute("SELECT total_resolved, correct_count FROM asset_stats WHERE target_asset = ?", (asset,)) as cursor_stats:
                    stats = await cursor_stats.fetchone()
                    
                if stats and stats['total_resolved'] >= config.MIN_SAMPLE_SIZE_FOR_RESET:
                    wr = (stats['correct_count'] / stats['total_resolved']) * 100
                    if wr < config.MIN_WINRATE_BEFORE_RESET:
                        state.asset_multipliers[asset] = config.IMPACT_MULTIPLIER
                        logging.warning(f"📉 WinRate актива {asset.upper()} ({wr:.1f}%) ниже порога. Множитель сброшен до {config.IMPACT_MULTIPLIER}")

            # 2. Калибровка глобального множителя (один раз за цикл на основе всей выборки)
            if all_errors:
                calibrate_multiplier(sum(all_errors) / len(all_errors), state) # Передаем state
            await conn.commit()

    await state.save_to_db() # Сохраняем состояние через state manager
    logging.info(f"System settings saved. New IMPACT_MULTIPLIER: {state.multiplier:.2f}") # Используем state.multiplier

async def _execute_vacuum():
    """Асинхронное выполнение VACUUM."""
    try:
        async with get_db_connection() as conn:
            await conn.execute("PRAGMA journal_mode=DELETE") # Отключаем WAL для VACUUM
            await conn.execute("VACUUM")
            await conn.execute("PRAGMA journal_mode=WAL") # Возвращаем WAL
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
            async with get_db_connection() as conn:
                
                # Получаем список каноничных ключей из конфига, которые НЕЛЬЗЯ удалять
                tracked_keys = []
                for k in config.TRACKED_KEYWORDS.keys():
                    key_parts = sorted(k.upper().replace(" ", "_").split("_"))
                    tracked_keys.append("_".join(key_parts))
                placeholders = ', '.join(['?'] * len(tracked_keys))

                # Удаляем старые события и прогнозы
                await conn.execute("DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
                await conn.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
                await conn.execute("DELETE FROM embeddings WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.EMBEDDING_RETENTION_DAYS,))
                
                # 1. Удаляем ключи с критически низким весом
                await conn.execute("DELETE FROM weights WHERE weight <= ?", (config.MIN_WEIGHT_THRESHOLD,))
                
                # 2. Удаляем "забытые" ключи, которых нет в последних прогнозах и нет в TRACKED_KEYWORDS
                async with conn.execute(f"""
                    DELETE FROM weights 
                    WHERE event_key NOT IN (SELECT DISTINCT event_key FROM predictions)
                    AND event_key NOT IN ({placeholders})
                """, tracked_keys) as cursor:
                    deleted_weights = cursor.rowcount
                
                await conn.commit()
            
            await _execute_vacuum()


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
    stop_words = {
        'reports', 'hit', 'triggers', 'massive', 'says', 'amid', 'following', 'after', 'due', 'warns', 'shows', 'proposes', 'plans', 'set', 'could', 'would', 'may', 'will', 'предложил', 'предлагает', 'может', 'планирует', 'хочет', 'объявил', 'ago', 'min', 'hours',
        'calendar', 'corporate', 'event', 'fiscal', 'announces', 'dividend', 'shareholder'
    }
    # Удаляем пунктуацию
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    # Удаляем одиночные цифры (часто это время или кол-во чего-то)
    cleaned = re.sub(r'\b\d+\b', '', cleaned)
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
                current_utc_ts = datetime.now(timezone.utc).timestamp()
                for msg in data.get('messages', []):
                    # Проверка на возраст сообщения (макс. 1 час для StockTwits)
                    pub_time_struct = time.strptime(msg.get('created_at'), '%Y-%m-%dT%H:%M:%SZ')
                    pub_timestamp = calendar.timegm(pub_time_struct)
                    if (current_utc_ts - pub_timestamp) / 3600 > 1.0:
                        continue

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
    try:
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
                    is_social = any(s in url for s in ["nitter", "reddit", "stocktwits"])
                    log_func = logging.warning if is_social else logging.error
                    
                    if "getaddrinfo" in error_str:
                        log_func(f"🌐 DNS/Connection Error for {url}: Source might be down or unreachable.")
                    else:
                        log_func(f"Feed error {url}: {e}")
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

                # Лимит 1 час для соцсетей (Reddit, Twitter/Nitter, StockTwits)
                is_social = any(s in entry.link.lower() or s in url.lower() for s in ["reddit.com", "nitter", "twitter.com", "x.com", "stocktwits.com"])
                effective_max_age = 1.0 if is_social else max_age_h

                if age_h <= effective_max_age:
                    fresh_entries.append(entry)
                else:
                    # Переводим в DEBUG, чтобы не спамить в консоль
                    logging.debug(f"Skipping old news: '{entry.title}' (Age: {age_h:.1f}h, Max: {effective_max_age}h)")
            # Если даты публикации нет, мы её не добавляем (более строгий подход к качеству данных)

        # 2. Сортируем по времени (самые свежие — первые) и берем в пределах лимита
        fresh_entries.sort(
            key=lambda x: calendar.timegm(x.get('published_parsed') or x.get('updated_parsed') or time.gmtime(0)), 
            reverse=True
        )
        max_entries_to_process = config.RSS_MAX_ENTRIES if is_market_active else config.RSS_MAX_ENTRIES_INACTIVE
        
        processed_count = 0
        for entry in fresh_entries:
            state.metrics.metrics["news_received"] += 1
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

            # 1. Быстрая проверка на дубликаты (URL и Fuzzy)
            use_semantic = config.USE_EMBEDDINGS and (config.GEMINI_API_KEY or config.OPENROUTER_API_KEY)
            fuzzy_threshold = config.DUPLICATE_TITLE_THRESHOLD if use_semantic else config.FALLBACK_DUPLICATE_THRESHOLD

            new_clean = clean_title(original_title)
            async with state.cache.cache_lock:
                if state.cache.is_url_processed(entry.link):
                    state.metrics.metrics["news_duplicate_url"] += 1
                    continue

                # 1. Быстрая проверка Set-based Hashing (Exact match по очищенному заголовку)
                if new_clean in state.cache.clean_titles:
                    state.metrics.metrics["news_duplicate_hash"] += 1
                    continue

                # Теперь передаем в функцию список УЖЕ ОЧИЩЕННЫХ залогов (clean_titles)
                if is_fuzzy_duplicate(original_title, state.cache.clean_titles, fuzzy_threshold):
                    state.metrics.metrics["news_duplicate_fuzzy"] += 1
                    state.cache.add_url(entry.link, original_title)
                    continue
                
                # Добавляем очищенную версию в быстрый кэш
                state.cache.clean_titles.add(new_clean)

            # Проверка в БД (перед тяжелым AI запросом эмбеддинга для экономии API)
            db_titles = await state.get_db_titles(hours=config.SEMANTIC_DEDUPLICATION_WINDOW)
            if is_fuzzy_duplicate(original_title, db_titles, fuzzy_threshold):
                state.metrics.metrics["news_duplicate_fuzzy"] += 1
                continue

            # 2. Получение эмбеддинга (БЕЗ блокировки, так как это сетевой запрос)
            new_embedding = None
            if use_semantic:
                new_embedding = await get_embedding(entry.title, model_rotator, state, session=session)
                
                # 3. Семантическая проверка (is_semantic_duplicate сам управляет своей блокировкой)
                if new_embedding and await is_semantic_duplicate(entry.title, new_embedding, state):
                    state.metrics.metrics["news_duplicate_semantic"] += 1
                    async with state.cache.cache_lock:
                        state.add_url(entry.link, entry.title, embedding=new_embedding)
                    continue

            # 4. Финальное добавление в кэш (если новость прошла все фильтры)
            async with state.cache_lock:
                # Считаем только реально новые новости, прошедшие фильтры дубликатов
                processed_count += 1
                if processed_count > max_entries_to_process:
                    logging.debug(f"Reached max_entries_to_process ({max_entries_to_process}) for feed {url}")
                    break
                    
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
                    pass
    except Exception as e:
        logging.error(f"Unhandled error in feed {url}: {e}")

async def news_worker(worker_id: int, session: aiohttp.ClientSession, state: GTSStateManager, rotator: ModelRotator):
    """Воркер для обработки новостей из очереди."""
    logging.info(f"Worker {worker_id} started.")
    while True:
        batch = []
        # Пытаемся собрать пакет новостей
        try:
            # Ждем первую новость
            item = await news_queue.get()
            batch.append(item)
            logging.info(f"📦 Worker {worker_id}: Начат сбор пакета (1/{config.AI_BATCH_SIZE})")
            
            # Накопление пакета: ждем до лимита времени или пока не наберется AI_BATCH_SIZE
            batch_start_time = time.time()
            while len(batch) < config.AI_BATCH_SIZE:
                elapsed = time.time() - batch_start_time
                wait_time = config.AI_BATCH_WAIT_SECONDS - elapsed
                
                if wait_time <= 0:
                    logging.info(f"⏱ Worker {worker_id}: Лимит времени ожидания исчерпан ({len(batch)}/{config.AI_BATCH_SIZE})")
                    break
                    
                try:
                    # Пытаемся забрать следующую новость из очереди с учетом оставшегося времени
                    next_item = await asyncio.wait_for(news_queue.get(), timeout=wait_time)
                    batch.append(next_item)
                    logging.info(f"➕ Worker {worker_id}: Добавлена новость ({len(batch)}/{config.AI_BATCH_SIZE})")
                except asyncio.TimeoutError:
                    logging.info(f"⏱ Worker {worker_id}: Ожидание новых сообщений истекло ({len(batch)}/{config.AI_BATCH_SIZE})")
                    break
            
            if len(batch) >= config.AI_BATCH_SIZE:
                logging.info(f"🚀 Worker {worker_id}: Пакет полностью укомплектован ({len(batch)}/{config.AI_BATCH_SIZE})")
            
            # Обрабатываем пакет
            prepared_batch = []
            for entry, m_data in batch:
                # Извлекаем максимально подробный текст новости
                body_text = entry.get("summary", "")
                # Проверяем наличие расширенного контента (content:encoded)
                if hasattr(entry, 'content') and entry.content:
                    content_val = entry.content[0].value
                    if len(content_val) > len(body_text):
                        body_text = content_val
                
                # Очистка от HTML-тегов и лишних пробелов для экономии контекста
                clean_body = re.sub(r'<[^>]+>', '', body_text).strip()
                
                prepared_batch.append({
                    "text": f"{entry.title}\n{clean_body}"[:4000], # Ограничиваем разумным пределом
                    "pub_time": entry.get('published') or entry.get('updated') or "Unknown",
                    "entry": entry,
                    "market_data": m_data
                })
            
            logging.info(f"Worker {worker_id}: Отправка пакета из {len(prepared_batch)} новостей на анализ ИИ ({rotator.get_active()['name']})")
            results = await ai_analyze_batch(prepared_batch, rotator, state, session)
            
            # Обрабатываем результаты. Если ИИ вернул меньше элементов, чем в батче,
            # мы логируем ошибку, но news_queue.task_done() выполнится для всех,
            # поэтому в будущем стоит реализовать возврат необработанных задач обратно в очередь.
            for i, analysis_result in enumerate(results):
                if i < len(prepared_batch):
                    await process_single_analysis_result(
                        prepared_batch[i]['entry'], 
                        prepared_batch[i]['market_data'], 
                        analysis_result, 
                        session, 
                        state
                    )
                else:
                    logging.warning(f"AI returned more results ({len(results)}) than batch size ({len(prepared_batch)})")
            
        except Exception as e:
            logging.error(f"Worker {worker_id} batch error: {e}")
        finally:
            for _ in range(len(batch)):
                news_queue.task_done()

async def process_single_analysis_result(entry: Any, market_data: Dict, analysis_result: Tuple, session: aiohttp.ClientSession, state: GTSStateManager):
    """Обработка результата анализа одной новости из пакета."""
    is_market_active = not market_data.get('is_stale', True)
    fng_val = market_data.get("fng_val", 50)
    
    # Извлекаем время публикации из RSS (published или updated)
    pub_time = entry.get('published') or entry.get('updated') or "Unknown Date"
    
    # Парсим в формат БД для EVENT_TYPE_LOOKBACK (относительно времени публикации)
    pub_struct = entry.get('published_parsed') or entry.get('updated_parsed')
    if pub_struct:
        # Преобразуем struct_time в datetime (UTC) и затем в строку для SQLite
        pub_dt = datetime(*pub_struct[:6], tzinfo=timezone.utc)
        pub_db_time = pub_dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        pub_db_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Определение источника до вызова AI: домен для статистики, display-имя для сообщений.
    source_obj = entry.get('source', {}) or {}
    source_display = str(source_obj.get('title') or "").strip()
    source_domain = (
        normalize_source_domain(source_obj.get('href') or "") or
        normalize_source_domain(entry.link) or
        normalize_source_domain(source_display)
    )
    if (not source_domain or source_domain in {"news.google.com", "google.com"}) and source_display:
        source_domain = normalize_source_domain(source_display) or source_domain
    if not source_domain and ' - ' in entry.title:
        source_domain = normalize_source_domain(entry.title.split(' - ')[-1])
    source_title = source_domain or source_display.lower() or "unknown"
    source_label = source_display or source_title

    try:
        score, event_type, entities, slug, is_black_swan, model_name, confidence, ai_summary, title_ru = analysis_result

        # Сброс флага Black Swan, если score новости недостаточно велик (защита от галлюцинаций ИИ)
        if is_black_swan and abs(score) < config.BLACK_SWAN_SCORE_THRESHOLD:
            logging.info(f"🦢 Понижение статуса Black Swan для {slug}: индивидуальный score {score:.2f} < {config.BLACK_SWAN_SCORE_THRESHOLD}")
            is_black_swan = False

        narrative_multiplier = 1.0

        # Фильтр по уровню уверенности
        if confidence < config.CONFIDENCE_THRESHOLD:
            logging.info(f"Skipping news '{entry.title}': Confidence {confidence:.2f} is below threshold {config.CONFIDENCE_THRESHOLD}")
            return

        raw_score = score  # Сохраняем оригинальный балл от ИИ (-10..10)
        narrative_multiplier = 1.0

        if abs(raw_score) < 0.1:
            logging.info(f"🔇 Trivial news ignored: {entry.title}")
            return

        # Slug Logic (Duplicates & Narrative Tracking)
        normalized_slug = slug.strip().lower() if slug else None
        async with state.cache.cache_lock:
            if normalized_slug:
                # Инкрементируем счетчик при каждом появлении
                state.narrative_counts[normalized_slug] += 1
                current_hits = state.narrative_counts[normalized_slug]

                if normalized_slug in state.cache.slugs:
                    delta_sec = time.time() - state.cache.slugs[normalized_slug]
                    
                    # Spam prevention: if news arrives too fast, it's a duplicate
                    if delta_sec < config.SLUG_SPAM_WINDOW:
                        logging.info(f"🐌 Slug spam prevention: {normalized_slug}")
                        state.metrics.metrics["news_duplicate_slug"] += 1
                        return
                    
                    # Narrative Boost logic (if enabled)
                    if delta_sec < config.SLUG_DUPLICATE_HOURS * 3600:
                        boost = (current_hits - 1) * config.NARRATIVE_BOOST_PER_HIT
                        narrative_multiplier = min(config.NARRATIVE_MAX_MULTIPLIER, 1.0 + boost)
                        if narrative_multiplier > 1.0:
                            logging.info(f"📈 Narrative Boost for {normalized_slug}: x{narrative_multiplier:.2f} (Hit #{current_hits})")
                
                state.slugs[normalized_slug] = time.time()
                state.slugs.move_to_end(normalized_slug)
                
                while len(state.slugs) > 1000: state.slugs.popitem(last=False)

        # Комбинированный Trust Factor: Статика + Динамика (из БД)
        base_trust = config.DEFAULT_TRUST_SCORE
        source_lookup = f"{source_title} {source_label}".lower()
        for s_key, s_weight in config.SOURCE_TRUST_LEVELS.items():
            if s_key.lower() in source_lookup:
                base_trust = s_weight
                break
        
        # Если по источнику есть статистика, корректируем базовый траст
        dynamic_adj = 1.0
        s_low = source_title.lower()
        if s_low in state.source_performance and isinstance(state.source_performance[s_low], dict):
            perf = state.source_performance[s_low]
            wr = perf.get("wr", 0.5)
            avg_alpha = perf.get("avg_alpha", 0.0)

            # Агрессивное подавление "токсичных" источников (отрицательная альфа или очень низкий WR)
            if wr < 0.30 or avg_alpha < -0.5:
                dynamic_adj = 0.4  # Снижаем влияние на 60% (случай с Bloomberg)
            elif wr < 0.45 or avg_alpha < 0:
                dynamic_adj = 0.75
            elif wr > 0.70 and avg_alpha > 0.1:
                dynamic_adj = 1.3  # Премиальные источники (случай с Digitimes)
            elif wr > 0.60:
                dynamic_adj = 1.15
        
        trust_factor = base_trust * dynamic_adj

        # ПРИМЕНЯЕМ ПОСТ-КАЛИБРОВКУ AI SCORE (MSF)
        model_sens = state.model_sensitivities.get(model_name, 1.0)
        
        # Safety Clip: если ИИ выдал уверенность > 1 (например 9 или 90)
        confidence = min(1.0, confidence / 10.0 if confidence > 1.0 else confidence)

        # Итоговый скор теперь учитывает индивидуальную амплитуду модели
        score *= (trust_factor * confidence * model_sens)
        
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
                    async with get_db_connection() as conn:
                        for asset in target_assets:
                            weight_key = (normalized_slug.upper(), asset.lower())
                            if weight_key not in state.weights:
                                state.weights[weight_key] = 1.0
                                await conn.execute("INSERT OR IGNORE INTO weights (event_key, target_asset, weight) VALUES (?, ?, 1.0)",
                                           (weight_key[0], weight_key[1]))
                                logging.info(f"🆕 DISCOVERED NARRATIVE: {weight_key[0]} now tracked for {asset}")
                        await conn.commit()

        # Обновляем баллы для каждого целевого актива
        for asset_name in target_assets:
            await state.scores.update_score(event_key, asset_name, score, is_market_active)

        # Фильтр значимости: для соцсетей (Reddit, X, StockTwits) повышаем порог в 2.5 раза
        is_social = any(s in source_lookup for s in ["reddit.com", "x.com", "twitter.com", "stocktwits.com"])
        effective_threshold = config.NEUTRAL_SCORE_THRESHOLD * 1.5 if is_social else config.NEUTRAL_SCORE_THRESHOLD

        # Дополнительная защита: социальные новости с низкой уверенностью обнуляются
        if is_social and confidence < 0.80:
            logging.info(f"🔇 Social Noise Filter: Muting low-confidence Reddit/X post: {slug}")
            return

        if abs(score) < effective_threshold:
            state.metrics.metrics["news_low_score"] += 1
            if is_social:
                logging.debug(f"Social Filter: Skipping {source_title} for {event_key} (Score {score:.2f} < {effective_threshold})")
            return # Используем return вместо continue, так как это функция

        # Находим самый волатильный актив (с максимальным накопленным баллом) для заголовка
        top_asset = "global"
        top_score = state.scores.get((event_key, "global"), 0.0)
        
        for asset_name in target_assets:
            current_a_score = state.scores.get((event_key, asset_name), 0.0)
            if abs(current_a_score) > abs(top_score):
                top_score = current_a_score
                top_asset = asset_name

        # Сигналы (Market/Risk) всегда базируются на глобальном индексе стресса (Global Risk Score),
        # в то время как расчет вероятности влияния (Impact) идет по самому волатильному активу.
        global_score = state.scores.get((event_key, "global"), 0.0)
        
        # Динамический расчет волатильности для сигналов
        price_history = market_data.get('price_history')
        
        top_asset_low = top_asset.lower()
        top_ticker = config.ASSET_TICKER_MAP.get(top_asset_low)
        top_vol = 1.0
        if price_history is not None and top_ticker and top_ticker in price_history.columns:
            top_vol = price_history[top_ticker].pct_change().tail(config.VOLATILITY_WINDOW).std() * 100
            if pd.isna(top_vol) or top_vol < 0.01: top_vol = 1.0
            
        top_multiplier = state.asset_multipliers.get(top_asset_low, state.multiplier)
        prob = predict_impact(top_score, top_multiplier, top_vol)
        
        # Сила сигнала в сигмах (Z-score) для обучения и внутренней логики
        top_z = min(abs(top_score) * top_multiplier, 10.0)

        market = market_signals(global_score)
        sig_type = generate_signal(prob, global_score)

        # Проверяем анти-спам ДО записи в базу, чтобы не плодить дубли
        can_send_alert = should_send(event_key, score, global_score, state, is_black_swan) 

        async with state.db_lock: # Используем лок из state manager
            try:
                async with get_db_connection() as conn:
                    # Сохраняем событие (link UNIQUE защитит от полных дублей)
                    async with conn.execute("""
                        INSERT INTO events (title, link, score, event, nasdaq, sp500, oil, soxs, gold, btc, vix, fear_greed, slug, is_black_swan, summary, title_ru, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (entry.title, entry.link, score, event_type, market["nasdaq"], market["sp500"], market["oil"], market["soxs"], market["gold"], market["btc"], market["vix"], fng_val, slug, 1 if is_black_swan else 0, ai_summary, title_ru, pub_db_time)) as cursor:
                        event_id = cursor.lastrowid
                    
                    for asset_name in target_assets:
                        a_low = asset_name.lower()
                        a_ticker = config.ASSET_TICKER_MAP.get(a_low)
                        a_vol = 1.0
                        if price_history is not None and a_ticker and a_ticker in price_history.columns:
                            a_vol = price_history[a_ticker].pct_change().tail(config.VOLATILITY_WINDOW).std() * 100
                            if pd.isna(a_vol) or a_vol < 0.01: a_vol = 1.0
                        
                        a_mult = state.asset_multipliers.get(a_low, state.multiplier)

                        # Для обучения сохраняем прогнозируемую силу конкретной новости в сигмах.
                        # Используем чистый score новости и вес, чтобы recalculate_learning.py работал корректно.
                        a_weight = state.learning.get_weight(event_key, a_low)
                        a_pred_z = min(abs(score * a_weight) * a_mult, 10.0)

                        await conn.execute("""
                            INSERT INTO predictions (event_id, event_key, score, predicted_impact, target_asset, resolved, event_type, is_black_swan, confidence, source_domain, model_name, timestamp)
                            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
                        """, (event_id, event_key, raw_score, a_pred_z, a_low, event_type, 1 if is_black_swan else 0, confidence, source_title, model_name, pub_db_time))
                    await conn.commit()

                if config.ENABLE_HOURLY_REPORT:
                    # Добавляем новость в список для часового отчета (только один раз для события)
                    state.add_news_for_summary({
                        "event_key": event_key,
                        "score": score,
                        "impact": top_z, 
                        "title": title_ru or entry.title,
                        "summary": ai_summary,
                        "link": entry.link,
                        "source": source_label
                    })

            except aiosqlite.IntegrityError:
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
                forecast_details.append(f"🌍 MARKET: {sig_type} (Stress Index: {global_score:+.2f})")

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
                f"📢 <b>Source:</b> {html.escape(source_label.upper())}\n"
                f"{divergence_tag}"
                f"{narrative_tag}"
                f"<b>Score:</b> {global_score:+.2f} (News: {score:+.2f}) | <b>Impact:</b> {prob:+.2f}%\n"
                f"-------------------\n"
                f"{forecast_str}\n"
                f"-------------------\n"
                f"{summary_part}"
                f"📰 <a href='{html.escape(entry.link)}'>{html.escape(title_ru or entry.title)}</a>"
            )
            state.metrics.metrics["news_sent_telegram"] += 1
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
                async with get_db_connection() as conn:
                    async with conn.execute("""
                        SELECT event_key, score, predicted_impact as impact, title_ru as title, summary, link, source_domain as source
                        FROM predictions p
                        JOIN events e ON p.event_id = e.id
                        WHERE p.timestamp >= datetime('now', '-1 hour')
                        GROUP BY e.link ORDER BY e.timestamp DESC
                    """) as cursor:
                        news_to_report = [dict(row) for row in await cursor.fetchall()]
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
        async with get_db_connection() as conn:
            async with conn.execute("""
                SELECT model_name, COUNT(*) as total, SUM(is_correct) as correct
                FROM predictions 
                WHERE resolved >= 1 AND timestamp >= datetime('now', '-6 hours')
                GROUP BY model_name
            """) as cursor:
                rows = await cursor.fetchall()
            
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
        summary_msg_parts.append(f"\n🧠 <b>EVENT:</b> {html.escape(news_item['event_key'])} | <b>Score:</b> {news_item['score']:.2f} | <b>Impact:</b> {news_item['impact']:.2f}σ")
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
        "queue_size": news_queue.qsize() if 'news_queue' in globals() else 0,
        "scores_in_ram": len(state.scores.scores),
        "uptime_seconds": int(time.time() - START_TIME),
        "ai_requests": state.metrics.metrics["ai_requests"],
        "active_model": model_rotator.get_active()["name"],
        "market_data_provider": state.market.market_data_status,
        "last_market_sync": datetime.fromtimestamp(state.market.last_market_data_time).strftime('%H:%M:%S') if state.market.last_market_data_time else "Never"
    }

async def check_ollama_status(session: aiohttp.ClientSession) -> bool:
    """Проверяет доступность локального сервера Ollama."""
    if not config.USE_LOCAL_OLLAMA and not config.OLLAMA_FALLBACK:
        return True

    logging.info(f"🔍 Проверка доступности Ollama: {config.OLLAMA_BASE_URL}")
    try:
        async with session.get(config.OLLAMA_BASE_URL, timeout=5) as resp:
            if resp.status == 200:
                logging.info("✅ Ollama доступна и работает.")
                return True
            logging.error(f"⚠️ Ollama ответила статусом {resp.status}")
    except Exception as e:
        logging.error(f"❌ Ошибка подключения к Ollama: {e}")
    return False

async def main():
    last_learning_run = 0
    last_cleanup_run = 0
    loop = asyncio.get_running_loop()

    # 1. Создаем постоянную сессию для воркеров и системных задач (Market Data, Telegram, AI)
    async with aiohttp.ClientSession() as persistent_session:
        workers = []
        try:
            # Инициализация БД (теперь асинхронная)
            await init_db()

            # Проверка Ollama, если она выбрана как основная модель
            if config.USE_LOCAL_OLLAMA or config.OLLAMA_FALLBACK:
                if not await check_ollama_status(persistent_session):
                    logging.critical("Критическая ошибка: Ollama не обнаружена. Завершение работы.")
                    return

            # Инициализация состояния из БД в асинхронном контексте
            await state.init_from_db()
            
            # Инициализация общего AI клиента
            if config.GEMINI_API_KEY:
                state.ai_client = genai.Client(api_key=config.GEMINI_API_KEY)
                
            # Запуск фонового репортера статистики
            asyncio.create_task(metrics_reporter_task(state))

            logging.info(f"🚀 Поставщик рыночных данных: {config.MARKET_DATA_PROVIDER.upper()}")
            
            # Запуск воркеров на основе конфигурации (1 для Free Gemini, 2+ для платных тарифов)
            workers = [asyncio.create_task(news_worker(i, persistent_session, state, model_rotator)) for i in range(config.NUM_WORKERS)]

            # Первичный запуск цикла обучения, чтобы обработать старые записи
            logging.info("Первичный запуск цикла обучения...")
            await learning_cycle(persistent_session, state)
            last_learning_run = time.time()
            
            logging.info("Первичная очистка базы данных...")
            await cleanup_db(state)
            last_cleanup_run = time.time()
            last_summary_run = time.time() # Инициализируем время последнего отчета

            while True:
                eligible_count = await count_eligible_predictions()
                time_to_next = max(0, config.LEARNING_INTERVAL - (time.time() - last_learning_run))
                minutes_left = int(time_to_next // 60)
                
                logging.info(f"📡 GTS 4.0 scanning... [До обучения: {minutes_left} мин | Готово новостей: {eligible_count}]")

                current_market_data = await get_market_data(persistent_session)
                if current_market_data:
                    state.last_market_data_time = time.time()
                    state.market_data_status = current_market_data.get('active_provider', 'unknown')
                else:
                    state.market_data_status = "FAILED"

                is_market_active = not current_market_data.get('is_stale', True)
                
                if not is_market_active:
                    logging.info("🌙 Night mode: Using slower decay factor to preserve sentiment.")

                # Универсальный мониторинг резких движений цен
                if is_market_active:
                    for asset, threshold in config.SHARP_MOVE_THRESHOLDS.items():
                        move = current_market_data.get(f"{asset}_change", 0.0)
                        if abs(move) >= threshold:
                            now = time.time()
                            if now - state.last_price_alert.get(asset, 0) > config.COOLDOWN:
                                direction = "🚀 РОСТ" if move > 0 else "🔻 ПАДЕНИЕ"
                                alert_msg = (
                                    f"⚠️ <b>РЕЗКОЕ ДВИЖЕНИЕ ЦЕНЫ: {asset.upper()}</b>\n"
                                    f"Направление: {direction} <b>{move:+.2f}%</b>\n"
                                    f"Окно мониторинга: ~{config.MARKET_LOOKBACK_HOURS}ч"
                                )
                                await send_telegram(persistent_session, alert_msg)
                                state.last_price_alert[asset] = now

                for key in list(state.scores.keys()):
                    await state.apply_decay(key, is_market_active)

                # 2. Создаем НОВУЮ сессию специально для этого цикла сканирования
                # DummyCookieJar гарантирует, что Google News не будет "узнавать" нас по куки
                async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as scan_session:
                    scan_tasks = []
                    for url in config.RSS_FEEDS:
                        # Создаем задачу, но не ждем её сразу, чтобы соблюсти паузу
                        task = asyncio.create_task(process_single_feed(url, scan_session, loop, current_market_data))
                        scan_tasks.append(task)
                        await asyncio.sleep(0.5) # Пауза между запросами (анти-бан)
                    
                    # Дожидаемся завершения всех задач сканирования ПРЕЖДЕ чем scan_session закроется
                    if scan_tasks:
                        await asyncio.gather(*scan_tasks)
                        logging.info(f"✅ Цикл сканирования лент завершен ({len(scan_tasks)} feeds)")

                # Дополнительно опрашиваем StockTwits для ключевых тикеров
                if config.SOCIAL_SEARCH_ENABLED:
                    async with aiohttp.ClientSession() as social_session:
                        social_tasks = []
                        for asset in ["NVDA", "BTC", "TSLA", "AMD"]:
                            social_tasks.append(asyncio.create_task(fetch_stocktwits(social_session, asset, current_market_data)))
                            await asyncio.sleep(1)
                        if social_tasks:
                            await asyncio.gather(*social_tasks)

                current_time = time.time()
                if current_time - last_learning_run >= config.LEARNING_INTERVAL:
                    await learning_cycle(persistent_session, state, raw_market_data=current_market_data)
                    last_learning_run = current_time
                if current_time - last_cleanup_run >= config.CLEANUP_INTERVAL:
                    await cleanup_db(state)
                    last_cleanup_run = current_time
                if config.ENABLE_HOURLY_REPORT and current_time - last_summary_run >= config.HOURLY_SUMMARY_INTERVAL:
                    await send_hourly_summary(persistent_session, state)
                    last_summary_run = current_time

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
