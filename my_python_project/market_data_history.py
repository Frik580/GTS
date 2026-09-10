"""Validated, fail-closed daily market-price ingestion.

Provider responses are normalized and validated in memory before a ticker's
history is replaced in SQLite.  A failed provider or suspicious price series
can therefore never partially overwrite the last known-good dataset.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

import aiohttp
import aiosqlite
import pandas as pd

import config
from http_utils import create_verified_ssl_context

logger = logging.getLogger("GTS.MarketHistory")


class PriceHistoryError(RuntimeError):
    """Raised when downloaded price history is unusable or unsafe."""


class PriceProviderUnavailable(PriceHistoryError):
    """Raised when a configured provider cannot be used."""


@dataclass(frozen=True)
class PriceHistory:
    ticker: str
    provider: str
    prices: pd.Series
    # Stored with the sync run so a later backtest can distinguish a raw
    # provider download from a history that passed the SOXS/SOXX safety gate.
    validation: str = "daily_price_validation"


@dataclass(frozen=True)
class PriceSyncResult:
    ticker: str
    status: str
    provider: Optional[str] = None
    row_count: int = 0
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    error: Optional[str] = None
    validation: Optional[str] = None


def _minimum_rows(ticker: str) -> int:
    if ticker.upper() in {"SOXX", "SOXS"}:
        return config.PRICE_HISTORY_CORE_MIN_ROWS
    return config.PRICE_HISTORY_MIN_ROWS


def _max_abs_daily_return(ticker: str) -> float:
    ticker = ticker.upper()
    if ticker == "SOXX":
        return config.SOXX_CALIBRATION_MAX_ABS_DAILY_RETURN
    if ticker == "SOXS":
        return config.SOXS_BACKTEST_MAX_ABS_DAILY_RETURN
    return config.PRICE_HISTORY_MAX_ABS_DAILY_RETURN


def normalize_daily_prices(prices: pd.Series) -> pd.Series:
    """Return a sorted UTC-indexed float series with one close per day."""
    if prices is None:
        return pd.Series(dtype=float)
    series = prices.copy() if isinstance(prices, pd.Series) else pd.Series(prices)
    if series.empty:
        return pd.Series(dtype=float)

    try:
        index = pd.to_datetime(series.index, utc=True, errors="raise").normalize()
    except Exception as exc:
        raise PriceHistoryError(f"invalid date index: {exc}") from exc

    numeric = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(numeric.to_numpy(), index=index, dtype=float)
    normalized = normalized.groupby(level=0).last().sort_index()
    return normalized


def _drop_incomplete_core_bar(
    ticker: str, series: pd.Series, current: datetime
) -> pd.Series:
    """Do not persist an in-progress US daily bar as an official close."""
    if series.empty or ticker.upper() not in {"SOXX", "SOXS"}:
        return series
    current_ny = current.astimezone(ZoneInfo("America/New_York"))
    close_with_grace = clock_time(
        16,
        config.PRICE_HISTORY_US_CLOSE_GRACE_MINUTES,
    )
    if (
        series.index[-1].date() == current_ny.date()
        and current_ny.time().replace(tzinfo=None) < close_with_grace
    ):
        return series.iloc[:-1]
    return series


def validate_daily_prices(
    ticker: str,
    prices: pd.Series,
    *,
    now: Optional[datetime] = None,
    min_rows: Optional[int] = None,
    require_fresh: bool = True,
) -> pd.Series:
    """Validate a complete daily series and return its normalized form."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    series = _drop_incomplete_core_bar(ticker, normalize_daily_prices(prices), current)
    required = _minimum_rows(ticker) if min_rows is None else min_rows
    if len(series) < required:
        raise PriceHistoryError(f"only {len(series)} rows; need at least {required}")
    if series.isna().any():
        raise PriceHistoryError("history contains non-numeric or missing closes")
    if not series.map(math.isfinite).all():
        raise PriceHistoryError("history contains non-finite closes")
    if (series <= 0).any():
        raise PriceHistoryError("history contains non-positive closes")

    last_day = series.index[-1].to_pydatetime()
    age_days = (current.astimezone(timezone.utc) - last_day).total_seconds() / 86400
    if age_days < -config.PRICE_HISTORY_FUTURE_TOLERANCE_DAYS:
        raise PriceHistoryError(f"latest bar is in the future: {series.index[-1].date()}")
    if require_fresh and age_days > config.PRICE_HISTORY_MAX_AGE_DAYS:
        raise PriceHistoryError(
            f"latest bar is stale by {age_days:.1f} days: {series.index[-1].date()}"
        )

    returns = series.pct_change().dropna().abs()
    if not returns.empty:
        worst_date = returns.idxmax()
        worst_return = float(returns.loc[worst_date])
        limit = _max_abs_daily_return(ticker)
        if worst_return > limit:
            raise PriceHistoryError(
                f"corporate-action discontinuity on {worst_date.date()}: "
                f"{worst_return:.2%} exceeds {limit:.0%}"
            )
    return series


def validate_soxs_against_soxx(
    soxs_prices: pd.Series, soxx_prices: pd.Series
) -> None:
    """Reject split-like SOXS jumps while allowing genuine -3x crash moves."""
    soxs = normalize_daily_prices(soxs_prices)
    soxx = normalize_daily_prices(soxx_prices)
    aligned = pd.concat(
        [soxs.pct_change().rename("soxs"), soxx.pct_change().rename("soxx")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise PriceHistoryError("SOXS/SOXX histories have no overlapping returns")
    candidates = aligned[aligned["soxs"].abs() >= config.SOXS_TRACKING_CHECK_MIN_ABS_RETURN]
    for index, row in candidates.iterrows():
        residual = abs(float(row["soxs"]) + 3.0 * float(row["soxx"]))
        if residual > config.SOXS_TRACKING_MAX_RESIDUAL:
            raise PriceHistoryError(
                f"SOXS corporate-action/tracking discontinuity on {index.date()}: "
                f"SOXS={row['soxs']:.2%}, SOXX={row['soxx']:.2%}, "
                f"-3x residual={residual:.2%}"
            )


def _parse_twelvedata_payload(ticker: str, payload: object) -> pd.Series:
    if not isinstance(payload, dict):
        raise PriceHistoryError("Twelve Data returned a non-object response")
    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message", "missing values")
        raise PriceHistoryError(f"Twelve Data error: {message}")
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise PriceHistoryError("Twelve Data returned no daily bars")

    points = {}
    for row in values:
        if not isinstance(row, dict) or "datetime" not in row or "close" not in row:
            raise PriceHistoryError("Twelve Data returned a malformed daily bar")
        points[row["datetime"]] = row["close"]
    return normalize_daily_prices(pd.Series(points, name=ticker))


def _parse_fred_csv(ticker: str, series_id: str, payload: str) -> pd.Series:
    """Parse the public FRED CSV export and omit its missing-value markers."""
    try:
        frame = pd.read_csv(StringIO(payload))
        if "DATE" not in frame.columns or series_id not in frame.columns:
            raise PriceHistoryError("FRED returned an unexpected CSV schema")
        values = pd.to_numeric(frame[series_id], errors="coerce")
        dates = pd.to_datetime(frame["DATE"], utc=True, errors="coerce")
        series = pd.Series(values.to_numpy(), index=dates, name=ticker)
        series = series[series.index.notna() & series.notna()]
        if series.empty:
            raise PriceHistoryError("FRED returned no numeric daily bars")
        return normalize_daily_prices(series)
    except PriceHistoryError:
        raise
    except Exception as exc:
        raise PriceHistoryError(f"FRED returned malformed CSV: {exc}") from exc


def _apply_yahoo_split_adjustments(
    ticker: str, closes: object, index: pd.DatetimeIndex, result: object
) -> pd.Series:
    """Normalize Yahoo raw closes to the latest share basis using split events."""
    series = normalize_daily_prices(pd.Series(closes, index=index, name=ticker))
    events = result.get("events", {}) if isinstance(result, dict) else {}
    splits = events.get("splits", {}) if isinstance(events, dict) else {}
    if not isinstance(splits, dict):
        return series

    for split in splits.values():
        if not isinstance(split, dict):
            continue
        try:
            numerator = float(split["numerator"])
            denominator = float(split["denominator"])
            split_ts = pd.to_datetime(split["date"], unit="s", utc=True).normalize()
            if numerator <= 0 or denominator <= 0:
                continue
            # To express pre-split prices in the current share basis, a 1:20
            # reverse split multiplies earlier closes by 20; a 2:1 split halves
            # them. Yahoo's raw Close is the appropriate input for this step.
            series.loc[series.index < split_ts] *= denominator / numerator
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return series


def _parse_yahoo_chart_payload(
    ticker: str,
    payload: object,
    *,
    soxx_prices: Optional[pd.Series] = None,
) -> pd.Series:
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise PriceHistoryError(f"Yahoo Chart error: {chart['error']}")
        result = chart["result"][0]
        timestamps = result["timestamp"]
        indicators = result["indicators"]
        adjusted = indicators.get("adjclose") or []
        adjusted_closes = adjusted[0].get("adjclose") if adjusted else None
        raw_closes = indicators["quote"][0]["close"]
    except PriceHistoryError:
        raise
    except (KeyError, IndexError, TypeError) as exc:
        raise PriceHistoryError(f"Yahoo Chart returned malformed data for {ticker}") from exc
    index = pd.to_datetime(timestamps, unit="s", utc=True)

    # Yahoo may return an adjclose array full of nulls for indices and yields.
    # For SOXS it is ambiguous whether ``close`` already includes a split.
    # When a SOXX reference is available, try every representation and accept
    # only the one that passes the independent -3x tracking check.  This avoids
    # double-applying a reverse split and writing an inflated price history.
    candidates = (("adjclose", adjusted_closes), ("close", raw_closes))
    events = result.get("events", {}) if isinstance(result, dict) else {}
    splits = events.get("splits", {}) if isinstance(events, dict) else {}
    if ticker.upper() == "SOXS" and raw_closes is not None and splits:
        split_adjusted = _apply_yahoo_split_adjustments(
            ticker, raw_closes, index, result
        )
        candidates = (
            (("adjclose", adjusted_closes), ("close", raw_closes),
             ("split-adjusted close", split_adjusted))
            if soxx_prices is not None
            else (("split-adjusted close", split_adjusted),
                  ("adjclose", adjusted_closes), ("close", raw_closes))
        )

    candidate_errors = []
    for label, closes in candidates:
        if closes is None:
            candidate_errors.append(f"{label} is missing")
            continue
        if len(timestamps) != len(closes):
            candidate_errors.append(f"{label} length differs from timestamps")
            continue
        # ``split-adjusted close`` is already a Series normalized to midnight,
        # while Yahoo's raw arrays use exchange-time timestamps.  Rebuilding a
        # Series with the raw timestamp index would reindex the normalized
        # candidate to all-NaN.  Preserve an existing Series' own index.
        candidate = closes if isinstance(closes, pd.Series) else pd.Series(
            closes, index=index, name=ticker
        )
        series = normalize_daily_prices(candidate)
        valid = series.notna() & series.map(math.isfinite) & (series > 0)
        invalid_count = int((~valid).sum())
        if invalid_count > config.YAHOO_DAILY_MAX_MISSING_CLOSES:
            candidate_errors.append(
                f"{label} contains {invalid_count} invalid closes"
            )
            continue
        cleaned = series[valid]
        if cleaned.empty:
            candidate_errors.append(f"{label} contains no usable closes")
            continue
        if invalid_count:
            logger.info(
                "%s Yahoo Chart %s: dropped %d invalid daily bar(s)",
                ticker,
                label,
                invalid_count,
            )
        if ticker.upper() == "SOXS" and soxx_prices is not None:
            try:
                validate_soxs_against_soxx(cleaned, soxx_prices)
            except PriceHistoryError as exc:
                candidate_errors.append(f"{label} fails SOXS/SOXX check: {exc}")
                continue
        return cleaned

    raise PriceHistoryError(
        "Yahoo Chart returned no complete close series: " + "; ".join(candidate_errors)
    )


def _parse_yahoo_intraday_payload(ticker: str, payload: object) -> pd.Series:
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise PriceHistoryError(f"Yahoo Chart error: {chart['error']}")
        result = chart["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except PriceHistoryError:
        raise
    except (KeyError, IndexError, TypeError) as exc:
        raise PriceHistoryError(f"Yahoo Chart returned malformed intraday data for {ticker}") from exc
    if len(timestamps) != len(closes):
        raise PriceHistoryError("Yahoo Chart intraday timestamps and closes differ in length")
    index = pd.to_datetime(timestamps, unit="s", utc=True)
    series = pd.Series(pd.to_numeric(closes, errors="coerce"), index=index, name=ticker)
    series = series.dropna()
    series = series[series.map(math.isfinite) & (series > 0)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) < 2:
        raise PriceHistoryError(f"Yahoo Chart returned only {len(series)} usable intraday bars")
    returns = series.pct_change().dropna().abs()
    if not returns.empty and float(returns.max()) > _max_abs_daily_return(ticker):
        worst_date = returns.idxmax()
        raise PriceHistoryError(
            f"intraday corporate-action discontinuity at {worst_date.isoformat()}"
        )
    return series.astype(float)


async def fetch_yahoo_chart_intraday(
    session: aiohttp.ClientSession, tickers: Iterable[str]
) -> pd.DataFrame:
    """Fetch a verified two-day intraday tail, tolerating individual symbol failure."""
    semaphore = asyncio.Semaphore(config.MARKET_INTRADAY_MAX_CONCURRENCY)

    async def fetch_one(ticker: str) -> tuple[str, Optional[pd.Series]]:
        params = {
            "range": "2d",
            "interval": "15m",
            "events": "div,splits",
            "includePrePost": "false",
        }
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        try:
            async with semaphore:
                async with session.get(
                    url,
                    params=params,
                    ssl=create_verified_ssl_context(),
                    headers={"User-Agent": "Mozilla/5.0 GTS-Market-Intraday/1.0"},
                    timeout=aiohttp.ClientTimeout(
                        total=config.PRICE_HISTORY_REQUEST_TIMEOUT
                    ),
                ) as response:
                    if response.status != 200:
                        raise PriceHistoryError(f"Yahoo Chart HTTP {response.status}")
                    series = _parse_yahoo_intraday_payload(ticker, await response.json())
                    return ticker, series
        except Exception as exc:
            logger.warning("%s intraday rejected from yahoo_chart: %s", ticker, exc)
            return ticker, None

    unique_tickers = tuple(dict.fromkeys(str(t) for t in tickers if t))
    fetched = await asyncio.gather(*(fetch_one(ticker) for ticker in unique_tickers))
    columns = {ticker: series for ticker, series in fetched if series is not None}
    if not columns:
        raise PriceHistoryError("Yahoo Chart returned no usable intraday series")
    return pd.DataFrame(columns).sort_index()


async def _fetch_twelvedata_daily(
    session: aiohttp.ClientSession, ticker: str
) -> pd.Series:
    if not config.MARKET_DATA_API_KEY:
        raise PriceProviderUnavailable("MARKET_DATA_API_KEY is not configured")
    if ticker.upper() in config.TWELVEDATA_UNSUPPORTED_TICKERS:
        raise PriceProviderUnavailable(
            "ticker is unavailable from the configured Twelve Data plan"
        )
    params = {
        "symbol": config.TWELVEDATA_SYMBOL_MAP.get(ticker.upper(), ticker),
        "interval": "1day",
        "outputsize": str(config.PRICE_HISTORY_OUTPUT_SIZE),
        "order": "ASC",
        "adjust": "all",
        "apikey": config.MARKET_DATA_API_KEY,
    }
    for attempt in range(config.TWELVEDATA_MAX_RETRIES + 1):
        async with session.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            ssl=create_verified_ssl_context(),
            timeout=aiohttp.ClientTimeout(total=config.PRICE_HISTORY_REQUEST_TIMEOUT),
        ) as response:
            if response.status == 200:
                return _parse_twelvedata_payload(ticker, await response.json())
            retryable = response.status == 429 or response.status >= 500
            if not retryable or attempt >= config.TWELVEDATA_MAX_RETRIES:
                raise PriceHistoryError(f"Twelve Data HTTP {response.status}")
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except (TypeError, ValueError):
                delay = 0.0
            if delay <= 0:
                delay = config.TWELVEDATA_RETRY_BASE_SECONDS * (2**attempt)
            delay = min(delay, config.TWELVEDATA_RETRY_MAX_SECONDS)
            logger.warning(
                "%s Twelve Data HTTP %s; retrying in %.1fs (%d/%d)",
                ticker,
                response.status,
                delay,
                attempt + 1,
                config.TWELVEDATA_MAX_RETRIES,
            )
        await asyncio.sleep(delay)


async def _fetch_yahoo_chart_daily(
    session: aiohttp.ClientSession,
    ticker: str,
    *,
    soxx_prices: Optional[pd.Series] = None,
) -> pd.Series:
    params = {
        "period1": "0",
        "period2": str(int(time.time()) + 86400),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    async with session.get(
        url,
        params=params,
        ssl=create_verified_ssl_context(),
        headers={"User-Agent": "Mozilla/5.0 GTS-Market-History/1.0"},
        timeout=aiohttp.ClientTimeout(total=config.PRICE_HISTORY_REQUEST_TIMEOUT),
    ) as response:
        if response.status != 200:
            raise PriceHistoryError(f"Yahoo Chart HTTP {response.status}")
        return _parse_yahoo_chart_payload(
            ticker, await response.json(), soxx_prices=soxx_prices
        )


async def _fetch_fred_daily(
    session: aiohttp.ClientSession, ticker: str
) -> pd.Series:
    series_id = config.FRED_SERIES_MAP.get(ticker.upper())
    if not series_id:
        raise PriceProviderUnavailable("no FRED series is configured for ticker")
    async with session.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        ssl=create_verified_ssl_context(),
        timeout=aiohttp.ClientTimeout(total=config.PRICE_HISTORY_REQUEST_TIMEOUT),
    ) as response:
        if response.status != 200:
            raise PriceHistoryError(f"FRED HTTP {response.status}")
        return _parse_fred_csv(ticker, series_id, await response.text())


async def _fetch_yfinance_daily(ticker: str) -> pd.Series:
    def download() -> pd.Series:
        import yfinance as yf

        cache_dir = Path(config.YFINANCE_CACHE_DIR).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
        frame = yf.download(
            ticker,
            period="max" if ticker.upper() in {"SOXX", "SOXS"} else "1y",
            interval="1d",
            auto_adjust=True,
            # yfinance's repair mode imports scipy at runtime. The validated
            # provider pipeline already rejects suspicious prices, so avoid an
            # optional dependency becoming a single point of failure.
            repair=False,
            progress=False,
            threads=False,
            timeout=config.PRICE_HISTORY_REQUEST_TIMEOUT,
        )
        if frame is None or frame.empty or "Close" not in frame:
            raise PriceHistoryError("yfinance returned no daily bars")
        close = frame["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return normalize_daily_prices(close)

    return await asyncio.get_running_loop().run_in_executor(None, download)


async def fetch_validated_price_history(
    session: aiohttp.ClientSession,
    ticker: str,
    providers: Optional[Sequence[str]] = None,
    *,
    soxx_prices: Optional[pd.Series] = None,
) -> PriceHistory:
    """Try configured providers in order and accept only a validated series."""
    ordered = tuple(providers or config.HISTORICAL_PRICE_PROVIDERS)
    errors = []
    for provider in ordered:
        provider = provider.strip().lower()
        try:
            if provider == "twelvedata":
                prices = await _fetch_twelvedata_daily(session, ticker)
            elif provider == "yahoo_chart":
                prices = await _fetch_yahoo_chart_daily(
                    session, ticker, soxx_prices=soxx_prices
                )
            elif provider == "fred":
                prices = await _fetch_fred_daily(session, ticker)
            elif provider == "yfinance":
                prices = await _fetch_yfinance_daily(ticker)
            else:
                raise PriceProviderUnavailable(f"unknown provider {provider!r}")
            prices = validate_daily_prices(ticker, prices)
            if ticker.upper() == "SOXS" and soxx_prices is not None:
                validate_soxs_against_soxx(prices, soxx_prices)
            validation = "daily_price_validation"
            if ticker.upper() == "SOXS" and soxx_prices is not None:
                validation += ";soxs_soxx_tracking"
            return PriceHistory(
                ticker=ticker,
                provider=provider,
                prices=prices,
                validation=validation,
            )
        except PriceProviderUnavailable as exc:
            errors.append(f"{provider}: {exc}")
            logger.debug("%s history provider %s skipped: %s", ticker, provider, exc)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            logger.warning("%s history rejected from %s: %s", ticker, provider, exc)
    raise PriceHistoryError("; ".join(errors) or "no historical-price providers configured")


async def ensure_price_sync_schema(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_sync_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            provider TEXT,
            status TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            first_date TEXT,
            last_date TEXT,
            error TEXT,
            validation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    async with conn.execute("PRAGMA table_info(price_sync_runs)") as cursor:
        existing = {row[1] for row in await cursor.fetchall()}
    if "validation" not in existing:
        await conn.execute("ALTER TABLE price_sync_runs ADD COLUMN validation TEXT")


async def replace_daily_prices_atomically(
    conn: aiosqlite.Connection, history: PriceHistory
) -> None:
    """Replace one validated ticker history, rolling back the ticker on failure."""
    prices = validate_daily_prices(history.ticker, history.prices)
    if history.ticker.upper() == "SOXS":
        soxx_prices = await _load_ticker_series(conn, "SOXX")
        validate_soxs_against_soxx(prices, soxx_prices)
    records = [
        (history.ticker, index.strftime("%Y-%m-%d"), float(value))
        for index, value in prices.items()
    ]
    await ensure_price_sync_schema(conn)
    await conn.execute("SAVEPOINT replace_daily_prices")
    try:
        await conn.execute("DELETE FROM daily_prices WHERE ticker = ?", (history.ticker,))
        await conn.executemany(
            "INSERT INTO daily_prices (ticker, date, close) VALUES (?, ?, ?)", records
        )
        await conn.execute(
            """
            INSERT INTO price_sync_runs
                (ticker, provider, status, row_count, first_date, last_date, validation)
            VALUES (?, ?, 'success', ?, ?, ?, ?)
            """,
            (
                history.ticker,
                history.provider,
                len(records),
                records[0][1],
                records[-1][1],
                history.validation,
            ),
        )
        await conn.execute("RELEASE SAVEPOINT replace_daily_prices")
    except Exception:
        await conn.execute("ROLLBACK TO SAVEPOINT replace_daily_prices")
        await conn.execute("RELEASE SAVEPOINT replace_daily_prices")
        raise


async def _load_ticker_series(
    conn: aiosqlite.Connection, ticker: str, *, days: Optional[int] = None
) -> pd.Series:
    sql = "SELECT date, close FROM daily_prices WHERE ticker = ?"
    params: list[object] = [ticker]
    if days is not None:
        sql += " AND date >= date('now', '-' || ? || ' days')"
        params.append(days)
    sql += " ORDER BY date ASC"
    async with conn.execute(sql, params) as cursor:
        rows = await cursor.fetchall()
    return normalize_daily_prices(pd.Series({row[0]: row[1] for row in rows}, name=ticker))


async def _cached_history_is_valid(conn: aiosqlite.Connection, ticker: str) -> bool:
    try:
        prices = await _load_ticker_series(conn, ticker)
        validated = validate_daily_prices(ticker, prices)
        if ticker.upper() == "SOXS":
            validate_soxs_against_soxx(
                validated, await _load_ticker_series(conn, "SOXX")
            )
        latest = validated.index[-1].to_pydatetime()
        age_days = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
        return age_days <= config.PRICE_HISTORY_REFRESH_AFTER_DAYS
    except PriceHistoryError:
        return False


async def _record_failed_sync(
    conn: aiosqlite.Connection, ticker: str, error: str
) -> None:
    await ensure_price_sync_schema(conn)
    await conn.execute(
        """
        INSERT INTO price_sync_runs (ticker, status, error)
        VALUES (?, 'failed', ?)
        """,
        (ticker, error[:1000]),
    )


async def sync_daily_price_histories(
    session: aiohttp.ClientSession,
    conn: aiosqlite.Connection,
    tickers: Iterable[str],
    *,
    force: bool = False,
) -> list[PriceSyncResult]:
    """Refresh invalid/stale histories without sacrificing last known-good data."""
    results = []
    unique_tickers = list(dict.fromkeys(str(t).upper() for t in tickers if t))
    # Refresh the SOXX reference before SOXS whenever both are requested.
    unique_tickers.sort(key=lambda ticker: ticker == "SOXS")
    for ticker in unique_tickers:
        if not force and await _cached_history_is_valid(conn, ticker):
            results.append(PriceSyncResult(ticker=ticker, status="cached"))
            continue
        try:
            soxx_prices = (
                await _load_ticker_series(conn, "SOXX")
                if ticker == "SOXS"
                else None
            )
            history = await fetch_validated_price_history(
                session, ticker, soxx_prices=soxx_prices
            )
            await replace_daily_prices_atomically(conn, history)
            await conn.commit()
            results.append(
                PriceSyncResult(
                    ticker=ticker,
                    status="success",
                    provider=history.provider,
                    row_count=len(history.prices),
                    first_date=history.prices.index[0].strftime("%Y-%m-%d"),
                    last_date=history.prices.index[-1].strftime("%Y-%m-%d"),
                    validation=history.validation,
                )
            )
        except Exception as exc:
            error = str(exc)
            await _record_failed_sync(conn, ticker, error)
            await conn.commit()
            results.append(PriceSyncResult(ticker=ticker, status="failed", error=error))
    return results


async def load_validated_daily_frame(
    conn: aiosqlite.Connection,
    tickers: Iterable[str],
    *,
    days: int = 400,
) -> pd.DataFrame:
    """Load only DB series that still pass the same production quality gate."""
    columns = {}
    for ticker in dict.fromkeys(str(t).upper() for t in tickers if t):
        try:
            full = await _load_ticker_series(conn, ticker)
            validated = validate_daily_prices(ticker, full)
            cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days)
            columns[ticker] = validated[validated.index >= cutoff]
        except PriceHistoryError as exc:
            logger.error("DB history rejected for %s: %s", ticker, exc)
    if "SOXS" in columns and "SOXX" in columns:
        try:
            validate_soxs_against_soxx(columns["SOXS"], columns["SOXX"])
        except PriceHistoryError as exc:
            logger.error("DB history rejected for SOXS: %s", exc)
            columns.pop("SOXS", None)
    return pd.DataFrame(columns).sort_index()
