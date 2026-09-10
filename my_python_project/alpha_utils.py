"""
Shared alpha / expected-move calculations used by learning, recalculate, and backtest.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("GTS.AlphaUtils")


def drop_tz(series: pd.Series) -> pd.Series:
    if series.index.tz is not None:
        series = series.copy()
        series.index = series.index.tz_convert(None)
    return series


def get_ewma_beta(target_rets: pd.Series, bench_rets: pd.Series) -> float:
    if len(target_rets) < 10:
        return 1.0
    alpha = 1 - config.EWMA_LAMBDA
    aligned = pd.concat([target_rets, bench_rets], axis=1).dropna()
    if len(aligned) < 10:
        return 1.0
    t_rets = aligned.iloc[:, 0]
    b_rets = aligned.iloc[:, 1]
    cov = t_rets.ewm(alpha=alpha).cov(b_rets).iloc[-1]
    var = b_rets.ewm(alpha=alpha).var().iloc[-1]
    beta = cov / var if var > 0 else 1.0
    return float(max(-config.BETA_CLIP, min(config.BETA_CLIP, beta)))


def _benchmark_move_pct(
    price_history: pd.DataFrame,
    bench_ticker: str,
    start_time: datetime,
    end_time: datetime,
) -> Optional[float]:
    if bench_ticker not in price_history.columns:
        return None
    b_ts = drop_tz(price_history[bench_ticker].dropna())
    if b_ts.empty:
        return None
    idx_at = b_ts.index.get_indexer([start_time], method="backfill")[0]
    idx_after = b_ts.index.get_indexer([end_time], method="backfill")[0]
    if idx_at == -1 or idx_after == -1 or idx_at == idx_after:
        return None
    b_at = float(b_ts.iloc[idx_at])
    b_after = float(b_ts.iloc[idx_after])
    if b_at == 0:
        return None
    return ((b_after - b_at) / b_at) * 100


def _rolling_beta_for_pair(
    target_ticker: str,
    bench_ticker: str,
    prediction_time: datetime,
    price_history: pd.DataFrame,
    lookback_days: int = 2,
) -> float:
    hist_end = prediction_time.replace(tzinfo=None) if prediction_time.tzinfo else prediction_time
    hist_start = hist_end - timedelta(days=lookback_days)
    if target_ticker not in price_history.columns or bench_ticker not in price_history.columns:
        return 1.0
    t_rets = drop_tz(price_history[target_ticker].dropna()).loc[hist_start:hist_end].pct_change().dropna()
    b_rets = drop_tz(price_history[bench_ticker].dropna()).loc[hist_start:hist_end].pct_change().dropna()
    return get_ewma_beta(t_rets, b_rets)


def calculate_expected_move(
    target_key: str,
    bench_cfg: Dict[str, Any],
    prediction_time: datetime,
    end_time: datetime,
    price_history: pd.DataFrame,
) -> float:
    """
    Compute expected benchmark-adjusted move (%) for an asset over [prediction_time, end_time].
    """
    try:
        if prediction_time.tzinfo is not None:
            prediction_time = prediction_time.replace(tzinfo=None)
        if end_time.tzinfo is not None:
            end_time = end_time.replace(tzinfo=None)

        bench_type = bench_cfg.get("type", "fixed")

        if bench_type == "leveraged":
            primary = bench_cfg["primary"]
            move = _benchmark_move_pct(price_history, primary, prediction_time, end_time)
            if move is None:
                return 0.0
            return move * bench_cfg["factor"]

        if bench_type == "multi_factor":
            target_ticker = config.ASSET_TICKER_MAP.get(target_key)
            if not target_ticker:
                return 0.0
            weights = bench_cfg.get("weights", [0.5, -0.5])
            factors = [bench_cfg["primary"]]
            if bench_cfg.get("secondary"):
                factors.append(bench_cfg["secondary"])

            expected = 0.0
            for i, bench_ticker in enumerate(factors):
                move = _benchmark_move_pct(price_history, bench_ticker, prediction_time, end_time)
                if move is None:
                    continue
                weight = weights[i] if i < len(weights) else 0.0
                beta = _rolling_beta_for_pair(target_ticker, bench_ticker, prediction_time, price_history)
                expected += weight * beta * move
            return expected

        if bench_type == "fixed":
            primary = bench_cfg["primary"]
            move = _benchmark_move_pct(price_history, primary, prediction_time, end_time)
            if move is None:
                return 0.0
            return move * bench_cfg.get("factor", 1.0)

        # rolling_beta (default)
        target_ticker = config.ASSET_TICKER_MAP.get(target_key)
        bench_ticker = bench_cfg["primary"]
        move = _benchmark_move_pct(price_history, bench_ticker, prediction_time, end_time)
        if move is None or not target_ticker:
            return move * bench_cfg.get("factor", 1.0) if move is not None else 0.0
        beta = _rolling_beta_for_pair(target_ticker, bench_ticker, prediction_time, price_history)
        return move * beta

    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("calculate_expected_move failed for %s: %s", target_key, exc)
        return 0.0


def calculate_signed_alpha(
    asset: str,
    ticker: str,
    prediction_time: datetime,
    end_time: datetime,
    price_history: pd.DataFrame,
) -> Optional[float]:
    """Z-normalized abnormal return for an asset over the evaluation window."""
    ts = drop_tz(price_history[ticker].dropna())
    if ts.empty:
        return None

    p_time = prediction_time.replace(tzinfo=None) if prediction_time.tzinfo else prediction_time
    idx_at = ts.index.get_indexer([p_time], method="backfill")[0]
    if idx_at == -1:
        return None

    actual_start = ts.index[idx_at]
    lookback_duration = end_time - prediction_time
    shifted_end = actual_start + lookback_duration
    if ts.index[-1] < shifted_end:
        return None

    idx_after = ts.index.get_indexer([shifted_end.replace(tzinfo=None) if shifted_end.tzinfo else shifted_end], method="backfill")[0]
    if idx_after == -1 or idx_at == idx_after:
        return None

    p_at = float(ts.iloc[idx_at])
    p_after = float(ts.iloc[idx_after])
    if p_at == 0:
        return None

    raw_change = ((p_after - p_at) / p_at) * 100
    b_cfg = config.ASSET_BENCHMARK_CONFIG.get(asset.lower())

    asset_rets = drop_tz(price_history[ticker].dropna()).loc[:shifted_end].pct_change().tail(config.VOLATILITY_WINDOW)
    realized_vol_raw = asset_rets.std() * 100
    if pd.isna(realized_vol_raw) or realized_vol_raw == 0:
        realized_vol_raw = 1.0

    vol_floor = config.GLOBAL_Z_ALPHA_VOL_FLOOR if asset.lower() == "global" else config.Z_ALPHA_VOL_FLOOR
    realized_vol = max(realized_vol_raw, vol_floor)

    alpha_val = raw_change
    if b_cfg:
        bench_key = b_cfg["primary"]
        if bench_key in price_history.columns and bench_key != ticker:
            expected = calculate_expected_move(
                asset.lower(),
                b_cfg,
                actual_start.to_pydatetime() if hasattr(actual_start, "to_pydatetime") else actual_start,
                shifted_end.to_pydatetime() if hasattr(shifted_end, "to_pydatetime") else shifted_end,
                price_history,
            )
            alpha_val = raw_change - expected

    return alpha_val / realized_vol
