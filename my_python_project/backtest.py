"""
Offline backtest for GTS predictions and SOXS quant decisions.

Usage:
    python backtest.py
    python backtest.py --walk-forward-train-ratio 0.7
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from db import get_db_connection
from market_data_history import PriceHistoryError, validate_soxs_against_soxx
from quant_calibration import (
    fit_from_history,
    load_daily_prices_from_db,
    load_daily_soxx_from_db,
)
from soxs_factor_backtest import run_ablation, print_report as print_ablation_report


def _safe_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(len(returns)))


def _calibration_bins(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    if len(predicted) == 0:
        return []
    bins = np.linspace(predicted.min(), predicted.max(), n_bins + 1)
    results = []
    for i in range(n_bins):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1] if i < n_bins - 1 else predicted <= bins[i + 1])
        if mask.sum() == 0:
            continue
        results.append({
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i + 1]),
            "count": int(mask.sum()),
            "avg_predicted": float(predicted[mask].mean()),
            "avg_actual": float(actual[mask].mean()),
            "mae": float(np.abs(predicted[mask] - actual[mask]).mean()),
        })
    return results


def _directional_winrate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    valid = df[df["is_correct"].notna()]
    if valid.empty:
        return 0.0
    return float(valid["is_correct"].mean() * 100)


def _evaluate_predictions(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    if df.empty:
        return {"label": label, "count": 0}

    df = df.copy()
    df["error"] = (df["actual_move"] - df["predicted_impact"]).abs()
    df["signed_error"] = df["actual_move"] - df["predicted_impact"]

    by_asset = {}
    for asset, group in df.groupby("target_asset"):
        by_asset[asset] = {
            "count": len(group),
            "winrate_pct": round(_directional_winrate(group), 2),
            "mae": round(float(group["error"].mean()), 3),
            "avg_signed_alpha": round(float(group["signed_alpha"].mean()), 3) if "signed_alpha" in group else None,
            "sharpe_alpha": round(_safe_sharpe(group["signed_alpha"].dropna()), 3) if "signed_alpha" in group else None,
        }

    calibration = _calibration_bins(
        df["predicted_impact"].to_numpy(),
        df["actual_move"].to_numpy(),
    )

    return {
        "label": label,
        "count": len(df),
        "winrate_pct": round(_directional_winrate(df), 2),
        "mae": round(float(df["error"].mean()), 3),
        "sharpe_signed_alpha": round(_safe_sharpe(df["signed_alpha"].dropna()), 3),
        "by_asset": by_asset,
        "calibration_bins": calibration,
    }


def _walk_forward_split(
    df: pd.DataFrame,
    train_ratio: float,
    purge_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split with a no-label-overlap purge and post-split embargo."""
    if df.empty:
        return df.copy(), df.copy()
    purge = config.SOXS_BACKTEST_PURGE_DAYS if purge_days is None else purge_days
    embargo = config.SOXS_BACKTEST_EMBARGO_DAYS if embargo_days is None else embargo_days
    df = df.copy()
    df["_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["_timestamp"]).sort_values("_timestamp")
    unique_days = pd.Index(df["_timestamp"].dt.normalize().unique()).sort_values()
    split_idx = int(len(unique_days) * train_ratio)
    if split_idx <= 0 or split_idx >= len(unique_days):
        return df.iloc[0:0].drop(columns="_timestamp"), df.drop(columns="_timestamp")
    split_time = pd.Timestamp(unique_days[split_idx])
    train_cutoff = split_time - pd.Timedelta(days=max(0, purge))
    test_cutoff = split_time + pd.Timedelta(days=max(0, embargo))
    train = df[df["_timestamp"] < train_cutoff].drop(columns="_timestamp")
    test = df[df["_timestamp"] >= test_cutoff].drop(columns="_timestamp")
    return train, test


def _independent_daily_decisions(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse correlated intraday snapshots to the final target for each UTC day."""
    if decisions_df.empty:
        return decisions_df.copy()
    df = decisions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["target_position"] = pd.to_numeric(df["target_position"], errors="coerce")
    df = df.dropna(subset=["timestamp", "target_position"])
    sort_columns = ["timestamp"] + (["id"] if "id" in df.columns else [])
    df = df.sort_values(sort_columns)
    df["decision_date"] = df["timestamp"].dt.normalize()
    daily = df.groupby("decision_date", as_index=False).tail(1).copy()
    daily = daily.sort_values("decision_date")
    return daily[daily["target_position"].ne(daily["target_position"].shift())].reset_index(drop=True)


def _performance_metrics(returns: pd.Series) -> Dict[str, Any]:
    returns = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"days": 0}
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float(equity.iloc[-1] ** (252.0 / len(returns)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "days": len(returns),
        "total_return_pct": round(total_return * 100.0, 3),
        "annualized_return_pct": round(annual_return * 100.0, 3),
        "annualized_volatility_pct": round(volatility * 100.0, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(float(drawdown.min()) * 100.0, 3),
        "positive_days_pct": round(float((returns > 0).mean()) * 100.0, 2),
    }


def simulate_soxs_strategy(
    decisions_df: pd.DataFrame,
    soxs_daily: pd.Series,
    commission_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> Dict[str, Any]:
    """Backtest target positions on actual SOXS closes with next-close execution."""
    decisions = _independent_daily_decisions(decisions_df)
    prices = pd.Series(soxs_daily, dtype=float).dropna().copy()
    if decisions.empty or len(prices) < 2:
        return {"days": 0, "message": "Insufficient SOXS prices or independent decisions"}

    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    suspicious_returns = prices.pct_change().abs()
    suspicious_dates = suspicious_returns[
        suspicious_returns > config.SOXS_BACKTEST_MAX_ABS_DAILY_RETURN
    ].index
    if len(suspicious_dates):
        return {
            "days": 0,
            "message": "SOXS history contains an unadjusted split/corporate-action discontinuity",
            "suspicious_dates": [date.strftime("%Y-%m-%d") for date in suspicious_dates],
        }
    price_frame = prices.rename("soxs_close").rename_axis("price_date").reset_index()
    signal_frame = decisions[["decision_date", "target_position"]].sort_values("decision_date")
    aligned = pd.merge_asof(
        price_frame.sort_values("price_date"),
        signal_frame,
        left_on="price_date",
        right_on="decision_date",
        direction="backward",
        allow_exact_matches=False,
    )
    aligned = aligned[aligned["decision_date"].notna()].copy()
    if len(aligned) < 2:
        return {"days": 0, "message": "No executable next-close SOXS decisions"}

    max_position = min(
        max(level["position"] for level in config.SOXS_POSITION_LEVELS),
        config.SOXS_MAX_ACTIONABLE_POSITION_PCT,
    )
    aligned["target_position"] = aligned["target_position"].clip(0.0, max_position)
    aligned["soxs_forward_return"] = aligned["soxs_close"].shift(-1) / aligned["soxs_close"] - 1.0
    aligned = aligned.dropna(subset=["soxs_forward_return"])

    commission = config.SOXS_BACKTEST_COMMISSION_BPS if commission_bps is None else commission_bps
    slippage = config.SOXS_BACKTEST_SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    cost_rate = (commission + slippage) / 10_000.0
    exposure = aligned["target_position"] / 100.0
    turnover = exposure.diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(float(exposure.iloc[0]))
    costs = turnover.fillna(0.0) * cost_rate
    strategy_returns = exposure * aligned["soxs_forward_return"] - costs

    buy_hold_returns = aligned["soxs_forward_return"].copy()
    if not buy_hold_returns.empty:
        buy_hold_returns.iloc[0] -= cost_rate

    strategy = _performance_metrics(strategy_returns)
    buy_hold = _performance_metrics(buy_hold_returns)
    strategy.update(
        {
            "active_days": int((exposure > 0).sum()),
            "trades": int((turnover.fillna(0.0) > 0).sum()),
            "turnover_pct": round(float(turnover.fillna(0.0).sum()) * 100.0, 2),
            "trading_cost_pct": round(float(costs.sum()) * 100.0, 4),
        }
    )
    return {
        "days": len(aligned),
        "independent_decisions": len(decisions),
        "execution": "next_available_close",
        "commission_bps": commission,
        "slippage_bps": slippage,
        "max_position_pct": max_position,
        "strategy": strategy,
        "buy_hold_soxs": buy_hold,
        "excess_return_pct": round(
            strategy.get("total_return_pct", 0.0) - buy_hold.get("total_return_pct", 0.0),
            3,
        ),
    }


async def load_predictions(resolved_min: int = 1) -> pd.DataFrame:
    async with get_db_connection() as conn:
        async with conn.execute(
            """
            SELECT id, event_key, score, target_asset, predicted_impact, actual_move,
                   signed_alpha, is_correct, confidence, timestamp, resolved
            FROM predictions
            WHERE resolved >= ?
            ORDER BY timestamp ASC
            """,
            (resolved_min,),
        ) as cursor:
            rows = [dict(r) for r in await cursor.fetchall()]
    return pd.DataFrame(rows)


async def load_quant_decisions() -> pd.DataFrame:
    async with get_db_connection() as conn:
        async with conn.execute(
            """
            SELECT id, bear_probability, bear_score,
                   COALESCE(shadow_target_position, target_position) AS target_position,
                   readiness_status, is_actionable, timestamp
            FROM quant_decisions
            ORDER BY timestamp ASC
            """
        ) as cursor:
            rows = [dict(r) for r in await cursor.fetchall()]
    return pd.DataFrame(rows)


async def evaluate_soxs_decisions(decisions_df: pd.DataFrame) -> Dict[str, Any]:
    if decisions_df.empty:
        return {"count": 0, "message": "No quant_decisions data"}

    daily_soxx = load_daily_soxx_from_db()
    daily_soxs = load_daily_prices_from_db("SOXS")
    if daily_soxx.empty or daily_soxs.empty:
        return {
            "count": len(decisions_df),
            "message": "Could not load reproducible SOXX and SOXS daily history",
        }

    decisions = decisions_df.to_dict("records")
    history = daily_soxx.to_frame(name="SOXX")
    cal = fit_from_history(decisions, history, daily_series=daily_soxx)
    try:
        validate_soxs_against_soxx(daily_soxs, daily_soxx)
        strategy = simulate_soxs_strategy(decisions_df, daily_soxs)
    except PriceHistoryError as exc:
        strategy = {"blocked": True, "message": str(exc)}
    ablation = run_ablation(forward_days=config.SOXS_CALIBRATION_FORWARD_DAYS)

    return {
        "count": len(decisions),
        "independent_daily_decisions": len(_independent_daily_decisions(decisions_df)),
        "calibration_fitted": cal.is_fitted,
        "calibration_blocked": cal.is_blocked,
        "calibration_reason": cal.block_reason,
        "calibration_samples": cal.sample_count,
        "calibration_training_samples": cal.training_count,
        "calibration_validation_samples": cal.validation_count,
        "validation_brier": cal.validation_brier,
        "baseline_brier": cal.baseline_brier,
        "strategy_backtest": strategy,
        "factor_ablation": ablation.get("factors", {}),
        "weight_hints": ablation.get("weight_hints", []),
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("GTS OFFLINE BACKTEST REPORT")
    print("=" * 60)

    for section in ("full", "train", "test"):
        data = report.get(section)
        if not data or data.get("count", 0) == 0:
            continue
        print(f"\n--- {data['label']} ({data['count']} predictions) ---")
        print(f"  WinRate:        {data['winrate_pct']}%")
        print(f"  MAE:            {data['mae']}")
        print(f"  Sharpe (alpha): {data['sharpe_signed_alpha']}")
        if data.get("by_asset"):
            print("  By asset:")
            for asset, stats in data["by_asset"].items():
                print(
                    f"    {asset}: n={stats['count']} WR={stats['winrate_pct']}% "
                    f"MAE={stats['mae']} Sharpe={stats['sharpe_alpha']}"
                )
        if data.get("calibration_bins"):
            print("  Calibration (predicted vs actual Z-move):")
            for b in data["calibration_bins"]:
                print(
                    f"    [{b['bin_low']:.2f}..{b['bin_high']:.2f}] "
                    f"n={b['count']} pred={b['avg_predicted']:.2f} actual={b['avg_actual']:.2f} mae={b['mae']:.2f}"
                )

    soxs = report.get("soxs")
    if soxs:
        print(f"\n--- SOXS Quant Decisions ---")
        for k, v in soxs.items():
            if k in ("factor_ablation", "weight_hints", "strategy_backtest"):
                continue
            print(f"  {k}: {v}")
        strategy_bt = soxs.get("strategy_backtest") or {}
        strategy = strategy_bt.get("strategy") or {}
        baseline = strategy_bt.get("buy_hold_soxs") or {}
        if strategy:
            print("\n  Actual SOXS strategy (next-close, net of costs):")
            print(
                f"    days={strategy_bt.get('days')} decisions={strategy_bt.get('independent_decisions')} "
                f"trades={strategy.get('trades')} costs={strategy.get('trading_cost_pct')}%"
            )
            print(
                f"    return={strategy.get('total_return_pct')}% Sharpe={strategy.get('sharpe')} "
                f"maxDD={strategy.get('max_drawdown_pct')}%"
            )
            print(
                f"    buy&hold SOXS={baseline.get('total_return_pct')}% "
                f"excess={strategy_bt.get('excess_return_pct')}%"
            )
        elif strategy_bt.get("message"):
            print(f"\n  Strategy backtest blocked: {strategy_bt['message']}")
            if strategy_bt.get("suspicious_dates"):
                print(f"    suspicious dates: {', '.join(strategy_bt['suspicious_dates'])}")
        hints = soxs.get("weight_hints") or []
        if hints:
            print("\n  Factor weight hints:")
            for h in hints:
                print(f"    - {h}")
        ablation = soxs.get("factor_ablation") or {}
        if ablation:
            print("\n  Factor ablation (corr factor->SOXX, negative=good):")
            for name, stats in ablation.items():
                if stats.get("count", 0) == 0:
                    continue
                print(
                    f"    {name}: n={stats['count']} avg_soxx={stats.get('avg_soxx_fwd_pct')}% "
                    f"corr={stats.get('corr_factor_soxx')}"
                )

    split = report.get("walk_forward_split")
    if split:
        print("\n--- Purged Walk-Forward Split ---")
        print(
            f"  raw={split['raw_count']} train={split['train_count']} test={split['test_count']} "
            f"purge={split['purge_days']}d embargo={split['embargo_days']}d"
        )

    wf = report.get("walk_forward_summary")
    if wf:
        print(f"\n--- Walk-Forward Summary ---")
        print(f"  Train winrate: {wf['train_winrate']}%")
        print(f"  Test winrate:  {wf['test_winrate']}%")
        print(f"  Degradation:   {wf['degradation_pp']} pp")


async def run_backtest(train_ratio: float = 0.7) -> Dict[str, Any]:
    df = await load_predictions(resolved_min=1)
    if df.empty:
        return {"full": {"label": "Full sample", "count": 0}}

    full = _evaluate_predictions(df, "Full sample")
    train_df, test_df = _walk_forward_split(df, train_ratio)
    train = _evaluate_predictions(train_df, f"Train (first {int(train_ratio * 100)}%)")
    test = _evaluate_predictions(test_df, f"Test (last {int((1 - train_ratio) * 100)}%)")

    decisions_df = await load_quant_decisions()
    soxs = await evaluate_soxs_decisions(decisions_df)

    wf_summary = None
    if train["count"] > 0 and test["count"] > 0:
        wf_summary = {
            "train_winrate": train["winrate_pct"],
            "test_winrate": test["winrate_pct"],
            "degradation_pp": round(train["winrate_pct"] - test["winrate_pct"], 2),
        }

    report = {
        "full": full,
        "train": train,
        "test": test,
        "soxs": soxs,
        "walk_forward_split": {
            "raw_count": len(df),
            "train_count": len(train_df),
            "test_count": len(test_df),
            "purge_days": config.SOXS_BACKTEST_PURGE_DAYS,
            "embargo_days": config.SOXS_BACKTEST_EMBARGO_DAYS,
        },
        "walk_forward_summary": wf_summary,
    }
    print_report(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="GTS offline backtest")
    parser.add_argument(
        "--walk-forward-train-ratio",
        type=float,
        default=0.7,
        help="Fraction of chronological data used for in-sample evaluation",
    )
    args = parser.parse_args()
    asyncio.run(run_backtest(train_ratio=args.walk_forward_train_ratio))


if __name__ == "__main__":
    main()
