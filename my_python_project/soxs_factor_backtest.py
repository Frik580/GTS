"""
SOXS factor ablation backtest — validates each quant input vs forward SOXX returns.

Usage:
    python soxs_factor_backtest.py
    python soxs_factor_backtest.py --forward-days 5
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from quant_calibration import load_daily_soxx_from_db, _soxx_forward_return_daily
from soxs_signal_utils import match_prediction_to_ticker, SOXS_TICKER_ALIASES


def _forward_return(daily: pd.Series, dt: datetime, days: int) -> Optional[float]:
    return _soxx_forward_return_daily(daily, dt, days)


def load_signal_predictions(db_path: str = config.DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT p.id, p.event_key, p.timestamp, p.capex_signal, p.guidance_signal,
               p.score, p.target_asset, p.is_correct, p.resolved,
               e.title, e.summary, e.slug
        FROM predictions p
        LEFT JOIN events e ON p.event_id = e.id
        WHERE (p.capex_signal IS NOT NULL AND p.capex_signal != 0)
           OR (p.guidance_signal IS NOT NULL AND p.guidance_signal != 0)
        ORDER BY p.timestamp ASC
        """,
        conn,
    )
    conn.close()
    return df


def load_quant_snapshots(db_path: str = config.DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT id, bear_score, bear_probability, target_position,
               capex_score, guidance_score, active_triggers, timestamp
        FROM quant_decisions
        WHERE bear_score IS NOT NULL
        ORDER BY timestamp ASC
        """,
        conn,
    )
    conn.close()
    return df


def evaluate_factor_rows(
    rows: pd.DataFrame,
    daily: pd.Series,
    forward_days: int,
    label_fn,
) -> Dict[str, Any]:
    """Generic evaluator: label_fn(row) -> factor value or None to skip."""
    if rows.empty:
        return {"count": 0}
    rows = rows.copy()
    rows["_timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce")
    rows = rows.dropna(subset=["_timestamp"]).sort_values("_timestamp")
    rows["_date"] = rows["_timestamp"].dt.normalize()
    rows = rows.groupby("_date", as_index=False).tail(1).sort_values("_timestamp")

    daily_clean = pd.Series(daily, dtype=float).dropna().copy()
    daily_clean.index = pd.to_datetime(daily_clean.index).tz_localize(None).normalize()
    daily_clean = daily_clean[~daily_clean.index.duplicated(keep="last")].sort_index()
    rets: List[float] = []
    factors: List[float] = []
    bearish_hits = 0
    n = 0
    last_label_end_idx = -1

    for _, row in rows.iterrows():
        factor = label_fn(row)
        if factor is None:
            continue
        dt = row["_timestamp"].to_pydatetime()
        idx_at = daily_clean.index.get_indexer([pd.Timestamp(dt).normalize()], method="backfill")[0]
        idx_after = idx_at + max(1, forward_days) if idx_at >= 0 else -1
        if idx_at == -1 or idx_after >= len(daily_clean) or idx_at <= last_label_end_idx:
            continue
        start_price = float(daily_clean.iloc[idx_at])
        end_price = float(daily_clean.iloc[idx_after])
        if start_price <= 0:
            continue
        ret = (end_price - start_price) / start_price * 100.0
        last_label_end_idx = idx_after
        n += 1
        rets.append(ret)
        factors.append(factor)
        if ret <= -config.SOXS_CALIBRATION_MULTI_DAY_THRESHOLD_PCT:
            bearish_hits += 1

    if n == 0:
        return {"count": 0}

    rets_arr = np.array(rets)
    factors_arr = np.array(factors)
    corr = (
        float(np.corrcoef(factors_arr, rets_arr)[0, 1])
        if n > 2 and np.std(factors_arr) > 1e-12 and np.std(rets_arr) > 1e-12
        else 0.0
    )

    return {
        "count": n,
        "avg_soxx_fwd_pct": round(float(rets_arr.mean()), 3),
        "bearish_rate_pct": round(bearish_hits / n * 100, 1),
        "corr_factor_soxx": round(corr, 3),
        "interpretation": "negative corr = higher factor -> lower SOXX (good for bear hedge)",
    }


def run_ablation(forward_days: int = 5) -> Dict[str, Any]:
    daily = load_daily_soxx_from_db()
    if daily.empty:
        return {"error": "No SOXX daily_prices in DB"}

    preds = load_signal_predictions()
    quant = load_quant_snapshots()
    results: Dict[str, Any] = {"forward_days": forward_days, "factors": {}}

    # --- Per-ticker CAPEX cuts (signal == -1) ---
    for ticker in config.SOXS_CAPEX_WEIGHTS:
        subset = preds[
            (preds["capex_signal"] == -1)
            & preds.apply(
                lambda r: match_prediction_to_ticker(
                    ticker, r["event_key"], r.get("title") or "", r.get("summary") or "", r.get("slug") or ""
                ),
                axis=1,
            )
        ]
        key = f"capex_cut_{ticker}"
        results["factors"][key] = evaluate_factor_rows(
            subset, daily, forward_days, lambda r: -1.0
        )

    # --- Per-ticker guidance downgrades ---
    for ticker in config.SOXS_GUIDANCE_WEIGHTS:
        subset = preds[
            (preds["guidance_signal"] == -1)
            & preds.apply(
                lambda r: match_prediction_to_ticker(
                    ticker, r["event_key"], r.get("title") or "", r.get("summary") or "", r.get("slug") or ""
                ),
                axis=1,
            )
        ]
        key = f"guidance_down_{ticker}"
        results["factors"][key] = evaluate_factor_rows(
            subset, daily, forward_days, lambda r: -1.0
        )

    # --- Divergence proxy: failed bearish news ---
    div_rows = preds[
        (preds["score"] >= config.SOXS_DIVERGENCE_MIN_SCORE)
        & (preds["target_asset"].isin(config.SOXS_DIVERGENCE_TARGET_ASSETS))
        & (preds["is_correct"] == 0)
        & (preds["resolved"] >= 1)
    ]
    results["factors"]["divergence_failed_bearish_news"] = evaluate_factor_rows(
        div_rows, daily, forward_days, lambda r: float(r["score"])
    )

    # --- Composite bear_score from quant snapshots ---
    if not quant.empty:
        results["factors"]["composite_bear_score"] = evaluate_factor_rows(
            quant, daily, forward_days, lambda r: float(r["bear_score"])
        )
        results["factors"]["composite_bear_prob"] = evaluate_factor_rows(
            quant, daily, forward_days, lambda r: float(r["bear_probability"])
        )

    # --- Weight recommendations ---
    results["weight_hints"] = _suggest_weight_adjustments(results["factors"])
    return results


def _suggest_weight_adjustments(factors: Dict[str, Dict[str, Any]]) -> List[str]:
    hints: List[str] = []
    for name, stats in factors.items():
        if stats.get("count", 0) < 5:
            hints.append(f"{name}: insufficient data (n={stats.get('count', 0)})")
            continue
        corr = stats.get("corr_factor_soxx", 0)
        if corr < -0.15:
            hints.append(f"{name}: useful bear signal (corr={corr}); consider increasing weight")
        elif corr > 0.15:
            hints.append(f"{name}: anti-predictive (corr={corr}); consider reducing weight or inverting")
        else:
            hints.append(f"{name}: weak signal (corr={corr}, n={stats['count']})")
    return hints


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("SOXS FACTOR ABLATION BACKTEST")
    print("=" * 60)
    if "error" in report:
        print("Error:", report["error"])
        return

    print(f"Forward window: {report['forward_days']} trading days")
    print(f"Bearish label threshold: SOXX <= -{config.SOXS_CALIBRATION_MULTI_DAY_THRESHOLD_PCT}%")

    for name, stats in report.get("factors", {}).items():
        if stats.get("count", 0) == 0:
            print(f"\n--- {name} --- no samples")
            continue
        print(f"\n--- {name} (n={stats['count']}) ---")
        print(f"  Avg SOXX forward:  {stats['avg_soxx_fwd_pct']:+.2f}%")
        print(f"  Bearish rate:      {stats['bearish_rate_pct']}%")
        print(f"  Corr factor->SOXX: {stats['corr_factor_soxx']}")

    hints = report.get("weight_hints", [])
    if hints:
        print("\n--- Weight hints ---")
        for h in hints:
            print(f"  - {h}")


def main():
    parser = argparse.ArgumentParser(description="SOXS factor ablation backtest")
    parser.add_argument("--forward-days", type=int, default=config.SOXS_CALIBRATION_FORWARD_DAYS)
    args = parser.parse_args()
    report = run_ablation(forward_days=args.forward_days)
    print_report(report)


if __name__ == "__main__":
    main()
