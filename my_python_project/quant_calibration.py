"""
Logistic calibration for SOXS bear probability based on historical quant_decisions vs SOXX returns.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("GTS.QuantCalibration")


class CalibrationBlockedError(RuntimeError):
    """Raised when historical validation shows that a calibrator is unsafe."""


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def logit(prob: float) -> float:
    p = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def anchor_calibrator(
    cal: "SOXSCalibrator",
    anchor_score: float = 0.0,
    anchor_prob_pct: Optional[float] = None,
) -> None:
    """Fix intercept so sigmoid(coef * anchor_score + intercept) == anchor_prob."""
    prob_pct = anchor_prob_pct if anchor_prob_pct is not None else config.SOXS_CALIBRATION_BASELINE_PROB
    cal.intercept = logit(prob_pct / 100.0) - cal.coef * anchor_score


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
    l2: float = 0.01,
) -> Tuple[float, float]:
    """Fit P(y=1) = sigmoid(a*x + b) via gradient descent."""
    if len(x) < 2:
        return config.SOXS_CALIBRATION_DEFAULT_COEF, logit(config.SOXS_CALIBRATION_BASELINE_PROB / 100.0)

    x_mean, x_std = float(np.mean(x)), float(np.std(x))
    if x_std < 1e-6:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    a, b = 0.0, 0.0
    for _ in range(epochs):
        z = a * x_norm + b
        p = sigmoid(z)
        grad_a = np.mean((p - y) * x_norm) + l2 * a
        grad_b = np.mean(p - y)
        a -= lr * grad_a
        b -= lr * grad_b

    a_raw = a / x_std
    b_raw = b - a * x_mean / x_std
    return float(a_raw), float(b_raw)


def fit_logistic_regression_fixed_baseline(
    x: np.ndarray,
    y: np.ndarray,
    baseline_prob: Optional[float] = None,
    lr: float = 0.05,
    epochs: int = 500,
    l2: float = 0.01,
) -> Tuple[float, float]:
    """Fit only coef while keeping score=0 anchored at baseline probability."""
    if len(x) < 2:
        coef = config.SOXS_CALIBRATION_DEFAULT_COEF
        intercept = logit((baseline_prob or config.SOXS_CALIBRATION_BASELINE_PROB) / 100.0)
        return coef, intercept

    p0 = (baseline_prob or config.SOXS_CALIBRATION_BASELINE_PROB) / 100.0
    intercept = logit(p0)
    scale = max(float(np.std(x)), 1.0)
    x_scaled = x / scale
    coef_scaled = config.SOXS_CALIBRATION_DEFAULT_COEF * scale

    for _ in range(epochs):
        z = coef_scaled * x_scaled + intercept
        p = sigmoid(z)
        grad_a = np.mean((p - y) * x_scaled) + l2 * coef_scaled
        coef_scaled -= lr * grad_a

    return float(coef_scaled / scale), intercept


class SOXSCalibrator:
    """Maps raw bear_score to calibrated probability using historical outcomes."""

    def __init__(self):
        self.coef: float = config.SOXS_CALIBRATION_DEFAULT_COEF
        self.intercept: float = logit(config.SOXS_CALIBRATION_BASELINE_PROB / 100.0)
        self.sample_count: int = 0
        self.training_count: int = 0
        self.validation_count: int = 0
        self.validation_brier: Optional[float] = None
        self.baseline_brier: Optional[float] = None
        self.is_fitted: bool = False
        self.is_blocked: bool = False
        self.block_reason: Optional[str] = None

    def calibrate_probability(self, bear_score: float) -> float:
        if self.is_blocked:
            raise CalibrationBlockedError(
                f"SOXS calibrator blocked: {self.block_reason or 'validation failure'}"
            )
        if not self.is_fitted:
            return min(100.0, max(0.0, round(30.0 + bear_score, 1)))
        prob = sigmoid(np.array([self.coef * bear_score + self.intercept]))[0] * 100.0
        return min(100.0, max(0.0, round(float(prob), 1)))

    def block(self, reason: str) -> None:
        self.is_fitted = False
        self.is_blocked = True
        self.block_reason = reason

    def normalize(self) -> None:
        """Anchor valid fitted params and block anti-predictive coefficients."""
        if not self.is_fitted:
            return
        if not np.isfinite(self.coef) or self.coef < config.SOXS_CALIBRATION_MIN_COEF:
            logger.error(
                "SOXS calibrator coef=%.4f is anti-predictive; calibration blocked",
                self.coef,
            )
            self.block("anti_predictive_coefficient")
            return
        anchor_calibrator(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coef": self.coef,
            "intercept": self.intercept,
            "sample_count": self.sample_count,
            "training_count": self.training_count,
            "validation_count": self.validation_count,
            "validation_brier": self.validation_brier,
            "baseline_brier": self.baseline_brier,
            "is_fitted": self.is_fitted,
            "is_blocked": self.is_blocked,
            "block_reason": self.block_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOXSCalibrator":
        cal = cls()
        cal.coef = float(data.get("coef", config.SOXS_CALIBRATION_DEFAULT_COEF))
        cal.intercept = float(data.get("intercept", config.SOXS_CALIBRATION_DEFAULT_INTERCEPT))
        cal.sample_count = int(data.get("sample_count", 0))
        cal.training_count = int(data.get("training_count", 0))
        cal.validation_count = int(data.get("validation_count", 0))
        cal.validation_brier = data.get("validation_brier")
        cal.baseline_brier = data.get("baseline_brier")
        cal.is_fitted = bool(data.get("is_fitted", False))
        cal.is_blocked = bool(data.get("is_blocked", False))
        cal.block_reason = data.get("block_reason")
        if cal.is_blocked:
            cal.is_fitted = False
        if cal.is_fitted:
            cal.normalize()
        return cal


def _soxx_forward_return(
    price_history: pd.DataFrame,
    decision_time: datetime,
    forward_hours: float,
) -> Optional[float]:
    ticker = config.ASSET_TICKER_MAP.get("soxx", "SOXX")
    if ticker not in price_history.columns:
        return None
    ts = price_history[ticker].dropna()
    if ts.index.tz is not None:
        ts = ts.copy()
        ts.index = ts.index.tz_convert(None)
    dt = decision_time.replace(tzinfo=None) if decision_time.tzinfo else decision_time
    end_dt = dt + timedelta(hours=forward_hours)
    idx_at = ts.index.get_indexer([dt], method="backfill")[0]
    idx_after = ts.index.get_indexer([end_dt], method="backfill")[0]
    if idx_at == -1 or idx_after == -1 or idx_at == idx_after:
        return None
    p_at = float(ts.iloc[idx_at])
    p_after = float(ts.iloc[idx_after])
    if p_at == 0:
        return None
    return ((p_after - p_at) / p_at) * 100


def load_daily_prices_from_db(ticker: str, db_path: Optional[str] = None) -> pd.Series:
    """Load reproducible adjusted daily closes for one ticker from SQLite."""
    import sqlite3

    path = db_path or config.DB_PATH
    try:
        conn = sqlite3.connect(path)
        rows = conn.execute(
            "SELECT date, close FROM daily_prices WHERE ticker = ? ORDER BY date",
            (ticker.upper(),),
        ).fetchall()
        conn.close()
    except Exception:
        return pd.Series(dtype=float)
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([float(r[1]) for r in rows], index=idx).sort_index()


def load_daily_soxx_from_db(db_path: Optional[str] = None) -> pd.Series:
    """Backward-compatible SOXX daily-price accessor."""
    return load_daily_prices_from_db("SOXX", db_path)


def _daily_forward_window(
    daily_series: pd.Series,
    decision_time: datetime,
    forward_trading_days: int,
) -> Optional[Tuple[float, datetime, datetime]]:
    if daily_series.empty:
        return None
    ts = daily_series.dropna().copy()
    ts.index = pd.to_datetime(ts.index).tz_localize(None).normalize()
    ts = ts[~ts.index.duplicated(keep="last")].sort_index()
    dt = pd.Timestamp(decision_time.replace(tzinfo=None) if decision_time.tzinfo else decision_time)
    idx_at = ts.index.get_indexer([dt.normalize()], method="backfill")[0]
    if idx_at == -1:
        return None
    idx_after = idx_at + max(1, forward_trading_days)
    if idx_after >= len(ts):
        return None
    p_at = float(ts.iloc[idx_at])
    p_after = float(ts.iloc[idx_after])
    if p_at <= 0 or not np.isfinite(p_at) or not np.isfinite(p_after):
        return None
    ret = ((p_after - p_at) / p_at) * 100
    return ret, ts.index[idx_at].to_pydatetime(), ts.index[idx_after].to_pydatetime()


def _soxx_forward_return_daily(
    daily_series: pd.Series,
    decision_time: datetime,
    forward_trading_days: int,
) -> Optional[float]:
    window = _daily_forward_window(daily_series, decision_time, forward_trading_days)
    return window[0] if window else None


def _intraday_forward_window(
    price_history: pd.DataFrame,
    decision_time: datetime,
    forward_hours: float,
) -> Optional[Tuple[float, datetime, datetime]]:
    ticker = config.ASSET_TICKER_MAP.get("soxx", "SOXX")
    if price_history.empty or ticker not in price_history.columns:
        return None
    ts = price_history[ticker].dropna().copy()
    if ts.empty:
        return None
    ts.index = pd.to_datetime(ts.index)
    if ts.index.tz is not None:
        ts.index = ts.index.tz_convert(None)
    ts = ts[~ts.index.duplicated(keep="last")].sort_index()
    dt = decision_time.replace(tzinfo=None) if decision_time.tzinfo else decision_time
    requested_end = dt + timedelta(hours=forward_hours)
    idx_at = ts.index.get_indexer([dt], method="backfill")[0]
    idx_after = ts.index.get_indexer([requested_end], method="backfill")[0]
    if idx_at == -1 or idx_after == -1 or idx_at == idx_after:
        return None
    p_at = float(ts.iloc[idx_at])
    p_after = float(ts.iloc[idx_after])
    if p_at <= 0 or not np.isfinite(p_at) or not np.isfinite(p_after):
        return None
    ret = ((p_after - p_at) / p_at) * 100
    return ret, ts.index[idx_at].to_pydatetime(), ts.index[idx_after].to_pydatetime()


def _forward_label_window(
    price_history: pd.DataFrame,
    daily_series: pd.Series,
    decision_time: datetime,
) -> Tuple[Optional[float], float, Optional[datetime], Optional[datetime]]:
    if config.SOXS_CALIBRATION_USE_MULTI_DAY_LABEL and not daily_series.empty:
        window = _daily_forward_window(
            daily_series, decision_time, config.SOXS_CALIBRATION_FORWARD_DAYS
        )
        if window:
            ret, start, end = window
            return ret, config.SOXS_CALIBRATION_MULTI_DAY_THRESHOLD_PCT, start, end
    window = _intraday_forward_window(
        price_history, decision_time, config.SOXS_CALIBRATION_FORWARD_HOURS
    )
    if not window:
        return None, config.SOXS_CALIBRATION_BEAR_THRESHOLD_PCT, None, None
    ret, start, end = window
    return ret, config.SOXS_CALIBRATION_BEAR_THRESHOLD_PCT, start, end


def _forward_return_for_label(
    price_history: pd.DataFrame,
    daily_series: pd.Series,
    decision_time: datetime,
) -> Tuple[Optional[float], float]:
    """Return (forward SOXX return %, bearish threshold %) for calibration label."""
    ret, threshold, _, _ = _forward_label_window(
        price_history, daily_series, decision_time
    )
    return ret, threshold


def build_calibration_samples(
    decisions: List[Dict[str, Any]],
    price_history: pd.DataFrame,
    daily_series: Optional[pd.Series] = None,
    holdout_cutoff: Optional[datetime] = None,
) -> pd.DataFrame:
    """Create daily, non-overlapping SOXS calibration observations."""
    daily = load_daily_soxx_from_db() if daily_series is None else daily_series
    cutoff = holdout_cutoff or (
        datetime.now() - timedelta(hours=config.SOXS_CALIBRATION_HOLDOUT_HOURS)
    )
    if not daily.empty:
        daily_returns = pd.Series(daily, dtype=float).sort_index().pct_change().abs()
        if (daily_returns > config.SOXX_CALIBRATION_MAX_ABS_DAILY_RETURN).any():
            logger.error("SOXS calibration skipped: SOXX history contains a split discontinuity")
            return pd.DataFrame()

    parsed: List[Tuple[datetime, Dict[str, Any], float]] = []
    for row in decisions:
        bear_score = row.get("bear_score")
        if bear_score is None:
            if not config.SOXS_USE_LEGACY_CALIBRATION_ROWS:
                continue
            bear_score = float(row.get("bear_probability", 30.0)) - 30.0
        try:
            dt = datetime.strptime(str(row.get("timestamp")), "%Y-%m-%d %H:%M:%S")
            score = float(bear_score)
        except (TypeError, ValueError):
            continue
        if dt >= cutoff or not np.isfinite(score):
            continue
        parsed.append((dt, row, score))

    # Hourly snapshots are correlated duplicates. Keep only the final state per UTC day.
    daily_rows: Dict[date, Tuple[datetime, Dict[str, Any], float]] = {}
    for item in sorted(parsed, key=lambda value: value[0]):
        daily_rows[item[0].date()] = item

    samples: List[Dict[str, Any]] = []
    last_label_end: Optional[datetime] = None
    for dt, _, score in daily_rows.values():
        ret, threshold, label_start, label_end = _forward_label_window(
            price_history, daily, dt
        )
        if ret is None or label_start is None or label_end is None:
            continue
        # Purge observations whose forward-return labels overlap a prior label.
        if last_label_end is not None and label_start <= last_label_end:
            continue
        samples.append(
            {
                "timestamp": dt,
                "bear_score": score,
                "forward_soxx_return_pct": ret,
                "label": 1 if ret <= -threshold else 0,
                "label_start": label_start,
                "label_end": label_end,
            }
        )
        last_label_end = label_end

    return pd.DataFrame(samples)


def fit_from_history(
    decisions: List[Dict[str, Any]],
    price_history: pd.DataFrame,
    daily_series: Optional[pd.Series] = None,
) -> SOXSCalibrator:
    """
    Fit on purged daily decisions and validate on the final chronological block.

    A non-positive coefficient or failure to beat the constant-probability baseline
    blocks the calibration instead of silently forcing a bullish coefficient.
    """
    cal = SOXSCalibrator()
    if not decisions or price_history.empty:
        cal.block_reason = "no_history"
        return cal

    samples = build_calibration_samples(
        decisions, price_history, daily_series=daily_series
    )
    cal.sample_count = len(samples)
    if len(samples) < config.SOXS_CALIBRATION_MIN_SAMPLES:
        cal.block_reason = "insufficient_independent_samples"
        logger.info(
            "SOXS calibration skipped: %d independent samples (need %d)",
            len(samples),
            config.SOXS_CALIBRATION_MIN_SAMPLES,
        )
        return cal

    validation_count = max(
        config.SOXS_CALIBRATION_MIN_VALIDATION_SAMPLES,
        int(np.ceil(len(samples) * config.SOXS_CALIBRATION_VALIDATION_RATIO)),
    )
    split_idx = len(samples) - validation_count
    if split_idx <= 0:
        cal.block_reason = "insufficient_training_samples"
        return cal

    validation = samples.iloc[split_idx:].copy()
    validation_start = validation["timestamp"].iloc[0]
    embargo_cutoff = validation_start - timedelta(days=config.SOXS_CALIBRATION_EMBARGO_DAYS)
    training = samples.iloc[:split_idx]
    training = training[training["label_end"] < embargo_cutoff]

    cal.training_count = len(training)
    cal.validation_count = len(validation)
    train_bears = int(training["label"].sum()) if not training.empty else 0
    validation_bears = int(validation["label"].sum()) if not validation.empty else 0
    train_bulls = len(training) - train_bears
    validation_bulls = len(validation) - validation_bears
    if (
        len(training) < 2
        or train_bears < config.SOXS_CALIBRATION_MIN_BEARISH_TRAIN
        or train_bulls < config.SOXS_CALIBRATION_MIN_BEARISH_TRAIN
        or validation_bears < config.SOXS_CALIBRATION_MIN_BEARISH_VALIDATION
        or validation_bulls < config.SOXS_CALIBRATION_MIN_BEARISH_VALIDATION
    ):
        cal.block_reason = "insufficient_class_balance"
        logger.warning(
            "SOXS calibration skipped: class balance train=%d/%d validation=%d/%d",
            train_bears,
            train_bulls,
            validation_bears,
            validation_bulls,
        )
        return cal

    y_validation = validation["label"].to_numpy(dtype=float)
    validation_probabilities: List[float] = []
    for row in validation.itertuples(index=False):
        fold_cutoff = row.timestamp - timedelta(days=config.SOXS_CALIBRATION_EMBARGO_DAYS)
        fold_training = samples[samples["label_end"] < fold_cutoff]
        fold_bears = int(fold_training["label"].sum()) if not fold_training.empty else 0
        fold_bulls = len(fold_training) - fold_bears
        if (
            fold_bears < config.SOXS_CALIBRATION_MIN_BEARISH_TRAIN
            or fold_bulls < config.SOXS_CALIBRATION_MIN_BEARISH_TRAIN
        ):
            cal.block_reason = "insufficient_walk_forward_class_balance"
            return cal
        fold_x = fold_training["bear_score"].to_numpy(dtype=float)
        fold_y = fold_training["label"].to_numpy(dtype=float)
        fold_coef, fold_intercept = fit_logistic_regression_fixed_baseline(fold_x, fold_y)
        if not np.isfinite(fold_coef) or fold_coef < config.SOXS_CALIBRATION_MIN_COEF:
            cal.coef = fold_coef
            cal.block("anti_predictive_coefficient")
            logger.error(
                "SOXS calibrator blocked in walk-forward fold: coef=%.4f",
                fold_coef,
            )
            return cal
        probability = float(
            sigmoid(np.array([fold_coef * float(row.bear_score) + fold_intercept]))[0]
        )
        validation_probabilities.append(probability)

    validation_prob = np.array(validation_probabilities, dtype=float)
    baseline_prob = np.full_like(
        y_validation, config.SOXS_CALIBRATION_BASELINE_PROB / 100.0
    )
    cal.validation_brier = float(np.mean((validation_prob - y_validation) ** 2))
    cal.baseline_brier = float(np.mean((baseline_prob - y_validation) ** 2))
    required_brier = cal.baseline_brier - config.SOXS_CALIBRATION_MIN_BRIER_IMPROVEMENT
    if cal.validation_brier >= required_brier:
        logger.error(
            "SOXS calibrator failed holdout: Brier %.4f vs baseline %.4f",
            cal.validation_brier,
            cal.baseline_brier,
        )
        cal.block("holdout_brier_not_better_than_baseline")
        return cal

    # Validation passed. Refit the deployable coefficient on every matured sample.
    x_all = samples["bear_score"].to_numpy(dtype=float)
    y_all = samples["label"].to_numpy(dtype=float)
    cal.coef, cal.intercept = fit_logistic_regression_fixed_baseline(x_all, y_all)
    cal.training_count = len(samples)
    cal.is_fitted = True
    cal.normalize()
    if cal.is_blocked:
        return cal

    accuracy = float(
        np.mean((validation_prob >= 0.5) == (y_validation >= 0.5))
    )
    logger.info(
        "SOXS calibrator validated: train=%d test=%d | Brier=%.3f baseline=%.3f | Acc=%.1f%% | coef=%.4f intercept=%.4f | P(score=0)=%.1f%%",
        len(training),
        len(validation),
        cal.validation_brier,
        cal.baseline_brier,
        accuracy * 100,
        cal.coef,
        cal.intercept,
        cal.calibrate_probability(0.0),
    )
    return cal


async def load_calibrator_from_db(conn) -> SOXSCalibrator:
    cal = SOXSCalibrator()
    keys = (
        "soxs_cal_coef",
        "soxs_cal_intercept",
        "soxs_cal_samples",
        "soxs_cal_training_samples",
        "soxs_cal_validation_samples",
        "soxs_cal_validation_brier",
        "soxs_cal_baseline_brier",
        "soxs_cal_fitted",
        "soxs_cal_blocked",
    )
    async with conn.execute(
        f"SELECT key, value FROM settings WHERE key IN ({','.join('?' * len(keys))})",
        keys,
    ) as cursor:
        rows = {r[0]: r[1] for r in await cursor.fetchall()}
    if not rows:
        return cal
    cal.coef = float(rows.get("soxs_cal_coef", config.SOXS_CALIBRATION_DEFAULT_COEF))
    cal.intercept = float(rows.get("soxs_cal_intercept", config.SOXS_CALIBRATION_DEFAULT_INTERCEPT))
    cal.sample_count = int(rows.get("soxs_cal_samples", 0))
    cal.training_count = int(rows.get("soxs_cal_training_samples", 0))
    cal.validation_count = int(rows.get("soxs_cal_validation_samples", 0))
    if float(rows.get("soxs_cal_validation_brier", -1.0)) >= 0:
        cal.validation_brier = float(rows["soxs_cal_validation_brier"])
    if float(rows.get("soxs_cal_baseline_brier", -1.0)) >= 0:
        cal.baseline_brier = float(rows["soxs_cal_baseline_brier"])
    cal.is_fitted = bool(int(rows.get("soxs_cal_fitted", 0)))
    cal.is_blocked = bool(int(rows.get("soxs_cal_blocked", 0)))
    if cal.is_blocked:
        cal.is_fitted = False
        cal.block_reason = "stored_validation_failure"
    if cal.is_fitted and cal.validation_count < config.SOXS_CALIBRATION_MIN_VALIDATION_SAMPLES:
        cal.block("legacy_calibrator_without_holdout")
    if cal.is_fitted:
        cal.normalize()
    return cal


async def save_calibrator_to_db(conn, cal: SOXSCalibrator) -> None:
    entries = [
        ("soxs_cal_coef", cal.coef),
        ("soxs_cal_intercept", cal.intercept),
        ("soxs_cal_samples", float(cal.sample_count)),
        ("soxs_cal_training_samples", float(cal.training_count)),
        ("soxs_cal_validation_samples", float(cal.validation_count)),
        ("soxs_cal_validation_brier", cal.validation_brier if cal.validation_brier is not None else -1.0),
        ("soxs_cal_baseline_brier", cal.baseline_brier if cal.baseline_brier is not None else -1.0),
        ("soxs_cal_fitted", 1.0 if cal.is_fitted else 0.0),
        ("soxs_cal_blocked", 1.0 if cal.is_blocked else 0.0),
    ]
    for key, value in entries:
        await conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
