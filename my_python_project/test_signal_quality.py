"""Unit tests for signal quality improvements (alpha_utils, SOXS calibration)."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import aiosqlite

import config
from alpha_utils import calculate_expected_move, get_ewma_beta
import quant_calibration
from quant_calibration import (
    CalibrationBlockedError,
    SOXSCalibrator,
    anchor_calibrator,
    build_calibration_samples,
    fit_from_history,
    fit_logistic_regression,
    load_calibrator_from_db,
    save_calibrator_to_db,
)
from quant_engine import SOXSQuantEngine
from soxs_signal_utils import (
    enrich_soxs_signals,
    match_soxs_ticker,
    match_prediction_to_ticker,
    divergence_weight,
)


def _make_price_series(start: float, changes: list, freq: str = "15min") -> pd.Series:
    idx = pd.date_range("2026-01-01 09:30", periods=len(changes) + 1, freq=freq)
    prices = [start]
    for ch in changes:
        prices.append(prices[-1] * (1 + ch))
    return pd.Series(prices, index=idx)


class TestAlphaUtils(unittest.TestCase):
    def test_get_ewma_beta_positive_correlation(self):
        rng = np.random.default_rng(42)
        target_returns = rng.normal(0.001, 0.01, 50)
        benchmark_returns = target_returns * 0.8 + rng.normal(0.0, 0.001, 50)
        target = pd.Series(100.0 * np.cumprod(1.0 + target_returns))
        bench = pd.Series(100.0 * np.cumprod(1.0 + benchmark_returns))
        beta = get_ewma_beta(target.pct_change().dropna(), bench.pct_change().dropna())
        self.assertGreater(beta, 0)

    def test_multi_factor_uses_secondary_benchmark(self):
        tip = _make_price_series(100, [0.001] * 30)
        dxy = _make_price_series(100, [-0.001] * 30)
        gold = _make_price_series(100, [0.0005] * 30)
        history = pd.DataFrame({"TIP": tip, "DX-Y.NYB": dxy, "GLD": gold})

        start = tip.index[10].to_pydatetime()
        end = tip.index[20].to_pydatetime()
        cfg = config.ASSET_BENCHMARK_CONFIG["gold"]

        expected = calculate_expected_move("gold", cfg, start, end, history)
        # With TIP up and DXY down, multi-factor expected move should be non-zero
        self.assertNotEqual(expected, 0.0)

    def test_leveraged_type(self):
        soxx = _make_price_series(200, [0.01] * 10)
        history = pd.DataFrame({"SOXX": soxx})
        start = soxx.index[0].to_pydatetime()
        end = soxx.index[5].to_pydatetime()
        cfg = config.ASSET_BENCHMARK_CONFIG["soxs"]
        result = calculate_expected_move("soxs", cfg, start, end, history)
        # SOXX rose ~5%, leveraged -3x => negative expected
        self.assertLess(result, 0)


class TestSOXSCalibration(unittest.TestCase):
    def test_logistic_regression_separates_classes(self):
        x = np.array([-10, -5, 0, 5, 10, 15, 20], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
        a, b = fit_logistic_regression(x, y, epochs=1000)
        cal = SOXSCalibrator()
        cal.coef, cal.intercept, cal.is_fitted = a, b, True
        cal.normalize()
        low = cal.calibrate_probability(-10)
        high = cal.calibrate_probability(20)
        self.assertLess(low, high)

    def test_baseline_anchor_at_zero_score(self):
        cal = SOXSCalibrator()
        cal.coef = 0.08
        cal.is_fitted = True
        cal.normalize()
        self.assertAlmostEqual(cal.calibrate_probability(0.0), config.SOXS_CALIBRATION_BASELINE_PROB, places=0)

    def test_normalize_blocks_negative_coef(self):
        cal = SOXSCalibrator()
        cal.coef = -0.13
        cal.intercept = 0.32
        cal.is_fitted = True
        cal.normalize()
        self.assertFalse(cal.is_fitted)
        self.assertTrue(cal.is_blocked)
        self.assertEqual(cal.block_reason, "anti_predictive_coefficient")
        with self.assertRaises(CalibrationBlockedError):
            cal.calibrate_probability(0.0)

    def test_fit_blocks_anti_predictive_training_relationship(self):
        timestamps = pd.date_range("2025-01-01", periods=24, freq="7D")
        samples = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bear_score": [20.0 if i % 2 == 0 else 0.0 for i in range(24)],
                "label": [0 if i % 2 == 0 else 1 for i in range(24)],
                "label_end": timestamps + pd.Timedelta(days=5),
            }
        )
        history = pd.DataFrame(
            {"SOXX": [100.0, 101.0]},
            index=pd.date_range("2025-01-01", periods=2),
        )
        with patch.object(quant_calibration, "build_calibration_samples", return_value=samples):
            cal = fit_from_history([{"bear_score": 1}], history)
        self.assertTrue(cal.is_blocked)
        self.assertEqual(cal.block_reason, "anti_predictive_coefficient")

    def test_fit_accepts_predictive_holdout(self):
        timestamps = pd.date_range("2025-01-01", periods=24, freq="7D")
        samples = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bear_score": [20.0 if i % 2 == 0 else 0.0 for i in range(24)],
                "label": [1 if i % 2 == 0 else 0 for i in range(24)],
                "label_end": timestamps + pd.Timedelta(days=5),
            }
        )
        history = pd.DataFrame(
            {"SOXX": [100.0, 101.0]},
            index=pd.date_range("2025-01-01", periods=2),
        )
        with patch.object(quant_calibration, "build_calibration_samples", return_value=samples):
            cal = fit_from_history([{"bear_score": 1}], history)
        self.assertTrue(cal.is_fitted)
        self.assertFalse(cal.is_blocked)
        self.assertLess(cal.validation_brier, cal.baseline_brier)

    def test_calibration_samples_are_daily_and_non_overlapping(self):
        dates = pd.date_range("2025-01-01", periods=80, freq="B")
        prices = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
        decisions = []
        for day in dates[:30]:
            decisions.append({"bear_score": 1.0, "timestamp": day.strftime("%Y-%m-%d 10:00:00")})
            decisions.append({"bear_score": 2.0, "timestamp": day.strftime("%Y-%m-%d 16:00:00")})
        samples = build_calibration_samples(
            decisions,
            prices.to_frame(name="SOXX"),
            daily_series=prices,
            holdout_cutoff=datetime(2030, 1, 1),
        )
        self.assertGreater(len(samples), 1)
        self.assertLess(len(samples), 30)
        for previous, current in zip(samples.iloc[:-1].itertuples(), samples.iloc[1:].itertuples()):
            self.assertGreater(current.label_start, previous.label_end)

    def test_calibration_rejects_unadjusted_soxx_split(self):
        dates = pd.date_range("2025-01-01", periods=20, freq="B")
        prices = pd.Series([100.0] * 10 + [40.0] * 10, index=dates)
        decisions = [
            {"bear_score": 5.0, "timestamp": dates[0].strftime("%Y-%m-%d 10:00:00")}
        ]
        samples = build_calibration_samples(
            decisions,
            prices.to_frame(name="SOXX"),
            daily_series=prices,
            holdout_cutoff=datetime(2030, 1, 1),
        )
        self.assertTrue(samples.empty)

    def test_fit_from_history_skips_legacy_rows(self):
        tip = _make_price_series(100, [0.0] * 40, freq="1h")
        soxx = _make_price_series(200, [-0.005] * 40, freq="1h")
        history = pd.DataFrame({"SOXX": soxx, "TIP": tip})
        decisions = [
            {"bear_score": None, "bear_probability": 55.0, "timestamp": "2026-01-02 10:00:00"},
            {"bear_score": 10.0, "bear_probability": 40.0, "timestamp": "2026-01-02 11:00:00"},
        ]
        cal = fit_from_history(decisions, history)
        self.assertFalse(cal.is_fitted)

    def test_fallback_linear_when_not_fitted(self):
        cal = SOXSCalibrator()
        self.assertFalse(cal.is_fitted)
        self.assertAlmostEqual(cal.calibrate_probability(0), 30.0)
        self.assertAlmostEqual(cal.calibrate_probability(10), 40.0)

    def test_unfitted_quant_engine_exposes_only_shadow_position(self):
        engine = SOXSQuantEngine(calibrator=SOXSCalibrator())
        res = engine.calculate_bear_probability({}, {}, 0, 0, False)
        self.assertEqual(res["calibration_state"], "SHADOW")
        self.assertEqual(res["target_position_percent"], 0.0)
        self.assertGreater(res["shadow_target_position_percent"], 0.0)

    def test_quant_engine_regime_gate_caps_bullish(self):
        cal = SOXSCalibrator()
        cal.coef = 0.1
        cal.is_fitted = True
        cal.normalize()
        engine = SOXSQuantEngine(calibrator=cal)
        # Moderate bear score -> ~50% raw position; gate caps to 20% without MA200
        res = engine.calculate_bear_probability({}, {}, divergence_instances=5, rotation_indicator=3, soxx_below_ma200=False)
        self.assertLessEqual(res["target_position_percent"], config.SOXS_REGIME_GATE_MAX_POSITION)
        self.assertTrue(res.get("regime_gated", False))

    def test_quant_engine_uses_calibrator(self):
        cal = SOXSCalibrator()
        cal.coef = 0.1
        cal.is_fitted = True
        cal.normalize()
        engine = SOXSQuantEngine(calibrator=cal)
        res = engine.calculate_bear_probability({}, {}, 0, 0, False)
        self.assertIn("bear_score", res)
        self.assertIn("calibration_fitted", res)
        self.assertTrue(res["calibration_fitted"])
        self.assertAlmostEqual(res["bear_probability"], config.SOXS_CALIBRATION_BASELINE_PROB, places=0)


class TestSOXSSignalUtils(unittest.TestCase):
    def test_nvidia_alias_matching(self):
        self.assertTrue(
            match_prediction_to_ticker("NVDA", "NVIDIA_HALVE_ASIA", "Nvidia cuts forecast", "")
        )
        self.assertTrue(
            match_prediction_to_ticker("AVGO", "ASML_GUIDANCE_UPGRADE", "", "Broadcom outlook raised")
        )

    def test_enrich_guidance_from_text(self):
        _, guidance = enrich_soxs_signals(
            ["NVIDIA"],
            "NVIDIA_GUIDANCE_CUT",
            "Nvidia lowers datacenter revenue outlook",
            "Company cut full-year guidance amid weak demand.",
            None,
            None,
        )
        self.assertEqual(guidance, -1)

    def test_divergence_weight_guidance_cut(self):
        w = divergence_weight({"guidance_signal": -1, "capex_signal": 0, "score": 0, "resolved": 1})
        self.assertEqual(w, config.SOXS_DIVERGENCE_GUIDANCE_WEIGHT)


class TestCalibrationPersistence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = await aiosqlite.connect(":memory:")
        await self.conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value REAL)")

    async def asyncTearDown(self):
        await self.conn.close()

    async def test_legacy_fitted_calibrator_is_blocked_without_holdout(self):
        await self.conn.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            [
                ("soxs_cal_coef", 0.1),
                ("soxs_cal_intercept", -0.8),
                ("soxs_cal_samples", 100),
                ("soxs_cal_fitted", 1),
            ],
        )
        cal = await load_calibrator_from_db(self.conn)
        self.assertTrue(cal.is_blocked)
        self.assertEqual(cal.block_reason, "legacy_calibrator_without_holdout")

    async def test_validated_calibrator_round_trip(self):
        cal = SOXSCalibrator()
        cal.coef = 0.1
        cal.is_fitted = True
        cal.sample_count = 30
        cal.training_count = 24
        cal.validation_count = 6
        cal.validation_brier = 0.15
        cal.baseline_brier = 0.21
        cal.normalize()
        await save_calibrator_to_db(self.conn, cal)
        loaded = await load_calibrator_from_db(self.conn)
        self.assertTrue(loaded.is_fitted)
        self.assertFalse(loaded.is_blocked)
        self.assertEqual(loaded.validation_count, 6)
        self.assertAlmostEqual(loaded.validation_brier, 0.15)


class TestWalkForwardConfig(unittest.TestCase):
    def test_holdout_constants_exist(self):
        self.assertGreater(config.WALK_FORWARD_HOLDOUT_HOURS, 0)
        self.assertGreater(config.WALK_FORWARD_MIN_CALIBRATION_AGE, 0)
        self.assertLess(config.WALK_FORWARD_MIN_CALIBRATION_AGE, config.WALK_FORWARD_HOLDOUT_HOURS)


if __name__ == "__main__":
    unittest.main()
