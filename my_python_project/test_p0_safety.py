"""Regression tests for P0 signal-safety defects."""
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

import config
import engine
from db import get_db_connection, init_db
from quant_calibration import SOXSCalibrator
from quant_engine import SOXSQuantEngine
from state_service import GTSStateManager


class TestConfigurationSafety(unittest.TestCase):
    def test_only_news_flag_is_not_inverted(self):
        with patch.dict(os.environ, {"P0_TEST_BOOL": "true"}):
            self.assertTrue(config._env_bool("P0_TEST_BOOL"))
        with patch.dict(os.environ, {"P0_TEST_BOOL": "false"}):
            self.assertFalse(config._env_bool("P0_TEST_BOOL", True))

    def test_actionable_soxs_alerts_are_opt_in_and_hard_capped(self):
        self.assertFalse(config.SOXS_ACTIONABLE_SIGNALS)
        self.assertLessEqual(config.SOXS_MAX_ACTIONABLE_POSITION_PCT, 20.0)


class TestLearningLifecycle(unittest.TestCase):
    def test_young_prediction_remains_unresolved(self):
        self.assertIsNone(engine._learning_resolution_window(0, 0.5, 1.0, 4.0))

    def test_prediction_advances_only_when_window_is_ready(self):
        self.assertEqual(engine._learning_resolution_window(0, 1.0, 1.0, 4.0), (1.0, 1))
        self.assertIsNone(engine._learning_resolution_window(1, 3.9, 1.0, 4.0))
        self.assertEqual(engine._learning_resolution_window(1, 4.0, 1.0, 4.0), (4.0, 2))


class TestPositionRestore(unittest.TestCase):
    def test_restore_uses_timestamp_of_actual_last_change(self):
        rows = [
            {"target_position": 20, "timestamp": "2026-09-01 10:00:00"},
            {"target_position": 20, "timestamp": "2026-09-02 10:00:00"},
            {"target_position": 50, "timestamp": "2026-09-03 10:00:00"},
            {"target_position": 50, "timestamp": "2026-09-04 10:00:00"},
        ]

        position, change_ts = engine._position_state_from_rows(rows)

        expected = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(position, 50.0)
        self.assertEqual(change_ts, expected)


class TestFailClosedInputs(unittest.TestCase):
    def test_missing_market_data_is_rejected(self):
        gaps = engine._soxs_market_data_gaps({}, None)
        self.assertIn("market_data", gaps)
        self.assertIn("rotation_indicator", gaps)

    def test_complete_market_data_is_accepted(self):
        market_data = {
            "soxx_current_price": 250.0,
            "soxx_ma200_value": 230.0,
            "soxx_below_ma200": False,
            "is_stale": False,
            "stale_map": {"soxx_change": False},
        }
        self.assertEqual(engine._soxs_market_data_gaps(market_data, 3), [])

    def test_stale_soxx_price_is_rejected(self):
        market_data = {
            "soxx_current_price": 250.0,
            "soxx_ma200_value": 230.0,
            "soxx_below_ma200": False,
            "is_stale": False,
            "stale_map": {"soxx_change": True},
        }
        self.assertIn("soxx_price_stale", engine._soxs_market_data_gaps(market_data, 3))

    def test_quant_engine_rejects_unknown_regime_inputs(self):
        quant = SOXSQuantEngine()
        with self.assertRaises(ValueError):
            quant.calculate_bear_probability({}, {}, 0, None, False)
        with self.assertRaises(ValueError):
            quant.calculate_bear_probability({}, {}, 0, 3, None)


class TestSOXSReadiness(unittest.TestCase):
    @staticmethod
    def _validated_calibrator() -> SOXSCalibrator:
        cal = SOXSCalibrator()
        cal.coef = 0.1
        cal.is_fitted = True
        cal.sample_count = config.SOXS_ACTIONABLE_MIN_SAMPLES
        cal.training_count = config.SOXS_ACTIONABLE_MIN_SAMPLES
        cal.validation_count = config.SOXS_ACTIONABLE_MIN_VALIDATION_SAMPLES
        cal.validation_brier = 0.10
        cal.baseline_brier = 0.20
        cal.normalize()
        return cal

    def test_unfitted_calibrator_is_shadow_only(self):
        readiness = engine._soxs_signal_readiness(
            SOXSCalibrator(), {"soxs_history_valid": True}
        )
        self.assertEqual(readiness["status"], "SHADOW")
        self.assertFalse(readiness["is_actionable"])

    def test_validated_model_requires_manual_promotion(self):
        readiness = engine._soxs_signal_readiness(
            self._validated_calibrator(), {"soxs_history_valid": True}
        )
        self.assertEqual(readiness["status"], "VALIDATED")
        self.assertIn("manual_promotion_disabled", readiness["blockers"])

    def test_actionable_requires_all_gates_and_caps_position(self):
        with patch.object(config, "SOXS_ACTIONABLE_SIGNALS", True):
            readiness = engine._soxs_signal_readiness(
                self._validated_calibrator(), {"soxs_history_valid": True}
            )
        self.assertEqual(readiness["status"], "ACTIONABLE")
        self.assertTrue(readiness["is_actionable"])
        self.assertEqual(
            engine._cap_actionable_soxs_position(120.0),
            config.SOXS_MAX_ACTIONABLE_POSITION_PCT,
        )


class TestSignalTaskFailClosed(unittest.IsolatedAsyncioTestCase):
    async def test_task_does_not_calculate_or_persist_without_market_data(self):
        state = Mock()
        state.get_last_capex_signals = AsyncMock(return_value={})
        state.get_last_guidance_signals = AsyncMock(return_value={})
        state.get_divergence_metrics_10d = AsyncMock(return_value=(0, []))
        state.get_rotation_ranking = AsyncMock(return_value=(None, 0.0, 0.0, 0.0))
        state.save_quant_decision = AsyncMock()
        guarded_quant = Mock()

        with (
            patch.object(engine, "refresh_soxs_calibrator", AsyncMock()),
            patch.object(engine, "get_market_data", AsyncMock(return_value={})),
            patch.object(engine, "quant_engine", guarded_quant),
        ):
            await engine.check_soxs_signals_task(None, state)

        guarded_quant.calculate_bear_probability.assert_not_called()
        state.save_quant_decision.assert_not_awaited()

    async def test_unfitted_calibration_is_persisted_as_shadow_without_alert(self):
        state = Mock()
        state.get_last_capex_signals = AsyncMock(return_value={})
        state.get_last_guidance_signals = AsyncMock(return_value={})
        state.get_divergence_metrics_10d = AsyncMock(return_value=(0, []))
        state.get_rotation_ranking = AsyncMock(return_value=(3, 0.0, 0.0, 0.0))
        state.save_quant_decision = AsyncMock()
        market_data = {
            "soxx_current_price": 250.0,
            "soxx_ma200_value": 230.0,
            "soxx_below_ma200": False,
            "soxs_history_valid": False,
            "is_stale": False,
            "stale_map": {"soxx_change": False},
        }
        send = AsyncMock()

        with (
            patch.object(engine, "refresh_soxs_calibrator", AsyncMock()),
            patch.object(engine, "get_market_data", AsyncMock(return_value=market_data)),
            patch.object(engine, "quant_engine", SOXSQuantEngine(SOXSCalibrator())),
            patch.object(engine, "send_telegram", send),
            patch.object(engine, "last_soxs_snapshot_ts", 0.0),
        ):
            await engine.check_soxs_signals_task(None, state)

        send.assert_not_awaited()
        state.save_quant_decision.assert_awaited_once()
        saved = state.save_quant_decision.await_args.kwargs
        self.assertEqual(saved["target_pos"], 0.0)
        self.assertGreaterEqual(saved["shadow_target_pos"], 0.0)
        self.assertEqual(saved["readiness_status"], "SHADOW")
        self.assertFalse(saved["is_actionable"])

    async def test_task_does_not_persist_when_calibration_is_blocked(self):
        state = Mock()
        state.get_last_capex_signals = AsyncMock(return_value={})
        state.get_last_guidance_signals = AsyncMock(return_value={})
        state.get_divergence_metrics_10d = AsyncMock(return_value=(0, []))
        state.get_rotation_ranking = AsyncMock(return_value=(3, 0.0, 0.0, 0.0))
        state.save_quant_decision = AsyncMock()
        market_data = {
            "soxx_current_price": 250.0,
            "soxx_ma200_value": 230.0,
            "soxx_below_ma200": False,
            "is_stale": False,
            "stale_map": {"soxx_change": False},
        }
        calibrator = SOXSCalibrator()
        calibrator.block("holdout_brier_not_better_than_baseline")

        with (
            patch.object(engine, "refresh_soxs_calibrator", AsyncMock()),
            patch.object(engine, "get_market_data", AsyncMock(return_value=market_data)),
            patch.object(engine, "quant_engine", SOXSQuantEngine(calibrator)),
        ):
            await engine.check_soxs_signals_task(None, state)

        state.save_quant_decision.assert_not_awaited()

    async def test_actionable_alert_is_promoted_only_after_all_gates_and_capped(self):
        state = Mock()
        state.get_last_capex_signals = AsyncMock(return_value={})
        state.get_last_guidance_signals = AsyncMock(return_value={})
        state.get_divergence_metrics_10d = AsyncMock(return_value=(10, []))
        state.get_rotation_ranking = AsyncMock(return_value=(10, 0.0, 0.0, 0.0))
        state.save_quant_decision = AsyncMock()
        market_data = {
            "soxx_current_price": 200.0,
            "soxx_ma200_value": 230.0,
            "soxx_below_ma200": True,
            "soxs_history_valid": True,
            "is_stale": False,
            "stale_map": {"soxx_change": False},
        }
        calibrator = SOXSCalibrator()
        calibrator.coef = 0.1
        calibrator.is_fitted = True
        calibrator.sample_count = config.SOXS_ACTIONABLE_MIN_SAMPLES
        calibrator.training_count = config.SOXS_ACTIONABLE_MIN_SAMPLES
        calibrator.validation_count = config.SOXS_ACTIONABLE_MIN_VALIDATION_SAMPLES
        calibrator.validation_brier = 0.10
        calibrator.baseline_brier = 0.20
        calibrator.normalize()
        send = AsyncMock()

        with (
            patch.object(config, "SOXS_ACTIONABLE_SIGNALS", True),
            patch.object(engine, "refresh_soxs_calibrator", AsyncMock()),
            patch.object(engine, "get_market_data", AsyncMock(return_value=market_data)),
            patch.object(engine, "quant_engine", SOXSQuantEngine(calibrator)),
            patch.object(engine, "send_telegram", send),
            patch.object(engine, "last_notified_position", -1.0),
            patch.object(engine, "last_applied_soxs_position", -1.0),
            patch.object(engine, "last_soxs_snapshot_ts", 0.0),
        ):
            await engine.check_soxs_signals_task(None, state)

        send.assert_awaited_once()
        state.save_quant_decision.assert_awaited_once()
        saved = state.save_quant_decision.await_args.kwargs
        self.assertEqual(saved["target_pos"], config.SOXS_MAX_ACTIONABLE_POSITION_PCT)
        self.assertTrue(saved["is_actionable"])
        self.assertEqual(saved["readiness_status"], "ACTIONABLE")


class TestQuantDecisionAudit(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_metadata_is_migrated_and_persisted(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "readiness.db")
            with patch.object(config, "DB_PATH", db_path):
                await init_db()
                manager = GTSStateManager()
                await manager.save_quant_decision(
                    bear_prob=45.0,
                    bear_score=15.0,
                    target_pos=0.0,
                    shadow_target_pos=50.0,
                    capex=-1.0,
                    guidance=0.0,
                    triggers=["test"],
                    readiness_status="SHADOW",
                    is_actionable=False,
                    block_reason="calibration_not_fitted",
                )
                async with get_db_connection() as conn:
                    async with conn.execute(
                        """
                        SELECT target_position, shadow_target_position,
                               readiness_status, is_actionable, block_reason
                        FROM quant_decisions
                        """
                    ) as cursor:
                        row = await cursor.fetchone()

        self.assertEqual(tuple(row), (0.0, 50.0, "SHADOW", 0, "calibration_not_fitted"))

if __name__ == "__main__":
    unittest.main()
