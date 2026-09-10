"""Regression tests for the P1 SOXS execution and walk-forward backtest."""
import unittest

import pandas as pd

from backtest import (
    _independent_daily_decisions,
    _walk_forward_split,
    simulate_soxs_strategy,
)


class TestIndependentDecisions(unittest.TestCase):
    def test_intraday_snapshots_collapse_and_unchanged_days_are_removed(self):
        decisions = pd.DataFrame(
            [
                {"id": 1, "timestamp": "2026-01-01 10:00:00", "target_position": 20},
                {"id": 2, "timestamp": "2026-01-01 16:00:00", "target_position": 50},
                {"id": 3, "timestamp": "2026-01-02 16:00:00", "target_position": 50},
                {"id": 4, "timestamp": "2026-01-03 16:00:00", "target_position": 20},
            ]
        )
        result = _independent_daily_decisions(decisions)
        self.assertEqual(result["target_position"].tolist(), [50, 20])


class TestSOXSExecution(unittest.TestCase):
    def test_signal_executes_at_next_close_without_lookahead(self):
        prices = pd.Series(
            [100.0, 50.0, 100.0, 110.0],
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )
        decisions = pd.DataFrame(
            [{"timestamp": "2026-01-02 12:00:00", "target_position": 100.0}]
        )
        result = simulate_soxs_strategy(decisions, prices, commission_bps=0, slippage_bps=0)
        self.assertAlmostEqual(result["strategy"]["total_return_pct"], 2.0, places=3)

    def test_turnover_cost_is_deducted(self):
        prices = pd.Series(
            [100.0, 100.0, 100.0, 100.0],
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )
        decisions = pd.DataFrame(
            [{"timestamp": "2026-01-01 12:00:00", "target_position": 100.0}]
        )
        result = simulate_soxs_strategy(decisions, prices, commission_bps=1, slippage_bps=5)
        self.assertAlmostEqual(result["strategy"]["trading_cost_pct"], 0.012, places=4)
        self.assertAlmostEqual(result["strategy"]["total_return_pct"], -0.012, places=3)

    def test_unadjusted_split_discontinuity_blocks_report(self):
        prices = pd.Series(
            [4.0, 40.0, 41.0],
            index=pd.date_range("2026-01-01", periods=3, freq="D"),
        )
        decisions = pd.DataFrame(
            [{"timestamp": "2025-12-31 12:00:00", "target_position": 100.0}]
        )
        result = simulate_soxs_strategy(decisions, prices, commission_bps=0, slippage_bps=0)
        self.assertEqual(result["days"], 0)
        self.assertIn("split", result["message"])


class TestPurgedWalkForward(unittest.TestCase):
    def test_split_has_purge_and_embargo_gap(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
                "value": range(10),
            }
        )
        train, test = _walk_forward_split(
            df, train_ratio=0.6, purge_days=2, embargo_days=1
        )
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 3)
        self.assertLess(
            pd.to_datetime(train["timestamp"]).max() + pd.Timedelta(days=3),
            pd.to_datetime(test["timestamp"]).min(),
        )


if __name__ == "__main__":
    unittest.main()
