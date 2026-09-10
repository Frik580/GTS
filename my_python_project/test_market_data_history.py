import ssl
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import aiosqlite
import pandas as pd

from http_utils import create_verified_ssl_context
from market_data_history import (
    PriceHistory,
    PriceHistoryError,
    _parse_fred_csv,
    _parse_twelvedata_payload,
    _parse_yahoo_chart_payload,
    _parse_yahoo_intraday_payload,
    fetch_validated_price_history,
    load_validated_daily_frame,
    replace_daily_prices_atomically,
    sync_daily_price_histories,
    validate_daily_prices,
    validate_soxs_against_soxx,
)


def valid_series(*, ticker="SOXX", end=None):
    del ticker
    end = end or (
        pd.Timestamp.now(tz="UTC").normalize() - pd.offsets.BDay(1)
    )
    index = pd.date_range(end=end, periods=220, freq="B", tz="UTC")
    return pd.Series([100.0 + i * 0.1 for i in range(len(index))], index=index)


class TestPriceValidation(unittest.TestCase):
    def test_verified_ssl_context_cannot_disable_certificate_checks(self):
        context = create_verified_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_valid_core_history_is_normalized(self):
        prices = valid_series()
        result = validate_daily_prices("SOXX", prices)
        self.assertEqual(len(result), 220)
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertIsNotNone(result.index.tz)

    def test_split_discontinuity_is_rejected(self):
        prices = valid_series(ticker="SOXS")
        prices.iloc[-1] *= 8
        with self.assertRaisesRegex(PriceHistoryError, "corporate-action discontinuity"):
            validate_daily_prices("SOXS", prices)

    def test_stale_history_is_rejected(self):
        old_end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=30)
        with self.assertRaisesRegex(PriceHistoryError, "stale"):
            validate_daily_prices("SOXX", valid_series(end=old_end))

    def test_in_progress_core_daily_bar_is_removed(self):
        current = datetime(2026, 9, 8, 18, 0, tzinfo=timezone.utc)  # 14:00 New York
        index = pd.date_range(end="2026-09-08", periods=220, freq="B", tz="UTC")
        prices = pd.Series([100.0 + i * 0.1 for i in range(220)], index=index)
        result = validate_daily_prices("SOXX", prices, now=current)
        self.assertEqual(result.index[-1].date().isoformat(), "2026-09-07")

    def test_soxs_split_like_jump_fails_tracking_check(self):
        soxx = valid_series()
        soxs = valid_series(ticker="SOXS")
        soxx.iloc[-1] = soxx.iloc[-2] * 1.06
        soxs.iloc[-1] = soxs.iloc[-2] * 0.05
        with self.assertRaisesRegex(PriceHistoryError, "tracking discontinuity"):
            validate_soxs_against_soxx(soxs, soxx)

    def test_genuine_inverse_crash_move_passes_tracking_check(self):
        soxx = valid_series()
        soxs = valid_series(ticker="SOXS")
        soxx.iloc[-1] = soxx.iloc[-2] * 0.80
        soxs.iloc[-1] = soxs.iloc[-2] * 1.60
        validate_soxs_against_soxx(soxs, soxx)

    def test_twelvedata_parser_rejects_error_payload(self):
        with self.assertRaisesRegex(PriceHistoryError, "quota exceeded"):
            _parse_twelvedata_payload(
                "SOXX", {"status": "error", "message": "quota exceeded"}
            )

    def test_fred_parser_drops_missing_value_markers(self):
        result = _parse_fred_csv(
            "^TNX",
            "DGS10",
            "DATE,DGS10\n2026-09-07,.\n2026-09-08,4.13\n2026-09-09,4.11\n",
        )
        self.assertEqual(result.tolist(), [4.13, 4.11])

    def test_yahoo_parser_prefers_adjusted_close(self):
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [1704067200, 1704153600],
                        "indicators": {
                            "quote": [{"close": [100.0, 800.0]}],
                            "adjclose": [{"adjclose": [12.5, 13.0]}],
                        },
                    }
                ],
            }
        }
        result = _parse_yahoo_chart_payload("SOXS", payload)
        self.assertEqual(result.tolist(), [12.5, 13.0])

    def test_yahoo_parser_falls_back_to_close_when_adjusted_has_gaps(self):
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [1704067200, 1704153600, 1704240000],
                        "indicators": {
                            "quote": [{"close": [100.0, None, 101.0]}],
                            "adjclose": [{"adjclose": [None, None, None]}],
                        },
                    }
                ],
            }
        }
        result = _parse_yahoo_chart_payload("^TNX", payload)
        self.assertEqual(result.tolist(), [100.0, 101.0])

    def test_yahoo_parser_split_adjusts_soxs_raw_close(self):
        split_date = 1704153600
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [1704067200, split_date],
                        "indicators": {
                            "quote": [{"close": [10.0, 200.0]}],
                            "adjclose": [{"adjclose": [10.0, 200.0]}],
                        },
                        "events": {
                            "splits": {
                                str(split_date): {
                                    "date": split_date,
                                    "numerator": 1,
                                    "denominator": 20,
                                }
                            }
                        },
                    }
                ],
            }
        }
        result = _parse_yahoo_chart_payload("SOXS", payload)
        self.assertEqual(result.tolist(), [200.0, 200.0])

    def test_yahoo_parser_uses_soxx_tracking_to_select_split_basis(self):
        split_date = 1704153600
        index = pd.to_datetime([1704067200, split_date], unit="s", utc=True)
        payload = {
            "chart": {
                "error": None,
                "result": [{
                    "timestamp": [1704067200, split_date],
                    "indicators": {
                        "quote": [{"close": [10.0, 200.0]}],
                        "adjclose": [{"adjclose": [10.0, 200.0]}],
                    },
                    "events": {"splits": {str(split_date): {
                        "date": split_date, "numerator": 1, "denominator": 20,
                    }}},
                }],
            }
        }
        soxx = pd.Series([100.0, 100.0], index=index)
        result = _parse_yahoo_chart_payload("SOXS", payload, soxx_prices=soxx)
        self.assertEqual(result.tolist(), [200.0, 200.0])

    def test_yahoo_parser_keeps_split_candidate_with_exchange_time_index(self):
        # Chart endpoints commonly use an intraday exchange timestamp.  The
        # split normalizer changes it to midnight, so the parser must not
        # reindex that Series back to the original intraday timestamps.
        split_date = 1704205800  # 2024-01-02 14:30 UTC
        payload = {
            "chart": {
                "error": None,
                "result": [{
                    "timestamp": [1704119400, split_date],
                    "indicators": {
                        "quote": [{"close": [10.0, 200.0]}],
                        "adjclose": [{"adjclose": [None, None]}],
                    },
                    "events": {"splits": {str(split_date): {
                        "date": split_date, "numerator": 1, "denominator": 20,
                    }}},
                }],
            }
        }
        result = _parse_yahoo_chart_payload("SOXS", payload)
        self.assertEqual(result.tolist(), [200.0, 200.0])

    def test_yahoo_intraday_parser_drops_missing_bars(self):
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [1704067200, 1704068100, 1704069000],
                        "indicators": {"quote": [{"close": [12.5, None, 13.0]}]},
                    }
                ],
            }
        }
        result = _parse_yahoo_intraday_payload("SOXS", payload)
        self.assertEqual(result.tolist(), [12.5, 13.0])


class TestProviderFallback(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_primary_provider_falls_back(self):
        prices = valid_series()
        with patch(
            "market_data_history._fetch_twelvedata_daily",
            new=AsyncMock(side_effect=PriceHistoryError("bad primary")),
        ), patch(
            "market_data_history._fetch_yahoo_chart_daily",
            new=AsyncMock(return_value=prices),
        ):
            result = await fetch_validated_price_history(
                object(), "SOXX", providers=("twelvedata", "yahoo_chart")
            )
        self.assertEqual(result.provider, "yahoo_chart")
        self.assertEqual(len(result.prices), 220)

    async def test_soxs_tracking_failure_falls_back_to_next_provider(self):
        soxx = valid_series()
        invalid_soxs = valid_series(ticker="SOXS")
        invalid_soxs.iloc[-1] = invalid_soxs.iloc[-2] * 0.05
        soxx.iloc[-1] = soxx.iloc[-2] * 1.06
        valid_soxs = valid_series(ticker="SOXS")
        valid_soxs.iloc[-1] = valid_soxs.iloc[-2] * 0.82

        with patch(
            "market_data_history._fetch_yahoo_chart_daily",
            new=AsyncMock(return_value=invalid_soxs),
        ), patch(
            "market_data_history._fetch_yfinance_daily",
            new=AsyncMock(return_value=valid_soxs),
        ):
            result = await fetch_validated_price_history(
                object(),
                "SOXS",
                providers=("yahoo_chart", "yfinance"),
                soxx_prices=soxx,
            )

        self.assertEqual(result.provider, "yfinance")


class TestAtomicPricePersistence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = await aiosqlite.connect(":memory:")
        await self.conn.execute(
            """
            CREATE TABLE daily_prices (
                ticker TEXT, date TEXT, close REAL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        await self.conn.commit()

    async def asyncTearDown(self):
        await self.conn.close()

    async def test_complete_history_replaces_old_rows(self):
        await self.conn.execute(
            "INSERT INTO daily_prices VALUES ('SOXX', '2000-01-01', 10.0)"
        )
        history = PriceHistory("SOXX", "test", valid_series())
        await replace_daily_prices_atomically(self.conn, history)
        await self.conn.commit()
        async with self.conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices WHERE ticker='SOXX'"
        ) as cursor:
            count, first_date, last_date = await cursor.fetchone()
        self.assertEqual(count, 220)
        self.assertNotEqual(first_date, "2000-01-01")
        self.assertIsNotNone(last_date)

    async def test_insert_failure_restores_previous_history(self):
        await self.conn.execute(
            "INSERT INTO daily_prices VALUES ('SOXX', '2000-01-01', 10.0)"
        )
        fail_date = valid_series().index[100].strftime("%Y-%m-%d")
        await self.conn.execute(
            f"""
            CREATE TRIGGER fail_price_insert BEFORE INSERT ON daily_prices
            WHEN NEW.date = '{fail_date}'
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END
            """
        )
        await self.conn.commit()

        with self.assertRaises(aiosqlite.IntegrityError):
            await replace_daily_prices_atomically(
                self.conn, PriceHistory("SOXX", "test", valid_series())
            )
        async with self.conn.execute(
            "SELECT date, close FROM daily_prices WHERE ticker='SOXX'"
        ) as cursor:
            rows = await cursor.fetchall()
        self.assertEqual(rows, [("2000-01-01", 10.0)])

    async def test_failed_refresh_keeps_old_rows_and_writes_audit(self):
        await self.conn.execute(
            "INSERT INTO daily_prices VALUES ('SOXS', '2000-01-01', 10.0)"
        )
        await self.conn.commit()
        with patch(
            "market_data_history.fetch_validated_price_history",
            new=AsyncMock(side_effect=PriceHistoryError("all providers failed")),
        ):
            results = await sync_daily_price_histories(object(), self.conn, ["SOXS"])

        self.assertEqual(results[0].status, "failed")
        async with self.conn.execute(
            "SELECT date, close FROM daily_prices WHERE ticker='SOXS'"
        ) as cursor:
            self.assertEqual(await cursor.fetchall(), [("2000-01-01", 10.0)])
        async with self.conn.execute(
            "SELECT status, error FROM price_sync_runs WHERE ticker='SOXS'"
        ) as cursor:
            status, error = await cursor.fetchone()
        self.assertEqual(status, "failed")
        self.assertIn("all providers failed", error)

    async def test_successful_sync_records_validation_audit(self):
        history = PriceHistory(
            "SOXX", "test", valid_series(), "daily_price_validation;unit_test"
        )
        await replace_daily_prices_atomically(self.conn, history)
        await self.conn.commit()
        async with self.conn.execute(
            "SELECT validation FROM price_sync_runs WHERE ticker='SOXX'"
        ) as cursor:
            self.assertEqual(
                (await cursor.fetchone())[0],
                "daily_price_validation;unit_test",
            )

    async def test_validated_frame_excludes_tracking_broken_soxs(self):
        soxx = valid_series()
        soxs = valid_series(ticker="SOXS")
        soxx.iloc[-1] = soxx.iloc[-2] * 1.06
        soxs.iloc[-1] = soxs.iloc[-2] * 0.05
        for ticker, series in (("SOXX", soxx), ("SOXS", soxs)):
            await self.conn.executemany(
                "INSERT INTO daily_prices VALUES (?, ?, ?)",
                [
                    (ticker, index.strftime("%Y-%m-%d"), float(value))
                    for index, value in series.items()
                ],
            )
        await self.conn.commit()
        frame = await load_validated_daily_frame(
            self.conn, ("SOXX", "SOXS"), days=400
        )
        self.assertIn("SOXX", frame.columns)
        self.assertNotIn("SOXS", frame.columns)


if __name__ == "__main__":
    unittest.main()
