import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import config
from db import get_db_connection, init_db
import numpy as np

def _drop_tz(series):
    if series.index.tz is not None:
        series = series.copy()
        series.index = series.index.tz_convert(None)
    return series

def get_ewma_beta(target_rets, bench_rets):
    if len(target_rets) < 10:
        return 1.0
    alpha = 1 - config.EWMA_LAMBDA
    cov = target_rets.ewm(alpha=alpha).cov(bench_rets).iloc[-1]
    var = bench_rets.ewm(alpha=alpha).var().iloc[-1]
    beta = cov / var if var > 0 else 1.0
    return max(-config.BETA_CLIP, min(config.BETA_CLIP, beta))

def calculate_expected_move(target_key, bench_cfg, prediction_time, b_move_raw, price_history):
    try:
        target_ticker = config.ASSET_TICKER_MAP.get(target_key)
        hist_end = prediction_time
        hist_start = hist_end - timedelta(days=2)

        if bench_cfg["type"] == "leveraged":
            return b_move_raw * bench_cfg["factor"]

        if bench_cfg["type"] == "multi_factor":
            return b_move_raw * bench_cfg["weights"][0]

        bench_ticker = bench_cfg["primary"]
        if target_ticker in price_history and bench_ticker in price_history:
            t_rets = _drop_tz(price_history[target_ticker].dropna()).loc[hist_start:hist_end].pct_change().dropna()
            b_rets = _drop_tz(price_history[bench_ticker].dropna()).loc[hist_start:hist_end].pct_change().dropna()
            beta = get_ewma_beta(t_rets, b_rets)
            return b_move_raw * beta

        return b_move_raw * bench_cfg.get("factor", 1.0)
    except Exception:
        return 0.0

def calculate_signed_alpha(asset, ticker, p_time, t_end, history):
    ts = _drop_tz(history[ticker].dropna())
    idx_at = ts.index.get_indexer([p_time], method='backfill')[0]
    idx_after = ts.index.get_indexer([t_end], method='backfill')[0]

    if idx_at == -1 or idx_after == -1 or idx_at == idx_after:
        return None

    p_at = float(ts.iloc[idx_at])
    p_after = float(ts.iloc[idx_after])
    if p_at == 0:
        return None

    raw_change = float(((p_after - p_at) / p_at) * 100)
    b_cfg = config.ASSET_BENCHMARK_CONFIG.get(asset.lower())

    # Расчет волатильности (используем окно ДО новости или ближайшее доступное)
    asset_rets = _drop_tz(history[ticker].dropna()).loc[:t_end].pct_change().tail(config.VOLATILITY_WINDOW)
    realized_vol_raw = asset_rets.std() * 100
    if pd.isna(realized_vol_raw) or realized_vol_raw == 0:
        realized_vol_raw = 1.0  # Дефолт 1%, если данных мало

    vol_floor = config.GLOBAL_Z_ALPHA_VOL_FLOOR if asset.lower() == "global" else config.Z_ALPHA_VOL_FLOOR
    realized_vol = max(realized_vol_raw, vol_floor)

    alpha_val = raw_change
    if b_cfg:
        bench_key = b_cfg["primary"]
        # Если бенчмарк совпадает с тикером (случай global), не вычитаем его
        if bench_key in history and bench_key != ticker:
            b_ts = _drop_tz(history[bench_key].dropna())
            idx_b_at = b_ts.index.get_indexer([p_time], method='backfill')[0]
            idx_b_after = b_ts.index.get_indexer([t_end], method='backfill')[0]

            if idx_b_at != -1 and idx_b_after != -1:
                b_at = float(b_ts.iloc[idx_b_at])
                b_after = float(b_ts.iloc[idx_b_after])
                if b_at != 0:
                    b_move = ((b_after - b_at) / b_at) * 100
                    expected = calculate_expected_move(asset.lower(), b_cfg, p_time, b_move, history)
                    alpha_val = (raw_change - expected)

    return alpha_val / realized_vol

def recalculate_all_stats():
    print("🚀 Запуск полного пересчета статистики GTS...")
    
    # Гарантируем, что структура БД актуальна (добавляем signed_alpha, если ее нет)
    init_db()
    
    with get_db_connection() as conn:
        # 1. Загружаем все разрешенные прогнозы
        query = "SELECT id, event_key, score, target_asset, timestamp, event_type, predicted_impact FROM predictions WHERE resolved >= 1"
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("⚠️ Нет данных для пересчета.")
            return

        # 2. Загружаем историю цен для всех активов
        assets = df['target_asset'].unique()
        assets_lower = [a.lower() for a in assets]
        
        # Собираем список реальных тикеров для загрузки
        download_tickers = [config.ASSET_TICKER_MAP[a] for a in assets_lower 
                           if a in config.ASSET_TICKER_MAP and config.ASSET_TICKER_MAP[a] != 'GLOBAL_REGIME']
        
        # Если есть глобальный режим, добавляем его компоненты
        if 'global' in assets_lower:
            regime_comps = ['^VIX', '^MOVE', 'DX-Y.NYB', 'HYG', '^TNX', '^IRX']
            download_tickers.extend(regime_comps)
        
        download_tickers = list(set(download_tickers))
        print(f"📈 Загрузка истории для тикеров: {download_tickers}")
        
        history = yf.download(download_tickers, period="60d", interval="15m", progress=False)['Close']

        # 2.1 Синтезируем GLOBAL_REGIME, если он нужен
        if 'global' in assets_lower:
            try:
                rets = history.pct_change().fillna(0)
                curve = history['^TNX'] - history['^IRX']
                curve_rets = curve.pct_change().fillna(0)
                
                w = config.GLOBAL_REGIME_WEIGHTS
                stress_rets = (rets['^VIX'] * w['vix'] + rets['^MOVE'] * w['move'] + 
                              rets['DX-Y.NYB'] * w['dxy'] + (rets['HYG'] * -1.0) * w['hyg'] + 
                              (curve_rets * -1.0) * w['growth'])
                
                history['GLOBAL_REGIME'] = (1 + stress_rets).cumprod() * 100
                print("🧬 Индекс GLOBAL_REGIME успешно воссоздан.")
            except Exception as e:
                print(f"⚠️ Не удалось воссоздать GLOBAL_REGIME: {e}")
        
        if history.empty:
            print("❌ Ошибка: Не удалось загрузить историю цен. Пересчет отменен.")
            return

        # 3. Подготовка
        cursor = conn.cursor()
        
        # Сохраняем текущие множители активов перед очисткой
        cursor.execute("SELECT target_asset, multiplier FROM asset_stats")
        multipliers = {r['target_asset']: r['multiplier'] for r in cursor.fetchall()}
        
        updates = []
        correct_count = 0
        print(f"Найдено {len(df)} прогнозов для анализа.")
        
        print("🧠 Пересчет направлений (is_correct)...")
        for _, row in df.iterrows():
            asset = row['target_asset']
            ticker = config.ASSET_TICKER_MAP.get(asset.lower())
            if not ticker or ticker not in history.columns:
                continue
                
            # Окно lookback зависит от типа события (берем Primary из конфига)
            lookback_h = config.EVENT_TYPE_LOOKBACK.get(row['event_type'], {"primary": 1})["primary"]
            
            p_time = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            t_end = p_time + timedelta(hours=lookback_h)
            
            # Ищем ближайшие цены в истории
            try:
                ts = history[ticker].dropna()
                # Убираем таймзоны для сравнения
                ts.index = ts.index.tz_localize(None)
                
                idx_at = ts.index.get_indexer([p_time], method='backfill')[0]
                idx_after = ts.index.get_indexer([t_end], method='backfill')[0]
                
                if idx_at != -1 and idx_after != -1 and idx_at != idx_after:
                    p_at = float(ts.iloc[idx_at])
                    p_after = float(ts.iloc[idx_after])
                    
                    signed_alpha = calculate_signed_alpha(asset, ticker, p_time, t_end, history)
                    if signed_alpha is None:
                        continue

                    actual_z = min(abs(signed_alpha), 10.0)
                    
                    # Корреляция
                    correlation = 1 if asset.lower() in ["oil", "vix", "soxs", "gold", "global"] else -1
                    
                    # НОВОЕ ПРАВИЛО: Направление верно, если знак совпадает
                    is_correct = 1 if (row['score'] * signed_alpha * correlation) > 0 else 0
                    if is_correct: correct_count += 1
                    
                    updates.append((is_correct, actual_z, signed_alpha, row['id']))
            except Exception as e:
                continue

        # 4. Массовое обновление флагов
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS temp_recalc_results")
        cursor.execute("""
            CREATE TEMP TABLE temp_recalc_results (
                id INTEGER PRIMARY KEY,
                is_correct INTEGER,
                actual_move REAL,
                signed_alpha REAL
            )
        """)
        cursor.executemany(
            "INSERT INTO temp_recalc_results (is_correct, actual_move, signed_alpha, id) VALUES (?, ?, ?, ?)",
            updates
        )
        cursor.execute("""
            UPDATE predictions
            SET is_correct = (SELECT is_correct FROM temp_recalc_results WHERE temp_recalc_results.id = predictions.id),
                actual_move = (SELECT actual_move FROM temp_recalc_results WHERE temp_recalc_results.id = predictions.id),
                signed_alpha = (SELECT signed_alpha FROM temp_recalc_results WHERE temp_recalc_results.id = predictions.id)
            WHERE id IN (SELECT id FROM temp_recalc_results)
        """)

        print(f"✅ Обновлено прогнозов: {len(updates)} (Верных: {correct_count})")

        # Очищаем статистику перед пересозданием
        cursor.execute("DELETE FROM source_stats")
        cursor.execute("DELETE FROM asset_stats")

        # 6. Пересчет статистики источников
        print("📊 Пересоздание Source Stats...")
        cursor.execute("""
            INSERT INTO source_stats (source_domain, total_resolved, correct_count, sum_error, sum_confidence, sum_alpha, sum_alpha_sq)
            SELECT 
                source_domain, 
                COUNT(*), 
                SUM(is_correct), 
                SUM(ABS(actual_move - predicted_impact)), 
                SUM(confidence),
                SUM(CASE WHEN score > 0 THEN signed_alpha ELSE -signed_alpha END),
                SUM(signed_alpha * signed_alpha)
            FROM predictions
            WHERE resolved >= 1 AND source_domain IS NOT NULL
            GROUP BY source_domain
        """)
        
        # 7. Пересчет статистики активов
        print("📉 Пересоздание Asset Stats...")
        cursor.execute("""
            INSERT INTO asset_stats (target_asset, total_resolved, correct_count, sum_error, multiplier)
            SELECT target_asset, COUNT(*), SUM(is_correct), SUM(ABS(actual_move - predicted_impact)), 0.3
            FROM predictions
            WHERE resolved >= 1
            GROUP BY target_asset
        """)
        
        # Восстанавливаем сохраненные множители для каждого актива
        for asset, mult in multipliers.items():
            cursor.execute("UPDATE asset_stats SET multiplier = ? WHERE target_asset = ?", (mult, asset))
        
        conn.commit()

    new_winrate = (correct_count / len(updates) * 100) if updates else 0
    print(f"✨ Готово! Новый WinRate системы: {new_winrate:.2f}%")
    print("ℹ️ Теперь можно запустить inspect_db.py для проверки.")

if __name__ == "__main__":
    try:
        recalculate_all_stats()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
