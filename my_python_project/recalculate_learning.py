import asyncio
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import config
from db import get_db_connection, init_db
from alpha_utils import calculate_signed_alpha

async def recalculate_all_stats():
    print("🚀 Запуск полного пересчета статистики GTS...")
    
    # Гарантируем, что структура БД актуальна (добавляем signed_alpha, если ее нет)
    await init_db()
    
    async with get_db_connection() as conn:
        # 1. Загружаем все разрешенные прогнозы
        query = """
            SELECT p.id, p.event_key, p.score, p.target_asset, p.timestamp, 
                   p.event_type, p.predicted_impact, p.source_domain, p.confidence,
                   p.confidence, p.model_name, e.slug 
            FROM predictions p
            LEFT JOIN events e ON p.event_id = e.id
            WHERE p.resolved >= 1
        """
        async with conn.execute(query) as cursor:
            rows = await cursor.fetchall()
            df = pd.DataFrame([dict(r) for r in rows])
        
        if df.empty:
            print("⚠️ Нет данных для пересчета.")
            return

        # 2. Загружаем историю цен для всех активов
        assets = df['target_asset'].unique()
        assets_lower = [a.lower() for a in assets]
        
        # Собираем список реальных тикеров для загрузки
        download_tickers = [config.ASSET_TICKER_MAP[a] for a in assets_lower 
                           if a in config.ASSET_TICKER_MAP and config.ASSET_TICKER_MAP[a] != 'GLOBAL_REGIME']
        
        # Если есть глобальный режим, добавляем его компоненты и бенчмарк для Z-Alpha
        if 'global' in assets_lower:
            regime_comps = ['^VIX', '^MOVE', 'DX-Y.NYB', 'HYG', '^TNX', '^IRX']
            download_tickers.extend(regime_comps)
            global_bench = config.ASSET_BENCHMARK_CONFIG.get("global", {}).get("primary")
            if global_bench and global_bench != "GLOBAL_REGIME":
                download_tickers.append(global_bench)
        
        # Multi-factor benchmarks (gold: TIP + DXY)
        for asset_cfg in config.ASSET_BENCHMARK_CONFIG.values():
            if asset_cfg.get("secondary"):
                download_tickers.append(asset_cfg["secondary"])
        
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

        # Сохраняем текущие множители активов перед очисткой
        async with conn.execute("SELECT target_asset, multiplier FROM asset_stats") as cursor:
            rows = await cursor.fetchall()
            multipliers = {r['target_asset']: r['multiplier'] for r in rows}
        
        # Загружаем текущие веса для точного пересчета прогноза
        async with conn.execute("SELECT event_key, target_asset, weight FROM weights") as cursor:
            rows = await cursor.fetchall()
            weights_map = {(r['event_key'], r['target_asset']): r['weight'] for r in rows}
        
        updates = []
        correct_count = 0
        print("🧠 Пересчет направлений (is_correct)...")
        for _, row in df.iterrows():
            asset = row['target_asset']
            asset_low = asset.lower()
            ticker = config.ASSET_TICKER_MAP.get(asset.lower())
            if not ticker or ticker not in history.columns:
                continue
                
            # Окно lookback зависит от типа события (берем Primary из конфига)
            lookback_h = config.EVENT_TYPE_LOOKBACK.get(row['event_type'], {"primary": 1})["primary"]
            
            p_time = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            t_end = p_time + timedelta(hours=lookback_h)
            
            try:
                signed_alpha = calculate_signed_alpha(asset, ticker, p_time, t_end, history)
                if signed_alpha is not None:

                    actual_z = min(abs(signed_alpha), 10.0)
                    
                    # Используем централизованную мапу корреляций
                    correlation = config.ASSET_CORRELATION_MAP.get(asset.lower(), -1)
                    
                    # ПРАВИЛО ОПРЕДЕЛЕНИЯ is_correct
                    if correlation == 0:
                        # Для нейтральной корреляции (Gold, VIX) направление верно, если знаки score и alpha совпадают
                        is_correct = 1 if (row['score'] * signed_alpha) > 0 else 0
                    else:
                        # Для стандартной корреляции (Risk-On/Off)
                        is_correct = 1 if (row['score'] * signed_alpha * correlation) > 0 else 0

                    if is_correct: correct_count += 1
                    
                    # Сохраняем только is_correct, actual_move (Z-score) и signed_alpha. НЕ ТРОГАЕМ predicted_impact.
                    updates.append((is_correct, actual_z, signed_alpha, row['id']))

            except Exception:
                continue

        # 4. Массовое обновление флагов
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS temp_recalc_results (
                id INTEGER PRIMARY KEY,
                is_correct INTEGER,
                actual_move REAL,
                signed_alpha REAL
            )
        """)
        await conn.execute("DELETE FROM temp_recalc_results")
        await conn.executemany("""
            INSERT INTO temp_recalc_results (is_correct, actual_move, signed_alpha, id) 
            VALUES (?, ?, ?, ?)
        """, updates)

        await conn.execute("""
            UPDATE predictions
            SET is_correct = (SELECT is_correct FROM temp_recalc_results WHERE temp_recalc_results.id = predictions.id),
                actual_move = (SELECT actual_move FROM temp_recalc_results WHERE temp_recalc_results.id = predictions.id),
                signed_alpha = (SELECT signed_alpha FROM temp_recalc_results WHERE temp_recalc_results.id = predictions.id)
            WHERE id IN (SELECT id FROM temp_recalc_results)
        """)
        await conn.execute("DROP TABLE temp_recalc_results")

        print(f"✅ Обновлено прогнозов: {len(updates)} (Верных: {correct_count})")

        # Очищаем статистику перед пересозданием
        await conn.execute("DELETE FROM source_stats")
        await conn.execute("DELETE FROM asset_stats")
        await conn.execute("DELETE FROM model_stats") # Также очищаем статистику моделей
        await conn.commit()

        # --- ПЕРЕСБОР АГРЕГИРОВАННОЙ СТАТИСТИКИ ---
        print("📊 Пересбор агрегированной статистики...")
        # Загружаем обновленные данные
        async with conn.execute("""
            SELECT target_asset, is_correct, predicted_impact, actual_move, source_domain, model_name, confidence, signed_alpha
            FROM predictions WHERE resolved >= 1 AND is_correct >= 0
        """) as cursor:
            stats_df = pd.DataFrame([dict(r) for r in await cursor.fetchall()])

        if not stats_df.empty:
            stats_df['error'] = abs(stats_df['actual_move'] - stats_df['predicted_impact'])

            # Статистика по активам
            asset_stats = stats_df.groupby('target_asset').agg(
                total_resolved=('is_correct', 'count'),
                correct_count=('is_correct', 'sum'),
                sum_error=('error', 'sum')
            ).reset_index()
            asset_records = [tuple(x) for x in asset_stats.to_numpy()]
            await conn.executemany(
                "INSERT INTO asset_stats (target_asset, total_resolved, correct_count, sum_error) VALUES (?, ?, ?, ?)",
                asset_records
            )

            # Статистика по источникам
            source_stats = stats_df.groupby('source_domain').agg(
                total_resolved=('is_correct', 'count'),
                correct_count=('is_correct', 'sum'),
                sum_error=('error', 'sum'),
                sum_confidence=('confidence', 'sum'),
                sum_alpha=('signed_alpha', 'sum'),
                sum_alpha_sq=('signed_alpha', lambda x: (x**2).sum())
            ).reset_index()
            source_records = [tuple(x) for x in source_stats.to_numpy()]
            await conn.executemany(
                "INSERT INTO source_stats (source_domain, total_resolved, correct_count, sum_error, sum_confidence, sum_alpha, sum_alpha_sq) VALUES (?, ?, ?, ?, ?, ?, ?)",
                source_records
            )
            await conn.commit()

    new_winrate = (correct_count / len(updates) * 100) if updates else 0
    print(f"✨ Готово! Общий WinRate системы после пересчета: {new_winrate:.2f}%")

if __name__ == "__main__":
    try:
        asyncio.run(recalculate_all_stats())
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
