import asyncio
import pandas as pd
import config
from db import get_db_connection, init_db
import yfinance as yf
from datetime import datetime, timedelta

def calculate_hbm_index_value():
    """
    Calculates the HBM Index value based on defined components and weights.
    Returns the current index value and its daily percentage change.
    """
    all_tickers = []
    for segment, tickers in config.HBM_INDEX_COMPONENTS.items():
        all_tickers.extend(tickers)

    if not all_tickers:
        return None, None

    # Fetch data for the last few days to ensure we have at least 2 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30) # Fetch 7 days for fast daily change calculation

    try:
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            print("⚠️ HBM Index: No data fetched from yfinance.")
            return None, None
    except Exception as e:
        print(f"⚠️ HBM Index: Error fetching data from yfinance: {e}")
        return None, None

    # Extract 'Close' prices. yfinance returns MultiIndex for multiple tickers.
    close_prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]

    # Calculate daily returns for each stock
    daily_returns = close_prices.pct_change().dropna()

    if daily_returns.empty:
        print("⚠️ HBM Index: Not enough data to calculate daily returns.")
        return None, None

    # Calculate weighted daily return for the HBM Index for each day
    index_returns_series = pd.Series(0.0, index=daily_returns.index)

    for segment, tickers in config.HBM_INDEX_COMPONENTS.items():
        segment_weight = config.HBM_INDEX_SEGMENT_WEIGHTS.get(segment, 0.0)
        if segment_weight == 0:
            continue
        num_stocks_in_segment = len(tickers)
        if num_stocks_in_segment == 0:
            continue
        stock_weight_in_segment = 1.0 / num_stocks_in_segment

        for ticker in tickers:
            if ticker in daily_returns.columns:
                index_returns_series += segment_weight * stock_weight_in_segment * daily_returns[ticker].fillna(0) # Fillna to handle missing daily returns for some stocks

    # Calculate the cumulative index value, starting from 100
    base_index_value = 100.0
    cumulative_index = (1 + index_returns_series).cumprod() * base_index_value

    if cumulative_index.empty:
        return base_index_value, 0.0 # Return base and 0 change if no cumulative data

    current_hbm_index = cumulative_index.iloc[-1]
    
    # Daily change is the last calculated weighted return
    daily_change_percent = index_returns_series.iloc[-1] * 100

    return current_hbm_index, daily_change_percent

async def inspect_gts():
    # Настраиваем Pandas, чтобы он не скрывал колонки и показывал текст полностью
    pd.set_option('display.max_columns', None)  # Показывать все колонки
    pd.set_option('display.expand_frame_repr', False)  # Не переносить таблицу на новую строку
    pd.set_option('display.max_colwidth', 50)  # Показывать текст в колонках полностью без обрезки

    # Инициализируем БД, чтобы автоматически добавить недостающие колонки (is_correct)
    await init_db()

    async with get_db_connection() as conn:
        print("--- ТЕКУЩИЕ ВЕСА (ПОСЛЕ ОБУЧЕНИЯ) ---")
        async with conn.execute("SELECT event_key, target_asset, weight FROM weights ORDER BY weight DESC") as cursor:
            weights = pd.DataFrame([dict(r) for r in await cursor.fetchall()])
        print(weights if not weights.empty else "Таблица весов пуста (используются дефолтные)")
        
        print("\n--- ГЛОБАЛЬНЫЕ ПРЕДЛОЖЕНИЯ ИИ (AI GLOBAL SUGGESTIONS) ---")
        async with conn.execute("SELECT keyword, asset, impact_direction, reasoning, timestamp FROM ai_global_suggestions ORDER BY timestamp DESC LIMIT 10") as cursor:
            suggestions = pd.DataFrame([dict(r) for r in await cursor.fetchall()])
            
        if not suggestions.empty:
            print(suggestions)
        else:
            print("Предложений пока нет. Дождитесь завершения цикла RESEARCH_INTERVAL.")

        print("\n--- ПОСЛЕДНИЕ 5 СОБЫТИЙ ---")
        async with conn.execute("SELECT title, score, nasdaq, sp500, oil, vix, soxs, gold, btc, timestamp FROM events ORDER BY timestamp DESC LIMIT 5") as cursor:
            events = pd.DataFrame([dict(r) for r in await cursor.fetchall()])
        print(events)
        
        print("\n--- АНАЛИЗ ОТКЛОНЕНИЙ (PREDICTED VS ACTUAL) ---")
        # Показываем последние 10 значимых прогнозов. 
        # Фильтруем шум (score < threshold), чтобы не видеть дефолтные 0 в actual_move.
        accuracy_query = f"""
            SELECT event_key, target_asset, score, predicted_impact, actual_move, 
                   (actual_move - predicted_impact) as error, is_correct, timestamp 
            FROM predictions 
            WHERE resolved = 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD} AND LOWER(target_asset) != 'hbm'
            ORDER BY timestamp DESC LIMIT 10
        """
        async with conn.execute(accuracy_query) as cursor:
            accuracy_df = pd.DataFrame([dict(r) for r in await cursor.fetchall()])

        if not accuracy_df.empty:
            print(accuracy_df)

        print("\n--- СТАТИСТИКА ПО АКТИВАМ И ТРЕНДЫ ---")
        # Загружаем все значимые разрешенные прогнозы для анализа *недавних* трендов
        # Исключаем шум из статистики, чтобы он не занижал WinRate и не искажал среднюю ошибку
        async with conn.execute(f"""
            SELECT target_asset, is_correct, actual_move, predicted_impact, timestamp, resolved
            FROM predictions WHERE resolved >= 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD} AND LOWER(target_asset) != 'hbm' AND is_correct >= 0
        """) as cursor:
            df_all = pd.DataFrame([dict(r) for r in await cursor.fetchall()])

        # Загружаем накопленную статистику по активам из новой таблицы asset_stats
        async with conn.execute("""
            SELECT target_asset, 
                   COALESCE(total_resolved, 0) as total_resolved, 
                   COALESCE(correct_count, 0) as correct_count, 
                   COALESCE(sum_error, 0.0) as sum_error,
                   COALESCE(multiplier, 0.0) as multiplier
            FROM asset_stats
            WHERE LOWER(target_asset) != 'hbm'
        """) as cursor:
            asset_stats_df = pd.DataFrame([dict(r) for r in await cursor.fetchall()])
        
        if not df_all.empty:
            df_all['error'] = abs(df_all['actual_move'] - df_all['predicted_impact'])
            
            stats_data = []
            # Итерируем по всем активам, которые есть либо в текущих прогнозах, либо в накопленной статистике
            all_assets_series = [df_all['target_asset']]
            if not asset_stats_df.empty:
                all_assets_series.append(asset_stats_df['target_asset'])
            
            all_unique_assets = pd.concat(all_assets_series).unique()
 
            for asset in all_unique_assets:
                if asset and asset.lower() == 'hbm':
                    continue

                # Общие показатели (накопленные)
                if not asset_stats_df.empty:
                    asset_total_stats = asset_stats_df[asset_stats_df['target_asset'] == asset]
                else:
                    asset_total_stats = pd.DataFrame() # Создаем пустой DataFrame, чтобы последующая логика работала
                if not asset_total_stats.empty:
                    total_cnt = int(asset_total_stats['total_resolved'].iloc[0])
                    total_wr = (float(asset_total_stats['correct_count'].iloc[0]) / total_cnt) * 100 if total_cnt > 0 else 0
                    total_err = float(asset_total_stats['sum_error'].iloc[0]) / total_cnt if total_cnt > 0 else 0
                else: # Если актива нет в asset_stats (например, новый актив)
                    total_cnt = 0
                    total_wr = 0
                    total_err = 0

                # Последние показатели (из df_all, который содержит только недавние прогнозы)
                asset_df_from_predictions = df_all[df_all['target_asset'] == asset].sort_values('timestamp')
                recent_window = max(5, int(len(asset_df_from_predictions) * 0.3)) # Window based on available recent data
                recent_df = asset_df_from_predictions.tail(recent_window)
                
                wr_trend_str = "---"
                err_trend_str = "---"
                wr_delta = 0
                err_delta = 0

                if not recent_df.empty and len(recent_df) >= 3:
                    recent_wr = recent_df['is_correct'].mean() * 100
                    recent_err = recent_df['error'].mean()
                    wr_delta = recent_wr - total_wr
                    err_delta = recent_err - total_err
                    wr_trend_str = f"{wr_delta:+.1f}%"
                    err_trend_str = f"{err_delta:+.2f}"
                
                # Формирование комментария
                comment = ""
                if total_cnt > 5 and total_wr == 0: 
                    comment = "❌ ИНВЕРСИЯ (Всё мимо)"
                elif total_wr > 70: 
                    comment = "💎 Отлично"
                elif total_wr < 45: comment = "⚠️ Слабо"
                else: comment = "🆗 Стабильно"
                
                if wr_delta > 5: comment += " | 📈 Улучшение точности"
                elif wr_delta < -5: comment += " | 📉 Точность падает"
                
                if err_delta < -3: comment += " | 🎯 Калибровка лучше"
                elif err_delta > 3: comment += " | 🌡 Разброс растет"

                stats_data.append({
                    "Asset": asset,
                    "Total": total_cnt,
                    "WinRate%": round(total_wr, 1),
                    "Multiplier": round(asset_total_stats['multiplier'].iloc[0], 2) if not asset_total_stats.empty else "---",
                    "WR_Trend": wr_trend_str,
                    "AvgError": f"{total_err:.2f} ({'✅' if total_err < 1.0 else '⚠️' if total_err < 3.0 else '🔥'})",
                    "Err_Trend": err_trend_str,
                    "Status/Comment": comment
                })
            
            stats_df = pd.DataFrame(stats_data)
            print(stats_df.sort_values("WinRate%", ascending=False).to_string(index=False))

        print("\n--- СТАТИСТИКА ПРОГНОЗОВ ---")
        async with conn.execute("SELECT COUNT(*) as total FROM predictions") as cursor:
            total = (await cursor.fetchone())['total']
        async with conn.execute("SELECT COUNT(*) as count FROM predictions WHERE resolved = 0") as cursor:
            pending = (await cursor.fetchone())['count']
        
        # Считаем общее количество обработанных записей (Фазы 1 и 2)
        async with conn.execute("SELECT COUNT(*) as count FROM predictions WHERE resolved >= 1") as cursor:
            all_resolved = (await cursor.fetchone())['count']

        # Загружаем все значимые resolved прогнозы для детального анализа трендов
        query = f"SELECT is_correct, actual_move FROM predictions WHERE resolved >= 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD} AND LOWER(target_asset) != 'hbm' AND is_correct >= 0 ORDER BY timestamp ASC"
        async with conn.execute(query) as cursor:
            df_sig = pd.DataFrame([dict(r) for r in await cursor.fetchall()])

        async with conn.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'") as cursor:
            curr_mult = await cursor.fetchone()
            multiplier_val = curr_mult[0] if curr_mult else config.IMPACT_MULTIPLIER

        print(f"Всего прогнозов в базе: {total}")
        print(f"Ожидают разрешения (pending): {pending}")
        print(f"Всего обработано (resolved): {all_resolved}")

        if not df_sig.empty:
            # Используем накопленную статистику для общего Win Rate
            async with conn.execute("""
                SELECT COALESCE(SUM(total_resolved), 0) as total_resolved_overall, 
                       COALESCE(SUM(correct_count), 0) as correct_count_overall
                FROM asset_stats
                WHERE LOWER(target_asset) != 'hbm'
            """) as cursor:
                overall_stats = await cursor.fetchone()

            trained_count_overall = overall_stats['total_resolved_overall']
            correct_count_overall = overall_stats['correct_count_overall']
            win_rate_total = (correct_count_overall / trained_count_overall * 100) if trained_count_overall > 0 else 0
            
            # Считаем "недавние" показатели (последние 20 значимых прогнозов) для выявления тренда
            recent_df = df_sig.tail(20)
            recent_count = len(recent_df)
            win_rate_recent = (recent_df['is_correct'].sum() / recent_count * 100) if recent_count > 0 else 0
            avg_move_recent = recent_df['actual_move'].mean()
            
            avg_move_total = df_sig['actual_move'].mean()
            wr_delta = win_rate_recent - win_rate_total
            am_delta = avg_move_recent - avg_move_total
            mult_delta = multiplier_val - config.IMPACT_MULTIPLIER

            print(f"Прошли обучение (значимые): {trained_count_overall} (накоплено)")
            print(f"Верных прогнозов (✅): {correct_count_overall} (накоплено)")
            print(f"Точность (Win Rate): {win_rate_total:.1f}% | Recent (20): {win_rate_recent:.1f}% ({wr_delta:+.1f}%)")
            print(f"Текущий множитель влияния (Multiplier): {multiplier_val:.4f} ({mult_delta:+.4f} к базе)")
            print(f"Среднее реальное движение: {avg_move_total:.2f} ({am_delta:+.2f} тренд)")
        else:
            print("Недостаточно данных для расчета статистики обучения.")

        # HBM Index
        hbm_index_val, hbm_daily_change = calculate_hbm_index_value()
        if hbm_index_val is not None:
            print(f"\n--- HBM Index ---")
            print(f"Current HBM Index Value: {hbm_index_val:.2f}")
            print(f"Daily Change: {hbm_daily_change:+.2f}%")
        else:
            print("\n--- HBM Index: Could not calculate ---")
            
        print("\n--- АНАЛИЗ ИСТОЧНИКОВ: WINRATE И УВЕРЕННОСТЬ ---")
        source_stats_query = """
            SELECT 
                COALESCE(NULLIF(source_domain, ''), '[unknown]') as Source, 
                total_resolved as Total, 
                ROUND((CAST(correct_count AS REAL) / total_resolved) * 100, 1) as "WinRate%", 
                ROUND(sum_alpha / total_resolved, 3) as "AvgAlpha",
                ROUND(
                    (sum_alpha / total_resolved) / 
                    NULLIF(SQRT(ABS(sum_alpha_sq / total_resolved - (sum_alpha / total_resolved) * (sum_alpha / total_resolved))), 0), 
                2) as "InfoRatio",
                ROUND(sum_error / total_resolved, 2) as AvgErr
            FROM source_stats
            WHERE total_resolved > 0
            ORDER BY "WinRate%" DESC
        """
        async with conn.execute(source_stats_query) as cursor:
            source_df = pd.DataFrame([dict(r) for r in await cursor.fetchall()])
        if not source_df.empty:
            print(source_df.to_string(index=False))
        else:
            print("Недостаточно данных для анализа источников.")

        print("\n--- ЭФФЕКТИВНОСТЬ МОДЕЛЕЙ (AI PERFORMANCE) ---")
        # Загружаем детальные данные для расчета трендов
        async with conn.execute("""
            SELECT p.model_name, p.is_correct, p.confidence, p.timestamp, ms.sensitivity
            FROM predictions p
            LEFT JOIN model_stats ms ON p.model_name = ms.model_name
            WHERE p.resolved >= 1
        """) as cursor:
            p_df = pd.DataFrame([dict(r) for r in await cursor.fetchall()])

        if not p_df.empty:
            model_perf_data = []
            for m_name, group in p_df.groupby("model_name"):
                group = group.sort_values('timestamp')
                total_cnt = len(group)
                correct_cnt = int(group['is_correct'].sum())
                total_wr = (correct_cnt / total_cnt) * 100
                
                # Расчет тренда (последние 15 прогнозов или 25% от выборки)
                recent_window = max(5, int(total_cnt * 0.25))
                recent_group = group.tail(recent_window)
                wr_trend = "---"
                
                if len(recent_group) >= 3:
                    recent_wr = (recent_group['is_correct'].sum() / len(recent_group)) * 100
                    wr_trend = f"{(recent_wr - total_wr):+.1f}%"

                sens = group['sensitivity'].iloc[0] if not pd.isna(group['sensitivity'].iloc[0]) else 1.0
                
                model_perf_data.append({
                    "Model": m_name,
                    "MSF(Sens)": round(sens, 3),
                    "Total": total_cnt,
                    "Correct": correct_cnt,
                    "AvgConf": round(group['confidence'].mean(), 2),
                    "WinRate%": round(total_wr, 1),
                    "WR_Trend": wr_trend
                })
            
            model_df = pd.DataFrame(model_perf_data).sort_values("WinRate%", ascending=False)
            print(model_df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(inspect_gts())
