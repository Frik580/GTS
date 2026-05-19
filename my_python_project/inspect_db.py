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
    start_date = end_date - timedelta(days=365) # Fetch 1 year of data for robust index calculation

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

def inspect_gts():
    # Настраиваем Pandas, чтобы он не скрывал колонки и показывал текст полностью
    pd.set_option('display.max_columns', None)  # Показывать все колонки
    pd.set_option('display.expand_frame_repr', False)  # Не переносить таблицу на новую строку
    pd.set_option('display.max_colwidth', 100)  # Увеличить ширину текста в колонках

    # Инициализируем БД, чтобы автоматически добавить недостающие колонки (is_correct)
    init_db()

    with get_db_connection() as conn:
        print("--- ТЕКУЩИЕ ВЕСА (ПОСЛЕ ОБУЧЕНИЯ) ---")
        weights = pd.read_sql("SELECT * FROM weights ORDER BY weight DESC", conn)
        print(weights if not weights.empty else "Таблица весов пуста (используются дефолтные)")
        
        print("\n--- ГЛОБАЛЬНЫЕ ПРЕДЛОЖЕНИЯ ИИ (AI GLOBAL SUGGESTIONS) ---")
        suggestions = pd.read_sql("SELECT keyword, asset, impact_direction, reasoning, timestamp FROM ai_global_suggestions ORDER BY timestamp DESC LIMIT 10", conn)
        if not suggestions.empty:
            print(suggestions)
        else:
            print("Предложений пока нет. Дождитесь завершения цикла RESEARCH_INTERVAL.")

        print("\n--- ПОСЛЕДНИЕ 5 СОБЫТИЙ ---")
        events = pd.read_sql("SELECT title, score, nasdaq, sp500, oil, vix, soxs, gold, btc, timestamp FROM events ORDER BY timestamp DESC LIMIT 5", conn)
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
        accuracy_df = pd.read_sql(accuracy_query, conn)
        if not accuracy_df.empty:
            print(accuracy_df)

        print("\n--- СТАТИСТИКА ПО АКТИВАМ И ТРЕНДЫ ---")
        # Загружаем все разрешенные прогнозы для анализа трендов
        # Исключаем шум из статистики, чтобы он не занижал WinRate и не искажал среднюю ошибку
        all_res_query = f"""
            SELECT target_asset, is_correct, actual_move, predicted_impact, timestamp 
            FROM predictions WHERE resolved = 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD} AND LOWER(target_asset) != 'hbm'
        """
        df_all = pd.read_sql(all_res_query, conn)
        
        if not df_all.empty:
            df_all['error'] = abs(df_all['actual_move'] - df_all['predicted_impact'])
            
            stats_data = []
            for asset in df_all['target_asset'].unique():
                if asset and asset.lower() == 'hbm':
                    continue
                asset_df = df_all[df_all['target_asset'] == asset].sort_values('timestamp')
                total_cnt = len(asset_df)
                
                # Общие показатели
                total_wr = asset_df['is_correct'].mean() * 100
                total_err = asset_df['error'].mean()
                
                # Последние показатели (последние 10 прогнозов или 30% данных)
                recent_window = max(5, int(total_cnt * 0.3))
                recent_df = asset_df.tail(recent_window)
                recent_wr = recent_df['is_correct'].mean() * 100
                recent_err = recent_df['error'].mean()
                
                # Расчет изменений
                wr_delta = recent_wr - total_wr
                err_delta = recent_err - total_err
                
                # Формирование комментария
                comment = ""
                if total_wr > 70: comment = "💎 Отлично"
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
                    "WR_Trend": f"{wr_delta:+.1f}%",
                    "AvgError": round(total_err, 2),
                    "Err_Trend": f"{err_delta:+.2f}",
                    "Status/Comment": comment
                })
            
            stats_df = pd.DataFrame(stats_data)
            print(stats_df.sort_values("WinRate%", ascending=False).to_string(index=False))

        print("\n--- СТАТИСТИКА ПРОГНОЗОВ ---")
        total = pd.read_sql("SELECT COUNT(*) as total FROM predictions", conn).iloc[0]['total']
        
        # Считаем общее количество обработанных записей (включая нейтральные)
        all_resolved = pd.read_sql("SELECT COUNT(*) as count FROM predictions WHERE resolved = 1", conn).iloc[0]['count']

        # Загружаем все значимые resolved прогнозы для детального анализа трендов
        query = f"SELECT is_correct, actual_move FROM predictions WHERE resolved = 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD} AND LOWER(target_asset) != 'hbm' ORDER BY timestamp ASC"
        df_sig = pd.read_sql(query, conn)

        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = 'impact_multiplier'")
        curr_mult = cursor.fetchone()
        multiplier_val = curr_mult[0] if curr_mult else config.IMPACT_MULTIPLIER

        print(f"Всего прогнозов в базе: {total}")
        print(f"Всего обработано (resolved): {all_resolved}")

        if not df_sig.empty:
            trained_count = len(df_sig)
            avg_move_total = df_sig['actual_move'].mean()
            correct_count = df_sig['is_correct'].sum()
            win_rate_total = (correct_count / trained_count * 100)
            
            # Считаем "недавние" показатели (последние 20 значимых прогнозов) для выявления тренда
            recent_df = df_sig.tail(20)
            recent_count = len(recent_df)
            win_rate_recent = (recent_df['is_correct'].sum() / recent_count * 100)
            avg_move_recent = recent_df['actual_move'].mean()
            
            wr_delta = win_rate_recent - win_rate_total
            am_delta = avg_move_recent - avg_move_total
            mult_delta = multiplier_val - config.IMPACT_MULTIPLIER

            print(f"Прошли обучение (значимые): {trained_count}")
            print(f"Верных прогнозов (✅): {correct_count}")
            print(f"Точность (Win Rate): {win_rate_total:.1f}% ({wr_delta:+.1f}% за последние 20)")
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
                source_domain as Source, 
                total_resolved as Total, 
                ROUND((CAST(correct_count AS REAL) / total_resolved) * 100, 1) as "WinRate%", 
                ROUND(sum_confidence / total_resolved, 2) as "AvgConf",
                ROUND(sum_error / total_resolved, 2) as AvgErr
            FROM source_stats
            WHERE total_resolved > 0
            ORDER BY "WinRate%" DESC, Total DESC
        """
        source_df = pd.read_sql(source_stats_query, conn)
        if not source_df.empty:
            print(source_df.to_string(index=False))
        else:
            print("Недостаточно данных для анализа источников.")

if __name__ == "__main__":
    inspect_gts()
