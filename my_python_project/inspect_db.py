import pandas as pd
import config
from db import get_db_connection, init_db

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
        events = pd.read_sql("SELECT title, score, event, timestamp FROM events ORDER BY timestamp DESC LIMIT 5", conn)
        print(events)
        
        print("\n--- АНАЛИЗ ОТКЛОНЕНИЙ (PREDICTED VS ACTUAL) ---")
        # Показываем последние 10 значимых прогнозов. 
        # Фильтруем шум (score < threshold), чтобы не видеть дефолтные 0 в actual_move.
        accuracy_query = f"""
            SELECT event_key, target_asset, score, predicted_impact, actual_move, 
                   (actual_move - predicted_impact) as error, is_correct, timestamp 
            FROM predictions 
            WHERE resolved = 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD}
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
            FROM predictions WHERE resolved = 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD}
        """
        df_all = pd.read_sql(all_res_query, conn)
        
        if not df_all.empty:
            df_all['error'] = abs(df_all['actual_move'] - df_all['predicted_impact'])
            
            stats_data = []
            for asset in df_all['target_asset'].unique():
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
        
        # Считаем статистику только по тем новостям, которые выше порога шума
        query = f"SELECT COUNT(*) as resolved, AVG(actual_move) as avg_move, SUM(is_correct) as correct FROM predictions WHERE resolved = 1 AND abs(score) >= {config.NEUTRAL_SCORE_THRESHOLD}"
        resolved_df = pd.read_sql(query, conn)
        
        resolved_count = resolved_df['resolved'].iloc[0]
        avg_move = resolved_df['avg_move'].iloc[0]
        avg_move_display = float(avg_move) if avg_move is not None else 0.0
        correct_count = resolved_df['correct'].iloc[0] if resolved_df['correct'].iloc[0] is not None else 0
        win_rate = (correct_count / resolved_count * 100) if resolved_count > 0 else 0

        print(f"Всего прогнозов в базе: {total}")
        print(f"Из них обработано (resolved): {resolved_count}")
        print(f"Верных прогнозов (✅): {correct_count}")
        print(f"Точность (Win Rate): {win_rate:.1f}%")
        print(f"Среднее реальное движение: {avg_move_display:.2f}")

if __name__ == "__main__":
    inspect_gts()
