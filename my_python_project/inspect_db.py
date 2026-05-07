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
        # Показываем последние 10 разрешенных прогнозов и их ошибку
        accuracy_query = """
            SELECT event_key, target_asset, score, predicted_impact, actual_move, 
                   (actual_move - predicted_impact) as error, is_correct, timestamp 
            FROM predictions 
            WHERE resolved = 1 
            ORDER BY timestamp DESC LIMIT 10
        """
        accuracy_df = pd.read_sql(accuracy_query, conn)
        if not accuracy_df.empty:
            print(accuracy_df)

        print("\n--- СТАТИСТИКА ПО АКТИВАМ ---")
        asset_stats_query = """
            SELECT target_asset, 
                   COUNT(*) as total_cases, 
                   ROUND(AVG(is_correct) * 100, 1) as win_rate_pct, 
                   ROUND(AVG(abs(actual_move - predicted_impact)), 2) as avg_abs_error
            FROM predictions 
            WHERE resolved = 1
            GROUP BY target_asset
        """
        asset_stats = pd.read_sql(asset_stats_query, conn)
        if not asset_stats.empty:
            print(asset_stats)

        print("\n--- СТАТИСТИКА ПРОГНОЗОВ ---")
        total = pd.read_sql("SELECT COUNT(*) as total FROM predictions", conn).iloc[0]['total']
        
        # Считаем статистику только по тем новостям, которые были признаны значимыми (score >= 0.5)
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
