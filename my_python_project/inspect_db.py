import pandas as pd
from db import get_db_connection, init_db

def inspect_gts():
    # Инициализируем БД, чтобы автоматически добавить недостающие колонки (is_correct)
    init_db()

    with get_db_connection() as conn:
        print("--- ТЕКУЩИЕ ВЕСА (ПОСЛЕ ОБУЧЕНИЯ) ---")
        weights = pd.read_sql("SELECT * FROM weights", conn)
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
        
        print("\n--- СТАТИСТИКА ПРОГНОЗОВ ---")
        total = pd.read_sql("SELECT COUNT(*) as total FROM predictions", conn).iloc[0]['total']
        resolved_df = pd.read_sql("SELECT COUNT(*) as resolved, AVG(actual_move) as avg_move, SUM(is_correct) as correct FROM predictions WHERE resolved = 1", conn)
        
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
