import sqlite3
import pandas as pd

def inspect_gts():
    conn = sqlite3.connect("gts.db")
    
    print("--- ТЕКУЩИЕ ВЕСА (ПОСЛЕ ОБУЧЕНИЯ) ---")
    weights = pd.read_sql("SELECT * FROM weights", conn)
    print(weights if not weights.empty else "Таблица весов пуста (используются дефолтные)")
    
    print("\n--- ПОСЛЕДНИЕ 5 СОБЫТИЙ ---")
    events = pd.read_sql("SELECT title, score, event, timestamp FROM events ORDER BY timestamp DESC LIMIT 5", conn)
    print(events)
    
    print("\n--- СТАТИСТИКА ПРОГНОЗОВ ---")
    total = pd.read_sql("SELECT COUNT(*) as total FROM predictions", conn).iloc[0]['total']
    resolved = pd.read_sql("SELECT COUNT(*) as resolved, AVG(actual_move) as avg_move FROM predictions WHERE resolved = 1", conn)
    
    avg_move = resolved['avg_move'].iloc[0]
    avg_move_display = float(avg_move) if avg_move is not None else 0.0

    print(f"Всего прогнозов в базе: {total}")
    print(f"Из них обработано (resolved): {resolved['resolved'].iloc[0]}")
    print(f"Среднее реальное движение: {avg_move_display:.2f}")
    
    conn.close()

if __name__ == "__main__":
    inspect_gts()
