import sqlite3
import config
from db import get_db_connection

def reset_event_keys(keys):
    if not keys:
        print("⚠️ Список ключей пуст.")
        return
        
    if isinstance(keys, str):
        keys = [keys]
        
    print(f"--- Сброс ключей: {', '.join(keys)} ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ', '.join(['?'] * len(keys))
        
        # 1. Удаляем из таблицы прогнозов (откуда init_state берет баллы)
        cursor.execute(f"DELETE FROM predictions WHERE event_key IN ({placeholders})", keys)
        rows_deleted = cursor.rowcount
        
        # 2. Удаляем из таблицы весов (если система успела на нем "обучиться")
        cursor.execute(f"DELETE FROM weights WHERE event_key IN ({placeholders})", keys)
        weights_deleted = cursor.rowcount
        
        conn.commit()
        print(f"✅ Удалено записей прогнозов: {rows_deleted}")
        print(f"✅ Удалено кастомных весов: {weights_deleted}")
        print("Теперь перезапустите engine.py, чтобы обнулить балл в RAM.")

def reset_long_keys(max_entities=2):
    """
    Находит и удаляет все ключи, в которых количество сущностей (частей, разделенных _) 
    превышает заданный порог.
    """
    print(f"--- Поиск и удаление ключей с количеством сущностей > {max_entities} ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Получаем все уникальные ключи из таблиц весов и прогнозов
        cursor.execute("SELECT event_key FROM weights UNION SELECT event_key FROM predictions")
        keys = [row[0] for row in cursor.fetchall() if row[0]]
        
        long_keys = [k for k in keys if len(k.split('_')) > max_entities]
        
        if not long_keys:
            print("🔍 Длинных ключей не обнаружено.")
            return

        for key_name in long_keys:
            cursor.execute("DELETE FROM predictions WHERE event_key = ?", (key_name,))
            cursor.execute("DELETE FROM weights WHERE event_key = ?", (key_name,))
            print(f"🗑 Удален длинный ключ: {key_name}")
            
        conn.commit()
        print(f"✅ Всего удалено уникальных длинных ключей: {len(long_keys)}")
        print("Перезапустите engine.py для обновления состояния в памяти.")

def reset_all_learning():
    """
    Полный сброс всего процесса обучения. 
    Удаляет все веса, сбрасывает множитель и очищает историю прогнозов.
    """
    print("⚠️ ВНИМАНИЕ: Запущен полный сброс обучения системы GTS...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Удаляем все накопленные веса событий
        cursor.execute("DELETE FROM weights")
        
        # 2. Сбрасываем глобальный множитель на значение из конфига
        cursor.execute("UPDATE settings SET value = ? WHERE key = 'impact_multiplier'", (config.IMPACT_MULTIPLIER,))
        
        # 3. Удаляем старые прогнозы, чтобы не обучаться на истории
        cursor.execute("DELETE FROM predictions")
        
        conn.commit()
        print("✅ Система обучения полностью сброшена.")
        print(f"✅ Глобальный множитель возвращен к: {config.IMPACT_MULTIPLIER}")
        print("🚀 Теперь вы можете запустить engine.py с чистого листа.")

if __name__ == "__main__":
    # Выберите нужное действие:
    
    # Вариант 1: Полный сброс
    # reset_all_learning()

    # Вариант 2: Сброс конкретных ключей
    # reset_event_keys(["OIL_US_IRAN"])

    # Вариант 3: Удаление ключей с > 2 сущностями (очистка базы согласно новому лимиту)
    reset_long_keys(max_entities=2)