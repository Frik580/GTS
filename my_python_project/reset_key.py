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

def reset_long_keys(max_entities=3):
    """
    Находит и удаляет все ключи, в которых количество сущностей (частей, разделенных _) 
    превышает заданный порог.
    """
    print(f"--- Поиск и удаление ключей с количеством сущностей > {max_entities} ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Получаем все уникальные ключи из таблиц
        cursor.execute("SELECT DISTINCT event_key FROM predictions")
        keys = [row['event_key'] for row in cursor.fetchall() if row['event_key']]
        
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

if __name__ == "__main__":
    # Теперь можно передать список ключей для сброса:
    reset_event_keys([
        "AI_ALIBABA", 
        "SOUTH", 
        "KOREA", 
        "CORP", 
        "GULF", 
        "OMAN", 
        "NORTH", 
        "RFK", 
        "COLOMBIA", 
        "BANK", 
        "DIAMONDBACK", 
        "DIAMONDBACK_OIL", 
        "RUSSIA", 
        "INDIA"
        ])
    
    # Или очистить все ключи, где больше 3 сущностей:
    # reset_long_keys(max_entities=3)