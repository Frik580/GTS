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
        
        # Оптимизированный поиск длинных ключей через SQL (если возможно) или фильтрация в Python
        cursor.execute("SELECT DISTINCT event_key FROM weights UNION SELECT DISTINCT event_key FROM predictions")
        long_keys = [row[0] for row in cursor.fetchall() if row[0] and len(row[0].split('_')) > max_entities]

        if not long_keys:
            print("🔍 Длинных ключей не обнаружено.")
            return

        placeholders = ', '.join(['?'] * len(long_keys))
        cursor.execute(f"DELETE FROM predictions WHERE event_key IN ({placeholders})", long_keys)
        cursor.execute(f"DELETE FROM weights WHERE event_key IN ({placeholders})", long_keys)

        conn.commit()
        print(f"✅ Всего удалено уникальных длинных ключей: {len(long_keys)}")

def deep_clean_db():
    """
    Выполняет глубокую оптимизацию:
    1. Удаляет осиротевшие эмбеддинги (для которых нет событий).
    2. Удаляет старые прогнозы согласно конфигу.
    3. Выполняет VACUUM.
    """
    print("🧹 Запуск глубокой очистки базы данных...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Удаляем старые записи
        cursor.execute("DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
        cursor.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
        
        # 2. Удаляем эмбеддинги, у которых нет соответствующих заголовков в events
        cursor.execute("DELETE FROM embeddings WHERE title NOT IN (SELECT title FROM events)")
        
        conn.commit()

    with get_db_connection() as conn:
        print("📦 Сжатие базы данных (VACUUM)...")
        conn.execute("VACUUM")
    
    print("✅ База данных полностью оптимизирована.")

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
        cursor.execute("DELETE FROM embeddings")
        cursor.execute("DELETE FROM events")
        
        # 4. Очищаем накопленную статистику и предложения
        cursor.execute("DELETE FROM source_stats")
        cursor.execute("DELETE FROM asset_stats")
        cursor.execute("DELETE FROM ai_global_suggestions")

        conn.commit()
        print("✅ Система обучения полностью сброшена.")
        print(f"✅ Глобальный множитель возвращен к: {config.IMPACT_MULTIPLIER}")
        print("🚀 Теперь вы можете запустить engine.py с чистого листа.")

def reset_multiplier_only():
    """
    Сбрасывает только глобальный множитель влияния до значения из конфига.
    Полезно, если система 'переобучилась' и задрала множитель слишком высоко.
    """
    print(f"--- Сброс IMPACT_MULTIPLIER до базового ({config.IMPACT_MULTIPLIER}) ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE settings SET value = ? WHERE key = 'impact_multiplier'", (config.IMPACT_MULTIPLIER,))
        conn.commit()
    print("✅ Множитель успешно сброшен в базе данных.")
    print("ℹ️ Не забудьте перезапустить engine.py, чтобы он подхватил новое значение.")

def reset_source_stats():
    """
    Сбрасывает накопленную статистику по источникам новостей.
    """
    print("--- Сброс статистики источников (Source Analysis) ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM source_stats")
        conn.commit()
    print("✅ Статистика источников успешно очищена.")

def reset_asset_stats():
    """
    Сбрасывает накопленную статистику по активам.
    """
    print("--- Сброс статистики активов (Asset Stats) ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM asset_stats")
        conn.commit()
    print("✅ Статистика активов успешно очищена.")

def clear_log():
    """
    Очищает содержимое лог-файла, указанного в config.py.
    """
    print(f"--- Очистка файла логов: {config.LOG_FILE} ---")
    try:
        with open(config.LOG_FILE, 'w', encoding='utf-8') as f:
            pass  # Открытие в режиме 'w' обнуляет файл
        print("✅ Лог-файл успешно очищен.")
    except Exception as e:
        print(f"❌ Ошибка: {e} (Возможно, файл занят запущенным engine.py)")

if __name__ == "__main__":
    # Выберите нужное действие:
    
    # Вариант 1: Полный сброс
    # reset_all_learning()

    # Вариант 6: Глубокая очистка без потери весов обучения
    # deep_clean_db()

    # Вариант 2: Сброс конкретных ключей
    # reset_event_keys(["OIL_US_IRAN"])

    # Вариант 3: Удаление ключей с > 2 сущностями (очистка базы согласно новому лимиту)
    # reset_long_keys(max_entities=2)

    # Вариант 4: Сброс только множителя
    # reset_multiplier_only()

    # Вариант 5: Сброс статистики источников
    # reset_source_stats()

    # Вариант 7: Очистка логов
    # clear_log()
