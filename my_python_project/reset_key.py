import asyncio
import config
from db import get_db_connection

async def reset_event_keys(keys):
    if not keys:
        print("⚠️ Список ключей пуст.")
        return
        
    if isinstance(keys, str):
        keys = [keys]
        
    print(f"--- Сброс ключей: {', '.join(keys)} ---")
    async with get_db_connection() as conn:
        placeholders = ', '.join(['?'] * len(keys))
        
        # 1. Удаляем из таблицы прогнозов (откуда init_state берет баллы)
        async with conn.execute(f"DELETE FROM predictions WHERE event_key IN ({placeholders})", keys) as cursor:
            rows_deleted = cursor.rowcount
        
        # 2. Удаляем из таблицы весов (если система успела на нем "обучиться")
        async with conn.execute(f"DELETE FROM weights WHERE event_key IN ({placeholders})", keys) as cursor:
            weights_deleted = cursor.rowcount
        
        await conn.commit()
        print(f"✅ Удалено записей прогнозов: {rows_deleted}")
        print(f"✅ Удалено кастомных весов: {weights_deleted}")
        print("Теперь перезапустите engine.py, чтобы обнулить балл в RAM.")

async def reset_long_keys(max_entities=None):
    """
    Находит и удаляет все ключи, в которых количество сущностей (частей, разделенных _) 
    превышает заданный порог (по умолчанию из config.MAX_ENTITY_PARTS).
    """
    if max_entities is None:
        max_entities = config.MAX_ENTITY_PARTS
    print(f"--- Поиск и удаление ключей с количеством сущностей > {max_entities} ---")
    async with get_db_connection() as conn:
        # Оптимизированный поиск длинных ключей через SQL (если возможно) или фильтрация в Python
        async with conn.execute("SELECT DISTINCT event_key FROM weights UNION SELECT DISTINCT event_key FROM predictions") as cursor:
            long_keys = [row[0] for row in await cursor.fetchall() if row[0] and len(row[0].split('_')) > max_entities]

        if not long_keys:
            print("🔍 Длинных ключей не обнаружено.")
            return

        placeholders = ', '.join(['?'] * len(long_keys))
        await conn.execute(f"DELETE FROM predictions WHERE event_key IN ({placeholders})", long_keys)
        await conn.execute(f"DELETE FROM weights WHERE event_key IN ({placeholders})", long_keys)

        await conn.commit()
        print(f"✅ Всего удалено уникальных длинных ключей: {len(long_keys)}")

async def deep_clean_db():
    """
    Выполняет глубокую оптимизацию:
    1. Удаляет осиротевшие эмбеддинги (для которых нет событий).
    2. Удаляет старые прогнозы согласно конфигу.
    3. Выполняет VACUUM.
    """
    print("🧹 Запуск глубокой очистки базы данных...")
    async with get_db_connection() as conn:
        # 1. Удаляем старые записи
        await conn.execute("DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
        await conn.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-' || ? || ' days')", (config.RETENTION_DAYS,))
        
        # 2. Удаляем эмбеддинги, у которых нет соответствующих заголовков в events
        await conn.execute("DELETE FROM embeddings WHERE title NOT IN (SELECT title FROM events)")
        
        await conn.commit()

    async with get_db_connection() as conn:
        print("📦 Сжатие базы данных (VACUUM)...")
        await conn.execute("VACUUM")
    
    print("✅ База данных полностью оптимизирована.")

async def reset_all_learning():
    """
    Полный сброс всего процесса обучения. 
    Удаляет все веса, сбрасывает множитель и очищает историю прогнозов.
    """
    print("⚠️ ВНИМАНИЕ: Запущен полный сброс обучения системы GTS...")
    async with get_db_connection() as conn:
        # 1. Удаляем все накопленные веса событий
        await conn.execute("DELETE FROM weights")
        
        # 2. Сбрасываем глобальный множитель на значение из конфига
        await conn.execute("UPDATE settings SET value = ? WHERE key = 'impact_multiplier'", (config.IMPACT_MULTIPLIER,))
        
        # 3. Удаляем старые прогнозы, чтобы не обучаться на истории
        await conn.execute("DELETE FROM predictions")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM events")
        
        # 4. Очищаем накопленную статистику и предложения
        await conn.execute("DELETE FROM source_stats")
        await conn.execute("DELETE FROM asset_stats")
        await conn.execute("DELETE FROM ai_global_suggestions")

        await conn.commit()
        print("✅ Система обучения полностью сброшена.")
        print(f"✅ Глобальный множитель возвращен к: {config.IMPACT_MULTIPLIER}")
        print("🚀 Теперь вы можете запустить engine.py с чистого листа.")

async def reset_multiplier_only():
    """
    Сбрасывает только глобальный множитель влияния до значения из конфига.
    Полезно, если система 'переобучилась' и задрала множитель слишком высоко.
    """
    print(f"--- Сброс IMPACT_MULTIPLIER до базового ({config.IMPACT_MULTIPLIER}) ---")
    async with get_db_connection() as conn:
        await conn.execute("UPDATE settings SET value = ? WHERE key = 'impact_multiplier'", (config.IMPACT_MULTIPLIER,))
        await conn.commit()
    print("✅ Множитель успешно сброшен в базе данных.")
    print("ℹ️ Не забудьте перезапустить engine.py, чтобы он подхватил новое значение.")

async def reset_source_stats():
    """
    Сбрасывает накопленную статистику по источникам новостей.
    """
    print("--- Сброс статистики источников (Source Analysis) ---")
    async with get_db_connection() as conn:
        await conn.execute("DELETE FROM source_stats")
        await conn.commit()
    print("✅ Статистика источников успешно очищена.")

async def reset_asset_stats():
    """
    Сбрасывает накопленную статистику по активам.
    """
    print("--- Сброс статистики активов (Asset Stats) ---")
    async with get_db_connection() as conn:
        await conn.execute("DELETE FROM asset_stats")
        await conn.commit()
    print("✅ Статистика активов успешно очищена.")

async def reset_model_stats():
    """
    Сбрасывает историю прогнозов (откуда берется WinRate моделей и активов).
    ВНИМАНИЕ: Это обнулит все таблицы в inspect_db, кроме весов обучения.
    """
    print("--- Сброс истории прогнозов (WinRate / Model Stats) ---")
    async with get_db_connection() as conn:
        await conn.execute("DELETE FROM predictions WHERE resolved >= 1")
        await conn.commit()
    print("✅ История прогнозов очищена. Статистика моделей обнулена.")

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

async def menu():
    print("\n--- GTS Utility Menu ---")
    print("1. Полный сброс всего обучения (reset_all_learning)")
    print("2. Сброс конкретных ключей (reset_event_keys)")
    print("3. Удаление длинных ключей (reset_long_keys)")
    print("4. Сброс только множителя (reset_multiplier_only)")
    print("5. Сброс статистики источников (reset_source_stats)")
    print("6. Сброс статистики активов (reset_asset_stats)")
    print("7. Сброс эффективности моделей (reset_model_stats)")
    print("8. Глубокая очистка базы (deep_clean_db)")
    print("9. Очистка лог-файла (clear_log)")
    print("0. Выход")

    choice = input("\nВыберите номер действия: ")

    if choice == '1': await reset_all_learning()
    elif choice == '2':
        keys = input("Введите ключи через запятую (например: BTC,OIL): ").split(',')
        await reset_event_keys([k.strip() for k in keys])
    elif choice == '3':
        limit = input(f"Введите лимит сущностей (по умолчанию {config.MAX_ENTITY_PARTS}): ")
        await reset_long_keys(int(limit) if limit else None)
    elif choice == '4': await reset_multiplier_only()
    elif choice == '5': await reset_source_stats()
    elif choice == '6': await reset_asset_stats()
    elif choice == '7': await reset_model_stats()
    elif choice == '8': await deep_clean_db()
    elif choice == '9': clear_log()
    elif choice == '0': print("Выход.")
    else: print("Неверный ввод.")

if __name__ == "__main__":
    asyncio.run(menu())
