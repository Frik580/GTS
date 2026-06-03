import sqlite3
import logging
from db import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(message)s')

def backfill_event_ids():
    logging.info("🚀 Начало миграции event_id для старых прогнозов...")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Получаем прогнозы без event_id
        cursor.execute("""
            SELECT id, timestamp, event_type, event_key 
            FROM predictions 
            WHERE event_id IS NULL
        """)
        preds = cursor.fetchall()
        
        if not preds:
            logging.info("✅ Все записи уже имеют event_id или база пуста.")
            return

        logging.info(f"Найдено {len(preds)} записей для обработки.")
        
        updated_count = 0
        ambiguous_count = 0
        failed_count = 0

        for p in preds:
            p_id, p_ts, p_type, p_key = p
            
            # Ищем кандидатов в таблице events
            # Совпадение по точному времени и типу события
            cursor.execute("""
                SELECT id, slug, title FROM events 
                WHERE timestamp = ? AND event = ?
            """, (p_ts, p_type))
            candidates = cursor.fetchall()
            
            match_id = None
            
            if len(candidates) == 1:
                # Идеальный случай: один таймстамп - одно событие нужного типа
                match_id = candidates[0]['id']
            elif len(candidates) > 1:
                # Неоднозначность: несколько новостей в одну секунду
                # Проверяем схожесть по slug или заголовку
                p_key_norm = p_key.upper() if p_key else ""
                
                for cand in candidates:
                    c_slug = (cand['slug'] or "").upper()
                    c_title = (cand['title'] or "").upper()
                    
                    # Если event_key содержится в slug или наоборот, это наше событие
                    if p_key_norm and (p_key_norm in c_slug or c_slug in p_key_norm or p_key_norm in c_title):
                        match_id = cand['id']
                        break
                
                if not match_id:
                    # Если по ключам не нашли, берем первый попавшийся (лучше, чем ничего)
                    # но помечаем как сомнительный
                    match_id = candidates[0]['id']
                    ambiguous_count += 1
            
            if match_id:
                cursor.execute("UPDATE predictions SET event_id = ? WHERE id = ?", (match_id, p_id))
                updated_count += 1
            else:
                failed_count += 1

        conn.commit()
        
        logging.info("\n--- РЕЗУЛЬТАТЫ МИГРАЦИИ ---")
        logging.info(f"✅ Успешно обновлено: {updated_count}")
        logging.info(f"⚠️ Обработано с неоднозначностью: {ambiguous_count}")
        logging.info(f"❌ Не удалось сопоставить: {failed_count}")
        
        # Проверка результата
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE event_id IS NULL")
        remaining = cursor.fetchone()[0]
        if remaining == 0:
            logging.info("✨ Миграция полностью завершена. Теперь learning cycle будет видеть все записи.")
        else:
            logging.info(f"ℹ️ Осталось {remaining} записей без ID (вероятно, исходные события были удалены при очистке).")

if __name__ == "__main__":
    try:
        backfill_event_ids()
    except Exception as e:
        logging.error(f"Ошибка миграции: {e}")