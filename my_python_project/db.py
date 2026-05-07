import sqlite3
from contextlib import contextmanager
import config

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row  # Позволяет обращаться к полям по именам
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            score REAL,
            event TEXT,
            nasdaq TEXT,
            oil TEXT,
            hbm TEXT,
            soxs TEXT,
            gold TEXT,
            btc TEXT,
            vix TEXT,
            fear_greed REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT,
            score REAL,
            predicted_impact REAL,
            actual_move REAL DEFAULT 0,
            target_asset TEXT,
            resolved INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weights (
            event_key TEXT PRIMARY KEY,
            weight REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_global_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT,
            asset TEXT,
            impact_direction TEXT,
            reasoning TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_link ON events(link)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolved)")

        # Словарь миграций: описываем колонки, которые должны быть в таблицах
        # Это позволяет добавлять новые активы просто дополняя этот список
        required_columns = {
            "events": {
                "nasdaq": "TEXT",
                "oil": "TEXT",
                "hbm": "TEXT",
                "soxs": "TEXT",
                "gold": "TEXT",
                "btc": "TEXT",
                "vix": "TEXT",
                "fear_greed": "REAL"
            },
            "predictions": {
                "actual_move": "REAL DEFAULT 0",
                "resolved": "INTEGER DEFAULT 0",
                "is_correct": "INTEGER DEFAULT 0"
            }
        }

        for table_name, columns in required_columns.items():
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [info[1] for info in cursor.fetchall()]
            
            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    except sqlite3.OperationalError:
                        # Безопасный пропуск, если колонка была добавлена другим процессом
                        pass
        
        # Создаем таблицу для системных настроек, если её нет
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value REAL
        )
        """)
        # Устанавливаем начальное значение множителя из конфига, если таблицы не было
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('impact_multiplier', ?)", (config.IMPACT_MULTIPLIER,))

        conn.commit()