import sqlite3
from contextlib import contextmanager

DB_PATH = "gts.db"

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
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
            soxs TEXT,
            vix TEXT,
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

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_link ON events(link)")
        conn.commit()