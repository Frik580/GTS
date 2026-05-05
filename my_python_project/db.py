import sqlite3

conn = sqlite3.connect("gts.db", check_same_thread=False)
cursor = conn.cursor()


def init_db():
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