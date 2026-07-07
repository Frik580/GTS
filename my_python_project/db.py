import aiosqlite
from contextlib import asynccontextmanager
import config

@asynccontextmanager
async def get_db_connection():
    async with aiosqlite.connect(config.DB_PATH, timeout=30) as conn:
        try:
            conn.row_factory = aiosqlite.Row  # Позволяет обращаться к полям по именам
            yield conn
        finally:
            pass # aiosqlite закрывает соединение автоматически в context manager

async def init_db():
    async with get_db_connection() as conn:
        # Устанавливаем режим WAL один раз при инициализации
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=-64000") # 64MB кэш
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            score REAL,
            event TEXT,
            nasdaq TEXT,
            sp500 TEXT,
            oil TEXT,
            soxs TEXT,
            gold TEXT,
            btc TEXT,
            vix TEXT,
            fear_greed REAL,
            slug TEXT,
            summary TEXT,
            title_ru TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            event_key TEXT,
            score REAL,
            predicted_impact REAL,
            actual_move REAL DEFAULT 0,
            is_correct INTEGER DEFAULT 0,
            target_asset TEXT,
            event_type TEXT,
            is_black_swan INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            confidence REAL DEFAULT 1.0,
            signed_alpha REAL DEFAULT 0,
            source_domain TEXT,
            model_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            title TEXT PRIMARY KEY,
            vector TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS weights (
            event_key TEXT,
            target_asset TEXT,
            weight REAL,
            PRIMARY KEY (event_key, target_asset)
        )
        """)

        # Таблица для калибровки чувствительности каждой модели
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS model_stats (
            model_name TEXT PRIMARY KEY,
            sensitivity REAL DEFAULT 1.0,
            total_resolved INTEGER DEFAULT 0
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_global_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT,
            asset TEXT,
            impact_direction TEXT,
            reasoning TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Таблица для долгосрочной статистики источников (накопительная)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS source_stats (
            source_domain TEXT PRIMARY KEY,
            total_resolved INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            sum_error REAL DEFAULT 0,
            sum_confidence REAL DEFAULT 0,
            sum_alpha REAL DEFAULT 0,
            sum_alpha_sq REAL DEFAULT 0
        )
        """)

        # Таблица для долгосрочной статистики по активам (накопительная)
        await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS asset_stats (
            target_asset TEXT PRIMARY KEY,
            total_resolved INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            sum_error REAL DEFAULT 0,
            multiplier REAL DEFAULT {config.IMPACT_MULTIPLIER}
        )
        """)

        # Таблица для хранения ежедневных исторических цен
        await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS daily_prices (
        ticker TEXT,
        date TEXT,
        close REAL,
        PRIMARY KEY (ticker, date)
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS quant_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bear_probability REAL,
        target_position REAL,
        capex_score REAL,
        guidance_score REAL,
        active_triggers TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Словарь миграций: описываем колонки, которые должны быть в таблицах
        # Это позволяет добавлять новые активы просто дополняя этот список
        required_columns = {
            "events": {
                "nasdaq": "TEXT",
                "sp500": "TEXT",
                "oil": "TEXT",
                "soxs": "TEXT",
                "gold": "TEXT",
                "btc": "TEXT",
                "vix": "TEXT",
                "fear_greed": "REAL",
                "slug": "TEXT",
                "is_black_swan": "INTEGER DEFAULT 0",
                "summary": "TEXT",
                "title_ru": "TEXT"
            },
            "predictions": {
                "event_id": "INTEGER",
                "actual_move": "REAL DEFAULT 0",
                "resolved": "INTEGER DEFAULT 0",
                "is_correct": "INTEGER DEFAULT 0",
                "target_asset": "TEXT",
                "event_type": "TEXT",
                "is_black_swan": "INTEGER DEFAULT 0",
                "confidence": "REAL DEFAULT 1.0",
                "signed_alpha": "REAL DEFAULT 0",
                "source_domain": "TEXT",
                "model_name": "TEXT",
                "capex_signal": "INTEGER DEFAULT 0",
                "guidance_signal": "INTEGER DEFAULT 0"
            },
            "asset_stats": {
                "multiplier": f"REAL DEFAULT {config.IMPACT_MULTIPLIER}"
            },
            "source_stats": {
                "sum_alpha": "REAL DEFAULT 0",
                "sum_alpha_sq": "REAL DEFAULT 0"
            }
        }

        # Миграция для таблицы weights (добавление target_asset)
        async with conn.execute("PRAGMA table_info(weights)") as cursor:
            if "target_asset" not in [info[1] for info in await cursor.fetchall()]:
                await conn.execute("ALTER TABLE weights ADD COLUMN target_asset TEXT DEFAULT 'global'")

        for table_name, columns in required_columns.items():
            async with conn.execute(f"PRAGMA table_info({table_name})") as cursor:
                existing_columns = [info[1] for info in await cursor.fetchall()]
            
            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    try:
                        await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    except Exception:
                        # Безопасный пропуск, если колонка была добавлена другим процессом
                        pass
        
        # Создаем таблицу для системных настроек, если её нет
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value REAL
        )
        """)
        # Устанавливаем начальное значение множителя из конфига, если таблицы не было
        await conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('impact_multiplier', ?)", (config.IMPACT_MULTIPLIER,))

        # Очистка существующих пустых значений в активах (миграция данных)
        await conn.execute("UPDATE predictions SET target_asset = 'global' WHERE target_asset IS NULL OR target_asset = ''")

        # Создание индексов (перенесено в конец, чтобы гарантировать наличие колонок после миграций)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_link ON events(link)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolved)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_event_key ON predictions(event_key)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_event_id ON predictions(event_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp ON embeddings(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_slug ON events(slug)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices (ticker, date DESC)")

        await conn.commit()