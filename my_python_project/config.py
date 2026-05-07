import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Database and Logs
DB_PATH = "gts.db"
LOG_FILE = "gts.log"

# Feeds
# Список ключевых слов для отслеживания. Можно менять, добавлять или удалять.
# Теперь это словарь: "Ключевое слово": Вес (приоритет)
# Формат: "Ключевое слово": (Вес, ["целевой_актив_1", "целевой_актив_2"])
# Доступные активы: "nasdaq", "oil", "soxs", "vix", "gold", "btc", "hbm", "global"
TRACKED_KEYWORDS = {
    "US Iran": (2.5, ["global", "oil", "vix"]), # Пример: влияет на общий риск и нефть
    "Nvidia": (1.8, ["hbm", "nasdaq", "soxs", "global"]), # Пример: влияет на полупроводники и Nasdaq
    "OpenAI": (1.8, ["hbm", "soxs"]), # Пример: влияет на AI и полупроводники
    "Oil": (1.5, ["oil", "global"]), # Пример: влияет на нефть и общий риск
    "Gold": (1.0, ["gold"]),
    "Bitcoin": (0.8, ["btc", "global"]),
    "Nasdaq": (1.0, ["nasdaq"]),
    "AI": (1.5, ["global", "hbm", "nasdaq", "soxs"]),
    "Trump": (2.2, ["global", "nasdaq", "oil"]),
    "Micron": (1.2, ["hbm", "soxs"]),
}

RSS_FEEDS = [f"https://news.google.com/rss/search?q={k.replace(' ', '+')}" for k in TRACKED_KEYWORDS.keys()]
RSS_MAX_ENTRIES = 3 # Максимальное количество новостей из одной ленты за один цикл сканирования

# Time Intervals (in seconds)
CHECK_INTERVAL = 300 # Интервал проверки новостей (5 минут)
COOLDOWN = 600 # Интервал между действиями (10 минут)
LEARNING_INTERVAL = 3600 # Интервал обучения (1 час)
CLEANUP_INTERVAL = 86400 # Интервал очистки (24 часа)
RESEARCH_INTERVAL = 86400 # Интервал глобального исследования ИИ (раз в сутки)
RETENTION_DAYS = 14 # Количество дней хранения данных

# AI Delays
AI_DELAY_JSON = 15 # Время ожидания ответа от модели при запросе JSON (15 секунд)
AI_DELAY_NO_JSON = 60 # Время ожидания ответа от модели при отсутствии JSON (60 секунд)

# Logic Factors
MARKET_LOOKBACK_HOURS = 4 # Количество часов для анализа рыночных данных вокруг события
DECAY_FACTOR = 0.95 # Увеличиваем скорость затухания для более быстрой адаптации к новым данным
MAX_SCORE_THRESHOLD = 25.0 # Увеличиваем порог для отправки сигналов, чтобы уменьшить количество ложных срабатываний
SCALING_FACTOR = 8.0 # Увеличиваем масштаб для более заметного влияния предсказаний
LEARNING_RATE = 0.05  # Увеличиваем скорость обучения для более быстрой адаптации
IMPACT_MULTIPLIER = 4.0 # Начальное значение. После старта система обучается и берет значение из БД.
LEARNING_THRESHOLD = 0.1 # Минимальное движение цены (%) для учета в обучении
MIN_WEIGHT_THRESHOLD = 0.5 # Порог веса, ниже которого ключ удаляется из БД
NEUTRAL_SCORE_THRESHOLD = 2.0 # Порог значимости. Все, что ниже, считается шумом и не участвует в обучении.

NON_FINANCIAL_SCORE_DECAY_FACTOR = 0.5 # Коэффициент снижения балла для нефинансовых/дипломатических новостей
# Рейтинг доверия источникам (Trust Factor)
SOURCE_TRUST_LEVELS = {
    "reuters": 1.3,        # Повышенное доверие
    "bloomberg": 1.3,
    "cnn": 1.2,
    "wsj": 1.2,
    "financial times": 1.2,
    "cnbc": 1.0,           # Стандарт
    "yahoo finance": 1.0,
    "reddit": 0.5,         # Пониженное доверие (высокий риск шума)
    "twitter": 0.4,
    "x.com": 0.4,
    "woxx.com": 0.4,
}
DEFAULT_TRUST_SCORE = 0.8  # Значение для неизвестных источников

# Thresholds for market signals (Empirical sensitivity)
SIGNAL_THRESHOLD_HIGH = 3.0  # For Indices (Nasdaq, SOXS)
SIGNAL_THRESHOLD_MED = 2.0   # For Commodities and VIX
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)
SIGNAL_THRESHOLD_BTC = 4.0   # For Crypto (Volatility buffer)
BTC_MIN_VOLATILITY_FOR_ALERT = 5.0 # Минимальное изменение цены BTC (%) для отправки уведомления



# Если твой Win Rate выше 60% — система работает отлично. 
# Если ниже 40% — значит, либо веса в config.py настроены неверно, либо рынок сейчас ведет себя иррационально.

# Средняя абсолютная ошибка (avg_abs_error): Чем ниже это число, тем лучше откалиброван ваш global_impact_multiplier. 
# Если ошибка везде большая (например, > 20), значит множитель в config.py требует ручной корректировки 
# или системе нужно больше времени на обучение.