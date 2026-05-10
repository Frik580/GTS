import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Database and Logs
DB_PATH = "gts.db"
LOG_FILE = "gts.log"

# Feeds
# Список ключевых слов для отслеживания. Можно менять, добавлять или удалять.
# Теперь это словарь: "Ключевое слово": Вес (приоритет)
# Формат: "Ключевое слово": (Вес, ["целевой_актив_1", "целевой_актив_2"])
# Доступные активы: "nasdaq", "oil", "soxs", "vix", "gold", "btc", "hbm", "global"
TRACKED_KEYWORDS = {
    "US Iran": (2.5, ["global", "oil", "vix", "btc"]), # Пример: влияет на общий риск и нефть
    "Nvidia": (1.8, ["hbm", "nasdaq", "soxs", "vix", "global"]), # Добавлен VIX для учета волатильности техов
    "OpenAI": (1.8, ["hbm", "soxs"]), # Пример: влияет на AI и полупроводники
    "Oil": (1.5, ["oil", "global", "vix"]), # Пример: влияет на нефть и общий риск
    "Gold": (1.0, ["gold"]),
    "BTC": (1.2, ["btc", "global"]), # Повышен вес для учета высокой волатильности
    "Nasdaq": (1.0, ["nasdaq"]),
    "AI": (1.5, ["hbm", "nasdaq", "soxs"]),
    "Trump policy economy": (2.2, ["global", "nasdaq", "oil", "vix"]),
    "MU": (1.2, ["hbm", "soxs"]),
    "Semiconductor": (1.5, ["hbm", "soxs", "nasdaq"]),
    "US Inflation": (2.0, ["global", "vix", "gold"]),
    "Intel": (1.3, ["hbm", "soxs"]),
    "AMD": (1.3, ["hbm", "soxs"]),
    "Broadcom": (1.2, ["hbm", "soxs"]),
    "Anthropic": (1.5, ["hbm", "soxs"]), # Исправлена опечатка
    "Qualcomm": (1.2, ["hbm", "soxs"]),
    "Hormuz": (2.0, ["oil", "vix", "global"]), # Фокус на геополитике в регионе
    "Yield": (1.8, ["global", "vix", "nasdaq"]), # Влияние на общий риск, волатильность и тех. сектор
    "Treasury": (1.5, ["global", "vix", "nasdaq"]), # Влияние на общий риск, волатильность и тех. сектор
    "Jerome Powell": (2.2, ["global", "vix", "nasdaq"]), # Прямое влияние на монетарную политику и рынки
}

# Основные RSS-ленты Yahoo Finance для расширения охвата рынка
YAHOO_FINANCE_FEEDS = [
    "https://finance.yahoo.com/news/rssindex", # Общая лента финансовых новостей
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA,AMD,AVGO,TSM,INTC,MU&region=US&lang=en-US", # Лента для полупроводников
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MU,SKHYNIX.KS,SAMSUNG.KS&region=US&lang=en-US", # Лента для памяти (HBM)
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ASML,AMAT,LRCX,KLAC&region=US&lang=en-US", # Лента для оборудования для производства чипов
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA,MSFT,GOOGL,AMZN,META&region=US&lang=en-US", # Лента для AI и крупных технологических компаний
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ASML,AMAT,LRCX,KLAC&region=US&lang=en-US", # Лента для оборудования для производства чипов (повтор для усиления охвата)
    "https://rsshub.app/bloomberg/topics/economics", # Лента для экономики
    "https://www.reuters.com/world/rss", # Актуальная лента мировых новостей
    "https://www.reuters.com/business/commodities/rss", # Актуальная лента сырьевых товаров
    "https://www.maritime-executive.com/rss", # Лента для морских новостей, включая новости о Ормузском проливе
    "https://www.federalreserve.gov/feeds/press_monetary.xml", # Лента для новостей Федеральной резервной системы США
    "https://home.treasury.gov/news/press-releases/rss" # Лента для новостей Министерства финансов США
]

RSS_FEEDS = [f"https://news.google.com/rss/search?q={k.replace(' ', '+')}" for k in TRACKED_KEYWORDS.keys()] + YAHOO_FINANCE_FEEDS
RSS_MAX_ENTRIES = 4 # Количество записей RSS для обработки в активное время рынка
RSS_MAX_ENTRIES_INACTIVE = RSS_MAX_ENTRIES*3 # Количество записей RSS для обработки, когда рынок неактивен (ночь/выходные)

# Time Intervals (in seconds)
CHECK_INTERVAL = 180 
COOLDOWN = 600 # Интервал между действиями (10 минут)
LEARNING_INTERVAL = 3600 # Интервал обучения (1 час)
MARKET_LOOKBACK_HOURS = 2 # Окно анализа реакции рынка
MAX_NEWS_AGE_HOURS = MARKET_LOOKBACK_HOURS*3 # Новость должна быть не старше окна анализа
CLEANUP_INTERVAL = 86400 # Интервал очистки (24 часа)
RESEARCH_INTERVAL = 86400 # Интервал глобального исследования ИИ (раз в сутки)
RETENTION_DAYS = 7 # Уменьшено для более быстрой ротации данных и компактности БД

# AI Delays
AI_DELAY_JSON = 15 # Время ожидания ответа от модели при запросе JSON (15 секунд)
AI_DELAY_NO_JSON = 60 # Время ожидания ответа от модели при отсутствии JSON (60 секунд)

# Logic Factors
DECAY_FACTOR = 0.9 
NIGHT_DECAY_FACTOR = 0.98 # Почти не снижаем балл, когда рынок закрыт, чтобы сохранить контекст к открытию
MAX_SCORE_THRESHOLD = 25.0 

# Коэффициенты нормализации для разных классов активов
ASSET_SCALING_FACTORS = {
    "global": 6.0,
    "nasdaq": 8.0,
    "oil": 7.0,
    "btc": 2.5,   # BTC волатилен, 1% изменения значит меньше
    "gold": 15.0, # Золото стабильно, 1% изменения — это сильный сигнал
    "vix": 4.0,   # VIX очень волатилен, снижаем фактор
    "soxs": 2.0,  # 3x плечо: 5% движения = 10 баллов импакта
    "hbm": 5.0    # Сектор чипов волатильнее индекса
}

LEARNING_RATE = 0.05  # Увеличено для более быстрой адаптации весов к изменениям рынка
IMPACT_MULTIPLIER = 4.0 # Начальное значение. После старта система обучается и берет значение из БД.
LEARNING_THRESHOLD = 0.3 # Порог рыночного движения (в %). Если цена изменилась меньше, обучение не проводится.
PIVOT_THRESHOLD = 5.0 # Порог "разворотной" новости, при котором накопленный балл обнуляется
MIN_WEIGHT_THRESHOLD = 0.8 # Повышено для автоматического удаления слабых/случайных связей
NEUTRAL_SCORE_THRESHOLD = 3.0 # Поднимаем порог для случайных сущностей, чтобы отсечь локальные новости
MAX_ENTITY_PARTS = 2 # Сокращаем длину ключа до 2 для лучшей группировки статистики
MIN_NEWS_SCORE_FOR_ALERT = 0.8 # Минимальный балл конкретной новости для отправки в Telegram

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
SIGNAL_THRESHOLD_MED = 2.5   # Повышено для VIX и Oil для фильтрации шума
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)
SIGNAL_THRESHOLD_BTC = 4.0   # For Crypto (Volatility buffer)
BTC_MIN_VOLATILITY_FOR_ALERT = 1.0 # Минимальное изменение цены BTC (%) для отправки уведомления



# Если твой Win Rate выше 60% — система работает отлично. 
# Если ниже 40% — значит, либо веса в config.py настроены неверно, либо рынок сейчас ведет себя иррационально.

# Средняя абсолютная ошибка (avg_abs_error): Чем ниже это число, тем лучше откалиброван ваш global_impact_multiplier. 
# Если ошибка везде большая (например, > 20), значит множитель в config.py требует ручной корректировки 
# или системе нужно больше времени на обучение.