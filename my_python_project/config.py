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
# Доступные активы: "nasdaq", "sp500", "oil", "soxs", "vix", "gold", "btc", "global"
TRACKED_KEYWORDS = {
    "US Iran": (2.5, ["global", "oil", "vix", "btc", "sp500"]), # Пример: влияет на общий риск и нефть
    "Nvidia": (1.8, ["nasdaq", "soxs", "vix", "global"]), # Добавлен VIX для учета волатильности техов
    "OpenAI": (1.8, ["soxs"]), # Пример: влияет на AI и полупроводники
    "Oil": (1.5, ["oil", "global", "vix"]), # Пример: влияет на нефть и общий риск
    "Gold": (0.8, ["gold"]), # Слегка повышаем вес, чтобы модель уделяла больше внимания золоту
    "BTC": (1.2, ["btc", "global"]), # Повышен вес для учета высокой волатильности
    "Nasdaq": (1.0, ["nasdaq"]),
    "AI Sector": (1.3, ["nasdaq", "soxs"]),
    "AI Infrastructure": (1.4, ["nasdaq", "soxs"]),
    "Trump Policy": (2.2, ["global", "nasdaq", "sp500", "oil", "vix"]),
    "MU": (1.2, ["soxs"]),
    "Semiconductor": (1.5, ["soxs", "nasdaq"]),
    "US Inflation": (2.0, ["global", "vix", "gold"]),
    "Intel": (1.3, ["soxs"]),
    "AMD": (1.3, ["soxs"]),
    "Broadcom": (1.2, ["soxs"]),
    "Anthropic": (1.5, ["soxs"]), # Исправлена опечатка
    "Qualcomm": (1.2, ["soxs"]),
    "Hormuz": (2.0, ["oil", "vix", "global"]), # Фокус на геополитике в регионе
    "Yield": (1.8, ["global", "vix", "nasdaq"]), # Влияние на общий риск, волатильность и тех. сектор
    "Treasury": (1.5, ["global", "vix", "nasdaq"]), # Влияние на общий риск, волатильность и тех. сектор
    "Jerome Powell": (2.2, ["global", "vix", "nasdaq"]), # Прямое влияние на монетарную политику и рынки
    "HBM": (1.5, ["soxs", "nasdaq"]),
    "HBM Memory": (1.5, ["soxs", "nasdaq"]), # Ключевой компонент для производства AI-ускорителей
    "Inflation": (2.0, ["global", "vix", "gold", "nasdaq", "sp500"]), # Добавлены макроэкономические факторы
    "Interest Rates": (2.2, ["global", "vix", "nasdaq", "sp500"]),
    "Recession": (2.5, ["global", "vix", "gold", "nasdaq", "sp500"]),
    "Geopolitical Tension": (2.5, ["global", "oil", "vix", "gold"]),
    "Earnings": (1.5, ["nasdaq", "sp500", "soxs"])
}

# HBM Index Configuration
HBM_INDEX_SEGMENT_WEIGHTS = {
    "HBM_MAKERS": 0.45,
    "AI_GPU": 0.30,
    "PACKAGING": 0.15,
    "EQUIPMENT": 0.10,
}

HBM_INDEX_COMPONENTS = {
    "HBM_MAKERS": ["000660.KS", "MU", "005930.KS"], # SK Hynix (000660.KS), Micron (MU), Samsung (005930.KS)
    "AI_GPU": ["NVDA", "AMD"], # NVIDIA (NVDA), AMD (AMD)
    "PACKAGING": ["TSM", "ASX"], # TSMC (TSM), ASE Technology Holding (ASX)
    "EQUIPMENT": ["ASML", "AMAT"], # ASML (ASML), Applied Materials (AMAT)
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
LEARNING_INTERVAL = 1800 # Интервал обучения (30 минут) - оптимально для баланса между адаптацией и стабильностью
MARKET_LOOKBACK_HOURS = 2 # Окно анализа реакции рынка
MAX_NEWS_AGE_HOURS = MARKET_LOOKBACK_HOURS*3 # Новость должна быть не старше окна анализа

# Адаптивные задержки обучения (в часах) в зависимости от типа события
EVENT_TYPE_LOOKBACK = {
    "military":   {"primary": 0.5, "secondary": 6},
    "economic":   {"primary": 0.5, "secondary": 4},
    "diplomatic": {"primary": 2, "secondary": 8},
    "tech":       {"primary": 1, "secondary": 3},
    "neutral":    {"primary": 2, "secondary": 2},
}
BLACK_SWAN_LOOKBACK_HOURS = 24 # Окно для оценки фундаментального сдвига при ЧП

CLEANUP_INTERVAL = 86400 # Интервал очистки (24 часа)
RESEARCH_INTERVAL = 86400 # Интервал глобального исследования ИИ (раз в сутки)
RETENTION_DAYS = 7 # Уменьшено для более быстрой ротации данных и компактности БД

# AI Delays
AI_DELAY_JSON = 4 # Оптимально для 15 RPM (бесплатный Gemini)
AI_DELAY_NO_JSON = 10 # Задержка для тяжелых/медленных моделей

# Logic Factors
DECAY_FACTOR = 0.9 # Оптимальный баланс: новость сохраняет 50% силы через 15-20 минут и затухает за 2-3 часа.
NIGHT_DECAY_FACTOR = 0.98 # Почти не снижаем балл, когда рынок закрыт, чтобы сохранить контекст к открытию
MAX_SCORE_THRESHOLD = 25.0 

# Коэффициенты нормализации для разных классов активов
ASSET_SCALING_FACTORS = { # Скорректированы для улучшения калибровки
    "global": 6.0, # Оставляем, так как это агрегированный показатель
    "nasdaq": 7.0, # Слегка уменьшаем, чтобы снизить переоценку влияния
    "sp500": 6.0,  # Слегка уменьшаем, чтобы снизить переоценку влияния
    "oil": 7.0,    # Оставляем
    "btc": 2.5,    # Оставляем, хорошо работает
    "gold": 7.0,   # Слегка уменьшаем, чтобы помочь с "разбросом" и улучшить калибровку
    "vix": 5.0,    # Увеличиваем, VIX очень волатилен и требует большего масштабирования
    "soxs": 3.0,   # Увеличиваем, SOXS 3x leveraged, требует большего масштабирования
}

LEARNING_RATE = 0.05  # Снижаем для большей стабильности, так как по основным активам точность начала падать
IMPACT_MULTIPLIER = 4.0 # Начальное значение. После старта система обучается и берет значение из БД.
LEARNING_THRESHOLD = 0.2 # Снижен порог рыночного движения, чтобы учиться на более мелких изменениях
PIVOT_THRESHOLD = 5.0 # Порог "разворотной" новости, при котором накопленный балл обнуляется
MIN_WEIGHT_THRESHOLD = 0.5 # Повышено для автоматического удаления слабых/случайных связей
NEUTRAL_SCORE_THRESHOLD = 2.5 # Еще выше порог для отсечения около-рыночного шума
MAX_ENTITY_PARTS = 2 # Сокращаем до 2 для лучшей группировки и консолидации весов
DUPLICATE_TITLE_THRESHOLD = 0.45 # Более агрессивный фильтр (чем ниже, тем сильнее подавление похожих новостей)

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
SIGNAL_THRESHOLD_HIGH = 3.5  # Повышаем порог для индексов, чтобы уменьшить количество ложных алертов
SIGNAL_THRESHOLD_MED = 2.5   # Повышено для VIX и Oil для фильтрации шума
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)
SIGNAL_THRESHOLD_BTC = 4.0   # For Crypto (Volatility buffer)
BTC_MIN_VOLATILITY_FOR_ALERT = 1.0 # Минимальное изменение цены BTC (%) для отправки уведомления



# Если твой Win Rate выше 60% — система работает отлично. 
# Если ниже 40% — значит, либо веса в config.py настроены неверно, либо рынок сейчас ведет себя иррационально.

# Средняя абсолютная ошибка (avg_abs_error): Чем ниже это число, тем лучше откалиброван ваш global_impact_multiplier. 
# Если ошибка везде большая (например, > 20), значит множитель в config.py требует ручной корректировки 
# или системе нужно больше времени на обучение.