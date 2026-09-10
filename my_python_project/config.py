import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag without inverting its meaning."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

# API Keys
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MARKET_DATA_API_KEY = os.getenv("MARKET_DATA_API_KEY") # Ключ от TwelveData или др.
# Включить/отключить провайдеров ИИ (отключайте исчерпанные, чтобы не тратить время на 429)
USE_GEMINI = os.getenv("USE_GEMINI", "False").lower() == "true"
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "False").lower() == "true"
USE_GROQ = os.getenv("USE_GROQ", "True").lower() == "true"
USE_CEREBRAS = os.getenv("USE_CEREBRAS", "True").lower() == "true"
# Включить/отключить использование DeepSeek
USE_DEEPSEEK = os.getenv("USE_DEEPSEEK", "False").lower() == "true"

# Провайдер данных: "twelvedata" или "yfinance" (фоллбек)
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "yfinance")

# Local Model Settings (Ollama)
USE_LOCAL_OLLAMA = os.getenv("USE_LOCAL_OLLAMA", "False").lower() == "true"
OLLAMA_FALLBACK = os.getenv("OLLAMA_FALLBACK", "False").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# Network Settings
HTTP_PROXY = os.getenv("HTTP_PROXY") # Оставьте пустым в .env, если прокси не нужен
# TLS verification is mandatory by default.  Never disable it for market data.
SSL_VERIFY = _env_bool("SSL_VERIFY", True)

# Database and Logs
DB_PATH = "gts.db"
LOG_FILE = "gts.log"

# Feeds
# Глобальный маппинг активов на тикеры
ASSET_TICKER_MAP = {
    "nasdaq": "^IXIC",
    "sp500": "^GSPC",
    "acwi": "ACWI",
    "tip": "TIP",
    "oil": "CL=F",
    "vix": "^VIX",
    "gold": "GLD",
    "btc": "BTC-USD",
    "soxs": "SOXS",
    "soxx": "SOXX",
    "global": "GLOBAL_REGIME"
}

# Тикеры для Leadership Fatigue / Rotation Indicator.
# Они должны синхронизироваться с историей при каждом запуске приложения.
ROTATION_TICKERS = ("NVDA", "AVGO", "ASML", "AMD", "MU", "INTC", "QCOM")

# Список ключевых слов для отслеживания. Можно менять, добавлять или удалять.
# Теперь это словарь: "Ключевое слово": Вес (приоритет)
# Формат: "Ключевое слово": (Вес, ["целевой_актив_1", "целевой_актив_2"])
# Доступные активы: "nasdaq", "sp500", "oil", "soxs", "vix", "gold", "btc", "global"
TRACKED_KEYWORDS = {
    "US Iran": (2.5, ["global", "oil", "vix", "btc", "sp500"]), # Пример: влияет на общий риск и нефть
    "Nvidia": (1.8, ["nasdaq", "soxs", "vix", "global"]), # Добавлен VIX для учета волатильности техов
    "OpenAI": (1.8, ["soxs", "global"]), # Пример: влияет на AI и полупроводники
    "Oil": (1.5, ["oil", "global", "vix"]), # Пример: влияет на нефть и общий риск
    "Gold": (0.8, ["gold"]), # Слегка повышаем вес, чтобы модель уделяла больше внимания золоту
    "BTC": (1.2, ["btc", "global"]), # Повышен вес для учета высокой волатильности
    "AI": (1.5, ["nasdaq", "soxs", "global"]),
    "SOXX": (1.4, ["soxs", "nasdaq", "global"]),
    "Computing": (1.3, ["nasdaq", "soxs", "global"]),
    "Nasdaq": (1.0, ["nasdaq"]),
    "AI Sector": (1.3, ["nasdaq", "soxs", "global"]),
    "AI Infrastructure": (1.4, ["nasdaq", "soxs", "global"]),
    "Trump Policy": (2.2, ["global", "nasdaq", "sp500", "oil", "vix"]),
    "MU": (1.2, ["soxs", "global"]),
    "Semiconductor": (1.5, ["soxs", "nasdaq", "global"]),
    "US Inflation": (2.0, ["global", "vix", "gold"]),
    "Intel": (1.3, ["soxs", "global"]),
    "AMD": (1.3, ["soxs", "global"]),
    "Broadcom": (1.2, ["soxs", "global"]),
    "Anthropic": (1.5, ["soxs", "nasdaq", "global"]), 
    "Qualcomm": (1.2, ["soxs", "global"]),
    "Fed": (2.2, ["global", "vix", "sp500", "nasdaq", "gold"]),
    "Hormuz": (2.0, ["oil", "vix", "global"]), # Фокус на геополитике в регионе
    "Yield": (1.8, ["global", "vix", "nasdaq"]), # Влияние на общий риск, волатильность и тех. сектор
    "Treasury": (1.5, ["global", "vix", "nasdaq"]), # Влияние на общий риск, волатильность и тех. сектор
    "HBM": (1.5, ["soxs", "nasdaq", "global"]),
    "HBM Memory": (1.5, ["soxs", "nasdaq", "global"]), # Ключевой компонент для производства AI-ускорителей
    "Inflation": (2.0, ["global", "vix", "gold", "nasdaq", "sp500"]), # Добавлены макроэкономические факторы
    "Interest Rates": (2.2, ["global", "vix", "nasdaq", "sp500"]),
    "Recession": (2.5, ["global", "vix", "gold", "nasdaq", "sp500"]),
    "Geopolitical Tension": (2.5, ["global", "oil", "vix", "gold"]),
    "Earnings": (1.5, ["nasdaq", "sp500"]),
    "Tech Earnings": (1.8, ["nasdaq", "soxs", "global"]),
    "Tech Upgrade": (1.7, ["nasdaq", "soxs", "global"]), # Повышение рейтингов акций тех. сектора
    "Tech Downgrade": (1.7, ["nasdaq", "soxs", "global"]), # Понижение рейтингов акций тех. сектора
    "Analyst Rating Tech": (1.5, ["nasdaq", "soxs", "global"]), # Аналитические рейтинги в тех. секторе
    "Investment Firm Tech": (1.3, ["nasdaq", "soxs", "global"]) # Новости от инвест. фирм по тех. сектору
}

# Нормализация сущностей для формирования консистентных ключей (event_key)
ENTITY_CANONICAL_MAP = {
    "USA": "US", "UNITED STATES": "US",
    "FED": "FED", "FEDERAL RESERVE": "FED", "US_FEDERAL_RESERVE": "FED", "POWELL": "FED", "JEROME_POWELL": "FED",
    "ФРС": "FED", "ФРС_США": "FED", "БЕЖЕВАЯ_КНИГА": "FED",
    "ECB": "ECB", "EUROPEAN_CENTRAL_BANK": "ECB", "LAGARDE": "ECB",
    "BOJ": "BOJ", "BANK_OF_JAPAN": "BOJ", "YEN": "BOJ",
    "IRAN": "IRAN", "IRANIAN": "IRAN", "TEHRAN": "IRAN",
    
    "BITCOIN": "BTC", "BTC": "BTC",
    "GOLD": "GOLD", "XAU": "GOLD",
    "OIL": "OIL", "CRUDE": "OIL",
    
    "NVDA": "NVIDIA", "NVIDIA": "NVIDIA", "BLACKWELL": "NVIDIA", "H100": "NVIDIA",
    "AMD": "AMD", "ADVANCED_MICRO_DEVICES": "AMD",
    "INTC": "INTEL", "INTEL": "INTEL",
    "AVGO": "BROADCOM", "BROADCOM": "BROADCOM",
    "ASML": "ASML", "ASML_HOLDING": "ASML",
    "TSMC": "TSM", "TSM": "TSM", "TAIWAN_SEMICONDUCTOR": "TSM",
    
    "HBM": "HBM", "HIGH_BANDWIDTH_MEMORY": "HBM", "SK_HYNIX": "HBM",
    "OPENAI": "OPENAI", "CHATGPT": "OPENAI", "SAM_ALTMAN": "OPENAI",
    "COMPUTE": "AI_INFRASTRUCTURE", "COMPUTING": "AI_INFRASTRUCTURE",
    "ANTHROPIC": "ANTHROPIC", "CLAUDE": "ANTHROPIC",
    
    "DONALD_TRUMP": "TRUMP", "MAGA": "TRUMP",
    "ALPHABET": "GOOGLE", "GOOGL": "GOOGLE",
    "MICROSOFT": "MSFT",
    
    "CPI": "INFLATION", "PCE": "INFLATION", "INFLATION": "INFLATION", "CONSUMER_PRICE_INDEX": "INFLATION",
    "PPI": "INFLATION", "PRODUCER_PRICE_INDEX": "INFLATION", "GDP": "ECONOMY", "GROWTH": "ECONOMY",
    "NONFARM_PAYROLLS": "EMPLOYMENT", "NFP": "EMPLOYMENT", "JOBS": "EMPLOYMENT", "UNEMPLOYMENT": "EMPLOYMENT",
    "YIELD": "TREASURY", "TREASURY": "TREASURY", "BOND": "TREASURY", "10Y": "TREASURY",
    
    "HORMUZ": "GEOPOLITICS_ME", "RED_SEA": "GEOPOLITICS_ME", "MIDDLE_EAST": "GEOPOLITICS_ME",
    "TAIWAN": "GEOPOLITICS_ASIA", "SOUTH_CHINA_SEA": "GEOPOLITICS_ASIA",
    "UKRAINE": "GEOPOLITICS_EU", "RUSSIA": "GEOPOLITICS_EU",
    
    "SEMICONDUCTOR": "SOXS",
    "OPEC": "OIL_SUPPLY", "INVENTORIES": "OIL_SUPPLY",
    "SHALE": "OIL_SUPPLY",
    
    "UPGRADE": "UPGRADE",
    "DOWNGRADE": "DOWNGRADE", "DECLINE": "DROP", "FALL": "DROP", "DECREASE": "DROP", "LOWER": "DROP",
    "INCREASE": "RISE", "SURGE": "RISE", "JUMP": "RISE", "HIGHER": "RISE",
    "EARNINGS": "EARNINGS", "QUARTERLY_RESULTS": "EARNINGS"
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




# Дополнительные прямые RSS-ленты для расширения охвата рынка
DIRECT_RSS_FEEDS = [
    "https://finance.yahoo.com/news/rss", # Общая лента финансовых новостей
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA,AMD,AVGO,TSM,INTC,MU&region=US&lang=en-US", # Лента для полупроводников
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MU,000660.KS,005930.KS&region=US&lang=en-US", # Лента для памяти (HBM)
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ASML,AMAT,LRCX,KLAC&region=US&lang=en-US", # Лента для оборудования для производства чипов
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA,MSFT,GOOGL,AMZN,META&region=US&lang=en-US", # Лента для AI и крупных технологических компаний
    "https://news.google.com/rss/search?q=site:bloomberg.com+economics&hl=en-US", # Экономика Bloomberg через Google
    "https://news.google.com/rss/search?q=site:bloomberg.com+markets&hl=en-US", # Рынки Bloomberg через Google
    "https://news.google.com/rss/search?q=site:reuters.com+world&hl=en-US", # Мировые новости Reuters через Google
    "https://news.google.com/rss/search?q=site:reuters.com+business&hl=en-US", # Бизнес новости Reuters через Google
    "https://news.google.com/rss/search?q=site:reuters.com+technology+semiconductor+OR+AI&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=site:reuters.com+commodities&hl=en-US", # Сырьевые товары Reuters через Google
    "https://news.google.com/rss/search?q=site:maritime-executive.com&hl=en-US", # Морские новости через Google (Ормузский пролив и логистика)
    "https://www.federalreserve.gov/feeds/press_monetary.xml", # Лента для новостей Федеральной резервной системы США
    "https://news.google.com/rss/search?q=site:home.treasury.gov+press+releases&hl=en-US", # Новости Минфина США через Google
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=40&output=atom", # Последние отчеты SEC EDGAR
    "https://news.google.com/rss/search?q=Deltaone+news&hl=en-US", # Замена Twitter на поиск новостей Deltaone
    "https://news.google.com/rss/search?q=unusual+whales+macro&hl=en-US", # Замена Twitter Unusual Whales
    "https://news.google.com/rss/search?q=FirstSquawk+news&hl=en-US",
    "https://www.tomshardware.com/feeds.xml", # Лента для новостей о технологиях и полупроводниках от Tom's Hardware
    "https://www.tomshardware.com/feeds/tag/semiconductors", # Лента для новостей о полупроводниках от Tom's Hardware
    "https://www.tomshardware.com/feeds/tag/artificial-intelligence", # Лента для новостей об искусственном интеллекте от Tom's Hardware
    "https://www.trendforce.com/feed/Semiconductors.html", # Лента для новостей о полупроводниках от TrendForce
    "https://www.digitimes.com/rss/daily.xml", # Лента для новостей о технологиях и полупроводниках от DigiTimes
    "https://www.theregister.com/software/ai_ml/headlines.atom", # Лента для новостей об ИИ от The Register
    # "https://www.eetimes.com/feed/", # Общая лента для новостей о технологиях от EE Times
    # "https://www.eetimes.com/tag/semiconductors/feed/" # Лента для новостей о полупроводниках от EE Times
    "https://news.google.com/rss/search?q=site:eetimes.com&hl=en-US", # Общая лента EE Times через Google
    "https://news.google.com/rss/search?q=site:eetimes.com+semiconductors&hl=en-US" # Лента о полупроводниках от EE Times через Google
]

# Настройки фильтрации источников
ONLY_SPECIFIC_SOURCES = True # Теперь только доверенные источники (Reuters, Bloomberg и т.д.)
SPECIFIC_SOURCES_LIST = [
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "federalreserve.gov",
    "treasury.gov",
    "sec.gov",
    "ecb.europa.eu",
    "asia.nikkei.com",
    "economist.com",

    # SEMI_SOURCES
    "semianalysis.com",
    "trendforce.com",
    "tomshardware.com",
    "digitimes.com",
    "theregister.com",
    "eetimes.com",
    "anandtech.com",
    "techpowerup.com",

    # FAST_SIGNAL
    "reddit.com",
    "stocktwits.com",
    "wccftech.com",

    # GEO_SOURCES
    "ukmto.org",
    "maritime-executive.com",
    "gcaptain.com",
    "hellenicshippingnews.com",
    "benzinga.com",
    "investors.com",
    "barrons.com",
    "seekingalpha.com"
]

# Настройки для поиска в соцсетях
SOCIAL_SEARCH_ENABLED = False # Временно отключаем, так как RSS от соцсетей может быть шумным и требует доработки фильтров
# Список надежных Nitter-инстансов (для Twitter RSS без API ключа)
NITTER_INSTANCES = [
    "nitter.net",
     "nitter.it",
     "nitter.poast.org",
    #   "nitter.privacydev.net",
    #    "nitter.no-logs.com",
        # "nitter.projectsegfau.lt"
        ]

# Логика формирования целевых запросов Google News
RSS_FEEDS = []
GOOGLE_BASE_URL = "https://news.google.com/rss/search?q="

if ONLY_SPECIFIC_SOURCES:
    _news_domains = [d for d in SPECIFIC_SOURCES_LIST if d not in ["x.com", "twitter.com", "reddit.com", "stocktwits.com"]]
    
    # Разбиваем список доменов на чанки по 10 штук, чтобы не превысить лимит длины URL
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    _domain_chunks = list(chunk_list(_news_domains, 10))
    
    for k in TRACKED_KEYWORDS.keys():
        keyword_q = k.replace(' ', '+')
        for chunk in _domain_chunks:
            _chunk_query = "+(" + "+OR+".join([f"site:{d}" for d in chunk]) + ")"
            RSS_FEEDS.append(f"{GOOGLE_BASE_URL}{keyword_q}{_chunk_query}+when:6h")

else:
    RSS_FEEDS = [f"{GOOGLE_BASE_URL}{k.replace(' ', '+')}+when:6h" for k in TRACKED_KEYWORDS.keys()]

# Добавляем поиск по соцсетям, если включено (теперь вне зависимости от ONLY_SPECIFIC_SOURCES)
if SOCIAL_SEARCH_ENABLED:
    for k in TRACKED_KEYWORDS.keys():
        keyword_q = k.replace(' ', '+')
        # Reddit Search RSS
        RSS_FEEDS.append(f"https://www.reddit.com/search.rss?q={keyword_q}&sort=new&t=hour")

RSS_FEEDS += DIRECT_RSS_FEEDS
RSS_MAX_ENTRIES = 25 # Увеличено для более глубокого сканирования активного рынка
RSS_MAX_ENTRIES_INACTIVE = 40 # Увеличено для более глубокого охвата за ночь

# Time Intervals (in seconds)
CHECK_INTERVAL = 420 # Увеличено до 7 минут, чтобы снизить риск Rate Limit (429)
COOLDOWN = 480 # Снижаем до 8 минут для более частых обновлений по сюжету
LEARNING_INTERVAL = 1800 # 30 минут — оптимально для накопления выборки цен
MARKET_LOOKBACK_HOURS = 2 # Увеличиваем до 2 часов: macro-alpha требует времени для проявления
MAX_NEWS_AGE_HOURS = 2 # Только свежие новости (было 4ч — пропускало старый backlog из RSS)
MAX_NEWS_AGE_HOURS_INACTIVE = 6 # Ночью (было 12ч)
# Не оценивать новости, опубликованные до старта движка (кроме grace для RSS-задержки)
ONLY_NEWS_AFTER_STARTUP = _env_bool("ONLY_NEWS_AFTER_STARTUP", True)
STARTUP_GRACE_MINUTES = int(os.getenv("STARTUP_GRACE_MINUTES", "20"))
DB_DEDUP_LOOKBACK_HOURS = 72  # Окно fuzzy-проверки по заголовкам из БД
DB_DEDUP_TITLE_LIMIT = 500    # Макс. заголовков из БД для fuzzy dedup
DB_URL_CACHE_LIMIT = 3000     # Сколько URL подгружать из БД при старте

# Адаптивные задержки обучения (в часах) в зависимости от типа события
EVENT_TYPE_LOOKBACK = {
    "military":   {"primary": 1, "secondary": 8},
    "economic":   {"primary": 1, "secondary": 6},
    "diplomatic": {"primary": 2, "secondary": 8},
    "tech":       {"primary": 1.5, "secondary": 4},
    "neutral":    {"primary": 2, "secondary": 4},
}
BLACK_SWAN_LOOKBACK_HOURS = 24 # Окно для оценки фундаментального сдвига при ЧП

CLEANUP_INTERVAL = 86400 # Интервал очистки (24 часа)
RESEARCH_INTERVAL = 86400 # Интервал глобального исследования ИИ (раз в сутки)
RETENTION_DAYS = 30 # Увеличено до месяца, чтобы система помнила начало затяжных конфликтов
EMBEDDING_RETENTION_DAYS = 1 # Векторы нужны только для дедупликации (1-3 дня достаточно)

# Параметры подгрузки контекста в RAM при старте
RAM_SCORE_LOOKBACK_DAYS = 7 # Загружаем баллы за неделю, чтобы видеть накопленный фон события
RAM_EMBEDDING_LOOKBACK_DAYS = 3
SLUG_DUPLICATE_HOURS = 48 # Увеличено до 2 суток, чтобы блокировать повторные обсуждения старых событий

# Dynamic Narrative Discovery
WEIGHT_DECAY_FACTOR = 0.999 # Ежедневный коэффициент затухания весов (EWMA) для борьбы с переобучением
USE_NARRATIVE_TRACKING = True 
NARRATIVE_AUTO_DISCOVERY = True # Автоматическое добавление новых сущностей в веса
MIN_NARRATIVE_STREAK = 3 # Сколько раз тема должна появиться за окно, чтобы стать ключом
NARRATIVE_DISCOVERY_WINDOW = 24 # Окно поиска новых тем (в часах)
NARRATIVE_BOOST_PER_HIT = 0.2 # +20% к силе новости за каждое повторение темы
NARRATIVE_MAX_MULTIPLIER = 2.0 # Максимальное усиление (2x)

# AI Delays
AI_DELAY_JSON = 3 # Немного сокращаем ожидание для повышения пропускной способности
AI_DELAY_NO_JSON = 10 # Задержка для тяжелых/медленных моделей
DEFAULT_AI_BATCH_SIZE = 25 # Размер пакета по умолчанию. Используется для провайдеров, не указанных в PROVIDER_BATCH_SIZES (например, DeepSeek). Увеличение экономит токены.
AI_BATCH_WAIT_SECONDS = 60 # Время ожидания для накопления пакета новостей
PROVIDER_BATCH_SIZES = {
    "gemini": 3,
    "groq": 5,
    "cerebras": 5,
    "nvidia": 5,
    "openai": 5,
    "openrouter": 5,
    "deepseek": 35 # Можно задать явно здесь
}
ONLY_PRIORITY_GEMINI = True # Если True, используются только модели Gemini из family_priority

# Concurrency Settings
GEMINI_CONCURRENCY = 1 # Бесплатный тариф требует последовательных запросов
GROQ_CONCURRENCY = 2
CEREBRAS_CONCURRENCY = 2
OPENROUTER_CONCURRENCY = 5 # Платные/быстрые модели могут обрабатываться параллельно
DEEPSEEK_CONCURRENCY = 2 # Лимит для DeepSeek API

ENABLE_HOURLY_REPORT = False # Включить/выключить отправку часового отчета в Telegram
HOURLY_SUMMARY_INTERVAL = 3600 # Интервал отправки часового отчета в Telegram (1 час)
NUM_WORKERS = 2 # Увеличиваем до 4, так как DeepSeek и OpenRouter могут работать параллельно с Gemini

# Logic Factors
ENABLE_TRIVIAL_FILTER = False # Включить/выключить фильтр тривиальных новостей
DECAY_FACTOR = 0.9 # Оптимальный баланс: новость сохраняет 50% силы через 15-20 минут и затухает за 2-3 часа.
NIGHT_DECAY_FACTOR = 0.98 # Почти не снижаем балл, когда рынок закрыт, чтобы сохранить контекст к открытию
MAX_SCORE_THRESHOLD = 25.0
DECAY_REFERENCE_SECONDS = 180 # Базовый интервал времени для расчета затухания

BLACK_SWAN_SCORE_THRESHOLD = 7.0 # Порог индивидуального скора новости для подтверждения статуса Black Swan
LEARNING_RATE = 0.005 # Повышаем шаг обучения для более быстрой адаптации к смене режима
ASYMMETRIC_LR_FACTOR = 3.0 # Более агрессивная коррекция при ошибке в направлении (is_correct = False)

# Multiplier Reset Logic
MIN_WINRATE_BEFORE_RESET = 40.0 # Порог WinRate (%), ниже которого множитель актива сбрасывается
MIN_SAMPLE_SIZE_FOR_RESET = 15 # Увеличим выборку для более точного сброса

IMPACT_MULTIPLIER = 0.3 # Базовая чувствительность к Z-score (Sigmas)
LEARNING_THRESHOLD = 0.7 # Минимальная уверенность ИИ для обучения на событии
PIVOT_THRESHOLD = 5.0 # Порог "разворотной" новости, при котором накопленный балл обнуляется
TRIVIAL_SCORE_THRESHOLD = 0.05 # Порог для отсеивания тривиальных новостей (балл от ИИ)
MIN_WEIGHT_THRESHOLD = 0.7 # Ослабляем порог для сохранения большего числа связей
NEUTRAL_SCORE_THRESHOLD = 1.8 # СНИЖАЕМ ПОРОГ, чтобы пропускать больше новостей
MAX_ENTITY_PARTS = 3 # Увеличено до 3, чтобы лучше обрабатывать сложные Slug от ИИ
DUPLICATE_TITLE_THRESHOLD = 0.82 # Баланс: ловит перефразировки, меньше ложных срабатываний на шаблонные статьи
FALLBACK_DUPLICATE_THRESHOLD = 0.55
FUZZY_MIN_WORD_JACCARD = 0.35 # Мин. пересечение слов — отсекает шаблоны MarketsMojo и т.п.
SEMANTIC_DEDUPLICATION_WINDOW = 720 # Увеличено до 30 дней (720ч) для борьбы с ре-индексацией старых новостей
SEMANTIC_DUPLICATE_THRESHOLD = 0.85 # Снижаем порог для более агрессивного поиска семантических дублей
USE_EMBEDDINGS = True # Включить/выключить семантическую дедупликацию через векторы
EMBEDDING_MODEL = "models/gemini-embedding-2" # Основная модель эмбеддингов (Gemini)
CONFIDENCE_THRESHOLD = 0.25 # СНИЖАЕМ ПОРОГ, чтобы дать шанс новостям с умеренной уверенностью
SLUG_SPAM_WINDOW = 7200 # 2 часа: если новость с тем же Slug пришла быстрее, она игнорируется как дубль

NON_FINANCIAL_SCORE_DECAY_FACTOR = 0.5 # Коэффициент снижения балла для нефинансовых/дипломатических новостей
# Рейтинг доверия источникам (Trust Factor)
SOURCE_TRUST_LEVELS = {
    # Official / primary
    "sec.gov": 1.15,
    "federalreserve.gov": 1.15,
    "treasury.gov": 1.15,

    # Tier 1 financial journalism
    "reuters.com": 1.0,
    "bloomberg.com": 1.0,
    "ft.com": 0.96,
    "wsj.com": 0.95,

    # Semiconductor specialists
    "semianalysis.com": 0.92,
    "trendforce.com": 0.87,
    "tomshardware.com": 0.82,
    "anandtech.com": 0.9,
    "servethehome.com": 0.84,

    # Supply chain / rumor-heavy
    "digitimes.com": 0.74,

    # Social
    "x.com": 0.25,
    "reddit.com": 0.3,
    "msn.com": 0.0,
    "aol.com": 0.0,
    "newsonair.gov.in": 0.0,
    "energynow.ca": 0.0,
    "energynow.com": 0.0
}
DEFAULT_TRUST_SCORE = 0.65  # Немного снижаем базу для фильтрации случайных источников

# Thresholds for market signals (Empirical sensitivity)
SIGNAL_THRESHOLD_HIGH = 3.5  # Повышаем порог для индексов, чтобы уменьшить количество ложных алертов
SIGNAL_THRESHOLD_MED = 2.5   # Повышено для VIX и Oil для фильтрации шума
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)
SIGNAL_THRESHOLD_BTC = 4.0   # For Crypto (Volatility buffer)
BTC_MIN_VOLATILITY_FOR_ALERT = 1.0 # Минимальное изменение цены BTC (%) для отправки уведомления
SHARP_MOVE_THRESHOLDS = {
    "oil": 2.0,
    "btc": 5.0,
    "nasdaq": 1.5
}

# Матрица корреляций активов относительно стандартного Risk-Off score от ИИ.
# -1: Актив падает при Risk-Off (Score > 0) -> например, NASDAQ.
#  1: Актив растет при Risk-Off (Score > 0) -> например, VIX, Gold, SOXS.
ASSET_CORRELATION_MAP = {
    "nasdaq": -1,
    "sp500": -1,
    "oil": 1,
    "vix": 0,    # Нейтральная корреляция: пусть Z-Alpha определяет направление
    "gold": 0,   # Нейтральная корреляция: золото зависит от DXY и ставок, а не только от риска
    "btc": -1,
    "soxs": 1,
    "global": 1  # Global Regime (Stress) растет при Risk-Off
}

# Конфигурация бенчмарков для расчета Alpha (Abnormal Return)
ASSET_BENCHMARK_CONFIG = {
    "nasdaq": {"primary": "^GSPC", "type": "rolling_beta"}, 
    "sp500":  {"primary": "ACWI", "type": "rolling_beta"}, # Смена на ACWI (MSCI World)
    "soxs":   {"primary": "SOXX", "type": "leveraged", "factor": -3.0}, # Прямая связь с полупроводниками
    "btc":    {"primary": "^GSPC", "type": "rolling_beta"}, # Меняем на динамическую бету к широкому рынку (S&P 500)
    # "oil":    {"primary": "DX-Y.NYB", "type": "rolling_beta"}, 
    "oil":    {"primary": "CL=F", "type": "fixed", "factor": 1.0}, # Для нефти оставляем простую модель
    "gold":   {"primary": "TIP", "secondary": "DX-Y.NYB", "type": "multi_factor", "weights": [0.5, -0.5]}, # 50% реальные ставки, 50% обратная корреляция с DXY
    # GLOBAL_REGIME — целевой ряд (факт); бенчмарк — широкий рынок (ожидание через rolling beta).
    # Раньше primary=GLOBAL_REGIME давал Z-Alpha=0 всегда (actual == expected).
    "global": {"primary": "^GSPC", "type": "rolling_beta"},
}

# Веса для композитного режима Global Regime
GLOBAL_REGIME_WEIGHTS = {
    "vix": 0.25,      # Equity Stress (Снижаем вес, т.к. VIX бывает обманчив)
    "move": 0.15,     # Bond Stress (^MOVE)
    "dxy": 0.15,      # Liquidity (Dollar Index) (Оставляем)
    "hyg": 0.25,      # Credit Stress (High Yield Corp) - Inverted (Повышаем, это важный индикатор)
    "growth": 0.20    # Growth Expectations (Yield Curve 10Y-3M) - Inverted (Повышаем, ключевой макро-фактор)
}
CLEANUP_INTERVAL = 86400 # Интервал очистки базы (24 часа)

# Параметры Квантовой Модели
EWMA_LAMBDA = 0.94 # Параметр затухания RiskMetrics
BETA_CLIP = 3.0 # Ограничение экстремальных значений беты
VOLATILITY_WINDOW = 40 # Окно для расчета реализованной волатильности
Z_ALPHA_VOL_FLOOR = 0.15 # Минимальная волатильность для расчета Z-Alpha (чтобы не учиться на микрошуме)
GLOBAL_Z_ALPHA_VOL_FLOOR = 0.5 # Минимальная волатильность для расчета глобального Z-Alpha (чтобы не учиться на микрошуме)
ALPHA_MIN_THRESHOLD = 0.05 

# --- ПАРАМЕТРЫ СТРАТЕГИИ SOXS v5.0 (BEAR PROBABILITY) ---

# Веса компаний для расчета совокупного CAPEX (сумма весов = 1.0)
SOXS_CAPEX_WEIGHTS = {
    "MSFT": 0.35,
    "META": 0.25,
    "AMZN": 0.25,
    "GOOGL": 0.15
}

# Веса компаний для расчета совокупного GUIDANCE чипмейкеров (сумма весов = 1.0)
SOXS_GUIDANCE_WEIGHTS = {
    "NVDA": 0.45,
    "AVGO": 0.25,
    "AMD": 0.15,
    "MU": 0.15
}

# Сила влияния компонентов на итоговый Bear Score
SOXS_FACTOR_WEIGHTS = {
    "capex": 25.0,
    "guidance": 30.0,
    "divergence": 20.0,    # Максимальное значение 1.0 (шкала 0..10 / 10)
    "rotation": 15.0,      # Максимальное значение 1.0 (шкала 0..10 / 10)
    "ma200_trend": 10.0    # Бонус за нахождение SOXX ниже MA200
}

# Пороговые значения вероятностей для изменения размера позиции
SOXS_POSITION_LEVELS = [
    {"limit": 30.0, "position": 0.0, "name": "0% Position (Bullish Regime)"},
    {"limit": 50.0, "position": 20.0, "name": "20% Position (Probationary Tactical Hedge)"},
    {"limit": 70.0, "position": 50.0, "name": "50% Position (Core Tactical Hedge)"},
    {"limit": 85.0, "position": 100.0, "name": "100% Position (Full Market Protection Plan)"},
    {"limit": 100.1, "position": 120.0, "name": "Aggressive Entry (120%+ Leveraged Cyclical Top)"}
]

# SOXS probability calibration (logistic on historical bear_score vs SOXX forward returns)
SOXS_CALIBRATION_MIN_SAMPLES = 20
SOXS_CALIBRATION_FORWARD_HOURS = 24
SOXS_CALIBRATION_BEAR_THRESHOLD_PCT = 1.0  # SOXX drop >= 1% over forward window => bearish label
SOXS_CALIBRATION_DEFAULT_COEF = 0.05       # Fallback linear-ish mapping when not enough data
SOXS_CALIBRATION_DEFAULT_INTERCEPT = 0.0
SOXS_CALIBRATION_INTERVAL = 86400          # Re-fit calibrator once per day
SOXS_CALIBRATION_BASELINE_PROB = 30.0      # bear_score=0 -> 30% (bullish baseline)
SOXS_CALIBRATION_MIN_COEF = 0.01           # Below this the fitted model is rejected
SOXS_CALIBRATION_HOLDOUT_HOURS = 48        # Exclude recent snapshots from calibration fit
SOXS_USE_LEGACY_CALIBRATION_ROWS = False   # Skip rows without bear_score when fitting

# SOXS position management
SOXS_POSITION_HYSTERESIS_HOURS = 48        # Min time between applied position changes
SOXS_POSITION_CONFIRM_CYCLES = 2           # Consecutive cycles at new level before applying
SOXS_REGIME_GATE_MAX_POSITION = 20.0       # Cap without SOXX below MA200 or strong fundamentals
SOXS_REGIME_GATE_MIN_BEAR_SCORE = 15.0     # Allow > gate max when bear score is high enough
SOXS_SNAPSHOT_INTERVAL = 3600              # Save quant snapshot every hour (even if position unchanged)

# SOXS divergence indicator (Price Confirmation v2)
SOXS_DIVERGENCE_MIN_SCORE = 4.0
SOXS_DIVERGENCE_TARGET_ASSETS = ("soxs", "nasdaq")
SOXS_DIVERGENCE_GUIDANCE_WEIGHT = 2
SOXS_DIVERGENCE_CAPEX_WEIGHT = 2
SOXS_DIVERGENCE_NEWS_WEIGHT = 1
SOXS_DIVERGENCE_LOOKBACK_DAYS = 10

# SOXS calibration label: multi-day forward window (trading days on daily_prices)
SOXS_CALIBRATION_FORWARD_DAYS = 5          # Label window for cyclical hedge (5 trading days)
SOXS_CALIBRATION_USE_MULTI_DAY_LABEL = True
SOXS_CALIBRATION_MULTI_DAY_THRESHOLD_PCT = 2.0  # SOXX drop >= 2% over N days => bearish
SOXS_CALIBRATION_VALIDATION_RATIO = 0.25   # Final chronological block used out-of-sample
SOXS_CALIBRATION_MIN_VALIDATION_SAMPLES = 5
SOXS_CALIBRATION_MIN_BEARISH_TRAIN = 2
SOXS_CALIBRATION_MIN_BEARISH_VALIDATION = 1
SOXS_CALIBRATION_EMBARGO_DAYS = 1          # Gap between fitted and validation samples
SOXS_CALIBRATION_MIN_BRIER_IMPROVEMENT = 0.0
SOXX_CALIBRATION_MAX_ABS_DAILY_RETURN = 0.5  # Reject likely unadjusted splits in labels

# Deployment promotion gate. A fitted calibrator is necessary but not sufficient:
# actionable position alerts require explicit opt-in and stronger OOS evidence.
SOXS_ACTIONABLE_SIGNALS = _env_bool("SOXS_ACTIONABLE_SIGNALS", False)
SOXS_ACTIONABLE_MIN_SAMPLES = max(
    SOXS_CALIBRATION_MIN_SAMPLES,
    int(os.getenv("SOXS_ACTIONABLE_MIN_SAMPLES", "100")),
)
SOXS_ACTIONABLE_MIN_VALIDATION_SAMPLES = max(
    SOXS_CALIBRATION_MIN_VALIDATION_SAMPLES,
    int(os.getenv("SOXS_ACTIONABLE_MIN_VALIDATION_SAMPLES", "20")),
)
SOXS_ACTIONABLE_MIN_BRIER_IMPROVEMENT = max(
    0.0,
    float(os.getenv("SOXS_ACTIONABLE_MIN_BRIER_IMPROVEMENT", "0.01")),
)
# Transitional hard cap for the 3x daily-reset ETF. Raising the env value cannot
# bypass it; changing the ceiling requires a deliberate reviewed code change.
SOXS_MAX_ACTIONABLE_POSITION_PCT = min(
    20.0,
    max(0.0, float(os.getenv("SOXS_MAX_ACTIONABLE_POSITION_PCT", "20"))),
)

# Reproducible SOXS strategy backtest assumptions
SOXS_BACKTEST_COMMISSION_BPS = 1.0
SOXS_BACKTEST_SLIPPAGE_BPS = 5.0
SOXS_BACKTEST_PURGE_DAYS = SOXS_CALIBRATION_FORWARD_DAYS
SOXS_BACKTEST_EMBARGO_DAYS = 1
SOXS_BACKTEST_MAX_ABS_DAILY_RETURN = 1.0  # >100% usually means an unadjusted split discontinuity

# Validated historical-price ingestion. Twelve Data is used when a key exists;
# direct Yahoo Chart and yfinance are independent fallback transports.
HISTORICAL_PRICE_PROVIDERS = tuple(
    provider.strip().lower()
    for provider in os.getenv(
        "HISTORICAL_PRICE_PROVIDERS", "twelvedata,yahoo_chart,fred,yfinance"
    ).split(",")
    if provider.strip()
)
# Twelve Data uses different symbols for several Yahoo index series.
TWELVEDATA_SYMBOL_MAP = {
    "^MOVE": "MOVE",
}
# These Yahoo indices are not available from the configured Twelve Data plan.
# Skip known 404 requests and use the Yahoo/yfinance fallback chain instead.
TWELVEDATA_UNSUPPORTED_TICKERS = {"DX-Y.NYB", "^TNX", "^IRX"}
YAHOO_DAILY_MAX_MISSING_CLOSES = 5
# Public FRED fallbacks. DXY uses the Fed broad dollar index as an explicitly
# documented proxy; the yield series are expressed in percentage points.
FRED_SERIES_MAP = {
    "DX-Y.NYB": "DTWEXBGS",
    "^TNX": "DGS10",
    "^IRX": "DGS3MO",
}
PRICE_HISTORY_OUTPUT_SIZE = 5000
PRICE_HISTORY_REQUEST_TIMEOUT = 30
# A rate-limit response from a paid/free market-data API is transient.  Keep
# retries bounded so one exhausted quota cannot stall the scheduler forever.
TWELVEDATA_MAX_RETRIES = min(
    4, max(0, int(os.getenv("TWELVEDATA_MAX_RETRIES", "2")))
)
TWELVEDATA_RETRY_BASE_SECONDS = max(
    0.25, float(os.getenv("TWELVEDATA_RETRY_BASE_SECONDS", "2"))
)
TWELVEDATA_RETRY_MAX_SECONDS = max(
    TWELVEDATA_RETRY_BASE_SECONDS,
    float(os.getenv("TWELVEDATA_RETRY_MAX_SECONDS", "30")),
)
PRICE_HISTORY_MIN_ROWS = 20
PRICE_HISTORY_CORE_MIN_ROWS = 200
PRICE_HISTORY_MAX_AGE_DAYS = 10
PRICE_HISTORY_REFRESH_AFTER_DAYS = 3
PRICE_HISTORY_FUTURE_TOLERANCE_DAYS = 1
PRICE_HISTORY_MAX_ABS_DAILY_RETURN = 2.0
SOXS_TRACKING_CHECK_MIN_ABS_RETURN = 0.50
SOXS_TRACKING_MAX_RESIDUAL = 0.30
PRICE_HISTORY_RAM_DAYS = 400
PRICE_HISTORY_US_CLOSE_GRACE_MINUTES = 15
MARKET_INTRADAY_MAX_CONCURRENCY = 6
YFINANCE_CACHE_DIR = os.getenv("YFINANCE_CACHE_DIR", ".cache/yfinance")

# SOXS signal lookback for capex/guidance aggregation
SOXS_SIGNAL_LOOKBACK_DAYS = 15

# Walk-forward learning: exclude recent predictions from weight/multiplier updates
WALK_FORWARD_HOLDOUT_HOURS = 48            # Minimum age before applying learning updates
WALK_FORWARD_MIN_CALIBRATION_AGE = 6       # Minimum age before multiplier calibration (phase 1)
