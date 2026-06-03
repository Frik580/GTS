import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MARKET_DATA_API_KEY = os.getenv("MARKET_DATA_API_KEY") # Ключ от TwelveData или др.
# Провайдер данных: "twelvedata" или "yfinance" (фоллбек)
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "yfinance")

# Network Settings
HTTP_PROXY = os.getenv("HTTP_PROXY") # Например: "http://user:pass@ip:port"
USE_PROXY = True if HTTP_PROXY else False

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
    "GDP": "ECONOMY", "GROWTH": "ECONOMY",
    "NONFARM_PAYROLLS": "EMPLOYMENT", "NFP": "EMPLOYMENT", "JOBS": "EMPLOYMENT", "UNEMPLOYMENT": "EMPLOYMENT",
    "YIELD": "TREASURY", "TREASURY": "TREASURY", "BOND": "TREASURY", "10Y": "TREASURY",
    
    "HORMUZ": "GEOPOLITICS_ME", "RED_SEA": "GEOPOLITICS_ME", "MIDDLE_EAST": "GEOPOLITICS_ME",
    "TAIWAN": "GEOPOLITICS_ASIA", "SOUTH_CHINA_SEA": "GEOPOLITICS_ASIA",
    "UKRAINE": "GEOPOLITICS_EU", "RUSSIA": "GEOPOLITICS_EU",
    
    "SEMICONDUCTOR": "SOXS",
    "OPEC": "OIL_SUPPLY", "INVENTORIES": "OIL_SUPPLY",
    "SHALE": "OIL_SUPPLY",
    
    "UPGRADE": "UPGRADE",
    "DOWNGRADE": "DOWNGRADE",
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




# Основные RSS-ленты Yahoo Finance для расширения охвата рынка
YAHOO_FINANCE_FEEDS = [
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
    "x.com",
    "twitter.com",
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
NITTER_INSTANCES = ["nitter.net", "nitter.it", "nitter.privacydev.net"]

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

    # Добавляем поиск по соцсетям, если включено
    if SOCIAL_SEARCH_ENABLED:
        for k in TRACKED_KEYWORDS.keys():
            keyword_q = k.replace(' ', '+')
            # Reddit Search RSS
            RSS_FEEDS.append(f"https://www.reddit.com/search.rss?q={keyword_q}&sort=new&t=hour")
            # Twitter via Nitter RSS (берем первый инстанс для примера)
            RSS_FEEDS.append(f"https://{NITTER_INSTANCES[0]}/search/rss?f=tweets&q={keyword_q}")
            # StockTwits не имеет RSS, он будет обрабатываться отдельным методом в engine.py

else:
    RSS_FEEDS = [f"{GOOGLE_BASE_URL}{k.replace(' ', '+')}+when:6h" for k in TRACKED_KEYWORDS.keys()]

RSS_FEEDS += YAHOO_FINANCE_FEEDS
RSS_MAX_ENTRIES = 15 # Увеличено до 15, чтобы находить доверенные источники внутри агрегаторов
RSS_MAX_ENTRIES_INACTIVE = 30 # Увеличено до 30 для более глубокого охвата за ночь

# Time Intervals (in seconds)
CHECK_INTERVAL = 300 # 5 минут — оптимальный баланс между скоростью и риском блокировки IP
COOLDOWN = 900 # Увеличиваем до 15 минут, чтобы не спамить повторами одного события
LEARNING_INTERVAL = 1800 # 30 минут — оптимально для накопления выборки цен
MARKET_LOOKBACK_HOURS = 2 # Увеличиваем до 2 часов: macro-alpha требует времени для проявления
MAX_NEWS_AGE_HOURS = 4 # Увеличено до 4ч для надежности захвата RSS
MAX_NEWS_AGE_HOURS_INACTIVE = 12 # Увеличено до 12ч для ночного периода

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

# Concurrency Settings
GEMINI_CONCURRENCY = 1 # Бесплатный тариф требует последовательных запросов
OPENROUTER_CONCURRENCY = 5 # Платные/быстрые модели могут обрабатываться параллельно
DEEPSEEK_CONCURRENCY = 2 # Лимит для DeepSeek API

ENABLE_HOURLY_REPORT = False # Включить/выключить отправку часового отчета в Telegram
HOURLY_SUMMARY_INTERVAL = 3600 # Интервал отправки часового отчета в Telegram (1 час)
NUM_WORKERS = 4 # Увеличиваем до 4, так как DeepSeek и OpenRouter могут работать параллельно с Gemini

# Logic Factors
DECAY_FACTOR = 0.9 # Оптимальный баланс: новость сохраняет 50% силы через 15-20 минут и затухает за 2-3 часа.
NIGHT_DECAY_FACTOR = 0.98 # Почти не снижаем балл, когда рынок закрыт, чтобы сохранить контекст к открытию
MAX_SCORE_THRESHOLD = 25.0
DECAY_REFERENCE_SECONDS = 180 # Базовый интервал времени для расчета затухания

BLACK_SWAN_SCORE_THRESHOLD = 7.0 # Порог индивидуального скора новости для подтверждения статуса Black Swan
LEARNING_RATE = 0.001 # Уменьшаем шаг обучения, чтобы веса не "прыгали" от одной ошибки
ASYMMETRIC_LR_FACTOR = 2.0 # Ускорение коррекции при ошибке в направлении (is_correct = False)

# Multiplier Reset Logic
MIN_WINRATE_BEFORE_RESET = 40.0 # Порог WinRate (%), ниже которого множитель актива сбрасывается
MIN_SAMPLE_SIZE_FOR_RESET = 15 # Увеличим выборку для более точного сброса

IMPACT_MULTIPLIER = 0.3 # Базовая чувствительность к Z-score (Sigmas)
LEARNING_THRESHOLD = 0.4 # Повышаем порог: учимся только на движениях > 0.4 сигмы
PIVOT_THRESHOLD = 5.0 # Порог "разворотной" новости, при котором накопленный балл обнуляется
MIN_WEIGHT_THRESHOLD = 0.8 # Чистим базу от слабых связей активнее
NEUTRAL_SCORE_THRESHOLD = 2.0 # Повышаем порог, чтобы игнорировать "слабые" перепечатки
MAX_ENTITY_PARTS = 3 # Увеличено до 3, чтобы лучше обрабатывать сложные Slug от ИИ
DUPLICATE_TITLE_THRESHOLD = 0.65 # Снижен порог для более агрессивной дедупликации
FALLBACK_DUPLICATE_THRESHOLD = 0.55 # Повышаем чувствительность для не-семантического поиска
SEMANTIC_DEDUPLICATION_WINDOW = 12 # Увеличено до 12ч для борьбы с перепечатками в разных часовых поясах
SEMANTIC_DUPLICATE_THRESHOLD = 0.81 # Снижен порог для склейки семантически схожих новостей с разными акцентами
USE_EMBEDDINGS = True # Включить/выключить семантическую дедупликацию через векторы
EMBEDDING_MODEL = "models/gemini-embedding-2" # Основная модель эмбеддингов (Gemini)
OPENROUTER_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free" # Высокопроизводительная альтернатива для OpenRouter
CONFIDENCE_THRESHOLD = 0.35 # Минимальная уверенность ИИ для принятия новости
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
}
DEFAULT_TRUST_SCORE = 0.65  # Немного снижаем базу для фильтрации случайных источников

# Thresholds for market signals (Empirical sensitivity)
SIGNAL_THRESHOLD_HIGH = 3.5  # Повышаем порог для индексов, чтобы уменьшить количество ложных алертов
SIGNAL_THRESHOLD_MED = 2.5   # Повышено для VIX и Oil для фильтрации шума
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)
SIGNAL_THRESHOLD_BTC = 4.0   # For Crypto (Volatility buffer)
BTC_MIN_VOLATILITY_FOR_ALERT = 1.0 # Минимальное изменение цены BTC (%) для отправки уведомления
OIL_SHARP_MOVE_THRESHOLD = 2.0 # Порог для оповещения о резком движении нефти (%)

# Конфигурация бенчмарков для расчета Alpha (Abnormal Return)
ASSET_BENCHMARK_CONFIG = {
    "nasdaq": {"primary": "^GSPC", "type": "rolling_beta"}, 
    "sp500":  {"primary": "ACWI", "type": "rolling_beta"}, # Смена на ACWI (MSCI World)
    "soxs":   {"primary": "SOXX", "type": "leveraged", "factor": -3.0}, # Прямая связь с полупроводниками
    "btc":    {"primary": "^IXIC", "secondary": "DX-Y.NYB", "type": "multi_factor", "weights": [0.7, -0.3]}, 
    "oil":    {"primary": "DX-Y.NYB", "type": "rolling_beta"}, 
    "gold":   {"primary": "TIP", "type": "rolling_beta"}, # TIP = Real Yields Proxy
    "global": {"primary": "GLOBAL_REGIME", "type": "fixed", "factor": 1.0}
}

# Веса для композитного режима Global Regime
GLOBAL_REGIME_WEIGHTS = {
    "vix": 0.35,      # Equity Stress
    "move": 0.20,     # Bond Stress (^MOVE)
    "dxy": 0.15,      # Liquidity (Dollar Index)
    "hyg": 0.20,      # Credit Stress (High Yield Corp) - Inverted
    "growth": 0.10    # Growth Expectations (Yield Curve 10Y-3M) - Inverted
}
CLEANUP_INTERVAL = 86400 # Интервал очистки базы (24 часа)

# Параметры Квантовой Модели
EWMA_LAMBDA = 0.94 # Параметр затухания RiskMetrics
BETA_CLIP = 3.0 # Ограничение экстремальных значений беты
VOLATILITY_WINDOW = 40 # Окно для расчета реализованной волатильности
Z_ALPHA_VOL_FLOOR = 0.05 # Минимальная волатильность (%) для Z-Alpha по обычным активам
GLOBAL_Z_ALPHA_VOL_FLOOR = 0.25 # Более высокий floor для GLOBAL_REGIME, чтобы не учиться на микрошуме
ALPHA_MIN_THRESHOLD = 0.05 

# Если твой Win Rate выше 60% — система работает отлично. 
# Если ниже 40% — значит, либо веса в config.py настроены неверно, либо рынок сейчас ведет себя иррационально.

# Средняя абсолютная ошибка (avg_abs_error): Чем ниже это число, тем лучше откалиброван ваш global_impact_multiplier. 
# Если ошибка везде большая (например, > 20), значит множитель в config.py требует ручной корректировки 
# или системе нужно больше времени на обучение.
