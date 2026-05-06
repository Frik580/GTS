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
TRACKED_KEYWORDS = {
    "US Iran": 2.5,
    # "Hormuz": 3.0,
    "Nvidia": 2.0,
    "OpenAI": 2.0,
    "Oil": 2.0,
    "Gold": 1.5,
    "Bitcoin": 1.2,
    "Nasdaq": 1.0
}

RSS_FEEDS = [f"https://news.google.com/rss/search?q={k.replace(' ', '+')}" for k in TRACKED_KEYWORDS.keys()]

# Time Intervals (in seconds)
CHECK_INTERVAL = 300
COOLDOWN = 600
LEARNING_INTERVAL = 3600
CLEANUP_INTERVAL = 86400
RETENTION_DAYS = 14

# AI Delays
AI_DELAY_JSON = 15
AI_DELAY_NO_JSON = 60

# Logic Factors
MARKET_LOOKBACK_HOURS = 4
DECAY_FACTOR = 0.95 # Увеличиваем скорость затухания для более быстрой адаптации к новым данным
MAX_SCORE_THRESHOLD = 25.0 # Увеличиваем порог для отправки сигналов, чтобы уменьшить количество ложных срабатываний
SCALING_FACTOR = 10.0 # Увеличиваем масштаб для более заметного влияния предсказаний
LEARNING_RATE = 0.05  # Увеличиваем скорость обучения для более быстрой адаптации
IMPACT_MULTIPLIER = 12.0 # Увеличиваем влияние предсказаний на итоговый балл

# Thresholds for market signals (Empirical sensitivity)
SIGNAL_THRESHOLD_HIGH = 3.0  # For Indices (Nasdaq, SOXS)
SIGNAL_THRESHOLD_MED = 2.0   # For Commodities and VIX
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)