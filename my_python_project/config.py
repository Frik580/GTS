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
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=US+Iran",
    "https://news.google.com/rss/search?q=Hormuz",
    "https://news.google.com/rss/search?q=Oil",
    "https://news.google.com/rss/search?q=Gold+Price+Analysis",
    "https://news.google.com/rss/search?q=Bitcoin+Crypto+News",
]

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
DECAY_FACTOR = 0.95
MAX_SCORE_THRESHOLD = 25.0
SCALING_FACTOR = 10.0