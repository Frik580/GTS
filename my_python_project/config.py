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
IMPACT_MULTIPLIER = 4.0 # Сбалансировано: MAX_SCORE (25) * 4.0 = 100%

# Thresholds for market signals (Empirical sensitivity)
SIGNAL_THRESHOLD_HIGH = 3.0  # For Indices (Nasdaq, SOXS)
SIGNAL_THRESHOLD_MED = 2.0   # For Commodities and VIX
SIGNAL_THRESHOLD_LOW = 1.5   # For Safe-havens (Gold)


# В системе GTS 4.0 влияние ключей (событий) на рыночные сигналы реализовано через механизм интенсивности, которая объединяет оценку нейросети и накопленные веса важности.

# 1. Механизм влияния (Формула)
# В файле d:\GTS\my_python_project\engine.py расчет происходит следующим образом: Интенсивность = AI Score (от -10 до 10) * Вес ключа

# AI Score: Определяется моделью Gemini. Положительный (0...10) — рост рисков/эскалация (Risk-Off). Отрицательный (-10...0) — разрядка/позитив (Risk-On).
# Вес ключа: Берется из config.TRACKED_KEYWORDS (или из таблицы weights в БД после обучения). Чем выше вес, тем сильнее новость «раскачивает» сигналы.
# 2. Как ключи влияют на активы
# Согласно функции market_signals, интенсивность влияет на активы по-разному в зависимости от порогов (thresholds):

# Актив	Реакция на Риск (Intensity > 0)	Реакция на Позитив (Intensity < 0)	Порог срабатывания
# Nasdaq	🔴 Bearish (Медвежий)	🟢 Bullish (Бычий)	Выше 3.0 / Ниже -2.0
# Gold	🟢 Bullish (Защитный актив)	🔴 Bearish (Сброс золота)	Выше 1.5 / Ниже -3.0
# Oil	🟢 Bullish (Ожидание дефицита)	🔴 Bearish (по умолчанию)	Выше 2.0
# VIX	🟢 Bullish (Рост страха)	⚪ Flat (Спокойствие)	Выше 2.0
# BTC	🔴 Bearish (Выход из риска)	🟢 Bullish (Покупка риска)	Выше 4.0 / Ниже -2.0
# SOXS	🟢 Bullish (Ставка на падение чипов)	🔴 Bearish (по умолчанию)	Выше 3.0
# HBM (AI)	🔴 Bearish (Падение тех-сектора)	🟢 Bullish (Рост тех-сектора)	Выше 2.0 / Ниже -2.0
# 3. Специализация ключей (Обучение)
# Хотя любой ключ технически влияет на все сигналы сразу, система обучения (learning_cycle) корректирует веса ключей, глядя на конкретные рынки:

# Ключи «US_IRAN», «HORMUZ», «GLOBAL»: Обучаются в первую очередь на индексах VIX и Nasdaq. Если при новости об Иране VIX не вырос, вес этих ключей будет снижаться.
# Ключи «NVIDIA», «OPENAI»: Обучаются на HBM Index и SOXS. Они сильнее всего «заточены» под сигналы полупроводникового сектора.
# Ключ «OIL»: Прямая корреляция с ценами на нефть (oil_change).
# Ключ «GOLD»: Прямая корреляция с котировками золота.
# Ключ «BTC»: Обучается на волатильности биткоина.
# Пример в цифрах:
# Новость: "NVIDIA анонсирует рекордную прибыль" (AI Score: -5.0 — это позитив/Risk-on).
# Ключ: NVIDIA (Вес: 2.0).
# Интенсивность: -5.0 * 2.0 = -10.0.
# Результат:
# Nasdaq: Bullish (так как -10.0 < -2.0).
# HBM: Bullish (так как -10.0 < -2.0).
# Gold: Bearish (так как -10.0 < -3.0).
# Если же новость будет про «Удар по нефтяным вышкам» (Score +8.0) с ключом US Iran (Вес 2.5), интенсивность станет +20.0, что переведет все сигналы в режим экстремального Risk-Off (Золото, Нефть, VIX — вверх; Насдак, БТК — вниз).