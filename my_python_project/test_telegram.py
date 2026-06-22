import asyncio
import aiohttp
import config
import sys
from engine import send_telegram

async def check_telegram_connection():
    """Проверяет настройки Telegram и отправляет тестовое сообщение."""
    print("--- 📡 GTS Telegram Diagnostic ---")
    
    if not config.BOT_TOKEN or not config.CHAT_ID:
        print("❌ ОШИБКА: TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не найдены в .env")
        return

    proxy = getattr(config, "HTTP_PROXY", None)
    print(f"Token: {config.BOT_TOKEN[:10]}... (скрыто)")
    print(f"Chat ID: {config.CHAT_ID}")
    print(f"Proxy: {proxy or 'Не используется (прямое соединение)'}")
    print("----------------------------------")

    # Создаем сессию с увеличенным таймаутом
    timeout = aiohttp.ClientTimeout(total=30, connect=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 0. Предварительная проверка прокси
        if proxy:
            print(f"🔍 Тестируем прокси через google.com...")
            try:
                async with session.get("https://www.google.com", proxy=proxy, timeout=10) as p_resp:
                    if p_resp.status == 200:
                        print("✅ Прокси успешно соединился с Google.")
                    else:
                        print(f"⚠️ Прокси работает, но Google вернул статус {p_resp.status}")
            except Exception as e:
                print(f"❌ Прокси не может соединиться даже с Google: {type(e).__name__}: {e}")
                print("Совет: Убедитесь, что прокси поддерживает протокол HTTP/HTTPS. Если это SOCKS5, нужны доп. библиотеки.")
                return

        # 1. Проверка самого токена
        url_me = f"https://api.telegram.org/bot{config.BOT_TOKEN}/getMe"
        try:
            print(f"📡 Попытка установки соединения...")
            async with session.get(url_me, proxy=proxy) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bot_name = data.get('result', {}).get('username')
                    print(f"✅ Токен валиден! Бот: @{bot_name}")
                    
                    # 2. Попытка отправки сообщения
                    print(f"Попытка отправки сообщения в чат {config.CHAT_ID}...")
                    await send_telegram(session, "🚀 <b>GTS System Check</b>\nСвязь установлена успешно. Конфигурация верна.")
                    print(f"✅ Сообщение отправлено. Проверьте ваш Telegram канал.")
                else:
                    error_text = await resp.text()
                    print(f"❌ Ошибка API (Статус {resp.status}): {error_text}")
        except Exception as e:
            print(f"❌ Критическая ошибка связи: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(check_telegram_connection())