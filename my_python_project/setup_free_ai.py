"""
Setup script for free AI providers (Groq / Cerebras).

Groq:  https://console.groq.com/keys  — бесплатно, без карты, ~14 400 req/day
Cerebras: https://cloud.cerebras.ai — бесплатно, 1M tokens/day
"""
from __future__ import annotations

import asyncio
import os
import re
import ssl
from pathlib import Path

import aiohttp
from dotenv import load_dotenv, set_key

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

TEST_PROMPT = 'Return JSON: {"ok": true}'


async def test_provider(name: str, url: str, key: str, model: str) -> bool:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "response_format": {"type": "json_object"},
    }
    conn = aiohttp.TCPConnector(ssl=SSL_CTX)
    async with aiohttp.ClientSession(connector=conn) as session:
        async with session.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=45),
        ) as resp:
            text = await resp.text()
            ok = resp.status == 200 and "choices" in text
            print(f"  {name}: HTTP {resp.status} {'OK' if ok else 'FAIL'}")
            if not ok:
                print(f"    {text[:200]}")
            return ok


def update_env(key: str, value: str) -> None:
    set_key(str(ENV_PATH), key, value)


async def setup_groq() -> bool:
    existing = os.getenv("GROQ_API_KEY", "").strip()
    if existing:
        print(f"Найден GROQ_API_KEY ({existing[:8]}...), проверяю...")
        ok = await test_provider(
            "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
            existing,
            "openai/gpt-oss-20b",
        )
        if ok:
            return True
        print("  Существующий ключ не работает.")

    print("\n=== Groq (рекомендуется) ===")
    print("1. Откройте https://console.groq.com/keys")
    print("2. Создайте API Key (бесплатно, карта не нужна)")
    print("3. Вставьте ключ ниже (начинается с gsk_)\n")
    key = input("GROQ_API_KEY: ").strip()
    if not key.startswith("gsk_"):
        print("Ключ должен начинаться с gsk_")
        return False

    ok = await test_provider(
        "Groq",
        "https://api.groq.com/openai/v1/chat/completions",
        key,
        "openai/gpt-oss-20b",
    )
    if ok:
        update_env("GROQ_API_KEY", key)
        update_env("USE_GROQ", "True")
        update_env("USE_GEMINI", "False")
        update_env("USE_OPENROUTER", "False")
        print("✅ Groq подключён и сохранён в .env")
        return True
    return False


async def setup_cerebras() -> bool:
    existing = os.getenv("CEREBRAS_API_KEY", "").strip()
    if existing:
        ok = await test_provider(
            "Cerebras",
            "https://api.cerebras.ai/v1/chat/completions",
            existing,
            "qwen-3.8-27b",
        )
        if ok:
            return True

    print("\n=== Cerebras (запасной) ===")
    print("1. Откройте https://cloud.cerebras.ai")
    print("2. Создайте API Key\n")
    key = input("CEREBRAS_API_KEY: ").strip()
    if not key:
        return False

    ok = await test_provider(
        "Cerebras",
        "https://api.cerebras.ai/v1/chat/completions",
        key,
        "qwen-3.8-27b",
    )
    if ok:
        update_env("CEREBRAS_API_KEY", key)
        update_env("USE_CEREBRAS", "True")
        print("✅ Cerebras подключён и сохранён в .env")
        return True
    return False


async def main():
    print("GTS — настройка бесплатного AI-провайдера\n")
    print("Текущий статус:")
    print(f"  Gemini:     {'включён' if os.getenv('USE_GEMINI','False').lower()=='true' else 'выключен'}")
    print(f"  OpenRouter: {'включён' if os.getenv('USE_OPENROUTER','False').lower()=='true' else 'выключен'}")
    print(f"  Groq key:   {'есть' if os.getenv('GROQ_API_KEY') else 'нет'}")
    print(f"  Cerebras:   {'есть' if os.getenv('CEREBRAS_API_KEY') else 'нет'}\n")

    if await setup_groq():
        return
    if await setup_cerebras():
        return
    print("\n❌ Не удалось подключить провайдера. Попробуйте позже или установите Ollama.")


if __name__ == "__main__":
    asyncio.run(main())
