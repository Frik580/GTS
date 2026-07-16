import asyncio
import logging
import json
import aiohttp
from google import genai
from db import get_db_connection
import config

# Настройка логирования для модуля исследования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Полная блокировка сообщения про AFC через глобальный фильтр
logging.getLogger().addFilter(lambda record: "AFC is enabled" not in record.getMessage())

logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)

client = genai.Client(api_key=config.GEMINI_API_KEY)

def init_model_pool():
    """Инициализирует список моделей для ротации."""
    pool = []
    try:
        all_models = list(client.models.list())
        models_list = [m.name for m in all_models if 'generateContent' in m.supported_actions]
        
        family_priority = {
            'gemini-3.1-flash': 1, 'gemini-3.1-pro': 2, 
            'gemini-3-flash': 3, 'gemini-3-pro': 4,
            'gemini-2.5-flash': 5, 'gemini-2.5-pro': 6,
            'gemini-2.0-flash': 7,
            'gemini-1.5-flash': 8, 'gemini-1.5-pro': 9,
            'gemini-1.0-pro': 10
        }
        
        for m_name in models_list:
            if any(spec in m_name.lower() for spec in ['-tts', '-image', 'robotics']):
                continue
            for fam, priority in family_priority.items():
                if fam in m_name:
                    pool.append({
                        "name": m_name,
                        "priority": priority,
                        "supports_json": any(v in m_name for v in ["1.5", "2.0", "2.5", "3", "latest"]),
                        "provider": "gemini"
                    })
        
        sorted_pool = sorted(pool, key=lambda x: x['priority'])

        if config.OPENROUTER_API_KEY:
            or_models = [
                {"name": "google/gemini-2.0-flash-lite-preview-02-05:free", "supports_json": True, "provider": "openrouter"},
                {"name": "tencent/hy3-preview:free", "supports_json": False, "provider": "openrouter"}
            ]
            for m in or_models:
                sorted_pool.append(m)

        if config.DEEPSEEK_API_KEY:
            ds_models = [
                {"name": "deepseek-v4-flash", "supports_json": True, "provider": "deepseek"},
                {"name": "deepseek-chat", "supports_json": True, "provider": "deepseek"},
                {"name": "deepseek-reasoner", "supports_json": False, "provider": "deepseek"}
            ]
            for m in ds_models:
                sorted_pool.append(m)
                logging.info(f"✅ Добавлена в пул ротации (DeepSeek): {m['name']}")

        return sorted_pool
    except Exception:
        return [{"name": "models/gemini-1.5-flash", "supports_json": True, "provider": "gemini"}]

model_pool = init_model_pool()
current_model_idx = 0
snoozed_providers = {} # provider_name -> snooze_until_timestamp

async def run_global_research():
    """Анализирует макро-триггеры и сохраняет предложения в БД."""
    global current_model_idx
    current_model_idx = 0  # Всегда начинаем с самого приоритетного провайдера (Gemini)
    # Переносим изменяемые активы в конец промпта, если они могут меняться
    static_research_instruction = """
    As a senior macro strategist, identify the top 15 global entities, geopolitical triggers, or economic factors.
    Return ONLY a JSON list of objects: [ { "keyword": "Entity Name", "asset": "target", "impact_direction": "bullish/bearish", "reasoning": "Short explanation" } ].
    """
    assets = ["nasdaq", "oil", "soxs", "vix", "gold", "btc"]
    prompt = f"{static_research_instruction} Focus on these assets over the next 30 days: {assets}."
    
    logging.info("--- Starting Global AI Research ---")
    max_retries = 3

    for attempt in range(max_retries):
        tried = 0
        start_idx = current_model_idx
        
        # Цикл поиска доступной модели
        while True:
            active = model_pool[current_model_idx]
            provider = active.get("provider")
            snooze_until = snoozed_providers.get(provider)

            if snooze_until and asyncio.get_event_loop().time() < snooze_until:
                current_model_idx = (current_model_idx + 1) % len(model_pool)
                if current_model_idx == start_idx:
                    logging.warning("[Research] All providers are snoozed. Waiting 60s.")
                    await asyncio.sleep(60)
                continue
            
            # Нашли доступную модель, выходим из цикла поиска
            break

        try:
            try:
                res_text = ""
                logging.info(f"[Research] Attempting to use model: {active['name']}")

                if active.get("provider") in ["openrouter", "deepseek"]:
                    api_url = "https://openrouter.ai/api/v1/chat/completions" if active.get("provider") == "openrouter" else "https://api.deepseek.com/chat/completions"
                    api_key = config.OPENROUTER_API_KEY if active.get("provider") == "openrouter" else config.DEEPSEEK_API_KEY
                    
                    payload = {
                        "model": active["name"],
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    if active["supports_json"]:
                        payload["response_format"] = {"type": "json_object"}
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            api_url,
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "HTTP-Referer": "https://gts-project.io",
                                "X-Title": "GTS Research",
                                "Content-Type": "application/json"
                            },
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=120, connect=15)
                        ) as resp:
                            if resp.status != 200:
                                raise Exception(f"API Error {resp.status}: {await resp.text()}")
                            res_json = await resp.json()
                            res_text = (res_json.get('choices', [{}])[0].get('message', {}).get('content') or "").strip()
                else:
                    gen_config = {"response_mime_type": "application/json"} if active["supports_json"] else {}
                    response = await asyncio.wait_for(client.aio.models.generate_content(
                        model=active["name"],
                        contents=prompt,
                        config=gen_config
                    ), timeout=120)
                    res_text = (response.text or "").strip()

                start, end = res_text.find('['), res_text.rfind(']') + 1
                if start == -1:
                    raise ValueError("No JSON list found")

                clean_json = res_text[start:end]
                # Убираем возможные артефакты markdown
                clean_json = clean_json.replace('```json', '').replace('```', '')
                
                suggestions = json.loads(clean_json)

                async with get_db_connection() as conn:
                    for s in suggestions:
                        await conn.execute("""
                            INSERT INTO ai_global_suggestions (keyword, asset, impact_direction, reasoning)
                            VALUES (?, ?, ?, ?)
                        """, (s['keyword'], s['asset'], s['impact_direction'], s['reasoning']))
                    await conn.commit()
                
                logging.info(f"✅ Research finished. Found {len(suggestions)} new suggestions.")
                return
            except Exception as e: # Эта вложенность нужна, чтобы поймать ошибку и правильно обработать ее ниже
                raise e
        except Exception as e:
            err_str = str(e)
            failed_provider = active.get("provider")
            logging.warning(f"⚠️ [Research] Model {active['name']} failed: {err_str[:200]}")
            if "429" in err_str or "rate limit" in err_str.lower():
                snooze_until = asyncio.get_event_loop().time() + 300 # Snooze for 5 mins
                snoozed_providers[failed_provider] = snooze_until
                logging.info(f"[Research] Snoozing provider '{failed_provider}' for 5 minutes.")
            
            current_model_idx = (current_model_idx + 1) % len(model_pool)
        await asyncio.sleep(60 * (attempt + 1))

    logging.error("❌ Global Research failed after all retries.")

if __name__ == "__main__":
    try:
        asyncio.run(run_global_research())
    except KeyboardInterrupt:
        pass