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

client = genai.Client(api_key=config.GEMINI_API_KEY)

def init_model_pool():
    """Инициализирует список моделей для ротации."""
    pool = []
    try:
        all_models = list(client.models.list())
        models_list = [m.name for m in all_models if 'generateContent' in m.supported_actions]
        
        family_priority = {
            'gemini-3.1-flash': 1, 'gemini-3.1-pro': 2, 'gemini-3-flash': 3,
            'gemini-1.5-flash': 8, 'gemini-1.5-pro': 9
        }
        
        for m_name in models_list:
            for fam, priority in family_priority.items():
                if fam in m_name:
                    pool.append({
                        "name": m_name,
                        "priority": priority,
                        "supports_json": any(v in m_name for v in ["1.5", "2.0", "3", "latest"]),
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

        return sorted_pool
    except Exception:
        return [{"name": "models/gemini-1.5-flash", "supports_json": True, "provider": "gemini"}]

model_pool = init_model_pool()
current_model_idx = 0

async def run_global_research():
    """Анализирует макро-триггеры и сохраняет предложения в БД."""
    global current_model_idx
    assets = ["nasdaq", "oil", "soxs", "vix", "gold", "btc", "hbm"]
    prompt = f"""
    As a senior macro strategist, identify the top 15 global entities, geopolitical triggers, or economic factors 
    that will most significantly impact these assets over the next 30 days: {assets}.
    
    Return ONLY a JSON list of objects:
    [
      {{
        "keyword": "Entity Name",
        "asset": "target asset from the list",
        "impact_direction": "bullish/bearish",
        "reasoning": "Short professional explanation"
      }}
    ]
    """
    
    logging.info("--- Starting Global AI Research ---")
    max_retries = 3

    for attempt in range(max_retries):
        tried = 0
        while tried < len(model_pool):
            try:
                active = model_pool[current_model_idx]
                res_text = ""

                if active.get("provider") == "openrouter":
                    payload = {
                        "model": active["name"],
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    if active["supports_json"]:
                        payload["response_format"] = {"type": "json_object"}
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                                "HTTP-Referer": "https://gts-project.io",
                                "X-Title": "GTS Research",
                                "Content-Type": "application/json"
                            },
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=120, connect=15)
                        ) as resp:
                            if resp.status != 200:
                                raise Exception(f"OpenRouter Error {resp.status}")
                            res_json = await resp.json()
                            res_text = res_json['choices'][0]['message']['content'].strip()
                else:
                    gen_config = {"response_mime_type": "application/json"} if active["supports_json"] else {}
                    response = await client.aio.models.generate_content(
                        model=active["name"],
                        contents=prompt,
                        config=gen_config
                    )
                    res_text = response.text.strip()

                start, end = res_text.find('['), res_text.rfind(']') + 1
                if start == -1:
                    raise ValueError("No JSON list found")

                clean_json = res_text[start:end]
                # Убираем возможные артефакты markdown
                clean_json = clean_json.replace('```json', '').replace('```', '')
                
                suggestions = json.loads(clean_json)

                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    for s in suggestions:
                        cursor.execute("""
                            INSERT INTO ai_global_suggestions (keyword, asset, impact_direction, reasoning)
                            VALUES (?, ?, ?, ?)
                        """, (s['keyword'], s['asset'], s['impact_direction'], s['reasoning']))
                    conn.commit()
                
                logging.info(f"✅ Research finished. Found {len(suggestions)} new suggestions.")
                return
            except Exception as e:
                logging.warning(f"⚠️ Model {model_pool[current_model_idx]['name']} failed: {e}")
                current_model_idx = (current_model_idx + 1) % len(model_pool)
                tried += 1
        
        await asyncio.sleep(60 * (attempt + 1))

    logging.error("❌ Global Research failed after all retries.")

if __name__ == "__main__":
    try:
        asyncio.run(run_global_research())
    except KeyboardInterrupt:
        pass