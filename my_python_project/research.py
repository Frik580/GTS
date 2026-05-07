import asyncio
import logging
import json
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
                        "supports_json": any(v in m_name for v in ["1.5", "2.0", "3", "latest"])
                    })
        return sorted(pool, key=lambda x: x['priority'])
    except Exception:
        return [{"name": "models/gemini-1.5-flash", "supports_json": True}]

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

                suggestions = json.loads(res_text[start:end])

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