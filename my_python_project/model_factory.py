import asyncio
import logging
from typing import List, Dict
import config

class GeminiDiscovery:
    @staticmethod
    def get_models(client) -> List[Dict]:
        pool = []
        family_priority = {
            'gemini-3.1-flash': 1,
            'gemini-3-flash': 2,
            'gemini-3-flash-live': 3,
            'gemini-2.5-flash': 4,
            'gemini-2.5-flash-lite': 5,
            'gemini-3.5-flash': 6
        }
        try:
            all_models = list(client.models.list())
            for m in all_models:
                if 'generateContent' in m.supported_actions and not any(s in m.name for s in ['-tts', '-image']):
                    # Определяем приоритет на основе имени модели
                    priority = 10  # По умолчанию для прочих моделей Gemini
                    for fam, p in family_priority.items():
                        if fam in m.name:
                            priority = p
                            break
                    
                    pool.append({
                        "name": m.name,
                        "supports_json": any(v in m.name for v in ["1.5", "2.0", "2.5", "3", "latest"]),
                        "provider": "gemini",
                        "priority": priority
                    })
        except Exception as e:
            logging.error(f"Gemini discovery error: {e}")
        return pool

class OpenRouterRegistry:
    @staticmethod
    def get_models() -> List[Dict]:
        if not config.OPENROUTER_API_KEY: return []
        return [
            {"name": "nvidia/nemotron-3-super-120b-a12b:free", "supports_json": True, "provider": "openrouter", "priority": 20},
            {"name": "openai/gpt-oss-120b:free", "supports_json": True, "provider": "openrouter", "priority": 20},
            {"name": "openrouter/free", "supports_json": False, "provider": "openrouter", "priority": 20}
        ]

class DeepSeekRegistry:
    @staticmethod
    def get_models() -> List[Dict]:
        if not config.DEEPSEEK_API_KEY: return []
        return [
            {"name": "deepseek-v4-flash", "supports_json": True, "provider": "deepseek", "priority": 30},
            {"name": "deepseek-reasoner", "supports_json": False, "provider": "deepseek", "priority": 30}
        ]

def init_model_pool():
    from google import genai
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    pool = []
    pool.extend(GeminiDiscovery.get_models(client))
    pool.extend(OpenRouterRegistry.get_models())
    pool.extend(DeepSeekRegistry.get_models())
    
    # Сортируем весь пул по приоритету перед использованием
    pool.sort(key=lambda x: x.get('priority', 999))
    
    if not pool:
        pool.append({"name": "models/gemini-1.5-flash", "supports_json": True, "provider": "gemini", "priority": 10})
    
    logging.info(f"✅ Пул моделей инициализирован: {len(pool)} доступно")
    return pool

class ModelRotator:
    def __init__(self, pool):
        self.pool = pool
        self._idx = 0
        self._lock = asyncio.Lock()

    def get_active(self) -> Dict:
        return self.pool[self._idx]

    async def rotate(self) -> Dict:
        async with self._lock:
            self._idx = (self._idx + 1) % len(self.pool)
            return self.get_active()