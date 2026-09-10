import asyncio
import logging
from typing import List, Dict
import config

class GeminiDiscovery:
    @staticmethod
    def get_models(client) -> List[Dict]:
        pool = []
        family_priority = {
            'gemini-3.5-flash': 1,
            'gemini-3.1-flash-lite': 2,
            'gemini-3-flash': 2,
            'gemini-2.5-flash': 4,
            'gemini-2.5-flash-lite': 5
        }
        try:
            all_models = list(client.models.list())
            for m in all_models:
                if 'generateContent' in m.supported_actions and not any(s in m.name for s in ['-tts', '-image']):
                    priority = None
                    for fam, p in family_priority.items():
                        if fam in m.name:
                            priority = p
                            break
                    
                    if priority is None:
                        if config.ONLY_PRIORITY_GEMINI:
                            continue
                        priority = 10

                    pool.append({
                        "name": m.name,
                        "supports_json": any(v in m.name for v in ["1.5", "2.0", "2.5", "3", "latest"]),
                        "provider": "gemini",
                        "priority": priority + 50  # ниже бесплатных Groq/Cerebras
                    })
        except Exception as e:
            logging.error(f"Gemini discovery error: {e}")
        return pool

class GroqRegistry:
    """Models are kept only when they are present in the provider catalog."""
    @staticmethod
    def get_models() -> List[Dict]:
        if not config.GROQ_API_KEY or not config.USE_GROQ:
            return []
        return []

class CerebrasRegistry:
    """Models are kept only when they are present in the provider catalog."""
    @staticmethod
    def get_models() -> List[Dict]:
        if not config.CEREBRAS_API_KEY or not config.USE_CEREBRAS:
            return []
        return []

class OpenRouterRegistry:
    @staticmethod
    def get_models() -> List[Dict]:
        if not config.OPENROUTER_API_KEY or not config.USE_OPENROUTER:
            return []
        return [
            {"name": "openrouter/free", "supports_json": False, "provider": "openrouter", "priority": 40},
            {"name": "google/gemma-4-26b-a4b-it:free", "supports_json": True, "provider": "openrouter", "priority": 42},
            {"name": "nvidia/nemotron-3-super-120b-a12b:free", "supports_json": True, "provider": "openrouter", "priority": 44},
            {"name": "nvidia/nemotron-3-ultra-550b-a55b:free", "supports_json": True, "provider": "openrouter", "priority": 45},
        ]

class DeepSeekRegistry:
    @staticmethod
    def get_models() -> List[Dict]:
        if not config.DEEPSEEK_API_KEY or not config.USE_DEEPSEEK:
            return []
        return [
            {"name": "deepseek-v4-flash", "supports_json": True, "provider": "deepseek", "priority": 60},
        ]

class OllamaRegistry:
    @staticmethod
    def get_models(priority: int = 0) -> List[Dict]:
        return [{"name": config.OLLAMA_MODEL, "supports_json": True, "provider": "ollama", "priority": priority}]

def init_model_pool():
    if config.USE_LOCAL_OLLAMA:
        logging.info(f"🚀 Используется только локальная модель Ollama: {config.OLLAMA_MODEL}")
        return OllamaRegistry.get_models()

    pool = []
    pool.extend(GroqRegistry.get_models())
    pool.extend(CerebrasRegistry.get_models())

    if config.USE_GEMINI and config.GEMINI_API_KEY:
        from google import genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        pool.extend(GeminiDiscovery.get_models(client))

    pool.extend(OpenRouterRegistry.get_models())
    pool.extend(DeepSeekRegistry.get_models())

    if config.OLLAMA_FALLBACK:
        pool.extend(OllamaRegistry.get_models(priority=100))

    pool.sort(key=lambda x: x.get('priority', 999))

    if not pool:
        logging.warning(
            "⚠️ Пул моделей пуст! Добавьте GROQ_API_KEY (бесплатно: console.groq.com/keys) "
            "или запустите: python setup_free_ai.py"
        )
        if config.USE_GEMINI and config.GEMINI_API_KEY:
            pool.append({"name": "models/gemini-2.0-flash", "supports_json": True, "provider": "gemini", "priority": 99})

    logging.info(f"✅ Пул моделей инициализирован: {len(pool)} доступно — {[m['name'] for m in pool[:3]]}")
    return pool

class ModelRotator:
    def __init__(self, pool):
        self.pool = pool
        self._idx = 0
        self._snoozed_providers = {}
        self._lock = asyncio.Lock()

    def get_active(self) -> Dict:
        if not self.pool:
            return {"name": "none", "provider": "none", "supports_json": False}
        return self.pool[self._idx]

    async def rotate(self, state=None, failed_provider: str = None) -> Dict:
        async with self._lock:
            if failed_provider:
                snooze_until = asyncio.get_event_loop().time() + 300
                self._snoozed_providers[failed_provider] = snooze_until
                logging.info(f"Snoozing provider '{failed_provider}' for 5 minutes due to rate limits.")

            if not self.pool:
                return self.get_active()

            start_idx = self._idx
            while True:
                self._idx = (self._idx + 1) % len(self.pool)
                next_model = self.pool[self._idx]
                provider = next_model.get("provider")

                snooze_until = self._snoozed_providers.get(provider)
                if snooze_until:
                    if asyncio.get_event_loop().time() < snooze_until:
                        if self._idx == start_idx:
                            snoozed_info = {p: f"{int(ts - asyncio.get_event_loop().time())}s left" for p, ts in self._snoozed_providers.items() if ts > asyncio.get_event_loop().time()}
                            logging.warning(f"All providers are currently rate-limited. Snoozed: {snoozed_info}. Waiting before next attempt.")
                            await asyncio.sleep(60)
                        continue
                break

            if state:
                await state.save_to_db()
            return self.get_active()
