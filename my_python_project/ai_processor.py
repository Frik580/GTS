import json
import hashlib
import re
import time
import logging
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict, Any
import config

class PromptBuilder:
    @staticmethod
    def build_analysis_prompt(text: str, pub_time: str) -> str:
        tags_hint = ", ".join([f'"{k}"' for k in config.TRACKED_KEYWORDS.keys()])
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
        return f"""
        Current Time: {current_time_utc} UTC
        Article Published At: {pub_time}
        Analyze this financial news snippet: "{text}"
        Identify key entities. Prioritize these tags: {tags_hint}.
        Return ONLY a JSON object:
        {{
          "primary_asset": "string",
          "score": number,
          "event_type": "military" | "economic" | "diplomatic" | "neutral" | "tech",
          "entities": ["list"],
          "slug": "string",
          "is_black_swan": boolean,
          "confidence": number,
          "summary": "RU text",
          "title_ru": "RU title"
        }}
        Scoring: POSITIVE (1 to 10) = RISK-OFF, NEGATIVE (-10 to -1) = RISK-ON.
        """

class ResponseParser:
    @staticmethod
    def parse(res_text: str) -> Optional[Dict]:
        try:
            start = res_text.find('{')
            end = res_text.rfind('}') + 1
            if start == -1 or end == 0: return None
            json_str = res_text[start:end].replace('```json', '').replace('```', '')
            return json.loads(json_str, strict=False)
        except:
            return None

    @staticmethod
    def validate_and_extract(data: Dict, model_name: str) -> Tuple:
        # Логика извлечения и нормализации данных (score, confidence и т.д.)
        score = float(data.get("score", 0))
        conf = float(data.get("confidence", 0.5))
        return (score, data.get("event_type"), data.get("entities", []), 
                data.get("slug"), data.get("is_black_swan", False), 
                model_name, conf, data.get("summary", ""), data.get("title_ru", ""))

class FallbackAnalyzer:
    @staticmethod
    async def run(text: str, state_learning: Any) -> Tuple:
        # Генерируем уникальный хеш от текста, чтобы разные новости в fallback-режиме
        # не считались одним и тем же сюжетом в антиспам-фильтре.
        text_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        slug = f"fallback_{text_hash}"
        return (0.0, "neutral", [], slug, False, "Fallback", 0.5, "", "")

class AIProvider:
    def __init__(self, rotator: Any, state: Any):
        self.rotator = rotator
        self.state = state

    async def call(self, prompt: str, session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
        active = self.rotator.get_active()
        provider = active.get("provider", "gemini")
        
        # Выбор лимитера
        limiter = getattr(self.state, f"{provider}_limiter")
        
        async with limiter:
            if provider == "gemini":
                # Логика вызова Gemini
                from google import genai
                client = genai.Client(api_key=config.GEMINI_API_KEY)
                response = await client.aio.models.generate_content(model=active["name"], contents=prompt)
                return response.text, active["name"]
            else:
                # Логика вызова OpenRouter/DeepSeek
                api_url = "https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter" else "https://api.deepseek.com/chat/completions"
                api_key = config.OPENROUTER_API_KEY if provider == "openrouter" else config.DEEPSEEK_API_KEY
                async with session.post(api_url, headers={"Authorization": f"Bearer {api_key}"}, 
                                        json={"model": active["name"], "messages": [{"role": "user", "content": prompt}]}) as resp:
                    res_json = await resp.json()
                    return res_json['choices'][0]['message']['content'], active["name"]

async def ai_analyze_refined(text: str, rotator: Any, state: Any, pub_time: str, session: aiohttp.ClientSession, source_title: str = "") -> Tuple:
    prompt = PromptBuilder.build_analysis_prompt(text, pub_time)
    provider = AIProvider(rotator, state)
    
    for attempt in range(3):
        try:
            res_text, model_name = await provider.call(prompt, session)
            data = ResponseParser.parse(res_text)
            if data:
                return ResponseParser.validate_and_extract(data, model_name)
            await rotator.rotate()
        except Exception as e:
            logging.warning(f"AI attempt {attempt} failed: {e}")
            await rotator.rotate()
            
    return await FallbackAnalyzer.run(text, state.learning)

async def get_embedding(text: str, rotator: Any, state: Any, session: aiohttp.ClientSession) -> Optional[List[float]]:
    # 1. Попытка через Gemini (основной)
    if config.GEMINI_API_KEY:
        async with state.gemini_limiter:
            try:
                from google import genai
                client = genai.Client(api_key=config.GEMINI_API_KEY)
                # Добавляем таймаут, чтобы не блокировать воркер при затупах сети
                res = await asyncio.wait_for(
                    client.aio.models.embed_content(model=config.EMBEDDING_MODEL, contents=text),
                    timeout=12
                )
                return res.embeddings[0].values
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logging.warning("⚠️ Квота Gemini Embedding исчерпана (1000/день). Используем фоллбек OpenRouter.")
                else:
                    logging.error(f"Gemini Embedding error: {e}")

    # 2. Фоллбек на OpenRouter (если Gemini недоступен или лимит исчерпан)
    if config.OPENROUTER_API_KEY:
        async with state.openrouter_limiter:
            try:
                url = "https://openrouter.ai/api/v1/embeddings"
                headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
                payload = {"model": config.OPENROUTER_EMBEDDING_MODEL, "input": text}
                async with session.post(url, headers=headers, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logging.info("✅ OpenRouter Embedding получен успешно.")
                        return data['data'][0]['embedding']
                    else:
                        logging.error(f"OpenRouter Embedding error status: {resp.status}")
            except Exception as e:
                logging.error(f"Fallback Embedding error: {e}")

    return None

async def is_semantic_duplicate(title: str, embedding: List[float], state: Any) -> bool:
    async with state.cache.cache_lock:
        for cached_title, (cached_emb, ts) in state.cache.embeddings.items():
            if time.time() - ts > config.SEMANTIC_DEDUPLICATION_WINDOW * 3600:
                continue
            sim = np.dot(embedding, cached_emb)
            if sim > config.SEMANTIC_DUPLICATE_THRESHOLD:
                logging.info(f"🧬 Semantic duplicate ({sim:.3f}): '{title}' ≈ '{cached_title}'")
                return True
    return False