import json
import re
import hashlib
import logging
import time
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict, Any

from pydantic import ValidationError

import config
from ai_models import NewsAnalysisResponse, BatchNewsResponse

try:
    import json_repair
except ImportError:
    json_repair = None

class PromptBuilder:
    @staticmethod
    def build_analysis_prompt(news_items: List[Dict[str, str]]) -> str:
        tags_hint = ", ".join([f'"{k}"' for k in config.TRACKED_KEYWORDS.keys()])
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
        
        
        formatted_news = ""
        for i, item in enumerate(news_items):
            formatted_news += f"ID: {i}\nTime: {item['pub_time']}\nContent: {item['text']}\n---\n"

        
        return f"""
        Current Time: {current_time_utc} UTC
        You are a senior macro strategist and financial analyst. 
        Your task is to perform a deep semantic analysis of the {len(news_items)} news items below.
        Do not perform simple keyword matching. Analyze the underlying economic intent and market sentiment.

        {formatted_news}

        STRICT SCORING LOGIC:
        - LOCAL/TRIVIAL news (local accidents, humor, memes, small community issues) MUST be score 0.
        - NO-ACTION/OPINION RULE: Articles that are pure editorials, "think pieces", broad market discussions, or "expert predictions" without a NEW factual event or data release MUST be score 0.
        - RECAPS/SUMMARIES: Articles that just summarize past events ("What happened in the markets yesterday", "Weekly review") MUST be score 0.
        - VAGUE STATEMENTS: News like "CEO is optimistic about the future" or "Company continues to innovate" without specific new data or product announcements MUST be score 0.
        - HARD NEWS ONLY: Only events that represent a change in current state (e.g., "X happened", "Data Y was released", "Official Z stated") should get a non-zero score.
        - INDIVIDUAL stock news (earnings, major upgrades) for key tickers like NVDA, MSFT, etc., should be between -4 and 4.
        Identify key entities. Use these tags for categorization reference only: {tags_hint}.
        Return ONLY a JSON object with key "items" containing a list of objects:
{{
        "items": [
        {{
          "id": integer (match the provided ID 0, 1, etc),
          "primary_asset": "string (MUST NOT be null. Use 'global' if no specific ticker is found)",
          "score": number,
          "event_type": "military" | "economic" | "diplomatic" | "neutral" | "tech",
          "entities": ["list"],
          "slug": "string (max 3 words, prioritize SUBJECT + CATEGORY, e.g., 'US_IRAN_CONFLICT')",
          "is_black_swan": boolean,
          "confidence": float (strictly 0.0 to 1.0),
          "summary": "RU text",
          "title_ru": "RU title",
          "capex_signal": 1 | 0 | -1 | null,
          "guidance_signal": 1 | 0 | -1 | null
        }}]
        }}
        STRICT RULE: You must return exactly {len(news_items)} populated objects in the "items" list. Do not return empty objects.
        STRICT LANGUAGE RULE: The "summary" and "title_ru" fields MUST be written strictly in the Russian language.
        IMPORTANT SCORING RULES:
        - POSITIVE (1 to 10): RISK-OFF. Events that increase systemic uncertainty, debt risk, or geopolitical friction (e.g., war, inflation spikes, missed earnings).
        - NEGATIVE (-10 to -1): RISK-ON. Events that drive economic growth, productivity, or liquidity (e.g., rate cuts, cooling inflation, peace).
        - TECH PROGRESS RULE: Technological breakthroughs (AI advancements, new product launches, efficiency gains) are fundamentally RISK-ON (Negative Score), even if they are "radical" or "disruptive".
        - SCORING SCALE: 0 = No impact; 1-2.4 = Minor noise; 2.5-4.9 = Significant market mover; 5.0-6.9 = Major trend shift; 7.0-10.0 = Black Swan.
        - CONTEXT OVER WORDS: A "radical change" in AI technology is a productivity booster (Negative Score), while a "radical change" in central bank policy might be a risk (Positive Score).
        - BLACK SWAN RULE: Set 'is_black_swan' to true ONLY for regime-changing catastrophes or massive systemic breakthroughs (e.g., OpenAI IPO, Fed pivot). If TRUE, the 'score' MUST be >= 7.0.
        - EXTREME CAUTION: Never exceed the -10 to 10 range. Most significant news should fall between 2.5 and 4.0.
        Analyze the INTENT and FACTUALITY. If the news is just a "reasoning" on a topic, score it 0.

        SPECIAL SOXS SIGNALS RULES:
        - If MSFT, META, AMZN, or GOOGL mention specific cloud or AI CAPEX changes: set capex_signal (1: increase/boost, -1: decrease/cut, 0: unchanged).
        - If NVDA, Broadcom (AVGO), AMD, or Micron (MU) change future financial guidance: set guidance_signal (1: upgrade/higher outlook, -1: downgrade/cut, 0: stable).
        - If the news does not contain those specific events, these fields MUST be null.
        """

class ResponseParser:
    @staticmethod
    def parse(res_text: str) -> Optional[Dict]:
        if not res_text:
            return None

        # 1. Используем json_repair, если он установлен
        if json_repair:
            try:
                data = json_repair.loads(res_text)
                if isinstance(data, dict) and "items" in data:
                    return data
            except Exception:
                pass

        # 2. Робастный поиск (fallback): перебор всех пар { ... }
        # Сначала очищаем от явных markdown-тегов, которые ИИ часто ставит СНАРУЖИ скобок
        cleaned_text = re.sub(r'```(?:json)?|```', '', res_text).strip()
        
        starts = [m.start() for m in re.finditer(r'\{', cleaned_text)]
        ends = [m.start() for m in re.finditer(r'\}', cleaned_text)]

        # Перебираем комбинации от самых длинных к коротким
        for s in starts:
            for e in reversed(ends):
                if e > s:
                    candidate = cleaned_text[s:e+1]
                    try:
                        # strict=False позволяет парсить JSON с управляющими символами (\n)
                        data = json.loads(candidate, strict=False)
                        if isinstance(data, dict) and "items" in data:
                            return data
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def validate_and_extract_batch(data: Dict, model_name: str) -> Dict[int, Tuple]:
        try:
            parsed_batch = BatchNewsResponse.model_validate(data)
            return {item.id: item.to_analysis_tuple(model_name) for item in parsed_batch.items}
        except ValidationError as e:
            short_data = (str(data)[:200] + "...") if len(str(data)) > 200 else str(data)
            error_details = []
            for err in e.errors()[:2]:
                loc = " -> ".join(str(v) for v in err['loc'])
                error_details.append(f"[{loc}]: {err['msg']} (input: {str(err.get('input'))[:50]}...)")
            logging.warning(f"AI Batch JSON validation failed ({model_name}). Errors: {'; '.join(error_details)}. Data snippet: {short_data}")
            return {}

class FallbackAnalyzer:
    @staticmethod
    async def run(text: str, state_learning: Any) -> Tuple:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        slug = f"fallback_{text_hash}"
        return (0.0, "neutral", [], slug, False, "Fallback", 0.5, "", "", None, None)

class AIProvider:
    def __init__(self, rotator: Any, state: Any):
        self.rotator = rotator
        self.state = state

    async def call(self, prompt: str, session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
        active = self.rotator.get_active()
        provider = active.get("provider", "gemini")
        limiter = getattr(self.state, f"{provider}_limiter")

        async with limiter:
            if provider == "gemini":
                kwargs = {"model": active["name"], "contents": prompt}
                if active.get("supports_json"):
                    kwargs["config"] = {"response_mime_type": "application/json"}
                response = await self.state.ai_client.aio.models.generate_content(**kwargs)
                return response.text, active["name"]

            if provider == "ollama":
                logging.info(f"🏠 [LOCAL AI] Отправка запроса к Ollama (модель: {active['name']})...")
                url = f"{config.OLLAMA_BASE_URL}/api/chat"
                payload = {
                    "model": active["name"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
                if active.get("supports_json"):
                    payload["format"] = "json"
                
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_body = await resp.text()
                        raise Exception(f"Ollama returned status {resp.status}: {error_body}")
                    res_json = await resp.json()
                    logging.info(f"✅ [LOCAL AI] Локальный анализ завершен ({active['name']})")
                    return res_json['message']['content'], active["name"]

            api_url = (
                "https://openrouter.ai/api/v1/chat/completions"
                if provider == "openrouter"
                else "https://api.deepseek.com/chat/completions"
            )
            api_key = config.OPENROUTER_API_KEY if provider == "openrouter" else config.DEEPSEEK_API_KEY
            payload = {"model": active["name"], "messages": [{"role": "user", "content": prompt}]}
            if active.get("supports_json"):
                payload["response_format"] = {"type": "json_object"}
            async with session.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise Exception(f"API {provider} returned status {resp.status}: {error_body}")
                
                res_json = await resp.json()
                
                # Проверка на наличие сообщения об ошибке внутри JSON (даже при статусе 200)
                if 'error' in res_json:
                    err_msg = res_json['error'].get('message', 'Unknown provider error')
                    err_code = res_json['error'].get('code', 'N/A')
                    raise Exception(f"{provider.upper()} API Error {err_code}: {err_msg}")

                if 'choices' not in res_json or not res_json['choices']:
                    raise Exception(f"Unexpected JSON structure from {provider}: {res_json}")
                
                return res_json['choices'][0]['message']['content'], active["name"]

async def ai_analyze_batch(
    news_batch: List[Dict], rotator: Any, state: Any, session: aiohttp.ClientSession
) -> List[Tuple]:
    """Анализирует пакет новостей за один вызов ИИ."""
    prompt = PromptBuilder.build_analysis_prompt(news_batch)
    provider = AIProvider(rotator, state)
    state.metrics.metrics["ai_requests"] += 1

    for attempt in range(3):
        model_name = rotator.get_active()["name"]
        try:
            start_time = time.time()
            # Добавляем жесткий таймаут на ответ от любого провайдера ИИ
            res_text, _ = await asyncio.wait_for(
                provider.call(prompt, session), timeout=90
            )
            state.metrics.ai_timings.append(time.time() - start_time)
            data = ResponseParser.parse(res_text)
            if data:
                results_map = ResponseParser.validate_and_extract_batch(data, model_name)
                
                # СТРОГАЯ ПРОВЕРКА: Количество ответов ИИ должно строго совпадать с размером батча
                if results_map and len(results_map) == len(news_batch):
                    try:
                        return [results_map[i] for i in range(len(news_batch))]
                    except KeyError:
                        logging.warning(f"AI returned invalid IDs (mapping failed) from {model_name}. Rotating...")
                else:
                    actual_count = len(results_map) if results_map else 0
                    logging.warning(f"AI incomplete response ({actual_count}/{len(news_batch)}) from {model_name}. Rotating...")

            await rotator.rotate(state)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Пытаемся вытянуть время ожидания из текста ошибки
                retry_match = re.search(r"retry in ([\d.]+)s", err_str)
                wait_info = f" (Retry in {retry_match.group(1)}s)" if retry_match else ""
                clean_error = f"429 Resource Exhausted{wait_info}"
            else:
                # Обрезаем слишком длинные сообщения об ошибках
                clean_error = (err_str[:150] + "...") if len(err_str) > 150 else err_str
            logging.warning(f"AI attempt {attempt + 1}/3 failed ({model_name}): {clean_error}")
            await rotator.rotate(state)

    return [await FallbackAnalyzer.run(item['text'], state.learning) for item in news_batch]

async def get_embedding(text: str, rotator: Any, state: Any, session: aiohttp.ClientSession) -> Optional[List[float]]:
    if config.GEMINI_API_KEY and state.ai_client:
        async with state.gemini_limiter:
            try:
                res = await asyncio.wait_for(
                    state.ai_client.aio.models.embed_content(model=config.EMBEDDING_MODEL, contents=text),
                    timeout=12,
                )
                return res.embeddings[0].values
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    pass # Лимит Gemini исчерпан, молча переходим к фоллбеку
                else:
                    logging.error(f"Gemini Embedding error: {e}")

    if config.OPENROUTER_API_KEY:
        async with state.openrouter_limiter:
            try:
                url = "https://openrouter.ai/api/v1/embeddings"
                headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
                payload = {"model": config.OPENROUTER_EMBEDDING_MODEL, "input": text}
                async with session.post(url, headers=headers, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data and data['data']:
                            return data['data'][0]['embedding']
                        else:
                            error_message = data.get('error', {}).get('message', 'Unknown error structure')
                            logging.error(f"OpenRouter Embedding error (200 OK): {error_message}")
                            # Возбуждаем исключение, чтобы оно было поймано и залогировано как Fallback Embedding error
                            raise Exception(f"OpenRouter API returned success status but contained an error: {error_message}")
                    else:
                        logging.error(f"OpenRouter Embedding error status: {resp.status}")
            except Exception as e:
                logging.error(f"Fallback Embedding error: {e}")

    return None

async def is_semantic_duplicate(title: str, embedding: List[float], state: Any) -> bool:
    emb = np.asarray(embedding, dtype=float)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    async with state.cache.cache_lock:
        for cached_title, (cached_emb, ts) in state.cache.embeddings.items():
            # FIX: Исключаем сравнение новости с самой собой, если она уже попала в кэш
            if title == cached_title:
                continue

            if time.time() - ts > config.SEMANTIC_DEDUPLICATION_WINDOW * 3600:
                continue
            cached = np.asarray(cached_emb, dtype=float)
            
            # Если размерности векторов не совпадают (смена модели или fallback), сравнение невозможно
            if emb.shape != cached.shape:
                continue

            # Нормализация для кэшированного вектора, если он еще не нормализован
            c_norm = np.linalg.norm(cached)
            if c_norm > 0:
                cached = cached / c_norm
            
            sim = float(np.dot(emb, cached))
            if sim > config.SEMANTIC_DUPLICATE_THRESHOLD:
                logging.info(f"Semantic duplicate ({sim:.3f}): '{title}' ~ '{cached_title}'")
                return True
    return False
