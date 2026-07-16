import asyncio
import aiohttp
import textwrap
from engine import ai_analyze, ModelRotator, init_model_pool, GTSStateManager
from typing import List, Dict

async def run_test_for_provider(provider_name: str, test_cases: List[Dict[str, str]]):
    """Запускает тесты для указанного провайдера."""
    state = GTSStateManager()
    full_pool = init_model_pool()
    provider_pool = [m for m in full_pool if m['provider'] == provider_name]

    if not provider_pool:
        print(f"--- 🚦 ПРОВАЙДЕР '{provider_name.upper()}' ПРОПУЩЕН (модели не найдены или API ключ отсутствует) ---\n")
        return

    rotator = ModelRotator(provider_pool)

    print(f"--- 🚀 ТЕСТИРОВАНИЕ ПРОВАЙДЕРА: {provider_name.upper()} (Модель: {rotator.get_active()['name']}) ---")
    
    async with aiohttp.ClientSession() as session:
        for i, case in enumerate(test_cases):
            print(f"\n--- കേസ് #{i+1}: {case['description']} ---")
            print(f"Текст: {case['text']}")

            try:
                res = await ai_analyze(case['text'], rotator, state, session=session)
                score, event_type, entities, slug, is_swan, model, conf, summary, title_ru, capex_sig, guidance_sig = res

                if score is not None:
                    print(f"✅ Успех!")
                    print(f"   - Модель: {model}")
                    print(f"   - Score: {score:.2f} | Тип: {event_type} | Уверенность: {conf:.2f}")
                    print(f"   - Slug: {slug} | Сущности: {entities}")
                    print(f"   - Black Swan: {'ДА' if is_swan else 'Нет'}")
                    print(f"   - Capex/Guidance: Capex={capex_sig}, Guidance={guidance_sig}")
                    print(f"   - Перевод: {title_ru}")
                    if summary:
                        print(f"   - Саммари:\n{textwrap.indent(summary, '     ')}")
                else:
                    print("❌ Модель не вернула результат.")
            except Exception as e:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА при тесте: {e}")
                # При ошибке (например, 429) переходим к следующему провайдеру
                break
    print("-" * 50 + "\n")

async def main():
    test_cases = [
        {"description": "Позитивная новость по акциям (Risk-On)", "text": "Nvidia reports record breaking earnings, stock jumps 10% in after-hours trading."},
        {"description": "Геополитика (Risk-Off)", "text": "US and Iran on brink of war after Hormuz strait incident, oil prices surge."},
        {"description": "Макроэкономика (Risk-Off)", "text": "Fed hints at another rate hike amid persistent inflation data."},
        {"description": "Сигнал по CAPEX", "text": "Microsoft announces major capex increase for AI cloud infrastructure to meet demand."},
        {"description": "Сигнал по Guidance", "text": "Micron (MU) cuts future guidance citing weak demand for memory chips."},
        {"description": "Нейтральная новость (обзор)", "text": "This is a weekly market recap and outlook for the next week. We will see what happens."},
        {"description": "Нейтральная новость (мнение)", "text": "Expert says that the stock market could go up or down in the next months depending on various factors."},
    ]

    # Последовательно тестируем каждого провайдера
    await run_test_for_provider("gemini", test_cases)
    await run_test_for_provider("openrouter", test_cases)
    await run_test_for_provider("deepseek", test_cases)

if __name__ == "__main__":
    # Добавляем обработку KeyboardInterrupt для чистого выхода
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nТестирование прервано пользователем.")