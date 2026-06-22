import asyncio
import aiohttp
from engine import ai_analyze, ModelRotator, init_model_pool, GTSStateManager

async def test_deepseek():
    state = GTSStateManager()
    # Инициализируем пул и оставляем только DeepSeek для теста
    full_pool = init_model_pool()
    ds_pool = [m for m in full_pool if m['provider'] == 'deepseek']
    
    if not ds_pool:
        print("❌ Ошибка: Модели DeepSeek не найдены в пуле. Проверьте API ключ в .env")
        return

    rotator = ModelRotator(ds_pool)
    test_text = "Nvidia reports record breaking earnings, stock jumps 10% in after-hours trading."
    
    print(f"🚀 Тестируем DeepSeek (модель: {rotator.get_active()['name']})...")
    
    async with aiohttp.ClientSession() as session:
        res = await ai_analyze(test_text, rotator, state, session=session)
        
    score, event_type, entities, slug, is_swan, model, conf, summary, title_ru, capex_sig, guidance_sig = res
    
    if score is not None:
        print(f"✅ Успех!")
        print(f"Модель: {model}")
        print(f"Скор: {score} | Тип: {event_type}")
        print(f"Сущности: {entities}")
        print(f"Перевод: {title_ru}")
    else:
        print("❌ Модель не вернула результат.")

if __name__ == "__main__":
    asyncio.run(test_deepseek())
