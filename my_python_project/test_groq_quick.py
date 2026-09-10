import asyncio
from dotenv import load_dotenv
from http_utils import create_session
from model_factory import init_model_pool, ModelRotator
from state_service import GTSStateManager
from ai_processor import ai_analyze_batch

load_dotenv()


async def main():
    state = GTSStateManager()
    rotator = ModelRotator(init_model_pool())
    active = rotator.get_active()
    print(f"Active: {active['provider']} / {active['name']}")

    batch = [{
        "text": "Fed raises interest rates by 25bps amid sticky inflation.",
        "pub_time": "2026-07-31 08:00",
        "entry": {},
        "market_data": {},
    }]
    async with create_session() as session:
        res = await ai_analyze_batch(batch, rotator, state, session)

    score, event_type, entities, slug, is_swan, model, conf, summary, title_ru, _, _ = res[0]
    print(f"Model: {model} | Score: {score} | Type: {event_type} | Conf: {conf}")
    print(f"Slug: {slug} | Title: {title_ru}")
    if model == "Fallback":
        print("FAIL: fallback used")
        raise SystemExit(1)
    print("OK: Groq integration works")


if __name__ == "__main__":
    asyncio.run(main())
