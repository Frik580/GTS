from google import genai
import config

def check_available_models():
    """Выводит список доступных моделей и их лимиты токенов."""
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    print(f"{'Display Name':<30} | {'Model Name':<40} | {'Input Limit':<12}")
    print("-" * 85)
    
    try:
        for m in client.models.list():
            if 'generateContent' in m.supported_actions or 'embedContent' in m.supported_actions:
                # input_token_limit - это размер контекстного окна (сколько текста модель примет)
                print(f"{m.display_name:<30} | {m.name:<40} | {m.input_token_limit:<12}")
    except Exception as e:
        print(f"Ошибка при получении списка моделей: {e}")
        if "API key not valid" in str(e):
            print("Проверьте ваш GEMINI_API_KEY в файле .env")

if __name__ == "__main__":
    print("Проверка доступных моделей Gemini...\n")
    check_available_models()
    print("\nПримечание: Лимиты RPM (запросы в минуту) зависят от вашего тарифа (Free/Pay-as-you-go) в AI Studio.")