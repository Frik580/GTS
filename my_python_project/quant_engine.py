# quant_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional

def analyze_soxs_strategy(
    score: float, 
    capex_sig: Optional[int], 
    guidance_sig: Optional[int], 
    market_data: Dict[str, Any]
) -> Tuple[int, List[str], str]:
    """
    Выполняет изолированный расчет сигналов и уровней для SOXS-стратегии.
    
    Входные параметры:
        - score: Итоговый балл новости от ИИ [-10..10]
        - capex_sig: Сигнал CAPEX от ИИ (1: рост, 0: стаб, -1: снижение, None)
        - guidance_sig: Сигнал Guidance от ИИ (1: апгрейд, 0: стаб, -1: даунгрейд, None)
        - market_data: Текущий срез рыночных данных из конвейера
        
    Возвращает:
        - soxs_level: Вычисленный уровень фазы (1, 2, 3, 4)
        - soxs_signals: Список сработавших триггерных строк для Telegram
        - soxs_verdict: Форматированный вердикт для Telegram-сообщения
    """
    soxs_signals = []
    soxs_level = 1  # По умолчанию - Уровень 1 (Шум)

    # 1. Проверяем сигналы сжатия (Уровень 2: Сжатие / Ротация)
    # Пример: если геополитическое напряжение или рыночный стресс растет, но direct-сигналов еще нет
    global_change = market_data.get("global_change", 0.0)
    if global_change > 1.5 and abs(score) > 3.0:
        soxs_signals.append("🌀 Рост глобального стресса (Сжатие)")
        soxs_level = max(soxs_level, 2)

    # 2. Проверка CAPEX (Уровень 3: Поворот)
    if capex_sig == -1:
        soxs_signals.append("❌ Снижение CAPEX гиперскейлеров")
        soxs_level = max(soxs_level, 3)
    
    # 3. Проверка Guidance NVDA/AVGO (Уровень 3: Поворот)
    if guidance_sig == -1:
        soxs_signals.append("📉 Ухудшение Guidance (NVDA/AVGO)")
        soxs_level = max(soxs_level, 3)
        
    # 4. Дивергенция (Уровень 3: Поворот)
    # Позитивная новость (Negative score) + падение индекса Nasdaq
    nasdaq_change = market_data.get("nasdaq_change", 0.0)
    if score < -3.5 and nasdaq_change < -0.5:
        soxs_signals.append("⚠️ Дивергенция (нет реакции на позитив)")
        soxs_level = max(soxs_level, 3)

    # 5. Медвежий режим рынка (Уровень 4: Медвежий рынок)
    if market_data.get('soxx_below_ma200'):
        soxs_signals.append("⚫ SOXX ниже 200-дневной средней")
        soxs_level = 4

    # --- ФОРМИРОВАНИЕ ТЕКСТОВОГО ВЕРДИКТА ---
    if soxs_level == 1:
        soxs_verdict = "🟡 <b>УРОВЕНЬ 1: ШУМ</b> (Не трогать SOXS)"
    elif soxs_level == 2:
        soxs_verdict = "🟠 <b>УРОВЕНЬ 2: СЖАТИЕ</b> (Пробная позиция 20-30%)"
    elif soxs_level == 3:
        soxs_verdict = "🔴 <b>УРОВЕНЬ 3: ПОВОРОТ</b> (Увеличить до ядра 50-100%)"
    elif soxs_level == 4:
        soxs_verdict = "⚫ <b>УРОВЕНЬ 4: МЕДВЕЖИЙ РЕЖИМ</b> (Агрессивный лонг SOXS)"
    else:
        soxs_verdict = "🟡 <b>УРОВЕНЬ 1: ШУМ</b> (Не трогать SOXS)"

    return soxs_level, soxs_signals, soxs_verdict