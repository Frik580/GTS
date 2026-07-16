import logging
from typing import Dict, List, Any, Tuple, Optional
import config

class SOXSQuantEngine:
    """
    Изолированный квантовый движок расчета циклической слабости полупроводников (SOXX)
    и динамического управления хеджирующей позицией SOXS.
    """
    def __init__(self):
        self.logger = logging.getLogger("GTS.QuantEngine")
        
    def calculate_bear_probability(
        self, 
        capex_signals: Dict[str, Tuple[int, Optional[str], Optional[str], Optional[str]]], 
        guidance_signals: Dict[str, Tuple[int, Optional[str], Optional[str], Optional[str]]], 
        divergence_instances: int,      # шкала 0..10 (Price Confirmation Divergence)
        rotation_indicator: int,         # шкала 0..10 (Leadership Fatigue)
        soxx_below_ma200: bool
    ) -> Dict[str, Any]:
        """
        Рассчитывает взвешенную вероятность медвежьего разворота на основе мультифакторной матрицы.
        """
        # 1. Рассчитываем взвешенный индекс CAPEX
        capex_score = 0.0
        for company, weight in config.SOXS_CAPEX_WEIGHTS.items():
            sig = capex_signals.get(company, (0, None, None, None))[0]  # Извлекаем только сам сигнал
            capex_score += sig * weight

        # 2. Рассчитываем взвешенный индекс Guidance
        guidance_score = 0.0
        for company, weight in config.SOXS_GUIDANCE_WEIGHTS.items():
            sig = guidance_signals.get(company, (0, None, None, None))[0] # Извлекаем только сам сигнал
            guidance_score += sig * weight

        # Нормализация поведенческих метрик (шкала 0-10 -> 0.0-1.0)
        norm_divergence = min(10.0, max(0.0, divergence_instances)) / 10.0
        norm_rotation = min(10.0, max(0.0, rotation_indicator)) / 10.0

        # 3. Вычисляем совокупный медвежий балл (Bear Score)
        # Отрицательные фундаментальные сдвиги (capex_score < 0) повышают ценность лонг-позиции SOXS (защита)
        w = config.SOXS_FACTOR_WEIGHTS
        
        capex_impact = -capex_score * w["capex"]
        guidance_impact = -guidance_score * w["guidance"]
        divergence_impact = norm_divergence * w["divergence"]
        rotation_impact = norm_rotation * w["rotation"]
        ma200_impact = w["ma200_trend"] if soxx_below_ma200 else 0

        bear_score = capex_impact + guidance_impact + divergence_impact + rotation_impact + ma200_impact

        self.logger.info(
            f"SOXS Quant Calculation: Capex={capex_impact:.2f}, Guidance={guidance_impact:.2f}, "
            f"Divergence={divergence_impact:.2f}, Rotation={rotation_impact:.2f}, MA200={ma200_impact:.2f} "
            f"-> Total Bear Score: {bear_score:.2f} (SOXX below MA200: {soxx_below_ma200})"
        )

        # 4. Нормализация сырого балла в диапазон вероятности [0% - 100%]
        # Нейтральный фон дает около 30% вероятности (базовый уровень)
        bear_prob = min(100.0, max(0.0, round(30.0 + bear_score, 1)))

        # 5. Определение размера позиции по шкале лимитов
        position_pct = 0.0
        decision_name = "0% Position (Bullish Regime)"
        for scale in config.SOXS_POSITION_LEVELS:
            if bear_prob < scale["limit"]:
                position_pct = scale["position"]
                decision_name = scale["name"]
                break

        # Сбор активных триггеров для понимания контекста
        active_triggers = []
        if capex_signals.get("MSFT", (0,))[0] == -1: active_triggers.append("❌ MSFT CAPEX Cut")
        if capex_signals.get("META", (0,))[0] == -1: active_triggers.append("❌ META CAPEX Cut")
        if capex_signals.get("AMZN", (0,))[0] == -1: active_triggers.append("❌ AMZN CAPEX Cut")
        if capex_signals.get("GOOGL", (0,))[0] == -1: active_triggers.append("❌ GOOGL CAPEX Cut")
        
        if guidance_signals.get("NVDA", (0,))[0] == -1: active_triggers.append("📉 NVDA guidance downgrade")
        if guidance_signals.get("AVGO", (0,))[0] == -1: active_triggers.append("📉 AVGO guidance downgrade")
        if guidance_signals.get("AMD", (0,))[0] == -1: active_triggers.append("📉 AMD guidance downgrade")
        if guidance_signals.get("MU", (0,))[0] == -1: active_triggers.append("📉 MU guidance downgrade")
        
        if divergence_instances > 4: active_triggers.append(f"⚠️ Price Confirmation Divergence ({divergence_instances}/10)")
        if rotation_indicator > 4: active_triggers.append(f"🔄 Leadership Fatigue: Rotation into 2nd tier ({rotation_indicator}/10)")
        if soxx_below_ma200: active_triggers.append("⚫ SOXX is Below its 200-day Moving Average")

        return {
            "bear_probability": bear_prob,
            "target_position_percent": position_pct,
            "verdict_name": decision_name,
            "active_triggers": active_triggers,
            "capex_score": round(capex_score, 2),
            "guidance_score": round(guidance_score, 2)
        }