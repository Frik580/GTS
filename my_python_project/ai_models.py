from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

class NewsAnalysisResponse(BaseModel):
    """
    Pydantic модель для валидации структурированного ответа от AI.
    """
    id: int
    primary_asset: str
    score: float = Field(..., ge=-10.0, le=10.0) # Ограничиваем ИИ диапазоном [-10, 10]
    event_type: str
    entities: List[str]
    slug: str
    is_black_swan: bool
    confidence: float = Field(..., ge=0.0, le=10.0) # Позволяем принять до 10, чтобы потом исправить
    summary: str
    title_ru: str
    capex_signal: Optional[int] = Field(None, description="1: increase, 0: stable, -1: decrease")
    guidance_signal: Optional[int] = Field(None, description="1: upgrade, 0: stable, -1: downgrade")

    def to_analysis_tuple(self, model_name: str) -> Tuple:
        safe_score = max(-10.0, min(10.0, self.score))
        safe_conf = self.confidence / 10.0 if self.confidence > 1.0 else max(0.0, self.confidence)
        
        return (
            safe_score,
            self.event_type,
            self.entities,
            self.slug,
            self.is_black_swan,
            model_name,
            safe_conf,
            self.summary,
            self.title_ru,
            # Добавляем новые параметры в возвращаемый кортеж
            self.capex_signal,
            self.guidance_signal
        )

class BatchNewsResponse(BaseModel):
    """Модель для валидации пакета новостей."""
    items: List[NewsAnalysisResponse]