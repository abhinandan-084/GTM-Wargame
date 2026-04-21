from pydantic import BaseModel, Field, field_validator, ValidationInfo
from datetime import date
from typing import Optional, List, Dict

class OEMTierConfig(BaseModel):
    rank: int = Field(..., ge=1, le=3)
    hill_k: float
    hill_n: float

class DataConfig(BaseModel):
    start_date: date
    weeks: int
    base_price: float
    random_state: int
    launch_spike: float
    market_leader_price_decay_factor: float

class PromoEvent(BaseModel):
    start_date: date
    end_date: date
    multiplier: float = Field(..., ge=0.0)

    @field_validator('end_date')
    def validate_dates(cls, end_date, info: ValidationInfo):
        start_date = info.data.get('start_date')
        if start_date and end_date < start_date:
            raise ValueError('end_date must be after start_date')
        return end_date
        