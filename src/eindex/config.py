"""Configuration models and loading utilities."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class SignalWeights(BaseModel):
    momentum: float = Field(gt=0.0, description="Momentum weight")
    low_vol: float = Field(gt=0.0, description="Low volatility weight")
    reversal: float = Field(gt=0.0, description="1-month reversal weight")

    @model_validator(mode="after")
    def validate_sum_to_one(self) -> "SignalWeights":
        total = self.momentum + self.low_vol + self.reversal
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Signal weights must sum to 1.0, got {total}")
        return self


class Config(BaseModel):
    start_date: str
    rebalance_freq: str
    lookback_cov_days: int = Field(gt=0)
    signal_weights: SignalWeights
    ic: float = Field(gt=0.0)
    risk_aversion: float = Field(gt=0.0)
    max_active_stock: float = Field(gt=0.0, le=1.0)
    max_active_sector: float = Field(gt=0.0, le=1.0)
    max_te_annual: float = Field(gt=0.0)
    max_turnover_oneway: float = Field(gt=0.0, le=1.0)
    cost_bps: float = Field(ge=0.0)
    min_weight: float = Field(ge=0.0, le=1.0)
    random_seed: int
    tickers: list[str]

    @field_validator("tickers")
    @classmethod
    def validate_tickers_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Ticker list cannot be empty")
        return v


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate config.yaml file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw_cfg: dict[str, Any] = yaml.safe_load(f)

    cfg = Config(**raw_cfg)
    logger.info("Successfully loaded and validated configuration from %s", config_path)
    return cfg
