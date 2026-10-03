"""Tests for configuration loading and validation."""

import pytest
import yaml
from pydantic import ValidationError

from eindex.config import Config, load_config


def test_load_valid_config(tmp_path):
    cfg_data = {
        "start_date": "2015-01-01",
        "rebalance_freq": "ME",
        "lookback_cov_days": 504,
        "signal_weights": {"momentum": 0.4, "low_vol": 0.3, "reversal": 0.3},
        "ic": 0.05,
        "risk_aversion": 10.0,
        "max_active_stock": 0.02,
        "max_active_sector": 0.03,
        "max_te_annual": 0.03,
        "max_turnover_oneway": 0.20,
        "cost_bps": 10.0,
        "min_weight": 0.0,
        "random_seed": 42,
        "tickers": ["RELIANCE.NS", "TCS.NS"],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data))

    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.risk_aversion == 10.0
    assert len(cfg.tickers) == 2


def test_invalid_signal_weights_raise(tmp_path):
    cfg_data = {
        "start_date": "2015-01-01",
        "rebalance_freq": "ME",
        "lookback_cov_days": 504,
        "signal_weights": {"momentum": 0.5, "low_vol": 0.5, "reversal": 0.5},
        "ic": 0.05,
        "risk_aversion": 10.0,
        "max_active_stock": 0.02,
        "max_active_sector": 0.03,
        "max_te_annual": 0.03,
        "max_turnover_oneway": 0.20,
        "cost_bps": 10.0,
        "min_weight": 0.0,
        "random_seed": 42,
        "tickers": ["RELIANCE.NS"],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data))

    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")
