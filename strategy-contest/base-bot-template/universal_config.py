#!/usr/bin/env python3
"""Small configuration helper for the simplified universal bot."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Cannot convert '{value}' to float") from exc


def _to_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Cannot convert '{value}' to int") from exc


@dataclass
class BotConfig:
    """Holds the minimal configuration required by the bot."""

    exchange: str = "paper"
    strategy: str = "dca"
    symbol: str = "BTC-USD"
    history: int = 200
    starting_cash: float = 1_000.0
    sleep_seconds: float = 2.0
    max_cycles: Optional[int] = None
    http_port: int = 8080
    control_port: int = 3010
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    exchange_params: Dict[str, Any] = field(default_factory=dict)
    bot_instance_id: Optional[str] = None
    user_id: Optional[str] = None
    bot_secret: Optional[str] = None
    base_url: Optional[str] = None
    database_url: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BotConfig":
        path = path or os.getenv("BOT_CONFIG", "bot.config.json")
        data: Dict[str, Any] = {}

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                file_data = json.load(handle)
                if not isinstance(file_data, dict):
                    raise ValueError("Configuration file must contain a JSON object")
                data.update(file_data)

        data.update(cls._env_overrides())
        return cls(**data)

    @staticmethod
    def _env_overrides() -> Dict[str, Any]:
        mapping = {
            "BOT_EXCHANGE": ("exchange", str),
            "BOT_STRATEGY": ("strategy", str),
            "BOT_SYMBOL": ("symbol", str),
            "BOT_HISTORY": ("history", _to_int),
            "BOT_STARTING_CASH": ("starting_cash", _to_float),
            "BOT_SLEEP": ("sleep_seconds", _to_float),
            "BOT_MAX_CYCLES": ("max_cycles", _to_int),
            "BOT_HTTP_PORT": ("http_port", _to_int),
            "BOT_CONTROL_PORT": ("control_port", _to_int),
            "BOT_INSTANCE_ID": ("bot_instance_id", str),
            "USER_ID": ("user_id", str),
            "BOT_SECRET": ("bot_secret", str),
            "BASE_URL": ("base_url", str),
            "POSTGRES_URL": ("database_url", str),
            "DATABASE_URL": ("database_url", str),
        }

        overrides: Dict[str, Any] = {}
        for env_name, (config_key, caster) in mapping.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                overrides[config_key] = caster(env_value)

        for env_name, config_key in (
            ("BOT_STRATEGY_PARAMS", "strategy_params"),
            ("BOT_EXCHANGE_PARAMS", "exchange_params"),
        ):
            env_value = os.getenv(env_name)
            if env_value:
                overrides[config_key] = json.loads(env_value)

        # Handle strategy-specific parameters
        strategy_params = overrides.get("strategy_params", {})

        # DCA strategy parameters
        if os.getenv("BASE_AMOUNT"):
            strategy_params["base_amount"] = _to_float(os.getenv("BASE_AMOUNT"))
        if os.getenv("INTERVAL_MINUTES"):
            strategy_params["interval_minutes"] = _to_int(os.getenv("INTERVAL_MINUTES"))

        # Momentum strategy parameters
        if os.getenv("MOMENTUM_THRESHOLD"):
            strategy_params["momentum_threshold"] = _to_float(os.getenv("MOMENTUM_THRESHOLD"))
        if os.getenv("MOMENTUM_PERIOD"):
            strategy_params["momentum_period"] = _to_int(os.getenv("MOMENTUM_PERIOD"))
        if os.getenv("VOLUME_THRESHOLD"):
            strategy_params["volume_threshold"] = _to_float(os.getenv("VOLUME_THRESHOLD"))

        # Grid strategy parameters
        if os.getenv("AMOUNT"):
            strategy_params["amount"] = _to_float(os.getenv("AMOUNT"))
        if os.getenv("GRID_SIZE"):
            strategy_params["grid_size"] = _to_float(os.getenv("GRID_SIZE"))
        if os.getenv("GRID_COUNT"):
            strategy_params["grid_count"] = _to_int(os.getenv("GRID_COUNT"))
        if os.getenv("MAX_ORDERS"):
            strategy_params["max_orders"] = _to_int(os.getenv("MAX_ORDERS"))

        # Scalping strategy parameters
        if os.getenv("TRADE_AMOUNT"):
            strategy_params["trade_amount"] = _to_float(os.getenv("TRADE_AMOUNT"))
        if os.getenv("SCALP_TARGET"):
            strategy_params["scalp_target"] = _to_float(os.getenv("SCALP_TARGET"))

        if strategy_params:
            overrides["strategy_params"] = strategy_params

        return overrides

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key == "strategy_params" and isinstance(value, dict):
                self.strategy_params.update(value)
            elif key == "exchange_params" and isinstance(value, dict):
                self.exchange_params.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)

        if isinstance(self.max_cycles, int) and self.max_cycles <= 0:
            self.max_cycles = None
