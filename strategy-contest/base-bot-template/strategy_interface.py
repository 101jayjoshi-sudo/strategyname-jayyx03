#!/usr/bin/env python3
"""Strategy base classes plus factory and built-in strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Any, Callable, Deque, Dict, List, Optional
from collections import deque

from exchange_interface import Exchange, MarketSnapshot


@dataclass
class Signal:
    """Instruction returned by a strategy."""

    action: str  # "buy", "sell", or "hold"
    size: float = 0.0
    reason: str = ""

    # Trade context fields for database logging
    target_price: Optional[float] = None     # Expected target price for this trade
    stop_loss: Optional[float] = None        # Stop loss price for this trade
    entry_price: Optional[float] = None      # Entry price (for sell signals)


@dataclass
class Portfolio:
    """Very small in-memory portfolio representation."""

    symbol: str
    cash: float
    quantity: float = 0.0

    def value(self, price: float) -> float:
        return self.cash + self.quantity * price


class BaseStrategy(ABC):
    """Base class every concrete strategy extends."""

    def __init__(self, *, config: Dict[str, Any], exchange: Exchange):
        self.config = config
        self.exchange = exchange

    def prepare(self) -> None:
        """Allow strategies to warm up. Optional."""

    def get_state(self) -> Dict[str, Any]:
        """Optional hook to expose serialisable strategy state."""
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Optional hook to restore serialised strategy state."""

    @abstractmethod
    def generate_signal(self, market: MarketSnapshot, portfolio: Portfolio) -> Signal:
        """Inspect the market snapshot and return a trading instruction."""

    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        """Hook for strategies to update internal state after a fill."""


# --- Simple strategy factory -------------------------------------------------

StrategyFactory = Callable[[Dict[str, Any], Exchange], BaseStrategy]
_STRATEGIES: Dict[str, StrategyFactory] = {}


def register_strategy(name: str, factory: StrategyFactory) -> None:
    _STRATEGIES[name] = factory


def create_strategy(name: str, *, config: Dict[str, Any], exchange: Exchange) -> BaseStrategy:
    try:
        factory = _STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_STRATEGIES)) or "<none>"
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {available}. "
            "Register custom strategies with register_strategy(name, factory)."
        ) from exc
    return factory(config, exchange)


def available_strategies() -> List[str]:
    return sorted(_STRATEGIES)


# --- Built-in strategies removed - use specific bot templates instead ---
# This base template provides only the infrastructure.
# Strategies are implemented in specific bot templates:
# - dca-bot-template: DCA + AdvancedDCA strategies
# - swing-bot-template: Swing trading strategies
# - momentum-bot-template: Momentum trading strategies

# No built-in strategies registered - templates will import their own
