#!/usr/bin/env python3
"""Adaptive momentum-reversion strategy template for the trading contest."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import math
import os
import sys

# Import base infrastructure from base-bot-template.
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "base-bot-template")
if not os.path.exists(BASE_PATH):
    # In Docker container, the base template is copied to /app/base
    BASE_PATH = "/app/base"

sys.path.insert(0, BASE_PATH)

from strategy_interface import BaseStrategy, Portfolio, Signal, register_strategy  # noqa: E402
from exchange_interface import MarketSnapshot  # noqa: E402


@dataclass
class _Indicators:
    fast_ema: float
    slow_ema: float
    rsi: float
    volatility: float
    momentum_pct: float


class AdaptiveMomentumStrategy(BaseStrategy):
    """Momentum strategy with volatility-aware risk controls."""

    def __init__(self, config: Dict[str, Any], exchange):
        super().__init__(config=config, exchange=exchange)
        self.fast_period = max(5, int(config.get("fast_period", 24)))
        self.slow_period = max(self.fast_period + 1, int(config.get("slow_period", 72)))
        self.rsi_period = max(5, int(config.get("rsi_period", 14)))
        self.rsi_buy = float(config.get("rsi_buy", 55.0))
        self.rsi_sell = float(config.get("rsi_sell", 45.0))
        self.momentum_threshold = float(config.get("momentum_threshold", 0.8))
        self.volatility_period = max(10, int(config.get("volatility_period", 48)))
        self.min_volatility = float(config.get("min_volatility", 0.003))
        self.stop_multiple = float(config.get("stop_multiple", 2.5))
        self.take_profit_multiple = float(config.get("take_profit_multiple", 3.0))
        self.trailing_stop_pct = float(config.get("trailing_stop_pct", 0.035))
        self.max_position_fraction = float(config.get("max_position_fraction", 0.35))
        self.risk_per_trade = float(config.get("risk_per_trade", 0.0125))
        self.min_trade_notional = float(config.get("min_trade_notional", 250.0))
        self.cooldown_minutes = max(0, int(config.get("cooldown_minutes", 120)))
        self.slippage_buffer = float(config.get("slippage_buffer", 0.0025))

        self._logger = logging.getLogger("strategy.adaptive_momentum")
        self._position_size = 0.0
        self._avg_entry_price = 0.0
        self._last_peak_price: Optional[float] = None
        self._last_signal_payload: Dict[str, Any] = {}
        self._last_trade_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

        # Public debugging hook consumed by dashboard logging when present.
        self.last_signal_data: Optional[Dict[str, Any]] = None

    def prepare(self) -> None:
        self._logger.info("AdaptiveMomentumStrategy ready with fast=%s slow=%s", self.fast_period, self.slow_period)

    def generate_signal(self, market: MarketSnapshot, portfolio: Portfolio) -> Signal:
        prices = market.prices
        now = self._ensure_aware(market.timestamp)

        # Expire cooldown automatically once the timeout elapses so new entries can form.
        if self._cooldown_until and now >= self._cooldown_until:
            self._cooldown_until = None

        if not self._enough_history(prices):
            self.last_signal_data = {"reason": "insufficient_history", "bars": len(prices)}
            return Signal("hold", reason="Insufficient history")

        indicators = self._compute_indicators(prices)
        volatility = max(indicators.volatility, self.min_volatility)
        equity_value = portfolio.cash + portfolio.quantity * market.current_price
        position_value = portfolio.quantity * market.current_price

        self.last_signal_data = {
            "price": round(market.current_price, 2),
            "fast_ema": round(indicators.fast_ema, 2),
            "slow_ema": round(indicators.slow_ema, 2),
            "rsi": round(indicators.rsi, 2),
            "volatility": round(volatility, 5),
            "momentum_pct": round(indicators.momentum_pct, 3),
            "position_value": round(position_value, 2),
            "cash": round(portfolio.cash, 2),
        }

        if portfolio.quantity <= 0:
            if self._cooldown_until and now < self._cooldown_until:
                self.last_signal_data["reason"] = "cooldown"
                return Signal("hold", reason="Cooling down")

            if self._should_enter_long(indicators):
                stop_price = market.current_price * (1 - self.stop_multiple * volatility)
                desired_notional = min(
                    equity_value * self.max_position_fraction,
                    portfolio.cash,
                )
                if desired_notional < self.min_trade_notional:
                    self.last_signal_data["reason"] = "notional_too_small"
                    return Signal("hold", reason="Capital too low")

                risk_per_unit = max(market.current_price - stop_price, market.current_price * 0.002)
                risk_budget = equity_value * self.risk_per_trade
                size_by_risk = risk_budget / risk_per_unit
                size_by_cash = desired_notional / market.current_price
                size = max(0.0, min(size_by_risk, size_by_cash))
                size *= max(0.0, 1.0 - self.slippage_buffer)

                if size <= 0:
                    self.last_signal_data["reason"] = "invalid_size"
                    return Signal("hold", reason="No capacity")

                target_price = market.current_price * (1 + self.take_profit_multiple * volatility)
                reason = f"Trend up (mom={indicators.momentum_pct:.2f}%, rsi={indicators.rsi:.1f})"
                self._last_signal_payload = {
                    "entry_price": market.current_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                }
                self.last_signal_data.update({
                    "action": "buy",
                    "size": size,
                    "stop": round(stop_price, 2),
                    "target": round(target_price, 2),
                })
                return Signal(
                    "buy",
                    size=size,
                    reason=reason,
                    target_price=target_price,
                    stop_loss=stop_price,
                )

            self.last_signal_data["reason"] = "entry_filters_not_met"
            return Signal("hold", reason="Entry filters not met")

        exit_signal = self._maybe_exit_position(
            market_price=market.current_price,
            indicators=indicators,
            volatility=volatility,
            portfolio=portfolio,
        )
        if exit_signal:
            return exit_signal

        self.last_signal_data["reason"] = "holding"
        return Signal("hold", reason="Holding position")

    def _should_enter_long(self, indicators: _Indicators) -> bool:
        return (
            indicators.fast_ema > indicators.slow_ema
            and indicators.momentum_pct >= self.momentum_threshold
            and indicators.rsi >= self.rsi_buy
        )

    def _maybe_exit_position(
        self,
        *,
        market_price: float,
        indicators: _Indicators,
        volatility: float,
        portfolio: Portfolio,
    ) -> Optional[Signal]:
        reasons: List[str] = []
        trailing_stop_price = None

        if self._avg_entry_price > 0:
            stop_price = self._avg_entry_price * (1 - self.stop_multiple * volatility)
        else:
            stop_price = market_price * (1 - self.stop_multiple * volatility)

        if self._last_peak_price is None:
            self._last_peak_price = market_price
        else:
            self._last_peak_price = max(self._last_peak_price, market_price)

        trailing_stop_price = self._last_peak_price * (1 - self.trailing_stop_pct)
        protective_stop = max(stop_price, trailing_stop_price)

        if market_price <= protective_stop:
            reasons.append("stop_loss")

        if indicators.fast_ema <= indicators.slow_ema:
            reasons.append("trend_reversal")

        if indicators.rsi <= self.rsi_sell:
            reasons.append("rsi_exit")

        take_profit_price = self._avg_entry_price * (1 + self.take_profit_multiple * volatility)
        if self._avg_entry_price > 0 and market_price >= take_profit_price:
            reasons.append("take_profit")

        if not reasons:
            return None

        size = max(0.0, portfolio.quantity)
        if size <= 0:
            return None

        reason_text = ",".join(reasons)
        self.last_signal_data.update({
            "action": "sell",
            "size": size,
            "stop": round(protective_stop, 2),
            "reasons": reason_text,
        })
        return Signal(
            "sell",
            size=size,
            reason=reason_text,
            stop_loss=protective_stop,
            entry_price=self._avg_entry_price,
        )

    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        timestamp = self._ensure_aware(timestamp)
        self._last_trade_time = timestamp
        self._cooldown_until = timestamp + timedelta(minutes=self.cooldown_minutes)

        if signal.action == "buy" and execution_size > 0:
            total_size = self._position_size + execution_size
            notional = self._avg_entry_price * self._position_size + execution_price * execution_size
            self._position_size = total_size
            self._avg_entry_price = notional / total_size if total_size > 0 else 0.0
            self._last_peak_price = execution_price
        elif signal.action == "sell" and execution_size > 0:
            self._position_size = max(0.0, self._position_size - execution_size)
            if self._position_size <= 0:
                self._avg_entry_price = 0.0
                self._last_peak_price = None

    def get_state(self) -> Dict[str, Any]:
        return {
            "position_size": self._position_size,
            "avg_entry_price": self._avg_entry_price,
            "last_peak_price": self._last_peak_price,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._position_size = float(state.get("position_size", 0.0))
        self._avg_entry_price = float(state.get("avg_entry_price", 0.0))
        self._last_peak_price = state.get("last_peak_price")
        cooldown = state.get("cooldown_until")
        if cooldown:
            self._cooldown_until = self._ensure_aware(datetime.fromisoformat(cooldown))
        last_trade = state.get("last_trade_time")
        if last_trade:
            self._last_trade_time = self._ensure_aware(datetime.fromisoformat(last_trade))

    def _enough_history(self, prices: List[float]) -> bool:
        needed = max(self.slow_period + 5, self.volatility_period + 5, self.rsi_period + 5)
        return len(prices) >= needed

    def _compute_indicators(self, prices: List[float]) -> _Indicators:
        fast_ema = self._ema(prices, self.fast_period)
        slow_ema = self._ema(prices, self.slow_period)
        rsi_value = self._rsi(prices, self.rsi_period)
        volatility = self._volatility(prices, self.volatility_period)
        momentum_pct = (fast_ema / slow_ema - 1.0) * 100 if slow_ema > 0 else 0.0
        return _Indicators(
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            rsi=rsi_value,
            volatility=volatility,
            momentum_pct=momentum_pct,
        )

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        if not values:
            return 0.0
        if len(values) <= period:
            return sum(values) / len(values)
        k = 2.0 / (period + 1)
        ema_value = sum(values[:period]) / period
        for price in values[period:]:
            ema_value = price * k + ema_value * (1.0 - k)
        return ema_value

    @staticmethod
    def _rsi(values: List[float], period: int) -> float:
        if len(values) <= period:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(-period, 0):
            change = values[i] - values[i - 1]
            if change >= 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100.0
        rs = (gains / period) / (losses / period)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _volatility(values: List[float], period: int) -> float:
        if len(values) <= period:
            return 0.0
        window = values[-period:]
        returns = []
        for prev, curr in zip(window, window[1:]):
            if prev > 0:
                returns.append((curr - prev) / prev)
        if not returns:
            return 0.0
        mean_return = sum(returns) / len(returns)
        squared = [(r - mean_return) ** 2 for r in returns]
        variance = sum(squared) / max(1, len(squared) - 1)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _ensure_aware(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)


# Register the strategy so UniversalBot can load it via config.
register_strategy("adaptive_momentum", lambda cfg, ex: AdaptiveMomentumStrategy(cfg, ex))
