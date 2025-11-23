#!/usr/bin/env python3
"""Adaptive Momentum trading strategy implementation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# Ensure the base template is importable both locally and inside Docker
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "base-bot-template")
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/app/base"
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

from strategy_interface import BaseStrategy, Portfolio, Signal, register_strategy  # noqa: E402
from exchange_interface import MarketSnapshot  # noqa: E402


@dataclass
class TradeContext:
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    opened_at: datetime
    peak_price: float


class AdaptiveMomentumStrategy(BaseStrategy):
    """Adaptive momentum system that trades only when trend strength is clear."""

    def __init__(self, config: Dict[str, Any], exchange):
        super().__init__(config=config, exchange=exchange)
        cfg = config or {}
        self.fast_period = max(2, int(cfg.get("fast_period", 12)))
        self.slow_period = max(self.fast_period + 1, int(cfg.get("slow_period", 48)))
        self.momentum_period = max(2, int(cfg.get("momentum_period", 24)))
        self.atr_period = max(2, int(cfg.get("atr_period", 14)))
        self.risk_per_trade = max(0.001, float(cfg.get("risk_per_trade", 0.02)))
        self.max_position_pct = min(0.55, float(cfg.get("max_position_pct", 0.5)))
        self.cooldown_hours = max(0.0, float(cfg.get("cooldown_hours", 6.0)))
        self.reward_risk = max(1.0, float(cfg.get("reward_risk", 2.5)))
        self.momentum_threshold = float(cfg.get("momentum_threshold", 0.75))
        self.min_slope_pct = float(cfg.get("min_slope_pct", 0.15))
        self.trailing_atr_multiplier = float(cfg.get("trailing_atr_multiplier", 1.2))
        self.stop_atr_multiplier = float(cfg.get("stop_atr_multiplier", 1.0))
        self.exit_momentum_floor = float(cfg.get("exit_momentum_floor", 0.0))
        self.min_hold_hours = max(0.0, float(cfg.get("min_hold_hours", 12.0)))
        self.trend_down_confirm = max(1, int(cfg.get("trend_down_confirm", 6)))
        self.stop_pct = float(cfg.get("stop_pct", 0.0))
        self.take_profit_pct = float(cfg.get("take_profit_pct", 0.0))
        self.breakout_lookback = max(5, int(cfg.get("breakout_lookback", 48)))
        self.breakout_buffer_pct = float(cfg.get("breakout_buffer_pct", 0.25))
        self.rsi_period = max(2, int(cfg.get("rsi_period", 14)))
        self.rsi_entry = float(cfg.get("rsi_entry", 55.0))
        self.rsi_exit = float(cfg.get("rsi_exit", 72.0))
        self.rsi_reset = max(0.0, float(cfg.get("rsi_reset", 5.0)))
        self.min_atr_pct = max(0.0, float(cfg.get("min_atr_pct", 0.003)))
        self.max_layers = max(1, int(cfg.get("max_layers", 2)))
        self.layer_step_pct = max(0.0, float(cfg.get("layer_step_pct", 0.02)))
        peak_cfg = cfg.get("peak_trail_pct", 0.0)
        self._peak_map: Dict[str, float] = {}
        if isinstance(peak_cfg, dict):
            self._peak_map = {str(k): float(v) for k, v in peak_cfg.items()}
            default_peak = self._peak_map.get("default")
            if default_peak is None and self._peak_map:
                default_peak = next(iter(self._peak_map.values()))
            self.peak_trail_pct = max(0.0, float(default_peak or 0.0))
        else:
            self.peak_trail_pct = max(0.0, float(peak_cfg))
        self._peak_default = self.peak_trail_pct
        self.simple_mode = bool(cfg.get("peak_mode", False))
        self.simple_trend_period = max(0, int(cfg.get("trend_period", 0)))
        self.simple_trend_buffer = max(0.0, float(cfg.get("trend_buffer_pct", 0.0)))
        self.reentry_buffer_pct = max(0.0, float(cfg.get("reentry_buffer_pct", 0.0)))
        self.reentry_expire_hours = max(0.0, float(cfg.get("reentry_expire_hours", 0.0)))
        self._trend_down_counter: int = 0
        self._regime_active: bool = False
        self._last_entry: Optional[datetime] = None
        self._layers: int = 0
        self._next_layer_price: Optional[float] = None
        self._trade: Optional[TradeContext] = None
        self._last_exit: Optional[datetime] = None
        self._reentry_level: Optional[float] = None
        self._reentry_expire_at: Optional[datetime] = None

    # --- Helpers ---------------------------------------------------------

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period or period <= 1:
            return None
        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for price in values[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _atr(values: list[float], period: int) -> Optional[float]:
        if len(values) < period + 1:
            return None
        trs = []
        for prev, curr in zip(values[-(period + 1):-1], values[-period:]):
            trs.append(abs(curr - prev))
        if not trs:
            return None
        return sum(trs) / len(trs)

    @staticmethod
    def _momentum(values: list[float], lookback: int) -> Optional[float]:
        if len(values) <= lookback:
            return None
        past = values[-lookback - 1]
        if past == 0:
            return None
        return (values[-1] / past - 1) * 100

    @staticmethod
    def _rsi(values: list[float], period: int) -> Optional[float]:
        if len(values) <= period:
            return None
        gains = 0.0
        losses = 0.0
        start = len(values) - period
        for idx in range(start, len(values)):
            change = values[idx] - values[idx - 1]
            if change >= 0:
                gains += change
            else:
                losses += -change
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _slope_pct(fast: float, slow: float) -> float:
        if slow == 0:
            return 0.0
        return (fast - slow) / slow * 100

    def _cooldown_active(self, now: datetime) -> bool:
        if not self._last_exit:
            return False
        return (now - self._last_exit) < timedelta(hours=self.cooldown_hours)

    def _simple_peak_signal(
        self,
        market: MarketSnapshot,
        portfolio: Portfolio,
        prices: list[float],
        price: float,
        now: datetime,
    ) -> Signal:
        drop_pct = self._symbol_peak_pct(market.symbol)
        if drop_pct <= 0:
            return Signal("hold", reason="Peak drop disabled")

        if portfolio.quantity > 0 and not self._trade:
            stop_loss = price * (1 - drop_pct)
            take_profit = price * (1 + max(drop_pct * self.reward_risk, 0.0))
            self._trade = TradeContext(
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                opened_at=now,
                peak_price=price,
            )

        if portfolio.quantity > 0 and self._trade:
            self._trade.peak_price = max(self._trade.peak_price, price)
            threshold = self._trade.peak_price * (1 - drop_pct)
            if price <= threshold:
                if self.reentry_buffer_pct > 0:
                    self._reentry_level = price * (1 - self.reentry_buffer_pct)
                    self._reentry_expire_at = (
                        now + timedelta(hours=self.reentry_expire_hours)
                        if self.reentry_expire_hours > 0
                        else None
                    )
                else:
                    self._reentry_level = None
                    self._reentry_expire_at = None
                return self._exit_signal("Peak trail hit", portfolio, reset_regime=True)
            return Signal("hold", reason="Holding core position")

        if self._reentry_level is not None:
            if self._reentry_expire_at and now >= self._reentry_expire_at:
                self._reentry_level = None
                self._reentry_expire_at = None
            elif price > self._reentry_level:
                return Signal("hold", reason="Waiting for deeper pullback")
            else:
                self._reentry_level = None
                self._reentry_expire_at = None

        if self._cooldown_active(now):
            return Signal("hold", reason="Cooldown active")

        if self.simple_trend_period > 1:
            if len(prices) < self.simple_trend_period:
                return Signal("hold", reason="Trend warmup")
            trend = self._ema(prices, self.simple_trend_period)
            if trend is None:
                return Signal("hold", reason="Trend unavailable")
            buffer_multiplier = max(0.0, 1.0 - self.simple_trend_buffer)
            threshold = trend * buffer_multiplier
            if price < threshold:
                return Signal("hold", reason="Trend filter")

        size = self._target_allocation_size(price, portfolio)
        if size <= 0:
            return Signal("hold", reason="No capacity")

        stop_loss = price * (1 - drop_pct)
        take_profit = (
            price * (1 + self.take_profit_pct)
            if self.take_profit_pct > 0
            else price * (1 + max(drop_pct * self.reward_risk, 0.0))
        )
        self._trade = TradeContext(
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss,
            opened_at=now,
            peak_price=price,
        )
        self._reentry_level = None
        self._reentry_expire_at = None
        self._regime_active = False
        self._layers = 1
        self._next_layer_price = None
        self._last_entry = now
        return Signal(
            "buy",
            size=size,
            reason=f"Peak cycle entry drop={drop_pct:.2%}",
            target_price=take_profit,
            stop_loss=stop_loss,
        )

    def _symbol_peak_pct(self, symbol: str) -> float:
        if self._peak_map:
            if symbol in self._peak_map:
                return max(0.0, self._peak_map[symbol])
            default = self._peak_map.get("default")
            if default is not None:
                return max(0.0, float(default))
        return max(0.0, self._peak_default)

    # --- Signal logic ----------------------------------------------------

    def generate_signal(self, market: MarketSnapshot, portfolio: Portfolio) -> Signal:
        prices = list(market.history)
        price = market.current_price
        now = market.timestamp if isinstance(market.timestamp, datetime) else datetime.utcnow().replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        if self.simple_mode:
            return self._simple_peak_signal(market, portfolio, prices, price, now)

        needed = max(self.slow_period, self.momentum_period, self.atr_period + 1)
        if len(prices) < needed:
            return Signal("hold", reason="Warming up")

        fast = self._ema(prices, self.fast_period)
        slow = self._ema(prices, self.slow_period)
        momentum = self._momentum(prices, self.momentum_period)
        atr = self._atr(prices, self.atr_period)
        rsi = self._rsi(prices, self.rsi_period)
        rsi_prev = None
        if len(prices) > self.rsi_period + 1:
            rsi_prev = self._rsi(prices[:-1], self.rsi_period)

        if fast is None or slow is None or momentum is None or atr is None or rsi is None:
            return Signal("hold", reason="Insufficient indicators")

        if rsi_prev is not None and rsi_prev < self.rsi_entry <= rsi:
            self._regime_active = True
        if self._regime_active and rsi <= (self.rsi_entry - self.rsi_reset):
            self._regime_active = False

        trend_up = fast > slow
        self._trend_down_counter = 0 if trend_up else self._trend_down_counter + 1
        down_confirmed = not trend_up and self._trend_down_counter >= self.trend_down_confirm
        if down_confirmed:
            self._regime_active = False

        if portfolio.quantity > 0 and self._trade:
            if price <= self._trade.stop_loss:
                return self._exit_signal("Stop loss hit", portfolio, reset_regime=True)
            if price >= self._trade.take_profit:
                return self._exit_signal("Take profit hit", portfolio, reset_regime=True)
            time_in_position = (now - self._trade.opened_at).total_seconds() / 3600
            if down_confirmed:
                return self._exit_signal("Trend reversed", portfolio, reset_regime=True)
            if (
                self.min_hold_hours > 0
                and time_in_position >= self.min_hold_hours
                and momentum < self.exit_momentum_floor
            ):
                return self._exit_signal("Momentum faded", portfolio, reset_regime=True)
            if rsi >= self.rsi_exit:
                return self._exit_signal("RSI exit", portfolio, reset_regime=True)
            if self.peak_trail_pct > 0:
                self._trade.peak_price = max(self._trade.peak_price, price)
                threshold = self._trade.peak_price * (1 - self.peak_trail_pct)
                if price <= threshold:
                    return self._exit_signal("Peak trail hit", portfolio, reset_regime=True)
            trail = price - atr * self.trailing_atr_multiplier
            if trail > self._trade.trailing_stop:
                self._trade.trailing_stop = trail
                self._trade.stop_loss = max(self._trade.stop_loss, trail)

            if self._should_pyramid(price):
                add_size = self._calculate_position_size(price, portfolio, stop_loss=self._trade.stop_loss)
                if add_size > 0:
                    return Signal(
                        "buy",
                        size=add_size,
                        reason="Pyramid add",
                        target_price=self._trade.take_profit,
                        stop_loss=self._trade.stop_loss,
                    )
            return Signal("hold", reason="Position managed")

        if self._cooldown_active(now):
            return Signal("hold", reason="Cooldown active")

        slope = self._slope_pct(fast, slow)
        if fast <= slow or momentum < self.momentum_threshold or slope < self.min_slope_pct:
            return Signal("hold", reason="No qualified trend")
        if not self._regime_active:
            return Signal("hold", reason="RSI trigger missing")

        if len(prices) > self.breakout_lookback:
            window = prices[-(self.breakout_lookback + 1):-1]
            recent_high = max(window) if window else max(prices[:-1] or prices)
        else:
            recent_high = max(prices[:-1] or prices)
        buffer_multiplier = 1 + (self.breakout_buffer_pct / 100.0)
        if price < recent_high * buffer_multiplier:
            return Signal("hold", reason="Breakout not confirmed")

        if price <= 0 or atr <= 0 or (atr / price) < self.min_atr_pct:
            return Signal("hold", reason="Volatility floor not met")

        stop_gap = self._stop_gap(price, atr)
        if stop_gap <= 0:
            return Signal("hold", reason="Invalid ATR")
        stop_loss = price - stop_gap
        if self.take_profit_pct > 0:
            take_profit = price * (1 + self.take_profit_pct)
        else:
            reward = stop_gap * self.reward_risk
            take_profit = price + reward

        size = self._calculate_position_size(price, portfolio, stop_loss=stop_loss)
        if size <= 0:
            return Signal("hold", reason="No capacity")

        self._trade = TradeContext(
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss,
            opened_at=now,
            peak_price=price,
        )
        self._regime_active = False
        self._last_entry = now
        return Signal(
            "buy",
            size=size,
            reason=f"Trend confirmed (mom={momentum:.2f} slope={slope:.2f})",
            target_price=take_profit,
            stop_loss=stop_loss,
        )

    def _exit_signal(self, reason: str, portfolio: Portfolio, *, reset_regime: bool = False) -> Signal:
        size = portfolio.quantity
        if size <= 0:
            return Signal("hold", reason="No position")
        self._last_exit = datetime.utcnow().replace(tzinfo=timezone.utc)
        self._trade = None
        if reset_regime:
            self._regime_active = False
        self._layers = 0
        self._next_layer_price = None
        return Signal("sell", size=size, reason=reason)

    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        if signal.action == "sell" and execution_size > 0:
            self._last_exit = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
            self._trade = None
        elif signal.action == "buy" and execution_size > 0 and self._trade:
            self._trade.entry_price = execution_price
            self._trade.peak_price = max(self._trade.peak_price, execution_price)
            self._schedule_next_layer(execution_price, base_entry=self._layers == 0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "trade": None
            if not self._trade
            else {
                "entry_price": self._trade.entry_price,
                "stop_loss": self._trade.stop_loss,
                "take_profit": self._trade.take_profit,
                "trailing_stop": self._trade.trailing_stop,
                "opened_at": self._trade.opened_at.isoformat(),
                "peak_price": self._trade.peak_price,
            },
            "last_exit": self._last_exit.isoformat() if self._last_exit else None,
            "last_entry": self._last_entry.isoformat() if self._last_entry else None,
            "trend_down_counter": self._trend_down_counter,
            "regime_active": self._regime_active,
            "layers": self._layers,
            "next_layer_price": self._next_layer_price,
            "reentry_level": self._reentry_level,
            "reentry_expire_at": self._reentry_expire_at.isoformat() if self._reentry_expire_at else None,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        trade_state = state.get("trade")
        if trade_state:
            opened_at = datetime.fromisoformat(trade_state["opened_at"])
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            self._trade = TradeContext(
                entry_price=float(trade_state["entry_price"]),
                stop_loss=float(trade_state["stop_loss"]),
                take_profit=float(trade_state["take_profit"]),
                trailing_stop=float(trade_state["trailing_stop"]),
                opened_at=opened_at,
                peak_price=float(trade_state.get("peak_price", trade_state["entry_price"])),
            )
        else:
            self._trade = None
        last_exit = state.get("last_exit")
        if last_exit:
            dt = datetime.fromisoformat(last_exit)
            self._last_exit = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        else:
            self._last_exit = None
        self._trend_down_counter = int(state.get("trend_down_counter", 0))
        self._regime_active = bool(state.get("regime_active", False))
        last_entry = state.get("last_entry")
        if last_entry:
            dt = datetime.fromisoformat(last_entry)
            self._last_entry = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        else:
            self._last_entry = None
        self._layers = int(state.get("layers", 0))
        self._next_layer_price = state.get("next_layer_price")
        self._reentry_level = state.get("reentry_level")
        expire_at = state.get("reentry_expire_at")
        if expire_at:
            dt = datetime.fromisoformat(expire_at)
            self._reentry_expire_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        else:
            self._reentry_expire_at = None

    def _stop_gap(self, price: float, atr: float) -> float:
        if self.stop_pct > 0:
            return price * self.stop_pct
        return atr * self.stop_atr_multiplier

    def _calculate_position_size(self, price: float, portfolio: Portfolio, stop_loss: Optional[float] = None) -> float:
        capital = portfolio.value(price)
        current_position_value = portfolio.quantity * price
        cap_notional = max(0.0, self.max_position_pct * capital - current_position_value)
        if cap_notional <= 0:
            return 0.0
        max_notional = min(cap_notional, portfolio.cash)
        if max_notional <= 0:
            return 0.0
        effective_stop = stop_loss
        if effective_stop is None:
            if self.stop_pct > 0:
                effective_stop = price * (1 - self.stop_pct)
            elif self._trade:
                effective_stop = self._trade.stop_loss
        if effective_stop is None:
            return 0.0
        per_unit_risk = price - effective_stop
        if per_unit_risk <= 0:
            return 0.0
        risk_budget = capital * self.risk_per_trade
        size_from_risk = risk_budget / per_unit_risk
        size_from_cash = max_notional / price if price > 0 else 0
        return min(size_from_cash, size_from_risk)

    def _target_allocation_size(self, price: float, portfolio: Portfolio) -> float:
        if price <= 0:
            return 0.0
        capital = portfolio.value(price)
        target_notional = self.max_position_pct * capital
        current_notional = portfolio.quantity * price
        additional_notional = max(0.0, target_notional - current_notional)
        additional_notional = min(additional_notional, portfolio.cash)
        if additional_notional <= 0:
            return 0.0
        return additional_notional / price

    def _schedule_next_layer(self, entry_price: float, *, base_entry: bool = False) -> None:
        if self.max_layers <= 1:
            self._layers = 1 if base_entry else self._layers
            self._next_layer_price = None
            return
        if base_entry or self._layers == 0:
            self._layers = 1
        else:
            self._layers = min(self.max_layers, self._layers + 1)
        if self._layers < self.max_layers:
            self._next_layer_price = entry_price * (1 + self.layer_step_pct)
        else:
            self._next_layer_price = None

    def _should_pyramid(self, price: float) -> bool:
        return (
            self.max_layers > 1
            and self._layers > 0
            and self._layers < self.max_layers
            and self._next_layer_price is not None
            and price >= self._next_layer_price
        )


def _factory(config: Dict[str, Any], exchange):
    return AdaptiveMomentumStrategy(config, exchange)


register_strategy("adaptive_momentum", _factory)
