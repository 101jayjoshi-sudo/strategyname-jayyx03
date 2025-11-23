#!/usr/bin/env python3
"""DCA trading strategies for DCA Bot Template.

Σημείωση: Η επιπλέον καταγραφή (structured logging) είναι τοπική ΜΟΝΟ για την
DcaStrategy. Δεν αλλάζει τίποτα στο base loop ούτε στις άλλες στρατηγικές.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from statistics import pstdev
from typing import Any, Deque, Dict, Optional
from collections import deque
import logging

# Import base infrastructure from base-bot-template
import sys
import os

# Handle both local development and Docker container paths
base_path = os.path.join(os.path.dirname(__file__), '..', 'base-bot-template')
if not os.path.exists(base_path):
    # In Docker container, base template is at /app/base/
    base_path = '/app/base'

sys.path.insert(0, base_path)

from strategy_interface import BaseStrategy, Signal, register_strategy
from exchange_interface import MarketSnapshot


# ----------------------------- DCA-only helpers -----------------------------

def _utc_iso(dt: datetime) -> str:
    """Return ISO timestamp with UTC tzinfo (seconds precision)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _as_bool(val, default: bool = False) -> bool:
    """Parse truthy strings/values to bool."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "on")


# --------------------------------- DCA --------------------------------------

class DcaStrategy(BaseStrategy):
    """Dollar-cost averaging: buy a fixed amount every N minutes.

    Επιπλέον (local-only) structured logging:
    - Ενεργοποίηση/Απενεργοποίηση μέσω:
        - config['strategy_local_logs'] ή ENV STRATEGY_LOCAL_LOGS (default: true)
    - Τα logs εμφανίζονται ως:
        [DCA/CYCLE], [DCA/DECISION], [DCA/ACTION], [DCA/TRACE]
    """

    def __init__(self, config: Dict[str, Any], exchange):
        super().__init__(config=config, exchange=exchange)
        self.interval_minutes = max(1, int(config.get("interval_minutes", 60)))
        self.base_amount = float(config.get("base_amount", 50.0))
        self._last_purchase: Optional[datetime] = None

        # --- DCA-local logging controls (δεν επηρεάζουν base/άλλες στρατηγικές) ---
        self._local_logs_enabled = _as_bool(
            config.get("strategy_local_logs", os.getenv("STRATEGY_LOCAL_LOGS", "true")), True
        )
        self._logger = logging.getLogger("strategy.dca")
        self._last_trace: Optional[Dict[str, Any]] = None

        # --- Spending limit controls ---
        self._starting_cash = float(config.get("starting_cash", 10000.0))
        self._db_client = config.get("db_client")  # DatabaseClient instance
        self._total_spent_cache: Optional[float] = None

        # --- State restoration from database ---
        self._restore_last_purchase_from_db()

    # --------------------------- state restoration utils ---------------------------

    def _restore_last_purchase_from_db(self) -> None:
        """Restore last purchase timestamp from database to prevent multiple buys on restart."""
        if not self._db_client or not hasattr(self._db_client, 'connection') or not self._db_client.connection:
            self._log_local("STATE", "No database connection - starting with empty state")
            return

        try:
            bot_id = getattr(self._db_client, 'bot_instance_id', None)
            if not bot_id:
                self._log_local("STATE", "No bot_instance_id - starting with empty state")
                return

            with self._db_client.connection.cursor() as cursor:
                # Get most recent buy trade timestamp for this bot
                cursor.execute(
                    "SELECT timestamp FROM bot_trades WHERE bot_id = %s AND side = 'buy' ORDER BY timestamp DESC LIMIT 1",
                    (bot_id,)
                )
                result = cursor.fetchone()

                if result:
                    last_trade_time = result['timestamp']

                    # Handle timezone conversion properly
                    # PostgreSQL timezone is Europe/Helsinki, but we need UTC for internal use
                    if last_trade_time.tzinfo is None:
                        # Assume timestamp from DB is in database timezone (Europe/Helsinki)
                        helsinki_tz = ZoneInfo("Europe/Helsinki")
                        last_trade_time = last_trade_time.replace(tzinfo=helsinki_tz)
                        self._log_local("STATE", f"Assuming DB timestamp is in Europe/Helsinki timezone")

                    # Convert to UTC for internal bot use
                    last_trade_time_utc = last_trade_time.astimezone(timezone.utc)
                    self._last_purchase = last_trade_time_utc

                    self._log_local("STATE", f"Restored last_purchase: DB={_utc_iso(last_trade_time)} -> UTC={_utc_iso(last_trade_time_utc)}")
                else:
                    self._log_local("STATE", "No previous buy trades found - starting fresh")

        except Exception as exc:
            self._log_local("STATE", f"Failed to restore state from database: {exc}")
            # Continue with None state - bot will work but may make extra purchases

    # --------------------------- local logging utils ---------------------------

    def _log_local(self, kind: str, msg: str) -> None:
        """Local, DCA-only logger (no-throw)."""
        if not self._local_logs_enabled:
            return
        try:
            self._logger.info(f"[DCA/{kind}] {msg}")
        except Exception:
            pass  # never let logging crash strategy

    def _build_trace(self, now: datetime, market: MarketSnapshot, portfolio) -> Dict[str, Any]:
        due = (self._last_purchase is None) or ((now - self._last_purchase) >= timedelta(minutes=self.interval_minutes))
        trace: Dict[str, Any] = {
            "now": _utc_iso(now),
            "last_purchase": _utc_iso(self._last_purchase) if self._last_purchase else None,
            "interval_minutes": self.interval_minutes,
            "due": due,
            "price": round(float(market.current_price), 2),
            "cash": round(float(portfolio.cash), 2),
            "base_amount": round(float(self.base_amount), 2),
        }
        return trace

    def get_last_trace(self) -> Optional[Dict[str, Any]]:
        """Optional external getter για debugging/telemetry."""
        return self._last_trace

    # --------------------------- spending limit utils ---------------------------

    def _get_total_spent(self) -> float:
        """Get total spent amount, using cache to avoid frequent DB calls."""
        if self._total_spent_cache is None and self._db_client:
            self._total_spent_cache = self._db_client.get_total_spent()
        return self._total_spent_cache or 0.0

    def _update_total_spent(self, amount: float) -> None:
        """Update total spent in database and local cache."""
        if self._db_client:
            self._db_client.update_total_spent(amount)
            if self._total_spent_cache is not None:
                self._total_spent_cache += amount

    def _check_spending_limit(self, amount: float) -> bool:
        """Check if we can spend the given amount without exceeding starting cash."""
        total_spent = self._get_total_spent()
        remaining_cash = self._starting_cash - total_spent
        can_spend = amount <= remaining_cash

        if self._local_logs_enabled:
            self._log_local("SPENDING", f"limit_check | starting=${self._starting_cash:,.2f} | spent=${total_spent:,.2f} | remaining=${remaining_cash:,.2f} | requested=${amount:,.2f} | can_spend={can_spend}")

        return can_spend

    # --------------------------------- logic ----------------------------------

    def generate_signal(self, market: MarketSnapshot, portfolio) -> Signal:
        # Always use current UTC time to avoid timezone issues with market.timestamp
        now = datetime.now(timezone.utc)

        # Build decision trace early & keep it
        trace = self._build_trace(now, market, portfolio)
        self._last_trace = trace

        # Cycle header (τοπικό)
        self._log_local("CYCLE", f"tick @ {trace['now']} | price=${trace['price']:,} | cash=${trace['cash']:,}")

        # Interval gate
        if not trace["due"]:
            self._log_local("DECISION", f"HOLD | reason=waiting_interval | next_in_min={self.interval_minutes}")
            self._log_local("TRACE", " | ".join(f"{k}={v}" for k, v in trace.items()))
            return Signal("hold", reason="Waiting for next interval")

        # Price validity
        if market.current_price <= 0:
            self._log_local("DECISION", "HOLD | reason=invalid_price")
            self._log_local("TRACE", " | ".join(f"{k}={v}" for k, v in trace.items()))
            return Signal("hold", reason="No valid price")

        # Cash check
        notional = min(self.base_amount, portfolio.cash)
        if notional <= 0:
            self._log_local("DECISION", "HOLD | reason=insufficient_cash")
            self._log_local("TRACE", " | ".join(f"{k}={v}" for k, v in trace.items()))
            return Signal("hold", reason="Insufficient cash")

        # Spending limit check
        if not self._check_spending_limit(notional):
            self._log_local("DECISION", "HOLD | reason=spending_limit_exceeded")
            self._log_local("TRACE", " | ".join(f"{k}={v}" for k, v in trace.items()))
            return Signal("hold", reason="Spending limit exceeded")

        # Compute order
        size = notional / market.current_price
        trace.update({"size": round(size, 8), "notional": round(notional, 2)})
        self._log_local("DECISION", f"BUY | reason=scheduled_dca | size={size:.8f} | notional=${notional:,.2f}")
        self._log_local("TRACE", " | ".join(f"{k}={v}" for k, v in trace.items()))

        return Signal("buy", size=size, reason="Scheduled DCA buy")

    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        # Make timestamp UTC-aware
        if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        if signal.action == "buy" and execution_size > 0:
            self._last_purchase = timestamp

            # Update total spent in database
            trade_amount = execution_size * execution_price
            self._update_total_spent(trade_amount)

            self._log_local(
                "ACTION",
                f"EXEC BUY {execution_size:.8f} @ ${execution_price:,.2f} | total=${trade_amount:,.2f} | last_purchase={_utc_iso(timestamp)}"
            )

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_purchase": _utc_iso(self._last_purchase) if self._last_purchase else None
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        value = state.get("last_purchase")
        if value:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self._last_purchase = dt


# ------------------------------ Advanced DCA --------------------------------

class AdvancedDcaStrategy(BaseStrategy):
    """Adaptive DCA strategy with volatility-aware spacing and take-profit bands.

    ENTERPRISE TIER FEATURE - Advanced DCA with sophisticated risk management.
    """

    def __init__(self, config: Dict[str, Any], exchange):
        super().__init__(config=config, exchange=exchange)
        self.base_amount = float(config.get("base_amount", 50.0))
        self.max_positions = int(config.get("max_positions", 10))
        self.min_minutes_between_buys = max(1, int(config.get("min_minutes_between_buys", 60)))
        self.base_drop_pct = max(0.1, float(config.get("base_drop_pct", 2.5)))
        self.volatility_window = max(5, int(config.get("volatility_window", 30)))
        self.volatility_factor = float(config.get("volatility_factor", 2.0))
        self.scale_factor = float(config.get("scale_factor", 1.5))
        self.take_profit_pct = float(config.get("take_profit_pct", 6.0))
        self.trailing_stop_pct = float(config.get("trailing_stop_pct", 3.0))
        self.drawdown_pause_pct = float(config.get("drawdown_pause_pct", 15.0))
        self.max_daily_buys = int(config.get("max_daily_buys", 4))

        self.entries: Deque[Dict[str, Any]] = deque()
        self.last_buy_at: Optional[datetime] = None
        self.trailing_high: Optional[float] = None
        self.daily_buy_counter: Dict[str, int] = {}

    def generate_signal(self, market: MarketSnapshot, portfolio) -> Signal:
        now = market.timestamp if isinstance(market.timestamp, datetime) else datetime.utcnow()

        if self._should_pause_for_drawdown(market):
            return Signal("hold", reason="Drawdown protection active")

        if portfolio.quantity > 0:
            sell_signal = self._maybe_take_profit(market, portfolio)
            if sell_signal:
                return sell_signal

        if not self._can_buy(now, portfolio):
            return Signal("hold", reason="Buy conditions not met")

        drop_threshold = self._dynamic_drop_threshold(market)
        drop_from_reference = self._price_drop_pct(market)

        if drop_from_reference < drop_threshold:
            reason = f"Drop {drop_from_reference:.2f}% < threshold {drop_threshold:.2f}%"
            return Signal("hold", reason=reason)

        size = self._position_size(market.current_price, portfolio)
        if size <= 0:
            return Signal("hold", reason="No capacity for additional position")

        self._record_daily_buy(now)
        return Signal("buy", size=size, reason=f"Price drop {drop_from_reference:.2f}% >= {drop_threshold:.2f}%")

    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        if signal.action == "buy" and execution_size > 0:
            entry = {
                "price": execution_price,
                "size": execution_size,
                "timestamp": timestamp.isoformat()
            }
            self.entries.append(entry)
            while len(self.entries) > self.max_positions:
                self.entries.popleft()
            self.last_buy_at = timestamp
            if self.trailing_high is None or execution_price > self.trailing_high:
                self.trailing_high = execution_price
        elif signal.action == "sell" and execution_size > 0:
            remaining = execution_size
            while self.entries and remaining > 0:
                position = self.entries.pop()
                if position["size"] > remaining:
                    position["size"] -= remaining
                    self.entries.append(position)
                    remaining = 0
                else:
                    remaining -= position["size"]
            self.trailing_high = execution_price

    def get_state(self) -> Dict[str, Any]:
        return {
            "entries": list(self.entries),
            "last_buy_at": self.last_buy_at.isoformat() if self.last_buy_at else None,
            "trailing_high": self.trailing_high,
            "daily_buy_counter": self.daily_buy_counter,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.entries = deque(state.get("entries", []))
        last_buy = state.get("last_buy_at")
        if last_buy:
            self.last_buy_at = datetime.fromisoformat(last_buy)
        self.trailing_high = state.get("trailing_high")
        self.daily_buy_counter = state.get("daily_buy_counter", {})

    # --- decision helpers -------------------------------------------------

    def _should_pause_for_drawdown(self, market: MarketSnapshot) -> bool:
        if not self.entries or self.drawdown_pause_pct <= 0:
            return False
        highest_entry = max(entry["price"] for entry in self.entries)
        drop_pct = (highest_entry - market.current_price) / highest_entry * 100
        return drop_pct >= self.drawdown_pause_pct

    def _maybe_take_profit(self, market: MarketSnapshot, portfolio) -> Signal | None:
        if not self.entries or self.take_profit_pct <= 0:
            return None
        total_size = sum(entry["size"] for entry in self.entries)
        if total_size <= 0:
            return None
        avg_entry = sum(entry["price"] * entry["size"] for entry in self.entries) / total_size
        gain_pct = (market.current_price - avg_entry) / avg_entry * 100
        if gain_pct < self.take_profit_pct:
            self._update_trailing_high(market.current_price)
            if self._trigger_trailing_stop(market.current_price):
                size = min(portfolio.quantity, total_size)
                return Signal("sell", size=size, reason="Trailing stop triggered")
            return None
        size = min(portfolio.quantity, total_size)
        return Signal("sell", size=size, reason=f"Take profit at {gain_pct:.2f}%")

    def _update_trailing_high(self, price: float) -> None:
        if self.trailing_high is None or price > self.trailing_high:
            self.trailing_high = price

    def _trigger_trailing_stop(self, price: float) -> bool:
        if self.trailing_high is None or self.trailing_stop_pct <= 0:
            return False
        drop_pct = (self.trailing_high - price) / self.trailing_high * 100
        return drop_pct >= self.trailing_stop_pct

    def _can_buy(self, now: datetime, portfolio) -> bool:
        if portfolio.cash <= 0:
            return False
        if len(self.entries) >= self.max_positions:
            return False
        if self.last_buy_at and now - self.last_buy_at < timedelta(minutes=self.min_minutes_between_buys):
            return False
        key = now.strftime("%Y-%m-%d")
        if self.daily_buy_counter.get(key, 0) >= self.max_daily_buys:
            return False
        return True

    def _record_daily_buy(self, now: datetime) -> None:
        key = now.strftime("%Y-%m-%d")
        self.daily_buy_counter[key] = self.daily_buy_counter.get(key, 0) + 1
        old_keys = [k for k in self.daily_buy_counter if k < key]
        for k in old_keys:
            del self.daily_buy_counter[k]

    def _dynamic_drop_threshold(self, market: MarketSnapshot) -> float:
        if len(market.prices) < self.volatility_window:
            return self.base_drop_pct
        window = market.prices[-self.volatility_window :]
        returns = []
        for prev, curr in zip(window, window[1:]):
            if prev > 0:
                returns.append((curr - prev) / prev)
        volatility = pstdev(returns) if len(returns) > 1 else 0.0
        return self.base_drop_pct * (1 + self.volatility_factor * volatility * 100)

    def _price_drop_pct(self, market: MarketSnapshot) -> float:
        current_price = market.current_price
        if not self.entries:
            recent_prices = market.prices[-self.volatility_window :] if len(market.prices) >= self.volatility_window else market.prices
            reference = max(recent_prices) if recent_prices else current_price
            if reference <= 0:
                return self.base_drop_pct
            return max(self.base_drop_pct, (reference - current_price) / reference * 100)
        reference_price = self.entries[-1]["price"]
        return (reference_price - current_price) / reference_price * 100

    def _position_size(self, price: float, portfolio) -> float:
        base_notional = min(self.base_amount, portfolio.cash)
        if base_notional <= 0 or price <= 0:
            return 0.0
        if self.max_positions <= 1:
            scale_multiplier = 1.0
        else:
            scale_multiplier = 1.0 + len(self.entries) * (self.scale_factor - 1.0) / (self.max_positions - 1)
        notional = min(portfolio.cash, base_notional * scale_multiplier)
        return notional / price


# Register DCA strategies at import time
register_strategy("dca", lambda cfg, ex: DcaStrategy(cfg, ex))
register_strategy("advanced_dca", lambda cfg, ex: AdvancedDcaStrategy(cfg, ex))
