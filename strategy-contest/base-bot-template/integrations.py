#!/usr/bin/env python3
"""Integration helpers for database logging and status callbacks."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urljoin
import hmac
import hashlib

import requests

try:  # psycopg2 is optional during local development
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - handled gracefully at runtime
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]


@dataclass
class StatusPayload:
    """Structured payload submitted to the main application."""

    status: str
    details: str
    bot_instance_id: str
    user_id: Optional[str]
    extra: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "details": self.details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "botId": self.bot_instance_id,
        }
        if self.user_id:
            payload["userId"] = self.user_id
        if self.extra:
            payload.update(self.extra)
        return payload


class StatusBroadcaster:
    """Send signed callbacks to the dashboard when the bot state changes."""

    def __init__(
        self,
        *,
        base_url: Optional[str],
        bot_instance_id: Optional[str],
        bot_secret: Optional[str],
        user_id: Optional[str],
        logger: logging.Logger,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.bot_instance_id = bot_instance_id
        self.bot_secret = bot_secret
        self.user_id = user_id
        self.logger = logger

    def send(self, status: str, details: str = "", extra: Optional[Dict[str, Any]] = None) -> bool:
        if not self.base_url or not self.bot_instance_id or not self.bot_secret:
            return False

        payload = StatusPayload(
            status=status.upper(),
            details=details,
            bot_instance_id=self.bot_instance_id,
            user_id=self.user_id,
            extra=extra or {},
        ).as_dict()

        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        signature = hmac.new(self.bot_secret.encode("utf-8"), serialized.encode("utf-8"), hashlib.sha256).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-Bot-Signature": signature,
            "X-Bot-Timestamp": str(int(time.time() * 1000)),
        }

        endpoint = f"/api/bots/{self.bot_instance_id}/status"
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))

        try:
            response = requests.post(url, headers=headers, data=serialized.encode("utf-8"), timeout=5)
        except Exception as exc:  # pragma: no cover - network failures handled at runtime
            self.logger.debug("Status callback failed: %s", exc)
            return False

        if response.status_code >= 400:
            self.logger.debug("Status callback error: %s %s", response.status_code, response.text)
            return False

        self.logger.info("Status callback sent: %s", status)
        return True


class DatabaseClient:
    """Very small PostgreSQL helper mirroring the swing bot capabilities."""

    def __init__(self, *, database_url: Optional[str], bot_instance_id: Optional[str], logger: logging.Logger) -> None:
        self.database_url = database_url
        self.bot_instance_id = bot_instance_id
        self.logger = logger
        self.connection = None

        if not self.database_url:
            self.logger.debug("No database URL provided; skipping DB integration")
            return
        if psycopg2 is None:
            self.logger.warning("psycopg2 not installed; database integration disabled")
            return

        self._connect()

    def _connect(self) -> None:
        if not self.database_url or psycopg2 is None:
            return
        try:
            self.connection = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
            self.connection.autocommit = True
            self.logger.info("Database connection established")
        except Exception as exc:  # pragma: no cover - depends on remote DB
            self.logger.warning("Database connection failed: %s", exc)
            self.connection = None

    def _execute(self, query: str, params: Optional[tuple] = None) -> None:
        if self.connection is None:
            return
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
        except Exception as exc:  # pragma: no cover - depends on remote DB
            self.logger.debug("Database query failed: %s", exc)
            try:
                self._connect()
            except Exception:  # pragma: no cover - reconnection best effort
                pass

    def update_bot_status(self, status: str, *, last_seen: Optional[datetime] = None) -> None:
        if not self.connection or not self.bot_instance_id:
            return
        query = (
            "UPDATE bots SET status = %s, last_seen_at = %s, updated_at = %s WHERE id = %s"
        )
        now = datetime.now(timezone.utc)
        params = (
            status.lower(),
            last_seen or now,
            now,
            self.bot_instance_id,
        )
        self._execute(query, params)

    def log_trade(
        self,
        *,
        side: str,
        amount: float,
        price: float,
        fees: float = 0.0,
        profit: Optional[float] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        external_trade_id: Optional[str] = None,
        reasoning: Optional[str] = None,
        strategy: Optional[str] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        entry_price: Optional[float] = None,
    ) -> None:
        if not self.connection or not self.bot_instance_id:
            return
        query = (
            "INSERT INTO bot_trades (bot_id, side, symbol, amount, price, fees, profit, exchange, external_trade_id, timestamp, reasoning, strategy, target_price, stop_loss, entry_price) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        params = (
            self.bot_instance_id,
            side.lower(),
            symbol or 'UNKNOWN',
            amount,
            price,
            fees,
            profit,
            exchange,
            external_trade_id,
            datetime.now(timezone.utc),  # Use timezone-aware UTC datetime
            reasoning,
            strategy,
            target_price,
            stop_loss,
            entry_price,
        )
        self._execute(query, params)

    def log_event(self, level: str, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.connection or not self.bot_instance_id:
            return
        query = (
            "INSERT INTO bot_logs (bot_id, level, message, timestamp, metadata) "
            "VALUES (%s, %s, %s, %s, %s)"
        )
        params = (
            self.bot_instance_id,
            level.upper(),
            message,
            datetime.now(timezone.utc),
            json.dumps(metadata or {}),
        )
        self._execute(query, params)

    def get_total_spent(self) -> float:
        """Get total amount spent by this bot from database."""
        if not self.connection or not self.bot_instance_id:
            return 0.0

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT total_spent FROM bots WHERE id = %s",
                    (self.bot_instance_id,)
                )
                result = cursor.fetchone()
                if result:
                    return float(result['total_spent'] or 0.0)
                return 0.0
        except Exception as exc:
            self.logger.debug("Failed to get total_spent: %s", exc)
            return 0.0

    def update_total_spent(self, amount: float) -> None:
        """Add amount to total_spent for this bot."""
        if not self.connection or not self.bot_instance_id:
            return

        query = "UPDATE bots SET total_spent = total_spent + %s WHERE id = %s"
        params = (amount, self.bot_instance_id)
        self._execute(query, params)

    def get_portfolio_quantity(self) -> float:
        """Get current portfolio quantity from database."""
        if not self.connection or not self.bot_instance_id:
            return 0.0
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT portfolio_quantity FROM bots WHERE id = %s",
                    (self.bot_instance_id,)
                )
                result = cursor.fetchone()
                if result:
                    return float(result['portfolio_quantity'] or 0.0)
                return 0.0
        except Exception as exc:
            self.logger.debug("Failed to get portfolio_quantity: %s", exc)
            return 0.0

    def update_portfolio_quantity(self, delta: float) -> None:
        """Add delta to portfolio_quantity for this bot (positive for buy, negative for sell)."""
        if not self.connection or not self.bot_instance_id:
            return

        query = "UPDATE bots SET portfolio_quantity = portfolio_quantity + %s WHERE id = %s"
        params = (delta, self.bot_instance_id)
        self._execute(query, params)

    def set_portfolio_quantity(self, quantity: float) -> None:
        """Set absolute portfolio_quantity for this bot."""
        if not self.connection or not self.bot_instance_id:
            return

        query = "UPDATE bots SET portfolio_quantity = %s WHERE id = %s"
        params = (quantity, self.bot_instance_id)
        self._execute(query, params)

    def get_buy_trades_count(self) -> int:
        """Get count of buy trades for this bot from database."""
        if not self.connection or not self.bot_instance_id:
            return 0

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM bot_trades WHERE bot_id = %s AND side = 'buy'",
                    (self.bot_instance_id,)
                )
                result = cursor.fetchone()
                if result:
                    return int(result['count'] or 0)
                return 0
        except Exception as exc:
            self.logger.debug("Failed to get buy trades count: %s", exc)
            return 0

    def get_total_invested(self) -> float:
        """Get total amount invested (sum of all buy trades) for this bot."""
        if not self.connection or not self.bot_instance_id:
            return 0.0

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT SUM(amount * price) as total FROM bot_trades WHERE bot_id = %s AND side = 'buy'",
                    (self.bot_instance_id,)
                )
                result = cursor.fetchone()
                if result and result['total']:
                    return float(result['total'])
                return 0.0
        except Exception as exc:
            self.logger.debug("Failed to get total invested: %s", exc)
            return 0.0

    def get_weighted_average_price(self) -> float:
        """Get weighted average entry price from buy trades."""
        if not self.connection or not self.bot_instance_id:
            return 0.0

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT SUM(amount * price) as total_cost, SUM(amount) as total_quantity FROM bot_trades WHERE bot_id = %s AND side = 'buy'",
                    (self.bot_instance_id,)
                )
                result = cursor.fetchone()
                if result and result['total_cost'] and result['total_quantity']:
                    total_cost = float(result['total_cost'])
                    total_quantity = float(result['total_quantity'])
                    if total_quantity > 0:
                        return total_cost / total_quantity
                return 0.0
        except Exception as exc:
            self.logger.debug("Failed to get weighted average price: %s", exc)
            return 0.0

    def get_currency_from_trades(self) -> str:
        """Get currency from symbol field in trades for this bot."""
        if not self.connection or not self.bot_instance_id:
            return ""

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT symbol FROM bot_trades WHERE bot_id = %s LIMIT 1",
                    (self.bot_instance_id,)
                )
                result = cursor.fetchone()
                if result and result['symbol']:
                    return str(result['symbol'])
                return ""
        except Exception as exc:
            self.logger.debug("Failed to get currency from trades: %s", exc)
            return ""

    def close(self) -> None:
        if self.connection:
            try:
                self.connection.close()
            finally:
                self.connection = None
