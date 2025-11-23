#!/usr/bin/env python3
"""Minimal Coinbase exchange adapter for the simplified universal bot."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from requests import RequestException

from exchange_interface import ExchangeRegistry, MarketSnapshot, TradeExecution


@dataclass
class CoinbaseExchange:
    """Fetch price data and submit basic orders to Coinbase Exchange."""

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    granularity: int = 900  # seconds (15 minutes)

    name: str = "coinbase"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("COINBASE_API_KEY")
        # Allow both legacy and documented env variable names.
        self.api_secret = self.api_secret or os.getenv("COINBASE_API_SECRET") or os.getenv("COINBASE_SECRET")
        self.api_passphrase = self.api_passphrase or os.getenv("COINBASE_API_PASSPHRASE") or os.getenv("COINBASE_PASSPHRASE")
        self.base_url = "https://api.exchange.coinbase.com"

    def fetch_market_snapshot(self, symbol: str, *, limit: int) -> MarketSnapshot:
        params = {
            "granularity": self.granularity,
            "limit": max(1, min(limit, 300)),
        }
        url = f"{self.base_url}/products/{symbol}/candles"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f"Failed to fetch Coinbase candles: {exc}") from exc

        raw_candles = response.json()
        if not raw_candles:
            raise RuntimeError(f"No candle data returned for {symbol}")

        # Coinbase returns candles newest-first; reverse to chronological order.
        candles = list(reversed(raw_candles))
        closes = [float(candle[4]) for candle in candles]
        latest_close = closes[-1]
        latest_timestamp = datetime.utcfromtimestamp(candles[-1][0])

        return MarketSnapshot(
            symbol=symbol,
            prices=closes,
            current_price=latest_close,
            timestamp=latest_timestamp,
        )

    def execute_trade(self, symbol: str, side: str, size: float, price: float) -> TradeExecution:
        if not (self.api_key and self.api_secret and self.api_passphrase):
            raise RuntimeError("Coinbase API credentials are required to execute trades")

        body: Dict[str, Any] = {
            "product_id": symbol,
            "side": side.lower(),
            "type": "market",
            "size": str(size),
        }
        path = "/orders"
        timestamp = str(time.time())
        message = timestamp + "POST" + path + json.dumps(body)
        secret = base64.b64decode(self.api_secret)
        signature = hmac.new(secret, message.encode("utf-8"), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode("utf-8")

        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, headers=headers, json=body, timeout=10)
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f"Failed to place Coinbase order: {exc}") from exc

        payload = response.json()
        executed_value = float(payload.get("executed_value", 0) or 0)
        filled_size = float(payload.get("filled_size") or body["size"])
        filled_price = executed_value / filled_size if executed_value > 0 and filled_size > 0 else price

        return TradeExecution(
            side=side.lower(),
            size=filled_size,
            price=filled_price,
            timestamp=datetime.utcnow(),
        )


# Register the exchange so it becomes available by name.
ExchangeRegistry.register("coinbase", lambda **kwargs: CoinbaseExchange(**kwargs))
