#!/usr/bin/env python3
"""Lightweight exchange abstractions for the simplified universal bot."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Protocol


@dataclass
class MarketSnapshot:
    """Minimal market view shared with strategies."""

    symbol: str
    prices: List[float]
    current_price: float
    timestamp: datetime

    @property
    def history(self) -> List[float]:
        """Convenience alias used by strategies."""
        return self.prices


@dataclass
class TradeExecution:
    """Outcome of a simulated or real trade."""

    side: str  # "buy" or "sell"
    size: float
    price: float
    timestamp: datetime


class Exchange(Protocol):
    """Protocol implemented by all exchanges."""

    name: str

    def fetch_market_snapshot(self, symbol: str, *, limit: int) -> MarketSnapshot:
        """Return the most recent price history for the symbol."""

    def execute_trade(self, symbol: str, side: str, size: float, price: float) -> TradeExecution:
        """Execute a trade at the provided price."""


class ExchangeRegistry:
    """Simple registry so the bot can instantiate exchanges by name."""

    _exchanges: Dict[str, Callable[..., Exchange]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., Exchange]) -> None:
        cls._exchanges[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> Exchange:
        if name not in cls._exchanges:
            available = ", ".join(sorted(cls._exchanges)) or "<none>"
            raise ValueError(f"Unknown exchange '{name}'. Available: {available}")
        return cls._exchanges[name](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._exchanges)


@dataclass
class PaperExchange:
    """Paper exchange that uses real market data but simulated trades."""

    name: str = "paper"
    coinbase_url: str = "https://api.exchange.coinbase.com"
    coingecko_url: str = "https://api.coingecko.com/api/v3/simple/price"
    _price_cache: Dict[str, float] = field(default_factory=dict)
    _cache_timestamp: Dict[str, datetime] = field(default_factory=dict)
    cache_duration_seconds: int = 30

    def fetch_market_snapshot(self, symbol: str, *, limit: int) -> MarketSnapshot:
        """Fetch real market data for paper trading simulation."""
        current_price = self._get_real_price(symbol)
        # Generate realistic price history around current price
        history = self._generate_realistic_history(current_price, limit)

        return MarketSnapshot(
            symbol=symbol,
            prices=history,
            current_price=current_price,
            timestamp=datetime.utcnow(),
        )

    def execute_trade(self, symbol: str, side: str, size: float, price: float) -> TradeExecution:
        # No slippage or fees - this exchange is a sandbox for strategies.
        return TradeExecution(side=side, size=size, price=price, timestamp=datetime.utcnow())

    def _get_real_price(self, symbol: str) -> float:
        """Get current real price with caching and multiple API fallbacks."""
        now = datetime.utcnow()

        # Check cache
        if (symbol in self._price_cache and
            symbol in self._cache_timestamp and
            (now - self._cache_timestamp[symbol]).total_seconds() < self.cache_duration_seconds):
            print(f"📊 Using cached price for {symbol}: ${self._price_cache[symbol]:,.2f}")
            return self._price_cache[symbol]

        # Try multiple APIs in order
        price = None
        last_error = None

        # Try 1: Coinbase API
        try:
            price = self._fetch_coinbase_price(symbol)
            if price:
                print(f"✅ Coinbase: Fetched real price for {symbol}: ${price:,.2f}")
        except Exception as e:
            print(f"❌ Coinbase API failed for {symbol}: {e}")
            last_error = e

        # Try 2: CoinGecko API (if Coinbase failed)
        if not price:
            try:
                price = self._fetch_coingecko_price(symbol)
                if price:
                    print(f"✅ CoinGecko: Fetched real price for {symbol}: ${price:,.2f}")
            except Exception as e:
                print(f"❌ CoinGecko API failed for {symbol}: {e}")
                last_error = e

        # If both APIs failed, try cached price (even if expired)
        if not price and symbol in self._price_cache:
            print(f"⚠️  All APIs failed for {symbol}, using expired cached price: ${self._price_cache[symbol]:,.2f}")
            return self._price_cache[symbol]

        # If we got a price, cache it
        if price:
            self._price_cache[symbol] = price
            self._cache_timestamp[symbol] = now
            return price

        # Complete failure
        raise Exception(f"All price APIs failed for {symbol}. Last error: {last_error}")

    def _fetch_coinbase_price(self, symbol: str) -> float:
        """Fetch price from Coinbase API."""
        import requests
        url = f"{self.coinbase_url}/products/{symbol}/ticker"
        print(f"🔗 Coinbase URL: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])

    def _fetch_coingecko_price(self, symbol: str) -> float:
        """Fetch price from CoinGecko API."""
        import requests

        # Convert symbol format (BTC-USD -> bitcoin vs usd)
        symbol_map = {
            'BTC-USD': ('bitcoin', 'usd'),
            'ETH-USD': ('ethereum', 'usd'),
            'DOT-USD': ('polkadot', 'usd'),
            'SOL-USD': ('solana', 'usd'),
            'ADA-USD': ('cardano', 'usd'),
        }

        if symbol not in symbol_map:
            raise Exception(f"Symbol {symbol} not supported by CoinGecko")

        coin_id, vs_currency = symbol_map[symbol]
        url = f"{self.coingecko_url}?ids={coin_id}&vs_currencies={vs_currency}"
        print(f"🔗 CoinGecko URL: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data[coin_id][vs_currency])

    def _generate_realistic_history(self, current_price: float, limit: int) -> List[float]:
        """Generate realistic price history around current price."""
        import random

        prices = []
        price = current_price * 0.99  # Start slightly below current
        volatility = 0.01  # 1% volatility for realistic movement

        for _ in range(limit):
            # Random walk around current price
            change = random.uniform(-volatility, volatility)
            price = max(0.01, price * (1 + change))
            prices.append(price)

        # Ensure last price is close to current
        prices[-1] = current_price
        return prices



# Register built-in exchanges.
ExchangeRegistry.register("paper", PaperExchange)
