#!/usr/bin/env python3
"""Contest-compliant backtest runner for the Adaptive Momentum strategy.

The script pulls **live Yahoo Finance data via yfinance.download()** every time it runs.
Optional pickled caches stored under analysis/cache/ simply prevent duplicate network
calls, but they can be deleted at any time—the data will be re-downloaded on demand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TEMPLATE = REPO_ROOT / "base-bot-template"
STRATEGY_TEMPLATE = REPO_ROOT / "your-strategy-template"
CACHE_DIR = REPO_ROOT / "analysis" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

for path in (BASE_TEMPLATE, STRATEGY_TEMPLATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from strategy_interface import MarketSnapshot, Portfolio  # noqa: E402
import your_strategy  # noqa: F401  (registers adaptive_momentum)
from your_strategy import AdaptiveMomentumStrategy  # noqa: E402

FEE_RATE = 0.001  # 10 bps round-trip (0.1%)


@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str
    price: float
    size: float
    reason: str


@dataclass
class BacktestResult:
    symbol: str
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    sharpe: float
    trades: List[Trade]
    equity_curve: List[float]


def fetch_data(symbol: str, start: str, end: str, interval: str, use_cache: bool) -> pd.DataFrame:
    cache_key = f"{symbol}_{start}_{end}_{interval}.pkl"
    cache_path = CACHE_DIR / cache_key
    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)
    data = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    data = data.tz_localize("UTC") if data.index.tz is None else data.tz_convert("UTC")
    if use_cache:
        data.to_pickle(cache_path)
    return data


def run_strategy(symbol: str, data: pd.DataFrame, strategy_cfg: Dict[str, float], starting_cash: float) -> BacktestResult:
    strategy = AdaptiveMomentumStrategy(strategy_cfg, exchange=None)
    portfolio = Portfolio(symbol=symbol, cash=starting_cash)
    trades: List[Trade] = []
    closes: List[float] = []
    equity_curve: List[float] = []

    for timestamp, row in data.iterrows():
        price = float(row["Close"])
        if math.isnan(price) or price <= 0:
            continue
        closes.append(price)
        snapshot = MarketSnapshot(symbol=symbol, prices=list(closes), current_price=price, timestamp=timestamp.to_pydatetime())
        signal = strategy.generate_signal(snapshot, portfolio)
        equity_curve.append(portfolio.value(price))
        if signal.action == "buy" and signal.size > 0:
            max_notional = 0.55 * portfolio.value(price)
            size = min(signal.size, portfolio.cash / price if price > 0 else 0)
            size = min(size, max_notional / price if price > 0 else 0)
            if size <= 0:
                continue
            notional = size * price
            fee = notional * FEE_RATE
            if notional + fee > portfolio.cash:
                continue
            portfolio.cash -= notional + fee
            portfolio.quantity += size
            trades.append(Trade(timestamp, symbol, "buy", price, size, signal.reason))
            strategy.on_trade(signal, price, size, timestamp.to_pydatetime())
        elif signal.action == "sell" and portfolio.quantity > 0:
            size = min(signal.size, portfolio.quantity)
            if size <= 0:
                continue
            notional = size * price
            fee = notional * FEE_RATE
            portfolio.cash += notional - fee
            portfolio.quantity -= size
            trades.append(Trade(timestamp, symbol, "sell", price, size, signal.reason))
            strategy.on_trade(signal, price, size, timestamp.to_pydatetime())

        equity_curve[-1] = portfolio.value(price)

    # Liquidate any remaining position at last price
    if portfolio.quantity > 0 and closes:
        last_price = closes[-1]
        notional = portfolio.quantity * last_price
        fee = notional * FEE_RATE
        portfolio.cash += notional - fee
        trades.append(Trade(data.index[-1], symbol, "sell", last_price, portfolio.quantity, "Forced liquidation"))
        portfolio.quantity = 0.0
        equity_curve.append(portfolio.cash)

    final_equity = portfolio.cash
    ret_pct = (final_equity - starting_cash) / starting_cash * 100
    max_dd = compute_max_drawdown(equity_curve)
    sharpe = compute_sharpe_ratio(equity_curve)

    return BacktestResult(
        symbol=symbol,
        final_equity=final_equity,
        return_pct=ret_pct,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        trades=trades,
        equity_curve=equity_curve,
    )


def compute_max_drawdown(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, drawdown)
    return max_dd


def compute_sharpe_ratio(equity: Sequence[float]) -> float:
    if len(equity) < 2:
        return 0.0
    returns = np.diff(equity) / np.array(equity[:-1])
    if returns.std(ddof=1) == 0:
        return 0.0
    # Annualise using 24*365 hourly periods for contest window
    sharpe = (returns.mean() / returns.std(ddof=1)) * math.sqrt(24 * 365)
    return float(sharpe)


def format_result(result: BacktestResult) -> Dict[str, Any]:
    return {
        "symbol": result.symbol,
        "final_equity": round(result.final_equity, 2),
        "return_pct": round(result.return_pct, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "sharpe": round(result.sharpe, 2),
        "trades": len(result.trades),
    }


def save_report(results: List[BacktestResult], output_markdown: Path, starting_cash: float) -> None:
    table_rows = [
        "| Symbol | Return % | Max DD % | Sharpe | Trades |",
        "|--------|----------|----------|--------|--------|",
    ]
    combined_equity = sum(r.final_equity for r in results)
    combined_return = (
        (combined_equity - len(results) * starting_cash) / (len(results) * starting_cash) * 100 if results else 0.0
    )
    combined_dd = max((r.max_drawdown_pct for r in results), default=0.0)
    combined_sharpe = sum(r.sharpe for r in results) / len(results) if results else 0.0
    total_trades = sum(len(r.trades) for r in results)

    for res in results:
        table_rows.append(
            f"| {res.symbol} | {res.return_pct:.2f}% | {res.max_drawdown_pct:.2f}% | {res.sharpe:.2f} | {len(res.trades)} |"
        )

    table_rows.append(
        f"| **Combined** | **{combined_return:.2f}%** | **{combined_dd:.2f}%** | **{combined_sharpe:.2f}** | **{total_trades}** |"
    )

    narrative = (
        "Backtests leverage live Yahoo Finance hourly OHLCV data fetched via yfinance.download() for each run. "
        "Optional caches in analysis/cache/ speed up subsequent executions but can be deleted at any time—"
        "the script will always re-fetch data if caches are missing."
    )

    output_markdown.write_text(
        "# Adaptive Momentum Backtest Report\n\n"
        + narrative
        + "\n\n## Summary\n\n"
        + "\n".join(table_rows)
        + "\n\nRun `python reports/backtest_runner.py` to regenerate this table."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Momentum contest backtester")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD"], help="Symbols to backtest")
    parser.add_argument("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-06-30", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1h", help="Data interval (must remain 1h for contest)")
    parser.add_argument("--cash", type=float, default=10000.0, help="Starting cash per symbol")
    parser.add_argument("--strategy-params", default="{}", help="JSON string of strategy overrides")
    parser.add_argument("--no-cache", action="store_true", help="Skip reading/writing cached Yahoo responses")
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "reports" / "backtest_report.md"), help="Path to markdown report output"
    )
    return parser.parse_args()


def load_strategy_params(raw: str) -> Dict[str, Any]:
    text = (raw or "{}").strip()
    if not text:
        return {}
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return json.loads(path.read_text())
    return json.loads(text)


if __name__ == "__main__":
    args = parse_args()
    strategy_params = load_strategy_params(args.strategy_params)
    results: List[BacktestResult] = []
    for symbol in args.symbols:
        df = fetch_data(symbol, args.start, args.end, args.interval, use_cache=not args.no_cache)
        result = run_strategy(symbol, df, strategy_params, starting_cash=args.cash)
        results.append(result)
        print(json.dumps(format_result(result)))

    save_report(results, Path(args.report), starting_cash=args.cash)
    combined_return = sum(r.return_pct for r in results) / len(results) if results else 0.0
    print(f"Combined average return: {combined_return:.2f}%")
