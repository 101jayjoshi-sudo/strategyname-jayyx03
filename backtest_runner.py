#!/usr/bin/env python3
"""Offline backtesting harness for the adaptive momentum strategy."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Configure import paths for strategy + infrastructure
ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "base-bot-template"
STRATEGY_PATH = ROOT / "your-strategy-template"

if not STRATEGY_PATH.exists():
    STRATEGY_PATH = ROOT / "adaptive-momentum-submission" / "your-strategy-template"
if not STRATEGY_PATH.exists():
    STRATEGY_PATH = Path(__file__).resolve().parent / "your-strategy-template"

if not BASE_PATH.exists():
    BASE_PATH = Path(__file__).resolve().parents[2] / "base-bot-template"

import sys

sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(STRATEGY_PATH))

from exchange_interface import MarketSnapshot  # type: ignore  # noqa: E402
from strategy_interface import Portfolio  # type: ignore  # noqa: E402
from your_strategy import AdaptiveMomentumStrategy  # type: ignore  # noqa: E402

CACHE_DIR = ROOT / "analysis" / "cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class BacktestMetrics:
    symbol: str
    starting_cash: float
    ending_equity: float
    pnl: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    win_rate: float

    def to_markdown_table_row(self) -> str:
        return (
            f"| {self.symbol} | ${self.starting_cash:,.2f} | ${self.ending_equity:,.2f} | "
            f"${self.pnl:,.2f} | {self.return_pct:.2f}% | {self.sharpe_ratio:.2f} | "
            f"{self.max_drawdown:.2f}% | {self.trade_count} | {self.win_rate:.2f}% |"
        )


def fetch_price_history(symbol: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{symbol.replace('-', '_')}_{start}_{end}_{interval}.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
    if data.empty:
        raise RuntimeError(f"No data returned for {symbol} {start}-{end} interval={interval}")

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    data.to_pickle(cache_file)
    return data


def iterate_market_snapshots(prices: pd.Series, symbol: str, history_window: int) -> Iterable[Tuple[datetime, MarketSnapshot]]:
    price_history: List[float] = []
    for timestamp, price in prices.items():
        price_history.append(float(price))
        if len(price_history) > history_window:
            price_history = price_history[-history_window:]
        snapshot = MarketSnapshot(
            symbol=symbol,
            prices=list(price_history),
            current_price=float(price),
            timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
        )
        yield timestamp.to_pydatetime().replace(tzinfo=timezone.utc), snapshot


def run_backtest(symbol: str, df: pd.DataFrame, *, starting_cash: float, strategy_config: Dict[str, float]) -> Tuple[BacktestMetrics, pd.DataFrame, pd.DataFrame]:
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df["Close"].xs(symbol, level=1, axis=1)
    else:
        close_key = "Close"
        alt_key = f"Close.{symbol}"
        if close_key in df.columns:
            close_series = df[close_key]
        elif alt_key in df.columns:
            close_series = df[alt_key]
        else:
            matches = [c for c in df.columns if c.lower().startswith("close")]
            if not matches:
                raise KeyError("Unable to identify Close column in downloaded data")
            close_series = df[matches[0]]
    prices = close_series.dropna().astype(float)

    strategy = AdaptiveMomentumStrategy(strategy_config, exchange=None)
    strategy.prepare()

    slow_period = int(strategy_config.get("slow_period", getattr(strategy, "slow_period", 72)))
    trend_window = int(strategy_config.get("trend_window", getattr(strategy, "trend_window", 36)))
    volatility_period = int(strategy_config.get("volatility_period", getattr(strategy, "volatility_period", 48)))
    history_window = max(slow_period * 3, trend_window * 3, volatility_period * 3, 200)

    portfolio = Portfolio(symbol=symbol, cash=starting_cash)
    equity_curve: List[Tuple[datetime, float]] = []
    trades: List[Dict[str, float]] = []

    position_qty = 0.0
    avg_entry_price = 0.0
    realized_pnl = 0.0
    wins = 0
    losses = 0

    for ts, snapshot in iterate_market_snapshots(prices, symbol, history_window):
        equity_value = portfolio.cash + portfolio.quantity * snapshot.current_price
        equity_curve.append((ts, equity_value))

        signal = strategy.generate_signal(snapshot, portfolio)
        if signal.action == "hold" or signal.size <= 0:
            continue

        price = snapshot.current_price
        if signal.action == "buy":
            affordable = portfolio.cash / price if price > 0 else 0.0
            size = min(signal.size, affordable)
            if size <= 0:
                continue

            cost = size * price
            portfolio.cash -= cost
            portfolio.quantity += size

            new_qty = position_qty + size
            avg_entry_price = ((avg_entry_price * position_qty) + (price * size)) / new_qty if new_qty > 0 else 0.0
            position_qty = new_qty

            strategy.on_trade(signal, price, size, ts)
            trades.append({
                "timestamp": ts.isoformat(),
                "symbol": symbol,
                "side": "buy",
                "price": price,
                "size": size,
                "reason": signal.reason,
            })
        elif signal.action == "sell":
            size = min(signal.size, portfolio.quantity)
            if size <= 0:
                continue

            proceeds = size * price
            portfolio.cash += proceeds
            portfolio.quantity -= size

            realized = (price - avg_entry_price) * size
            realized_pnl += realized
            if realized > 0:
                wins += 1
            elif realized < 0:
                losses += 1

            position_qty = max(0.0, position_qty - size)
            if position_qty <= 0:
                avg_entry_price = 0.0

            strategy.on_trade(signal, price, size, ts)
            trades.append({
                "timestamp": ts.isoformat(),
                "symbol": symbol,
                "side": "sell",
                "price": price,
                "size": size,
                "reason": signal.reason,
                "realized_pnl": realized,
            })

    if equity_curve:
        final_equity = equity_curve[-1][1]
    else:
        final_equity = starting_cash

    equity_series = pd.Series({ts: val for ts, val in equity_curve})
    returns = equity_series.pct_change().dropna()

    sharpe = 0.0
    if not returns.empty:
        hourly_factor = math.sqrt(24 * 365)
        sharpe = (returns.mean() / returns.std()) * hourly_factor if returns.std() > 0 else 0.0

    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = drawdowns.min() * 100 if not drawdowns.empty else 0.0

    trade_count = len(trades)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    metrics = BacktestMetrics(
        symbol=symbol,
        starting_cash=starting_cash,
        ending_equity=final_equity,
        pnl=final_equity - starting_cash,
        return_pct=((final_equity / starting_cash) - 1) * 100,
        sharpe_ratio=sharpe,
        max_drawdown=abs(max_drawdown),
        trade_count=trade_count,
        win_rate=win_rate,
    )

    trade_df = pd.DataFrame(trades)
    equity_df = equity_series.rename("equity").to_frame()
    return metrics, trade_df, equity_df


def write_markdown_report(results: List[BacktestMetrics], out_file: Path) -> None:
    out_lines = [
        "# Adaptive Momentum – Six-Month Backtest",
        "",
        "Period: 2024-01-01 to 2024-06-30 (hourly bars, Yahoo Finance data)",
        "Starting capital per run: $10,000",
        "",
        "| Symbol | Starting Cash | Final Equity | PnL | Return % | Sharpe | Max Drawdown | Trades | Win Rate |",
        "|--------|---------------|--------------|-----|----------|--------|--------------|--------|----------|",
    ]
    out_lines.extend(metric.to_markdown_table_row() for metric in results)
    aggregate = sum(metric.pnl for metric in results)
    out_lines.extend([
        "",
        f"Combined PnL: ${aggregate:,.2f}",
        f"Generated on: {datetime.now(timezone.utc).isoformat()}",
    ])
    out_file.write_text("\n".join(out_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run six-month backtests for the adaptive momentum strategy")
    parser.add_argument("--output", default=str(ROOT / "reports" / "backtest-report.md"), help="Path to Markdown summary")
    parser.add_argument("--config", default="{}", help="JSON string with strategy parameter overrides")
    parser.add_argument("--config-file", help="Path to JSON file with strategy overrides")
    args = parser.parse_args()

    strategy_config: Dict[str, float] = {}

    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            raise SystemExit(f"Config file not found: {config_path}")
        try:
            file_payload = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(file_payload, dict):
                raise ValueError("Config file must contain a JSON object")
            strategy_config.update(file_payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Invalid JSON in --config-file: {exc}")

    try:
        inline_config = json.loads(args.config)
        if not isinstance(inline_config, dict):
            raise ValueError("Inline --config must be a JSON object")
        strategy_config.update(inline_config)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid JSON for --config: {exc}")

    symbols = ["BTC-USD", "ETH-USD"]
    start, end = "2024-01-01", "2024-07-01"
    results: List[BacktestMetrics] = []

    report_dir = Path(args.output).resolve().parent
    report_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        df = fetch_price_history(symbol, start=start, end=end, interval="1h")
        metrics, trades, curve = run_backtest(symbol, df, starting_cash=10000.0, strategy_config=strategy_config)
        results.append(metrics)

        trades.to_csv(report_dir / f"{symbol.replace('-', '_').lower()}_trades.csv", index=False)
        curve.to_csv(report_dir / f"{symbol.replace('-', '_').lower()}_equity_curve.csv")

    write_markdown_report(results, Path(args.output))

    summary_path = report_dir / "backtest_summary.json"
    summary = {metric.symbol: asdict(metric) for metric in results}
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Report written to {args.output}")
    for metric in results:
        print(metric)


if __name__ == "__main__":
    main()
