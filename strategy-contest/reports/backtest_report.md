# Adaptive Momentum Backtest Report

Backtests leverage live Yahoo Finance hourly OHLCV data fetched via yfinance.download() for each run. Optional caches in analysis/cache/ speed up subsequent executions but can be deleted at any time—the script will always re-fetch data if caches are missing.

## Summary

| Symbol | Return % | Max DD % | Sharpe | Trades |
|--------|----------|----------|--------|--------|
| BTC-USD | 26.81% | 10.99% | 1.80 | 78 |
| ETH-USD | 30.49% | 14.98% | 1.69 | 34 |
| **Combined** | **28.65%** | **14.98%** | **1.75** | **112** |

Run `python reports/backtest_runner.py` to regenerate this table.