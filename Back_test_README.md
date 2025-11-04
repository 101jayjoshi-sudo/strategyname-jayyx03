# Backtesting Toolkit

The `backtest_runner.py` script reproduces the six-month evaluation required by
the contest. It pulls hourly OHLCV data for BTC-USD and ETH-USD from Yahoo
Finance (via `yfinance`), replays the strategy logic exactly as implemented in
`your_strategy.py`, and exports a Markdown report plus CSV artifacts.

## Environment Setup

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell
pip install -r base-bot-template/requirements.txt
pip install pandas numpy yfinance
```

## Running the Backtest

```bash
python backtest_runner.py --output backtest-report.md
```

Optional parameter overrides can be supplied either inline or via a JSON file:

```bash
# Inline JSON overrides
python backtest_runner.py --config '{"cooldown_minutes": 90, "momentum_threshold": 0.6}'

# Using a reusable config file
python backtest_runner.py --config-file configs/enhanced_params.json
```

Outputs are written alongside the runner:

- `backtest-report.md` – aggregate Markdown summary
- `backtest_summary.json` – machine-readable metrics
- `btc_usd_trades.csv`, `eth_usd_trades.csv` – trade blotters
- `btc_usd_equity_curve.csv`, `eth_usd_equity_curve.csv` – equity curves

Artifacts are cached under `analysis/cache/` so repeated runs are instant. Remove
the cache directory to force a fresh download.
