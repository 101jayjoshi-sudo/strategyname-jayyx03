# Submission (jayyx03) (entry number 36)

Thank you for reviewing this contest deliverable. The archive contains the
complete Adaptive Momentum trading bot, backtest evidence, and tooling to
reproduce the January–June 2024 evaluation on BTC-USD and ETH-USD.

## Package Contents

```
your-strategy-template/
├── your_strategy.py        # Strategy implementation registering "adaptive_momentum"
├── startup.py              # Entry point wiring into UniversalBot
├── Dockerfile              # Container build
├── requirements.txt        # Strategy-specific dependency notes
└── README.md               # Parameter documentation & usage guidance

analysis/
├── backtest_runner.py      # Six-month replay harness (hourly Yahoo Finance data)
└── README.md               # Setup instructions for analysis tooling

reports/
├── backtest-report.md      # Human-readable summary of BTC/ETH results
├── backtest_summary.json   # Machine-readable metrics
├── btc_usd_trades.csv      # Trade blotter (BTC-USD)
├── eth_usd_trades.csv      # Trade blotter (ETH-USD)
├── btc_usd_equity_curve.csv
└── eth_usd_equity_curve.csv
```

## Quick Start

1. Create a Python environment and install dependencies:
   ```powershell
   python -m venv .venv
   . .venv\Scripts\Activate.ps1
   pip install -r base-bot-template/requirements.txt
   ```
2. Run the bot locally (defaults shown):
   ```powershell
   set BOT_STRATEGY=adaptive_momentum
   set BOT_SYMBOL=BTC-USD
   set BOT_STARTING_CASH=10000
   python your-strategy-template/startup.py
   ```
   The dashboard endpoints are exposed on ports 8080 (status/performance) and 3010 (control plane).
3. Build & run via Docker if preferred:
   ```powershell
   docker build -t adaptive-momentum-bot -f your-strategy-template/Dockerfile .
   docker run --rm -p 8080:8080 -p 3010:3010 \
     -e BOT_STRATEGY=adaptive_momentum \
     -e BOT_SYMBOL=BTC-USD \
     -e BOT_STARTING_CASH=10000 \
     adaptive-momentum-bot
   ```

## Backtest Summary (Jan–Jun 2024)

- Starting cash: $10,000 per asset (BTC-USD, ETH-USD)
- Combined PnL: **$2,063.82** on $20,000 capital
- Max drawdown: **< 6%** on both assets
- Trades executed: 172 (BTC-USD), 180 (ETH-USD)
- Sharpe ratios: 3.86 (BTC-USD), 1.45 (ETH-USD)

Full metrics and trade logs are available in the `reports/` directory.

## Reproducing the Backtest

1. Install research dependencies:
   ```powershell
   pip install -r base-bot-template/requirements.txt pandas numpy yfinance
   ```
2. Run the analysis script:
   ```powershell
   python analysis/backtest_runner.py --output reports/backtest-report.md
   ```
   Hourly price data is downloaded once and cached under `analysis/cache/` (not included in the archive).

## Support

Please reach out if anything is unclear or if you encounter issues running the
code. The package is self-contained and ready for contest submission.

