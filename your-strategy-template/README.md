# Adaptive Momentum Strategy Template

Production-ready trading template designed for the contest infrastructure. The
strategy reacts to medium-term momentum shifts while enforcing volatility-aware
risk limits to keep drawdown under control and satisfy the "at least 10 trades"
requirement during the January–June 2024 evaluation window.

## Strategy Overview

- **Signal core** – trade long-only when the fast EMA is above the slow EMA,
  positive momentum exceeds a configurable threshold, and RSI confirms the
  trend. This avoids chasing weak moves and keeps the bot active during strong
  swings.
- **Risk controls** – position sizing uses a risk-per-trade budget, max
  position fraction, and minimum notional threshold. Hard, trailing, and
  volatility-adjusted stops are always attached.
- **Trade exits** – positions close when protective stops trigger, fast EMA
  drops below the slow EMA, RSI shows loss of momentum, or a dynamic take-profit
  target is reached.
- **Anti-churn cooldown** – after every trade the bot enforces a configurable
  cooldown window in minutes to let the market evolve before re-entering.

The implementation lives in [`your_strategy.py`](your_strategy.py) and registers
its handle as `adaptive_momentum` with the shared `BaseStrategy` factory.


## Running Locally

```bash
# Install base infrastructure dependencies
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r base-bot-template/requirements.txt

# Run the bot using the adaptive momentum strategy
set BOT_STRATEGY=adaptive_momentum
set BOT_SYMBOL=BTC-USD
set BOT_STARTING_CASH=10000
python your-strategy-template/startup.py
```

The bot exposes the standard dashboard endpoints on ports 8080 and 3010, just
like the reference template.

## Docker Image

```
docker build -t adaptive-momentum-bot -f your-strategy-template/Dockerfile .
docker run --rm -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=adaptive_momentum \
  -e BOT_SYMBOL=BTC-USD \
  -e BOT_STARTING_CASH=10000 \
  adaptive-momentum-bot
```

## Backtesting

A standalone research script (`analysis/backtest_runner.py`) is provided to
replay six months of Coinbase minute data for BTC-USD and ETH-USD, producing the
PnL, Sharpe ratio, drawdown series, and trade blotter required by the contest.
See `analysis/README.md` for detailed instructions and reproducibility notes.

## Deliverables Checklist

- `your_strategy.py` – strategy implementation and registration ✔️
- `startup.py` – entry point ✔️
- `Dockerfile` – container recipe ✔️
- `requirements.txt` – dependency notes ✔️
- `README.md` – you are here ✔️
- Backtest package – generated under `reports/` (see analysis docs) ✔️
