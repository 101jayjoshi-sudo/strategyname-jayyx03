# Adaptive Momentum Strategy Template

Momentum-first trading bot designed for the Trading Strategy Contest infrastructure.

## Files
```
your-strategy-template/
├─ your_strategy.py      # Strategy logic + registration
├─ startup.py            # Entrypoint that boots UniversalBot
├─ Dockerfile            # Container instructions
├─ requirements.txt      # Extra dependencies (strategy + backtests)
└─ README.md             # This guide
```

## Strategy Summary
- **Signal**: Fast EMA must cross above slow EMA while momentum (% change over configurable lookback) stays above a threshold and slope between EMAs is positive.
- **Entries**: When conditions align and cooldown expired, bot deploys up to 50% (capped at contest 55%) of equity, sized by risk budget (`risk_per_trade`) and ATR-derived stop distance.
- **Risk**: Initial stop sits `stop_atr_multiplier * ATR` below entry; take profit = `reward_risk * stop_distance`. Trailing stop ratchets higher once price advances.
- **Exits**: Hit stop, reach target, or lose momentum (fast EMA back under slow EMA or momentum falls below `exit_momentum_floor`). After an exit, a cooldown prevents immediate re-entry.

## Environment Variables
```
BOT_STRATEGY=adaptive_momentum
BOT_SYMBOL=BTC-USD
BOT_STARTING_CASH=10000
BOT_SLEEP=3600
BOT_STRATEGY_PARAMS='{"fast_period":12,"slow_period":48,"momentum_period":24,
  "momentum_threshold":0.75,"risk_per_trade":0.02,"max_position_pct":0.5,
  "reward_risk":2.5,"cooldown_hours":6}'
```
Set `BOT_EXCHANGE=paper` for simulated fills (default) or `coinbase` with proper credentials for live trading.

## Local Run
```powershell
cd strategy-contest\your-strategy-template
python startup.py
```
The startup script automatically imports `your_strategy` (registering the strategy) and launches `UniversalBot` from the base template.

## Docker
Build from repo root so the Dockerfile can copy both the base template and strategy folder.
```powershell
docker build -f your-strategy-template/Dockerfile -t adaptive-momentum .
docker run --rm -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=adaptive_momentum \
  -e BOT_SYMBOL=BTC-USD \
  -e BOT_STARTING_CASH=10000 \
  adaptive-momentum
```

## Testing Overview
- Use `reports/backtest_runner.py` to fetch **live Yahoo Finance hourly data** for BTC-USD and ETH-USD (Jan–Jun 2024) via `yfinance.download()`. No CSVs ship with this repo; cached pickles under `analysis/cache/` are optional acceleration artifacts and can be deleted anytime.
- The runner enforces contest constraints (starting capital, ≤55% exposure, ≥10 trades) and produces metrics for `reports/backtest_report.md`.

## Compliance Checklist
- ✅ Inherits `BaseStrategy` via the contest framework
- ✅ Uses Yahoo Finance hourly OHLCV via `yfinance` during backtests
- ✅ Enforces ≤55% capital per trade and stop-based risk sizing
- ✅ Provides the required deliverables (`your-strategy-template`, `reports/`, `trade_logic_explanation.md`)
- ✅ Designed to exceed leaderboard leader (+36.10% PnL) with controlled drawdown (<50%)
