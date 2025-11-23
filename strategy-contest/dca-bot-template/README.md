# DCA Bot Template

Dollar Cost Averaging trading bot with full dashboard integration and enterprise features.

## Features

### Basic DCA Strategy (All Tiers)
- **Time-based DCA**: Purchases fixed amounts at regular intervals
- **Configurable intervals**: Minutes, hours, or days between purchases
- **Cash management**: Never exceeds available balance
- **Simple & reliable**: Perfect for beginners

### Advanced DCA Strategy (Enterprise Tier Only)
- **Volatility-aware spacing**: Dynamic intervals based on market volatility
- **Drawdown protection**: Pauses buying during major price drops
- **Position scaling**: Increases position size as price drops
- **Take profit bands**: Automatic profit-taking at target percentages
- **Trailing stops**: Protects gains with dynamic stop-losses
- **Daily limits**: Maximum purchases per day for risk control

## Configuration

### Basic Configuration
```json
{
  "exchange": "paper",
  "strategy": "dca",
  "symbol": "BTC-USD",
  "starting_cash": 1000.0,
  "sleep_seconds": 3600,
  "strategy_params": {
    "interval_minutes": 60,
    "base_amount": 50.0
  }
}
```

### Enterprise Configuration
```json
{
  "strategy": "advanced_dca",
  "strategy_params": {
    "base_amount": 75,
    "max_positions": 6,
    "min_minutes_between_buys": 45,
    "base_drop_pct": 2.0,
    "volatility_factor": 1.8,
    "take_profit_pct": 5.5,
    "trailing_stop_pct": 2.5,
    "drawdown_pause_pct": 15.0,
    "max_daily_buys": 4
  }
}
```

## Environment Variables

```bash
# Core Configuration
BOT_EXCHANGE=paper|coinbase
BOT_STRATEGY=dca|advanced_dca
BOT_SYMBOL=BTC-USD
BOT_STARTING_CASH=1000.0
BASE_AMOUNT=50.0
INTERVAL_MINUTES=60
BOT_SLEEP=3600

# Dashboard Integration
BOT_INSTANCE_ID=your-bot-id
USER_ID=your-user-id
BOT_SECRET=your-hmac-secret
BASE_URL=https://your-app.com
POSTGRES_URL=postgresql://...

# Exchange API (for live trading)
BOT_EXCHANGE_PARAMS='{"api_key":"...","api_secret":"..."}'
```

## Quick Start

### Prerequisites

This template inherits from `base-bot-template`. Ensure the base template exists in the parent directory:

```
your-project/
├── base-bot-template/      # Required infrastructure
└── dca-bot-template/       # This template
```

### Local Development

**Basic DCA:**
```bash
BOT_STRATEGY=dca python startup.py
```

**Advanced DCA (Enterprise):**
```bash
BOT_STRATEGY=advanced_dca BOT_SYMBOL=ETH-USD python startup.py
```

### Docker Deployment

**Build (from repository root):**
```bash
docker build -f dca-bot-template/Dockerfile -t dca-bot .
```

**Run Basic DCA:**
```bash
docker run -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=dca \
  -e BOT_SYMBOL=BTC-USD \
  -e BOT_STARTING_CASH=1000 \
  dca-bot
```

**Run Advanced DCA:**
```bash
docker run -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=advanced_dca \
  -e BOT_SYMBOL=ETH-USD \
  -e BOT_STARTING_CASH=2000 \
  -e BOT_STRATEGY_PARAMS='{"base_amount":75,"max_positions":6,"take_profit_pct":5.5}' \
  dca-bot
```

### Production Deployment

**With Dashboard Integration:**
```bash
docker run -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=advanced_dca \
  -e BOT_INSTANCE_ID=bot-abc123 \
  -e USER_ID=user-456 \
  -e BOT_SECRET=your-hmac-secret \
  -e BASE_URL=https://your-app.com \
  -e POSTGRES_URL=postgresql://... \
  -e BOT_EXCHANGE_PARAMS='{"api_key":"...","api_secret":"..."}' \
  dca-bot
```

## Dashboard Integration

Full compatibility with the main app dashboard:

- **Performance Metrics**: Real-time P&L, positions, trade history
- **Settings Management**: Hot configuration reload via dashboard
- **Bot Controls**: Start/stop/pause/restart from dashboard
- **Live Logs**: Structured log output with trade details
- **Status Reporting**: Real-time status updates via callbacks

### Advanced Settings Features

- **Settings History**: Complete audit trail of all configuration changes with timestamps
- **Settings Restore**: Rollback to any previous configuration (excludes API keys for security)
- **Last Saved Badge**: Real-time display of when settings were last saved
- **Time Calculator**: Interactive interval calculator with presets (15min, 1hr, 1day, 3days, 1week, 2weeks)
- **Smart Validation**: 14-day maximum interval limit with warnings for long periods
- **Reset to Defaults**: Properly clears all fields to empty state
- **Exchange Configuration**: Reusable component with conditional paper trading support
- **Connection Testing**: Unified bot connection testing with consistent messaging

## API Endpoints

### Health Check (Port 8080)
- `GET /health` - Bot status and available strategies

### Control API (Port 3010, HMAC Authenticated)
- `GET /performance` - Real-time performance metrics
- `GET /settings` - Current configuration
- `POST /settings` - Hot configuration reload with validation
- `POST /commands` - Bot control (start/stop/pause/restart)
- `GET /logs` - Recent trading logs

### Dashboard Settings API
- `GET /api/bots/[id]/settings/dca` - Retrieve DCA bot settings from database
- `POST /api/bots/[id]/settings/dca` - Save DCA settings with validation and metadata
- `GET /api/bots/[id]/settings/dca/history` - Complete settings history with change tracking
- **Validation**: Required fields (baseAmount, intervalMinutes), interval limits (max 20160 minutes)
- **Metadata**: Automatic timestamps, configuration IDs, change detection

## Strategies

| Strategy | Tier | Description |
|----------|------|-------------|
| `dca` | Basic | Simple time-based dollar cost averaging |
| `advanced_dca` | Enterprise | Sophisticated adaptive DCA with risk management |

## Enterprise Features

Advanced DCA strategy includes:
- **Smart Timing**: Volatility-based purchase intervals
- **Risk Management**: Drawdown protection and daily limits
- **Profit Optimization**: Take profit bands with trailing stops
- **Position Scaling**: Dynamic position sizing based on price action
- **Advanced Analytics**: Detailed performance metrics and reporting

Perfect for professional traders and institutional users requiring sophisticated automation.