# Base Bot Template – Universal Trading Infrastructure

The foundation for all trading bot templates, providing complete dashboard integration
and enterprise-grade infrastructure without any trading strategies.

## What's Included

- **Universal orchestration (`universal_bot.py`)** – loads config, wires exchange +
  strategy via factory pattern, runs in-memory portfolio, tracks performance metrics,
  and provides full dashboard integration.
- **Dual HTTP architecture** – port `8080` exposes health/status, port `3010`
  provides authenticated control plane for dashboard integration.
- **Dashboard integration (`http_endpoints.py`)** – complete compatibility with
  main app dashboard including performance, settings, commands, and logs endpoints.
- **Configuration management (`universal_config.py`)** – dataclass with JSON/env
  overrides supporting all bot types and dashboard requirements.
- **Strategy framework (`strategy_interface.py`)** – abstract base classes and
  factory pattern for clean strategy separation.
- **Exchange abstraction (`exchange_interface.py`)** – protocol and paper exchange,
  with Coinbase adapter available.
- **Enterprise features (`integrations.py`)** – PostgreSQL logging, status callbacks,
  and HMAC authentication for production deployment.
- **Minimal dependencies** – only `requests` + optional `psycopg2-binary`.

## Template Inheritance Architecture

This base template provides **infrastructure only** - no trading strategies included.
Strategies are implemented in specific bot templates that inherit this infrastructure:

```
base-bot-template/           # Infrastructure only
├── universal_bot.py         # Core bot orchestration
├── http_endpoints.py        # Dashboard API endpoints
├── integrations.py          # Database & callbacks
├── universal_config.py      # Configuration management
├── strategy_interface.py    # Strategy framework
├── exchange_interface.py    # Exchange abstraction
└── requirements.txt

dca-bot-template/           # DCA strategies
├── dca_strategy.py         # DCA + AdvancedDCA implementations
├── startup.py              # DCA bot main
└── Dockerfile             # Inherits base infrastructure

swing-bot-template/         # Swing strategies
├── swing_strategy.py       # Swing trading implementation
├── startup.py              # Swing bot main
└── Dockerfile             # Inherits base infrastructure

momentum-bot-template/      # Momentum strategies
├── momentum_strategy.py    # Momentum implementation
├── startup.py              # Momentum bot main
└── Dockerfile             # Inherits base infrastructure
```

## Benefits of Template Inheritance

✅ **Clean separation** – Strategy bugs don't affect other bot types
✅ **Easy debugging** – Focused, strategy-specific codebases
✅ **Lean deployments** – Only the needed strategy in each image
✅ **Consistent integration** – Same dashboard experience across all bots
✅ **Enterprise features** – All bots get full infrastructure capabilities

## Dashboard Integration Features

### Performance Monitoring
- Real-time P&L tracking with currency formatting
- Portfolio metrics (cash, positions, unrealized gains)
- Trade history and execution details
- Risk metrics and performance analytics

### Settings Management
- Hot configuration reload without restart
- Dashboard-compatible field mapping
- Strategy-specific parameter validation
- Exchange API key management

### Bot Control
- Start/stop/pause/restart commands
- State management and reporting
- Status callbacks to main app
- Real-time status updates via SSE

### Enterprise Logging
- Structured log output for debugging
- Trade execution logs with P&L
- Error analysis with context
- Database integration for audit trails

## Usage in Bot Templates

Bot templates inherit this infrastructure by importing and extending:

```python
# In your bot template startup.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'base-bot-template'))

# Import your strategies (this registers them)
import your_strategy

# Import base infrastructure
from universal_bot import UniversalBot

def main():
    bot = UniversalBot()
    bot.run()
```

## Environment Variables

All bot templates support these configuration options:

```bash
# Core Configuration
BOT_EXCHANGE=paper|coinbase
BOT_STRATEGY=your_strategy_name
BOT_SYMBOL=BTC-USD
BOT_STARTING_CASH=1000.0
BOT_SLEEP=60

# Dashboard Integration
BOT_INSTANCE_ID=your-bot-id
USER_ID=your-user-id
BOT_SECRET=your-hmac-secret
BASE_URL=https://your-app.com
POSTGRES_URL=postgresql://...

# HTTP Configuration
BOT_HTTP_PORT=8080
BOT_CONTROL_PORT=3010

# Strategy Parameters (JSON)
BOT_STRATEGY_PARAMS='{"param1": "value1"}'

# Exchange Parameters (JSON)
BOT_EXCHANGE_PARAMS='{"api_key": "...", "api_secret": "..."}'
```

## API Endpoints

### Health Check (Port 8080)
- `GET /health` - Bot status and available strategies
- `GET /settings` - Current configuration (read-only)

### Control API (Port 3010, HMAC Authenticated)
- `GET /performance` - Real-time performance metrics
- `GET /settings` - Current configuration with dashboard mapping
- `POST /settings` - Hot configuration reload
- `POST /commands` - Bot control (start/stop/pause/restart)
- `GET /logs` - Recent trading logs

## HMAC Authentication

Control endpoints require HMAC-SHA256 authentication:

```bash
# Headers required for POST requests
X-Bot-Signature: <hmac-sha256-hex-digest>
X-Bot-Timestamp: <millisecond-epoch>
```

## Development

**⚠️ IMPORTANT:** This base template should not be run directly - it has no trading strategies.
Use it as infrastructure for building specific bot templates.

### Creating a New Bot Template

Follow these steps to create a new bot template:

1. **Create Template Directory**
```bash
mkdir my-strategy-bot-template
cd my-strategy-bot-template
```

2. **Create Strategy Implementation**
```python
# my_strategy.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'base-bot-template'))

from strategy_interface import BaseStrategy, Signal, register_strategy
from exchange_interface import MarketSnapshot

class MyStrategy(BaseStrategy):
    def generate_signal(self, market: MarketSnapshot, portfolio) -> Signal:
        # Your trading logic here
        return Signal("hold", reason="Custom logic")

# Register your strategy
register_strategy("my_strategy", lambda cfg, ex: MyStrategy(cfg, ex))
```

3. **Create Startup Script**
```python
# startup.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'base-bot-template'))

import my_strategy  # This registers your strategies
from universal_bot import UniversalBot

def main():
    print("🤖 Starting My Strategy Bot...")
    bot = UniversalBot()
    bot.run()

if __name__ == "__main__":
    main()
```

4. **Create Dockerfile**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ../base-bot-template/ /app/base/
COPY . /app/
RUN pip install -r /app/base/requirements.txt
EXPOSE 8080 3010
CMD ["python", "/app/startup.py"]
```

5. **Test Your Template**
```bash
BOT_STRATEGY=my_strategy python startup.py
```

## Next Steps

- Use `dca-bot-template` for dollar cost averaging strategies
- Use `swing-bot-template` for swing trading strategies
- Use `momentum-bot-template` for momentum-based strategies
- Create custom templates following the inheritance pattern

The base infrastructure ensures consistent dashboard integration and enterprise
features across all bot types while maintaining clean separation of trading logic.