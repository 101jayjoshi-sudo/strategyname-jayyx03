# DCA Bot Manual - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [DCA Bot Pipeline & Flow](#dca-bot-pipeline--flow)
3. [Startup Phase Analysis](#startup-phase-analysis)
4. [Run Phase Analysis](#run-phase-analysis)
5. [DCA Strategy Deep Dive](#dca-strategy-deep-dive)
6. [Infinite Loop Operations](#infinite-loop-operations)
7. [Trading Logic & Decision Making](#trading-logic--decision-making)
8. [Advanced DCA Strategy (Enterprise)](#advanced-dca-strategy-enterprise)
9. [Dashboard Settings Management](#dashboard-settings-management)
10. [Configuration Parameters](#configuration-parameters)
11. [Spending Limit Controls](#spending-limit-controls)
12. [Risk Management](#risk-management)

---

## Overview

The **DCA (Dollar-Cost Averaging) Bot** is a systematic trading bot that implements time-based cryptocurrency purchasing strategies. It operates on the principle of buying fixed amounts at regular intervals, regardless of price fluctuations, to minimize the impact of volatility and reduce average purchase costs over time.

### Core Philosophy
- **Time-based purchases**: Buys at predetermined intervals
- **Fixed amount investing**: Consistent investment regardless of price
- **Emotion-free trading**: Removes psychological bias from investment decisions
- **Long-term accumulation**: Focuses on gradual position building

---

## DCA Bot Pipeline & Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DCA BOT PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

1. STARTUP PHASE
   ├── Load configuration from environment variables
   ├── Initialize Universal Bot infrastructure
   ├── Connect to exchange (Paper/Coinbase)
   ├── Load DCA strategy (dca or advanced_dca)
   ├── Setup portfolio tracker
   └── Initialize HTTP servers (ports 8080, 3010)

2. RUN PHASE
   ├── Start health check server (port 8080)
   ├── Start control API server (port 3010)
   ├── Begin status broadcasting to dashboard
   └── Enter infinite trading loop

3. INFINITE LOOP CYCLE
   ├── Fetch market snapshot (current price + history)
   ├── Execute DCA strategy logic
   ├── Generate trading signal (buy/hold/sell)
   ├── Execute trades if signal requires action
   ├── Update portfolio state
   ├── Log results and broadcast status
   ├── Sleep for configured interval
   └── Repeat cycle
```

---

## Startup Phase Analysis

### 🚀 What Happens During Startup

#### 1. **Configuration Loading**
```python
# Environment variables loaded:
BOT_SYMBOL=BTC-USD          # Trading pair
BOT_STARTING_CASH=1000      # Initial cash balance
BOT_SLEEP=900               # Sleep between cycles (seconds)
BOT_EXCHANGE=paper          # Exchange type
BOT_STRATEGY=dca            # Strategy selection
```

#### 2. **Infrastructure Initialization**
- **Universal Bot**: Core bot framework with shared functionality
- **Exchange Connection**: Paper trading or live Coinbase integration
- **Portfolio Tracker**: Manages cash, positions, and P&L tracking
- **Strategy Loading**: Instantiates DCA strategy with configuration

#### 3. **Component Setup**
- **HTTP Server (8080)**: Health checks and public endpoints
- **Control Server (3010)**: Authenticated settings updates via HMAC
- **Status Broadcasting**: Real-time updates to dashboard
- **Database Integration**: Optional PostgreSQL for logging and state

#### 4. **Strategy Preparation**
```python
class DcaStrategy:
    def __init__(self, config, exchange):
        self.interval_minutes = 60      # Default buy interval
        self.base_amount = 50.0         # Default purchase amount
        self._last_purchase = None      # Track last buy time
```

---

## Run Phase Analysis

### ⚡ Server Initialization

#### **HTTP Server (Port 8080)**
- **Purpose**: Public health checks and status
- **Endpoints**: `/health`, `/status`
- **Access**: Open to dashboard for monitoring

#### **Control Server (Port 3010)**
- **Purpose**: Secure settings updates
- **Authentication**: HMAC-SHA256 with BOT_SECRET
- **Endpoints**: `/settings` (POST), `/commands` (POST)
- **VPC Only**: Internal network communication

#### **Status Broadcasting**
- **Real-time updates** to dashboard
- **Portfolio changes** and trade executions
- **Error reporting** and bot health status

---

## DCA Strategy Deep Dive

### 📊 Basic DCA Strategy Logic

#### **Core Principle**: Time-Based Purchasing
The basic DCA strategy follows a simple rule: **"Buy a fixed amount every N minutes"**

```python
def generate_signal(self, market, portfolio):
    now = current_time

    # Check if enough time has passed since last purchase
    if last_purchase_time + interval_minutes < now:
        if portfolio.cash >= base_amount:
            size = base_amount / current_price
            return Signal("buy", size=size, reason="Scheduled DCA buy")

    return Signal("hold", reason="Waiting for next interval")
```

#### **Decision Flow**:
1. **Time Check**: Has enough time passed since last buy?
2. **Cash Check**: Do we have sufficient funds?
3. **Price Check**: Is current price valid (> 0)?
4. **Execute**: Calculate size and place buy order

### 🎯 What the DCA Bot is "Hunting"

The DCA bot is NOT hunting for:
- ❌ Price patterns or technical indicators
- ❌ Market reversals or breakouts
- ❌ Support/resistance levels
- ❌ Volume spikes or momentum

The DCA bot IS hunting for:
- ✅ **Time intervals** - Fixed schedule adherence
- ✅ **Cash availability** - Ensuring funds for purchases
- ✅ **Consistent execution** - Regular accumulation regardless of price

### 📈 When Does DCA Bot Buy?

#### **Basic DCA Strategy**:
```
BUY CONDITIONS (ALL must be true):
├── Time elapsed ≥ interval_minutes since last purchase
├── portfolio.cash ≥ base_amount
├── current_price > 0 (valid market data)
└── No external pause/stop commands
```

#### **Buy Logic Example**:
- **Interval**: Every 60 minutes
- **Amount**: $50 per purchase
- **Price**: Current market price (no waiting for "good" prices)

### 💰 When Does DCA Bot Sell?

#### **Basic DCA Strategy**:
- **NEVER sells automatically** - Pure accumulation strategy
- Only sells on **manual commands** via dashboard
- Focus is on **building position over time**

---

## Infinite Loop Operations

### 🔄 Loop Cycle Breakdown

#### **Each 15-minute cycle (configurable)**:

```python
while True:
    # 1. MARKET DATA COLLECTION
    snapshot = exchange.fetch_market_snapshot(
        symbol=BOT_SYMBOL,
        limit=history_length
    )

    # 2. STRATEGY SIGNAL GENERATION
    signal = dca_strategy.generate_signal(snapshot, portfolio)

    # 3. TRADE EXECUTION (if required)
    if signal.action == "buy":
        execution = execute_trade(signal, current_price)
        if execution:
            strategy.on_trade(signal, execution.price, execution.size)

    # 4. PORTFOLIO UPDATE
    portfolio.update_positions()

    # 5. STATUS BROADCASTING
    broadcast_status_to_dashboard()

    # 6. CYCLE COMPLETION
    log_cycle_completion()
    sleep(BOT_SLEEP_seconds)
```

#### **Cycle Timeline**:
- **Fetch Market Data**: ~1-2 seconds
- **Strategy Processing**: <1 second
- **Trade Execution**: 2-5 seconds (if buying)
- **Status Updates**: 1-2 seconds
- **Sleep Period**: 900 seconds (15 minutes default)

### 📊 Market Data Analysis

#### **What Data is Collected**:
```python
MarketSnapshot {
    symbol: "BTC-USD"
    current_price: 45230.50
    timestamp: 2024-09-30T11:30:00Z
    prices: [45100, 45150, 45200, 45230]  # Recent history
    volumes: [1250, 1180, 1340, 1220]     # Volume data
}
```

#### **How Data is Used**:
- **current_price**: For position sizing calculations
- **timestamp**: For interval timing decisions
- **price history**: Not used in basic DCA (used in Advanced DCA)

---

## Advanced DCA Strategy (Enterprise)

### 🚀 Enhanced Features

The Advanced DCA strategy adds sophisticated risk management and adaptive timing:

#### **Key Enhancements**:
1. **Volatility-Aware Spacing**: Adjusts buy frequency based on market volatility
2. **Price Drop Targeting**: Buys more aggressively during price drops
3. **Take Profit Bands**: Automatic profit taking at configured levels
4. **Trailing Stop Loss**: Dynamic stop loss that follows price up
5. **Position Scaling**: Larger purchases as drawdown increases
6. **Daily Buy Limits**: Risk management through frequency caps

### 🎯 Advanced DCA Hunting Strategy

#### **What Advanced DCA Hunts For**:
- ✅ **Price drops** ≥ 2.5% from recent highs
- ✅ **Volatility opportunities** - More aggressive during high volatility
- ✅ **Drawdown recovery** - Larger purchases during significant drops
- ✅ **Take profit zones** - 6%+ gains from average entry
- ✅ **Trailing stop triggers** - 3% drops from recent highs

#### **Buy Conditions (Advanced)**:
```python
BUY CONDITIONS:
├── Price dropped ≥ dynamic_threshold% from reference price
├── Time elapsed ≥ min_minutes_between_buys
├── Daily buy count < max_daily_buys
├── Portfolio capacity < max_positions
├── Cash available for scaled position size
└── Not in drawdown protection mode
```

#### **Sell Conditions (Advanced)**:
```python
SELL CONDITIONS (ANY can trigger):
├── Take profit: Gain ≥ take_profit_pct% from average entry
├── Trailing stop: Price dropped ≥ trailing_stop_pct% from high
└── Drawdown protection: Loss ≥ drawdown_pause_pct%
```

### 📊 Dynamic Threshold Calculation

```python
def _dynamic_drop_threshold(self, market):
    base_threshold = 2.5%  # Minimum drop required

    # Calculate recent volatility
    volatility = calculate_volatility(market.prices, window=30)

    # Adjust threshold based on volatility
    dynamic_threshold = base_threshold * (1 + volatility_factor * volatility)

    return dynamic_threshold  # Higher volatility = higher threshold
```

---

## Dashboard Settings Management

### 🎛️ Advanced Settings Interface

The DCA bot includes a sophisticated dashboard interface for real-time configuration management:

#### **Settings History & Audit Trail**
- **Complete Change Tracking**: Every configuration change is logged with timestamps
- **Settings Restore**: Rollback to any previous configuration (API keys excluded for security)
- **Last Saved Badge**: Real-time display showing when settings were last saved
- **Change Detection**: Automatic detection and highlighting of modified parameters

#### **Interactive Time Calculator**
- **Visual Calculator**: Interactive widget with days/hours/minutes inputs
- **Quick Presets**: One-click buttons for common intervals:
  - 15 minutes, 1 hour, 1 day, 3 days, 1 week, 2 weeks
- **Real-time Conversion**: Automatic conversion between time units
- **Smart Validation**: 14-day maximum limit (20,160 minutes) with warnings

#### **Enhanced User Experience**
- **Reset to Defaults**: Properly clears all fields to empty state
- **Field Validation**: Real-time validation with clear error messages
- **Exchange Configuration**: Reusable component with conditional paper trading support
- **Connection Testing**: Unified bot connection testing with status feedback

#### **API Integration**
```python
# Dashboard Settings API Endpoints
GET  /api/bots/[id]/settings/dca         # Retrieve current settings
POST /api/bots/[id]/settings/dca         # Save settings with validation
GET  /api/bots/[id]/settings/dca/history # Complete settings history
```

#### **Settings Persistence Flow**
```
User Input → Real-time Validation → Database Storage → Bot Update
    ↓              ↓                      ↓              ↓
Frontend    Client-side checks    PostgreSQL flags    Hot reload
Validation      ↓                   JSON field          via API
               Server-side              ↓
               validation         Change tracking
```

#### **Required Fields Validation**
- **baseAmount**: Purchase amount per interval (must be > 0)
- **intervalMinutes**: Time between purchases (1-20,160 minutes)
- **botSymbol**: Trading pair symbol
- **botStartingCash**: Initial cash balance

#### **Metadata Tracking**
```json
{
  "baseAmount": 50.0,
  "intervalMinutes": 60,
  "lastSaved": "2024-09-30T18:30:00Z",
  "configId": "dca-config-abc123",
  "changedFields": ["baseAmount", "intervalMinutes"]
}
```

---

## Configuration Parameters

### 📋 Basic DCA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOT_SYMBOL` | `BTC-USD` | Trading pair to purchase |
| `BOT_STARTING_CASH` | `1000` | Initial cash balance |
| `BOT_SLEEP` | `900` | Seconds between cycles (15 min) |
| `interval_minutes` | `60` | Minutes between purchases |
| `base_amount` | `50.0` | Fixed purchase amount USD |

### 🚀 Advanced DCA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_positions` | `10` | Maximum number of open positions |
| `min_minutes_between_buys` | `60` | Minimum time between purchases |
| `base_drop_pct` | `2.5` | Base price drop threshold % |
| `volatility_window` | `30` | Lookback window for volatility |
| `volatility_factor` | `2.0` | Volatility sensitivity multiplier |
| `scale_factor` | `1.5` | Position size scaling factor |
| `take_profit_pct` | `6.0` | Take profit threshold % |
| `trailing_stop_pct` | `3.0` | Trailing stop loss % |
| `drawdown_pause_pct` | `15.0` | Drawdown protection trigger % |
| `max_daily_buys` | `4` | Maximum purchases per day |

---

## Spending Limit Controls

### 💰 Automatic Spending Management

The DCA bot includes sophisticated spending limit controls to prevent overspending beyond configured budgets. This feature ensures that bots respect their `starting_cash` parameter and never exceed their intended investment limits.

#### **How It Works**:

```python
# Each time the bot considers a purchase:
total_spent = get_total_spent_from_database()
remaining_cash = starting_cash - total_spent
if purchase_amount <= remaining_cash:
    execute_purchase()
else:
    skip_purchase("Spending limit exceeded")
```

### 🎯 Key Features

#### **Real-time Spending Tracking**:
- **Database Integration**: Total spent amount is tracked in the `bots.total_spent` field
- **Automatic Updates**: Each buy trade automatically increments the total spent amount
- **Cache Optimization**: Local caching reduces database queries during trading cycles

#### **Pre-Purchase Validation**:
- **Budget Check**: Validates available budget before every purchase decision
- **Detailed Logging**: Comprehensive logging of spending decisions and limits
- **Graceful Handling**: Bot continues running but skips purchases when limit is reached

#### **Dynamic Budget Management**:
- **Configuration Updates**: Starting cash can be adjusted via dashboard settings
- **Immediate Effect**: Changes take effect on the next trading cycle
- **Historical Preservation**: Past spending history is maintained for audit purposes

### 📊 Spending Limit Logic

#### **Decision Flow**:
```
Trading Cycle Start
    ↓
Market Data Collection
    ↓
Time Interval Check (✓)
    ↓
Cash Availability Check (✓)
    ↓
💰 SPENDING LIMIT CHECK
    ├── Get total_spent from database
    ├── Calculate: remaining = starting_cash - total_spent
    ├── Check: purchase_amount <= remaining
    ├── If TRUE: Continue to purchase
    └── If FALSE: Skip purchase (log reason)
    ↓
Purchase Execution (if approved)
    ↓
Update total_spent in database
    ↓
Continue to next cycle
```

### 🔧 Implementation Details

#### **Database Schema**:
```sql
-- bots table enhancement
ALTER TABLE bots ADD COLUMN total_spent DECIMAL(15,2) NOT NULL DEFAULT 0.00;

-- Index for performance
CREATE INDEX idx_bots_total_spent ON bots(total_spent);
```

#### **Strategy Integration**:
```python
# DCA Strategy enhanced with spending controls
class DcaStrategy(BaseStrategy):
    def __init__(self, config, exchange):
        # ... existing initialization ...
        self._starting_cash = float(config.get("starting_cash", 10000.0))
        self._db_client = config.get("db_client")
        self._total_spent_cache = None

    def _check_spending_limit(self, amount):
        total_spent = self._get_total_spent()
        remaining_cash = self._starting_cash - total_spent
        return amount <= remaining_cash

    def generate_signal(self, market, portfolio):
        # ... time and cash checks ...

        # NEW: Spending limit check
        if not self._check_spending_limit(notional):
            return Signal("hold", reason="Spending limit exceeded")

        # ... proceed with purchase ...
```

### 📈 Example Scenarios

#### **Scenario 1: Normal Operation**
```
Starting Cash: $10,000
Total Spent: $8,500
Purchase Amount: $1,000
Remaining: $1,500

Result: ✅ Purchase approved ($1,000 <= $1,500)
```

#### **Scenario 2: Limit Reached**
```
Starting Cash: $10,000
Total Spent: $9,800
Purchase Amount: $1,000
Remaining: $200

Result: ❌ Purchase skipped ($1,000 > $200)
Log: "HOLD | reason=spending_limit_exceeded"
```

#### **Scenario 3: Budget Reduction**
```
Original Starting Cash: $10,000
Total Spent: $8,200
NEW Starting Cash: $8,000 (reduced via dashboard)
Remaining: $8,000 - $8,200 = -$200

Result: ❌ All future purchases blocked until budget increased
```

### 🚨 Error Handling & Logging

#### **Comprehensive Logging**:
```
[DCA/SPENDING] limit_check | starting=$10,000.00 | spent=$8,500.00 | remaining=$1,500.00 | requested=$1,000.00 | can_spend=true
[DCA/DECISION] BUY | reason=scheduled_dca | size=0.00826688 | notional=$1,000.00
[DCA/ACTION] EXEC BUY 0.00826688 @ $120,964.55 | total=$1,000.00 | last_purchase=2025-10-03T14:51:08+00:00
```

#### **Database Error Handling**:
- **Connection Failures**: Gracefully falls back to allowing purchases (fail-open)
- **Query Errors**: Logs warnings but doesn't crash the trading cycle
- **Reconnection Logic**: Automatic database reconnection on connection loss

### ⚙️ Configuration

#### **Required Parameters**:
- `starting_cash`: Maximum total amount the bot can spend
- `db_client`: Database client instance for spending tracking

#### **Optional Parameters**:
- `strategy_local_logs`: Enable detailed spending limit logging (default: true)

#### **Environment Variables**:
```bash
BOT_STARTING_CASH=10000    # Maximum spending limit
STRATEGY_LOCAL_LOGS=true   # Enable detailed logging
```

### 🎛️ Dashboard Integration

#### **Real-time Monitoring**:
- **Total Spent Display**: Shows cumulative spending across all trades
- **Remaining Budget**: Real-time calculation of available funds
- **Spending History**: Complete audit trail of all purchases

#### **Budget Management**:
- **Starting Cash Updates**: Immediate effect on next trading cycle
- **Spending Reset**: Manual reset of total_spent for fresh starts
- **Limit Alerts**: Dashboard warnings when approaching spending limits

### 🛡️ Security & Safety

#### **Fail-Safe Design**:
- **Conservative Approach**: Errs on the side of preventing overspending
- **Audit Trail**: Complete database record of all spending decisions
- **Rollback Support**: Can restore previous spending states if needed

#### **Prevention of Overspending**:
- **Double-Check Logic**: Validates spending limits before every purchase
- **Database Consistency**: Atomic updates ensure accurate spending tracking
- **Real-time Updates**: Immediate reflection of purchases in spending totals

---

## Risk Management

### 🛡️ Built-in Protections

#### **Cash Management**:
- Never exceeds available cash balance
- Reserves funds for future purchases
- Automatic position sizing based on available capital

#### **Position Limits**:
- Maximum position count (Advanced DCA)
- Daily purchase frequency limits
- Minimum time between purchases

#### **Drawdown Protection**:
- Pauses buying during severe drawdowns (>15%)
- Trailing stop loss for position protection
- Take profit bands to secure gains

#### **Technical Safeguards**:
- Invalid price data handling
- Exchange connection error recovery
- Graceful degradation during outages

### ⚠️ Risk Considerations

#### **Market Risks**:
- DCA doesn't protect against prolonged bear markets
- No stop-loss in basic DCA strategy
- Continued buying during price declines

#### **Operational Risks**:
- Exchange API failures
- Network connectivity issues
- Configuration errors

#### **Liquidity Risks**:
- Large positions may impact market price
- Exchange liquidity limitations
- Slippage during volatile periods

---

## Example Trading Scenarios

### 📈 Scenario 1: Basic DCA in Sideways Market

```
Time: 00:00 | Price: $45,000 | Action: BUY $50 (0.00111 BTC)
Time: 01:00 | Price: $44,800 | Action: BUY $50 (0.00112 BTC)
Time: 02:00 | Price: $45,200 | Action: BUY $50 (0.00111 BTC)
Time: 03:00 | Price: $45,100 | Action: BUY $50 (0.00111 BTC)

Result: Accumulated 0.00445 BTC for $200
Average cost: $44,975 (vs market avg $45,025)
```

### 📉 Scenario 2: Advanced DCA During Price Drop

```
Time: 00:00 | Price: $45,000 | Action: HOLD (no trigger)
Time: 01:00 | Price: $43,875 | Action: BUY $50 (2.5% drop)
Time: 01:30 | Price: $42,750 | Action: BUY $75 (5% drop, scaled)
Time: 02:00 | Price: $43,200 | Action: HOLD (< 60min since last)
Time: 03:00 | Price: $46,800 | Action: SELL 50% (take profit at 6%+)

Result: Strategic accumulation during dip + profit taking
```

---

## Monitoring & Maintenance

### 📊 Key Metrics to Watch

#### **Performance Indicators**:
- Average purchase price vs market price
- Total position size and value
- Unrealized P&L percentage
- Purchase frequency and timing

#### **Health Indicators**:
- Bot uptime and cycle completion
- Exchange connectivity status
- Cash balance and utilization
- Error rates and failed transactions

#### **Risk Indicators**:
- Maximum drawdown experienced
- Position concentration risk
- Daily/weekly purchase frequency
- Portfolio correlation with market

### 🔧 Operational Maintenance

#### **Regular Tasks**:
- Monitor cash balance for depletion
- Review and adjust interval timing
- Check exchange API limits
- Validate configuration parameters

#### **Troubleshooting**:
- Exchange connection failures → Check API credentials
- Missing purchases → Verify cash balance and timing
- Price data errors → Validate symbol configuration
- High error rates → Check network connectivity

---

## Conclusion

The DCA Bot represents a disciplined, systematic approach to cryptocurrency accumulation. By removing emotional decision-making and maintaining consistent purchasing patterns, it aims to reduce the impact of market volatility and achieve better average entry prices over time.

**Key Success Factors**:
- Consistent execution regardless of market conditions
- Proper cash management and position sizing
- Regular monitoring and parameter adjustment
- Long-term perspective and patience

**Best Use Cases**:
- Long-term cryptocurrency accumulation
- Volatile market environments
- Hands-off investment approaches
- Risk-averse trading strategies

The bot excels in environments where time-based discipline outperforms emotional trading decisions, making it an excellent tool for systematic wealth building in cryptocurrency markets.