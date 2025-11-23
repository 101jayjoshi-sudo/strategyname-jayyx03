#!/usr/bin/env python3
"""Simplified universal bot runner with strategy/exchange switching."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional

# Import enhanced logging system
from enhanced_logging import (
    setup_enhanced_logging,
    get_trade_logger,
    get_performance_logger,
    log_trade_execution,
    log_strategy_signal,
    log_bot_status,
    log_performance_metrics
)

# Ensure built-in exchanges/strategies are registered.
import coinbase_exchange  # noqa: F401

from exchange_interface import ExchangeRegistry, TradeExecution
from http_endpoints import BotControlServer, BotHTTPServer
from integrations import DatabaseClient, StatusBroadcaster
from strategy_interface import Portfolio, Signal, available_strategies, create_strategy
from universal_config import BotConfig
# Generated ENV schema from registry - replaces settings_mapping.py
from env_schema import (
    STRATEGY_MAPPINGS,
    map_dashboard_to_env_vars,
    validate_dashboard_settings,
    apply_settings_with_scope_check,
)


class UniversalBot:
    """Tiny orchestration layer that wires config, exchange, strategy, and HTTP endpoints."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self.config = BotConfig.load(config_path)
        self._http_server: Optional[BotHTTPServer] = None
        self._control_server: Optional[BotControlServer] = None
        self._cycle = 0
        self._running = False
        self._paused = False
        self._stop_requested = False
        self._restart_requested = False
        self._last_price: Optional[float] = None
        self._last_portfolio_value: Optional[float] = None
        self._last_snapshot_at: Optional[datetime] = None
        self._last_signal: Optional[Signal] = None
        self._last_execution: Optional[TradeExecution] = None
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._avg_entry_price = 0.0
        self._trades: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._started_at = datetime.utcnow()

        self._configure_logging()

        # Setup log file path based on bot instance ID
        log_file_path = None
        if hasattr(self.config, 'bot_instance_id') and self.config.bot_instance_id:
            log_file_path = f"/app/logs/bot-{self.config.bot_instance_id}.log"
        else:
            log_file_path = "/app/logs/universal-bot.log"

        self.logger = setup_enhanced_logging(
            log_level="INFO",
            log_file=log_file_path,
            detail_logging=True,  # Enable detailed logging by default
            logger_name="universal-bot"
        )

        # Log where logs are being saved
        if log_file_path:
            self.logger.info(f"📁 Logs are being saved to: {log_file_path}")
            self.logger.info(f"📁 Log rotation: 10MB max size, 5 backup files")

        self.trade_logger = get_trade_logger()
        self.performance_logger = get_performance_logger()
        self._last_applied_env_vars: Dict[str, str] = {}

        self.exchange = None
        self.strategy = None
        self._db_client = DatabaseClient(
            database_url=self.config.database_url,
            bot_instance_id=self.config.bot_instance_id,
            logger=self.logger,
        )

        # Initialize portfolio and restore position from database (source of truth)
        self.portfolio = Portfolio(symbol=self.config.symbol, cash=self.config.starting_cash)
        self._restore_portfolio_from_database()
        self._status_broadcaster = StatusBroadcaster(
            base_url=self.config.base_url,
            bot_instance_id=self.config.bot_instance_id,
            bot_secret=self.config.bot_secret,
            user_id=self.config.user_id,
            logger=self.logger,
        )

        self._build_components()

    def _restore_portfolio_from_database(self) -> None:
        """Restore portfolio position from database to handle bot restarts."""
        try:
            if self._db_client:
                db_quantity = self._db_client.get_portfolio_quantity()
                if db_quantity > 0:
                    self.portfolio.quantity = db_quantity
                    self.logger.info(f"Restored portfolio position from database: {db_quantity:.8f}")
                else:
                    self.logger.info("No existing position found in database - starting fresh")
            else:
                self.logger.warning("No database client - cannot restore position")
        except Exception as exc:
            self.logger.warning(f"Failed to restore portfolio position from database: {exc}")

    def _build_components(self) -> None:
        print(">>>>>>>>> STARTUP IS STARTING")
        print("Initializing bot components...")

        with self._lock:
            print("Creating exchange connection...")
            self.exchange = ExchangeRegistry.create(self.config.exchange, **self.config.exchange_params)

            print(f"Loading strategy: {self.config.strategy}")

            # Prepare strategy config with additional universal bot parameters
            strategy_config = dict(self.config.strategy_params)
            strategy_config["starting_cash"] = self.config.starting_cash
            strategy_config["db_client"] = self._db_client

            self.strategy = create_strategy(
                self.config.strategy,
                config=strategy_config,
                exchange=self.exchange,
            )

            print("Preparing strategy...")
            self.strategy.prepare()
            self.portfolio.symbol = self.config.symbol

            print("Bot components initialized successfully")
            self.logger.info(
                "Bot ready with exchange=%s strategy=%s symbol=%s",
                self.config.exchange,
                self.config.strategy,
                self.config.symbol,
            )
        print(">>>>>>>>> STARTUP COMPLETED")
        print()

    def _check_configuration_complete(self) -> bool:
        """
        Check if bot has received configuration from UI.
        Returns True if ready to start trading.
        """
        # Primary check: configuration flag file (UI configuration received)
        config_flag_file = "/app/state/config_received.flag"
        if os.path.exists(config_flag_file):
            print("✅ Configuration flag found - proceeding with trading")
            return True

        # Secondary check: persisted configuration exists
        persisted_config_file = "/app/state/config.json"
        if os.path.exists(persisted_config_file):
            print("✅ Persisted configuration found - proceeding with trading")
            return True

        # Tertiary check: Essential trading parameters must be explicitly set
        # This allows bots to start if manually configured via ENV
        symbol = getattr(self.config, 'symbol', None)
        starting_cash = getattr(self.config, 'starting_cash', None)

        # Check if these are default values (indicating no configuration)
        has_symbol = symbol and symbol != 'BTC-USD'  # Default symbol
        has_cash = starting_cash and starting_cash != 1000.0  # Default cash

        if has_symbol or has_cash:
            print("✅ Configuration found via environment - proceeding with trading")
            return True

        print("❌ Configuration missing - entering waiting state")
        return False

    def _wait_for_configuration(self) -> None:
        """Wait for user configuration before starting trading."""
        print("⏳ WAITING FOR CONFIGURATION - Bot is ready but waiting for user settings from UI")
        print("💡 Trading will start automatically once configuration is received via settings API")
        print("🛑 Bot will not trade with default values until configured")

        # Report waiting status
        self._report_state("waiting_for_config", "Waiting for user configuration")

        # Create state directory
        os.makedirs("/app/state", exist_ok=True)

        # Wait indefinitely for configuration
        while not self._check_configuration_complete():
            # Check every 30 seconds
            time.sleep(30)

            if self._stop_requested:
                break

        if not self._stop_requested:
            print("🔥 Configuration received! Starting trading...")
            self._report_state("running", "Configuration received, starting trading")

    def run(self) -> None:
        """Run until max_cycles is reached (or indefinitely)."""
        print(">>>>>>>>> RUN IS STARTING")
        print("Starting HTTP servers...")

        if not self._http_server:
            self._http_server = BotHTTPServer(self, port=self.config.http_port)
            self._http_server.start()
            print(f"HTTP server started on port {self.config.http_port}")
            self.logger.info("HTTP endpoints available on port %s", self.config.http_port)

        if not self._control_server:
            self._control_server = BotControlServer(
                self,
                port=self.config.control_port,
                bot_secret=self.config.bot_secret,
            )
            self._control_server.start()
            print(f"Control server started on port {self.config.control_port}")
            self.logger.info("Control endpoints available on port %s", self.config.control_port)

        print("Bot servers are ready")

        # Check for configuration before starting trading
        if not self._check_configuration_complete():
            self._wait_for_configuration()

        if self._stop_requested:
            return

        print("Bot is now ready for trading")
        self._running = True
        self._report_state("running", "Bot loop started")
        print(">>>>>>>>> RUN COMPLETED")
        print()

        print(">>>>>>>>> ENTERING INFINITE LOOP")
        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"Loop cycle #{cycle_count} starting...")

                # Debug: Show database portfolio_quantity for easier debugging
                if self._db_client:
                    try:
                        db_portfolio_qty = self._db_client.get_portfolio_quantity()
                        self.logger.info(f"📊 DB portfolio_quantity: {db_portfolio_qty:.8f} | Memory portfolio: {self.portfolio.quantity:.8f}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to get DB portfolio_quantity: {e}")
                        self.logger.info(f"📊 DB connection failed | Memory portfolio: {self.portfolio.quantity:.8f}")
                else:
                    self.logger.info(f"📊 No DB client | Memory portfolio: {self.portfolio.quantity:.8f}")

                with self._lock:
                    snapshot = self.exchange.fetch_market_snapshot(
                        self.config.symbol,
                        limit=self.config.history,
                    )
                    
                    

                    # --- DEBUG ACTIVE STRATEGY PARAMS (SCALPING ONLY) ---
                    self.loop_counter = getattr(self, "loop_counter", 0) + 1

                    # Debug strategy detection first
                    strategy_name = getattr(self.config, 'strategy', 'unknown')
                    if self.loop_counter % 1 == 1:  # Show strategy detection every 20 loops
                        self.logger.info(f"[DEBUG] Strategy detection: config.strategy='{strategy_name}' | Bot strategy class: {getattr(self.strategy, '__class__', type('?', (), {})).__name__}")

                    # Check multiple ways to detect scalping
                    is_scalping = (
                        (hasattr(self.config, 'strategy') and 'scalping' in str(self.config.strategy).lower()) or
                        'scalping' in str(getattr(self.strategy, '__class__', type('?', (), {})).__name__).lower()
                    )

                    if is_scalping and self.loop_counter % 1 == 0:  # εμφάνιση κάθε loop για scalping
                        p = getattr(self.strategy, "__dict__", {})
                        self.logger.info(
                            f"[DEBUG] buy_thr={p.get('buy_threshold')} | "
                            f"RSI=[{p.get('rsi_min')},{p.get('rsi_max')}] RSI_thr={p.get('rsi_threshold')} | "
                            f"MA=[{p.get('short_ma_period')},{p.get('long_ma_period')}] | "
                            f"vol_conf={p.get('enable_volume_confirmation')} vol_thr={p.get('volume_threshold')} | "
                            f"scalp_target={p.get('scalp_target')} trade_amt={p.get('trade_amount')} | "
                            f"stop_loss={p.get('stop_loss')} trail={p.get('trailing_profit_threshold')} | "
                            f"strategy={getattr(self.strategy, '__class__', type('?', (), {})).__name__}"
                        )
                    # -----------------------------------                    
                    
                    
                    execution: Optional[TradeExecution] = None
                    if self._paused:
                        signal = Signal("hold", reason="paused")
                    else:
                        signal = self.strategy.generate_signal(snapshot, self.portfolio)

                        # Enhanced strategy signal logging
                        scalping_data = None
                        if hasattr(self.strategy, 'last_signal_data'):
                            scalping_data = self.strategy.last_signal_data

                        log_strategy_signal(
                            self.logger,
                            strategy_name=self.config.strategy,
                            signal_action=signal.action,
                            signal_reason=signal.reason,
                            market_price=snapshot.current_price,
                            symbol=snapshot.symbol,
                            scalping_data=scalping_data
                        )

                        execution = self._apply_signal(signal, snapshot.current_price, snapshot.symbol)
                        if execution:
                            self.strategy.on_trade(signal, execution.price, execution.size, execution.timestamp)
                            self._last_execution = execution

                    self._last_signal = signal
                    self._update_portfolio_metrics(snapshot)

                    if not self._paused:
                        if self.config.max_cycles is not None and self._cycle >= self.config.max_cycles:
                            self.logger.info("Reached max_cycles=%s", self.config.max_cycles)
                            break
                        self._cycle += 1

                    portfolio_value = (
                        self._last_portfolio_value
                        if self._last_portfolio_value is not None
                        else self.portfolio.value(snapshot.current_price)
                    )

                    # Enhanced bot status logging
                    log_bot_status(
                        self.logger,
                        status=self._current_state().upper(),
                        portfolio_cash=self.portfolio.cash,
                        portfolio_quantity=self.portfolio.quantity,
                        portfolio_value=portfolio_value,
                        symbol=self.config.symbol,
                        current_price=snapshot.current_price,
                        cycle=self._cycle,
                        bot_type=getattr(self.config, 'strategy', None)
                    )

                    # Performance logging every 10 cycles
                    if self._cycle % 10 == 0:
                        total_pnl = self._realized_pnl + self._unrealized_pnl
                        win_rate = self._calculate_win_rate()
                        log_performance_metrics(
                            self.performance_logger,
                            realized_pnl=self._realized_pnl,
                            unrealized_pnl=self._unrealized_pnl,
                            total_pnl=total_pnl,
                            win_rate=win_rate,
                            total_trades=len(self._trades),
                            avg_entry_price=self._avg_entry_price
                        )

                self._heartbeat()

                if self._stop_requested:
                    self.logger.info("Stop requested; exiting loop")
                    break

                if self._restart_requested:
                    self._perform_restart()

                print(f"Loop cycle #{cycle_count} completed")
                print()

                if self.config.sleep_seconds > 0:
                    print(f"Sleeping for {self.config.sleep_seconds} seconds...")
                    time.sleep(self.config.sleep_seconds)
                    print("Sleep completed, starting next cycle...")
                    print()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self._running = False
            self._report_state("stopped", "Bot loop stopped")
            if self._http_server:
                self._http_server.stop()
                self._http_server = None
            if self._control_server:
                self._control_server.stop()
                self._control_server = None
            if self._db_client:
                self._db_client.close()

    def _perform_restart(self) -> None:
        with self._lock:
            if not self._restart_requested:
                return
            self.logger.info("Restarting bot components")
            self._restart_requested = False
            self.portfolio = Portfolio(symbol=self.config.symbol, cash=self.config.starting_cash)
            self._cycle = 0
            self._last_price = None
            self._last_portfolio_value = None
            self._last_snapshot_at = None
            self._last_signal = None
            self._last_execution = None
            self._realized_pnl = 0.0
            self._unrealized_pnl = 0.0
            self._avg_entry_price = 0.0
            self._trades.clear()
            self._build_components()
        self._report_state("running", "Bot restarted")

    def _current_state(self) -> str:
        if self._stop_requested:
            return "stopping"
        if self._restart_requested:
            return "restarting"
        if not self._running:
            return "stopped"
        if self._paused:
            return "paused"
        return "running"

    def _heartbeat(self) -> None:
        if self._db_client:
            self._db_client.update_bot_status(self._current_state(), last_seen=datetime.utcnow())

    def _update_portfolio_metrics(self, snapshot) -> None:
        self._last_price = snapshot.current_price
        self._last_snapshot_at = snapshot.timestamp
        market_value = self.portfolio.quantity * snapshot.current_price
        self._last_portfolio_value = self.portfolio.cash + market_value
        if self.portfolio.quantity > 0:
            self._unrealized_pnl = (snapshot.current_price - self._avg_entry_price) * self.portfolio.quantity
        else:
            self._unrealized_pnl = 0.0

    def _report_state(self, status: str, details: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        self.logger.debug("Status update: %s %s", status, details)
        if self._status_broadcaster:
            self._status_broadcaster.send(status, details, extra or {})
        if self._db_client:
            self._db_client.update_bot_status(status, last_seen=datetime.utcnow())

    def _log_command(self, command: str, status: str, metadata: Optional[Dict[str, Any]]) -> None:
        payload = {
            "command": command,
            "status": status,
            "state": self._current_state(),
        }
        if metadata:
            payload["metadata"] = metadata
        self.logger.info("Command '%s' handled with status=%s", command, status)
        if self._db_client:
            self._db_client.log_event("INFO", f"command:{command}", metadata=payload)

    def handle_command(self, command: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"start", "stop", "pause", "resume", "restart"}
        if command not in allowed:
            return {
                "status": "error",
                "command": command,
                "message": f"Unknown command: {command}. Valid commands: {', '.join(sorted(allowed))}",
            }

        response: Dict[str, Any] = {
            "status": "ok",
            "command": command,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with self._lock:
            if command in {"start", "resume"}:
                if not self._paused:
                    response["message"] = "Bot already running"
                else:
                    self._paused = False
                    response["message"] = "Bot resumed"
                    self._report_state("running", "Resumed via command", {"source": "command"})
            elif command == "pause":
                if self._paused:
                    response["message"] = "Bot already paused"
                else:
                    self._paused = True
                    response["message"] = "Bot paused"
                    self._report_state("paused", "Paused via command", {"source": "command"})
            elif command == "stop":
                if self._stop_requested:
                    response["message"] = "Stop already requested"
                else:
                    self._stop_requested = True
                    response["message"] = "Stop requested"
                    self._report_state("stopping", "Stop command received", {"source": "command"})
            elif command == "restart":
                if self._restart_requested:
                    response["message"] = "Restart already scheduled"
                else:
                    self._restart_requested = True
                    response["message"] = "Restart scheduled"
                    self._report_state("restarting", "Restart command received", {"source": "command"})

            response["state"] = self._current_state()

        self._log_command(command, response["status"], metadata)
        return response

    def _apply_signal(self, signal: Signal, price: float, symbol: str):
        if signal.action == "hold" or signal.size <= 0:
            return None

        if signal.action == "buy":
            return self._handle_buy(signal, price, symbol)
        if signal.action == "sell":
            return self._handle_sell(signal, price, symbol)

        self.logger.warning("Unknown signal action '%s'", signal.action)
        return None

    def _handle_buy(self, signal: Signal, price: float, symbol: str):
        # Scalping bots use exact trade_amount, bypassing cash constraints
        if getattr(self.config, 'strategy', '').lower() in ['scalping', 'advanced_scalping']:
            size = signal.size
            if size <= 0:
                self.logger.debug("Skipping buy - invalid signal size")
                return None
        else:
            # Traditional cash-based logic for other strategies
            affordable_size = self.portfolio.cash / price if price > 0 else 0.0
            size = min(signal.size, affordable_size)
            if size <= 0:
                self.logger.debug("Skipping buy - insufficient cash")
                return None

        previous_quantity = self.portfolio.quantity
        execution = self.exchange.execute_trade(symbol, "buy", size, price)
        cost = execution.size * execution.price
        self.portfolio.cash -= cost
        self.portfolio.quantity += execution.size
        total_cost_before = self._avg_entry_price * previous_quantity
        new_quantity = self.portfolio.quantity
        if new_quantity > 0:
            self._avg_entry_price = (total_cost_before + cost) / new_quantity
        else:
            self._avg_entry_price = 0.0

        # Update database with new portfolio quantity
        if self._db_client:
            self._db_client.set_portfolio_quantity(self.portfolio.quantity)

        self._record_trade(execution, signal, realized_pnl=None)

        # Enhanced trade logging
        log_trade_execution(
            self.trade_logger,
            action="BUY",
            symbol=symbol,
            size=execution.size,
            price=execution.price,
            reason=signal.reason,
            portfolio_value=self.portfolio.value(price)
        )
        return execution

    def _handle_sell(self, signal: Signal, price: float, symbol: str):
        available = self.portfolio.quantity
        size = min(signal.size, available)
        if size <= 0:
            self.logger.debug("Skipping sell - no position")
            return None

        execution = self.exchange.execute_trade(symbol, "sell", size, price)
        proceeds = execution.size * execution.price
        cost_basis = self._avg_entry_price * execution.size
        realized = proceeds - cost_basis
        self._realized_pnl += realized
        self.portfolio.cash += proceeds
        self.portfolio.quantity -= execution.size
        if self.portfolio.quantity <= 0:
            self._avg_entry_price = 0.0

        # Update database with new portfolio quantity
        if self._db_client:
            self._db_client.set_portfolio_quantity(self.portfolio.quantity)

        self._record_trade(execution, signal, realized_pnl=realized)

        # Enhanced trade logging
        log_trade_execution(
            self.trade_logger,
            action="SELL",
            symbol=symbol,
            size=execution.size,
            price=execution.price,
            reason=signal.reason,
            portfolio_value=self.portfolio.value(price),
            pnl=realized
        )
        return execution

    def _record_trade(self, execution: TradeExecution, signal: Signal, realized_pnl: Optional[float]) -> None:
        trade = {
            "side": execution.side,
            "size": execution.size,
            "price": execution.price,
            "timestamp": execution.timestamp.isoformat(),
            "reason": signal.reason,
        }
        if realized_pnl is not None:
            trade["realized_pnl"] = realized_pnl
        self._trades.append(trade)
        if self._db_client:
            self._db_client.log_trade(
                side=execution.side,
                amount=execution.size,
                price=execution.price,
                profit=realized_pnl,
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                reasoning=signal.reason,
                strategy=getattr(self.config, 'strategy', 'unknown'),
                target_price=getattr(signal, 'target_price', None),
                stop_loss=getattr(signal, 'stop_loss', None),
                entry_price=getattr(signal, 'entry_price', None),
            )



    def apply_settings(self, updates: Dict[str, object]) -> None:
        with self._lock:
            if not updates:
                return

            # Create configuration received flag to signal waiting bot
            try:
                os.makedirs("/app/state", exist_ok=True)
                config_flag_file = "/app/state/config_received.flag"
                with open(config_flag_file, 'w') as f:
                    f.write(f"{datetime.now().isoformat()}\n")
                    f.write(f"Bot received configuration from UI\n")
                    f.write(f"Settings keys: {list(updates.keys())}\n")
                print(f"✅ Configuration flag created: {config_flag_file}")
            except Exception as e:
                print(f"⚠️ Could not create configuration flag: {e}")

            previous_exchange = self.config.exchange
            previous_strategy = self.config.strategy
            previous_symbol = self.config.symbol
            previous_port = self.config.http_port
            previous_control_port = self.config.control_port
            previous_secret = self.config.bot_secret
            previous_database_url = self.config.database_url
            previous_bot_id = self.config.bot_instance_id
            previous_base_url = self.config.base_url
            previous_user_id = self.config.user_id

            applied_keys = set()

            if "config" in updates and isinstance(updates["config"], dict):
                dashboard_config = updates["config"]
                universal_updates: Dict[str, Any] = {}

                def has_value(value: Any) -> bool:
                    return value not in (None, "", [])

                strategy_override = dashboard_config.get("botStrategy") or dashboard_config.get("strategy")
                if has_value(strategy_override):
                    universal_updates["strategy"] = str(strategy_override).lower()

                strategy_candidate = strategy_override or self.config.strategy
                strategy_key = str(strategy_candidate).lower() if strategy_candidate else ""
                env_vars: Dict[str, str] = {}

                if has_value(strategy_override) and strategy_key not in STRATEGY_MAPPINGS:
                    raise ValueError(f"Unsupported strategy: {strategy_key}")

                if strategy_key in STRATEGY_MAPPINGS:
                    validate_dashboard_settings(strategy_key, dashboard_config)
                    env_vars = map_dashboard_to_env_vars(strategy_key, dashboard_config)

                self._last_applied_env_vars = env_vars
                env_keys_display = "none" if not env_vars else ", ".join(sorted(env_vars))
                strategy_label = strategy_key or self.config.strategy
                self.logger.info(
                    "Validated dashboard settings for strategy=%s (env vars: %s)",
                    strategy_label,
                    env_keys_display,
                )

                if has_value(dashboard_config.get("cryptoSymbol")):
                    universal_updates["symbol"] = str(dashboard_config["cryptoSymbol"]).replace('/', '-')
                if has_value(dashboard_config.get("botSymbol")):
                    universal_updates["symbol"] = str(dashboard_config["botSymbol"]).replace('/', '-')

                for cash_key in ("tradeAmount", "botStartingCash"):
                    if has_value(dashboard_config.get(cash_key)):
                        coerced_cash = self._coerce_dashboard_value(dashboard_config[cash_key])
                        if isinstance(coerced_cash, (int, float)):
                            universal_updates["starting_cash"] = float(coerced_cash)

                if has_value(dashboard_config.get("botSleep")):
                    coerced_sleep = self._coerce_dashboard_value(dashboard_config["botSleep"])
                    if isinstance(coerced_sleep, (int, float)):
                        universal_updates["sleep_seconds"] = float(coerced_sleep)

                if has_value(dashboard_config.get("botExchange")):
                    universal_updates["exchange"] = str(dashboard_config["botExchange"])

                exchange_params = dict(self.config.exchange_params)
                if has_value(dashboard_config.get("coinbaseApiKey")):
                    exchange_params["api_key"] = str(dashboard_config["coinbaseApiKey"])
                if has_value(dashboard_config.get("coinbaseSecret")):
                    exchange_params["api_secret"] = str(dashboard_config["coinbaseSecret"])
                if exchange_params != self.config.exchange_params:
                    universal_updates["exchange_params"] = exchange_params

                strategy_params = dict(self.config.strategy_params)
                strategy_mappings = {
                    "rsiBuyThreshold": "rsi_buy_threshold",
                    "rsiSellThreshold": "rsi_sell_threshold",
                    "maxTradesPerHour": "max_trades_per_hour",
                    "maxTradesPerDay": "max_trades_per_day",
                    "maxHoldings": "max_holdings",
                    "swingWindow": "swing_window",
                    "swingDiffThreshold": "swing_diff_threshold",
                    "sellPercentage": "sell_percentage",
                    "trailingProfitThreshold": "trailing_profit_threshold",
                }
                int_pref_keys = {"swingWindow", "maxTradesPerHour", "maxTradesPerDay"}

                for dashboard_key, universal_key in strategy_mappings.items():
                    if not has_value(dashboard_config.get(dashboard_key)):
                        continue
                    prefer_int = dashboard_key in int_pref_keys
                    coerced_value = self._coerce_dashboard_value(dashboard_config[dashboard_key], prefer_int=prefer_int)
                    if coerced_value is None:
                        continue
                    if prefer_int and isinstance(coerced_value, float):
                        coerced_value = int(coerced_value)
                    strategy_params[universal_key] = coerced_value

                strategy_params = self._apply_strategy_specific_params(strategy_key, dashboard_config, strategy_params)

                if "checkInterval" in dashboard_config and has_value(dashboard_config["checkInterval"]):
                    interval_value = self._coerce_dashboard_value(dashboard_config["checkInterval"], prefer_int=True)
                    if isinstance(interval_value, (int, float)):
                        universal_updates["sleep_seconds"] = int(interval_value) * 60

                if strategy_params != self.config.strategy_params:
                    universal_updates["strategy_params"] = strategy_params
                    params_display = "none" if not strategy_params else ", ".join(sorted(strategy_params))
                    self.logger.info(
                        "Strategy parameters mapped for %s: %s",
                        strategy_label,
                        params_display,
                    )

                if "isEnabled" in dashboard_config:
                    enabled = bool(dashboard_config["isEnabled"])
                    if not enabled and not self._paused:
                        self._paused = True
                        self._report_state("paused", "Paused via settings")
                    elif enabled and self._paused:
                        self._paused = False
                        self._report_state("running", "Resumed via settings")

                self.config.update(universal_updates)
                applied_keys = set(universal_updates.keys())
            else:
                self.config.update(updates)
                applied_keys = set(updates.keys())
                self._last_applied_env_vars = {}

            if previous_port != self.config.http_port and self._http_server:
                self._http_server.stop()
                self._http_server = BotHTTPServer(self, port=self.config.http_port)
                self._http_server.start()
                self.logger.info("HTTP endpoints moved to port %s", self.config.http_port)

            if (previous_control_port != self.config.control_port or previous_secret != self.config.bot_secret) and self._control_server:
                self._control_server.stop()
                self._control_server = BotControlServer(
                    self,
                    port=self.config.control_port,
                    bot_secret=self.config.bot_secret,
                )
                self._control_server.start()
                self.logger.info("Control endpoints moved to port %s", self.config.control_port)

            if previous_exchange != self.config.exchange or previous_strategy != self.config.strategy or previous_symbol != self.config.symbol or "exchange_params" in applied_keys or "strategy_params" in applied_keys:
                self.logger.info(
                    "Rebuilding components (exchange: %s->%s, strategy: %s->%s, symbol: %s->%s)",
                    previous_exchange,
                    self.config.exchange,
                    previous_strategy,
                    self.config.strategy,
                    previous_symbol,
                    self.config.symbol,
                )
                self._build_components()

            if "starting_cash" in applied_keys:
                self.portfolio.cash = float(self.config.starting_cash)

            if (
                previous_database_url != self.config.database_url
                or previous_bot_id != self.config.bot_instance_id
                or previous_base_url != self.config.base_url
                or previous_secret != self.config.bot_secret
                or previous_user_id != self.config.user_id
            ):
                if self._db_client:
                    self._db_client.close()
                self._db_client = DatabaseClient(
                    database_url=self.config.database_url,
                    bot_instance_id=self.config.bot_instance_id,
                    logger=self.logger,
                )
                self._status_broadcaster = StatusBroadcaster(
                    base_url=self.config.base_url,
                    bot_instance_id=self.config.bot_instance_id,
                    bot_secret=self.config.bot_secret,
                    user_id=self.config.user_id,
                    logger=self.logger,
                )


    def _apply_strategy_specific_params(
        self,
        strategy: str,
        dashboard_config: Dict[str, Any],
        current_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = dict(current_params)

        def set_param(param_key: str, dashboard_key: str, *, prefer_int: bool = False) -> None:
            if dashboard_key not in dashboard_config:
                return
            value = dashboard_config[dashboard_key]
            if value in (None, ""):
                return
            coerced = self._coerce_dashboard_value(value, prefer_int=prefer_int)
            if coerced is None:
                return
            params[param_key] = coerced

        normalized = (strategy or "").lower()

        if normalized in {"grid", "advanced_grid"}:
            set_param("amount", "amount")
            set_param("grid_size", "gridSize")
            set_param("grid_count", "gridCount", prefer_int=True)
            set_param("max_orders", "maxOrders", prefer_int=True)
        elif normalized in {"dca", "advanced_dca"}:
            set_param("base_amount", "baseAmount")
            set_param("interval_minutes", "intervalMinutes", prefer_int=True)
        elif normalized in {"momentum", "advanced_momentum"}:
            set_param("base_amount", "baseAmount")
            set_param("momentum_threshold", "momentumThreshold")
            set_param("momentum_period", "momentumPeriod", prefer_int=True)
            set_param("volume_threshold", "volumeThreshold")
        elif normalized in {"scalping", "advanced_scalping"}:
            set_param("trade_amount", "tradeAmount")
            set_param("scalp_target", "scalpTarget")

            # Filter parameters (gates & scoring)
            set_param("buy_threshold", "buyThreshold")
            set_param("short_ma_period", "shortMaPeriod", prefer_int=True)
            set_param("long_ma_period", "longMaPeriod", prefer_int=True)
            set_param("rsi_threshold", "rsiThreshold")
            set_param("rsi_min", "rsiMin")
            set_param("rsi_max", "rsiMax")
            set_param("enable_volume_confirmation", "enableVolumeConfirmation")
            set_param("volume_threshold", "volumeThreshold")

        return params

    def _coerce_dashboard_value(self, value: Any, *, prefer_int: bool = False) -> Any:
        if isinstance(value, (int, float)):
            return int(value) if prefer_int else value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if prefer_int:
                try:
                    return int(candidate)
                except ValueError:
                    try:
                        return int(float(candidate))
                    except ValueError:
                        return candidate
            try:
                return int(candidate)
            except ValueError:
                try:
                    return float(candidate)
                except ValueError:
                    return candidate
        return value

    def get_settings(self) -> Dict[str, object]:
        with self._lock:
            data = self.config.to_dict()

        # Mask sensitive data
        if data.get("bot_secret"):
            data["bot_secret_set"] = True
            data["bot_secret"] = None
        if data.get("database_url"):
            data["database_url_set"] = True
            data["database_url"] = None

        # Add universal bot metadata
        data["available_strategies"] = available_strategies()
        data["available_exchanges"] = ExchangeRegistry.available()

        # Map universal-bot settings to dashboard-expected format for compatibility
        exchange_params = self.config.exchange_params or {}
        strategy_params = self.config.strategy_params or {}


        # Create dashboard-compatible settings mapping
        trade_amount_value = strategy_params.get("trade_amount", self.config.starting_cash)
        dashboard_settings = {
            # Core trading config (map universal -> dashboard format)
            "cryptoSymbol": self.config.symbol.replace('-', '/'),
            "botSymbol": self.config.symbol.replace('-', '/'),
            "tradeAmount": trade_amount_value,
            "botStartingCash": self.config.starting_cash,
            "botSleep": self.config.sleep_seconds,
            "botExchange": self.config.exchange,
            "botStrategy": self.config.strategy,

            # Exchange API (extract from exchange_params)
            "coinbaseApiKey": exchange_params.get("api_key", ""),
            "coinbaseSecret": exchange_params.get("api_secret", ""),

            # Strategy-specific params (map from strategy_params with defaults)
            "rsiBuyThreshold": strategy_params.get("rsi_buy_threshold", 30),
            "rsiSellThreshold": strategy_params.get("rsi_sell_threshold", 70),
            "maxTradesPerHour": strategy_params.get("max_trades_per_hour", 10),
            "maxTradesPerDay": strategy_params.get("max_trades_per_day", 24),
            "maxHoldings": strategy_params.get("max_holdings", self.config.starting_cash),
            "amount": strategy_params.get("amount"),
            "gridSize": strategy_params.get("grid_size"),
            "gridCount": strategy_params.get("grid_count"),
            "maxOrders": strategy_params.get("max_orders"),
            "baseAmount": strategy_params.get("base_amount"),
            "momentumThreshold": strategy_params.get("momentum_threshold"),
            "momentumPeriod": strategy_params.get("momentum_period"),
            "volumeThreshold": strategy_params.get("volume_threshold"),
            "scalpTarget": strategy_params.get("scalp_target"),

            # Technical analysis defaults
            "swingWindow": strategy_params.get("swing_window", 7),
            "swingDiffThreshold": strategy_params.get("swing_diff_threshold", 0.02),
            "sellPercentage": strategy_params.get("sell_percentage", 0.5),
            "trailingProfitThreshold": strategy_params.get("trailing_profit_threshold", 0.01),

            # Scheduling
            "isEnabled": not self._paused and self._running,
            "checkInterval": str(int(self.config.sleep_seconds / 60)) if self.config.sleep_seconds >= 60 else "1",

            # Applied env vars (debugging)
            "latestEnvVars": sorted(self._last_applied_env_vars.keys()),
        }


        # Add dashboard settings to response
        data["dashboardSettings"] = dashboard_settings

        return data

    def _format_signal(self, signal: Optional[Signal]) -> Optional[Dict[str, Any]]:
        if not signal:
            return None
        return {
            "action": signal.action,
            "size": signal.size,
            "reason": signal.reason,
        }

    def _format_execution(self, execution: Optional[TradeExecution]) -> Optional[Dict[str, Any]]:
        if not execution:
            return None
        return {
            "side": execution.side,
            "size": execution.size,
            "price": execution.price,
            "timestamp": execution.timestamp.isoformat(),
        }

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            latest_price = self._last_price
            portfolio_value = self._last_portfolio_value
            return {
                "running": self._running and not self._stop_requested,
                "state": self._current_state(),
                "cycle": self._cycle,
                "symbol": self.config.symbol,
                "cash": self.portfolio.cash,
                "quantity": self.portfolio.quantity,
                "latest_price": latest_price,
                "portfolio_value": portfolio_value,
                "realized_pnl": self._realized_pnl,
                "unrealized_pnl": self._unrealized_pnl,
                "avg_entry_price": self._avg_entry_price if self.portfolio.quantity > 0 else None,
                "paused": self._paused,
                "http_port": self.config.http_port,
                "control_port": self.config.control_port,
                "bot_instance_id": self.config.bot_instance_id,
                "user_id": self.config.user_id,
                "last_signal": self._format_signal(self._last_signal),
                "last_execution": self._format_execution(self._last_execution),
            }

    def get_performance(self) -> Dict[str, Any]:
        with self._lock:
            current_price = self._last_price or 0.0
            quantity = self.portfolio.quantity
            cash = self.portfolio.cash
            market_value = quantity * current_price
            portfolio_value = cash + market_value
            total_pnl = self._realized_pnl + self._unrealized_pnl

            # Format currency values (detect EUR/USD from symbol)
            symbol_parts = self.config.symbol.replace('-', '/').split('/')
            quote_currency = symbol_parts[1] if len(symbol_parts) > 1 else "EUR"
            locale = "en-US" if quote_currency == "USD" else "de-DE"

            def format_currency(value: float) -> str:
                from locale import localeconv
                try:
                    if quote_currency == "USD":
                        return f"${value:,.2f}"
                    else:
                        return f"€{value:,.2f}"
                except:
                    return f"{quote_currency}{value:.2f}"

            # Calculate time ago for last run
            def time_ago(dt: datetime) -> str:
                if not dt:
                    return "Never"
                now = datetime.utcnow()
                diff = now - dt

                if diff.days > 0:
                    return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
                elif diff.seconds >= 3600:
                    hours = diff.seconds // 3600
                    return f"{hours} hour{'s' if hours != 1 else ''} ago"
                elif diff.seconds >= 60:
                    minutes = diff.seconds // 60
                    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                else:
                    return f"{diff.seconds} second{'s' if diff.seconds != 1 else ''} ago"

            # Match swing-bot-template structure exactly
            return {
                "data": {
                    "timestamp": self._last_snapshot_at.strftime('%Y-%m-%d %H:%M:%S') if self._last_snapshot_at else datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    "bot_info": {
                        "name": self.config.strategy.upper(),
                        "full_name": f"{self.config.strategy.title()} Trading Bot",
                        "symbol": self.config.symbol.replace('-', '/'),
                        "currency": quote_currency,
                        "is_running": self._running and not self._paused,
                        "demo_mode": self.config.exchange == "paper"
                    },
                    "marketData": {
                        "current_price": round(current_price, 2),
                        "priceFormatted": format_currency(current_price)
                    },
                    "positions": {
                        "has_active_position": quantity > 0,
                        "total_position_size": round(quantity, 6),
                        "average_entry_price": round(self._avg_entry_price, 2) if quantity > 0 else 0.0,
                        "entryPriceFormatted": format_currency(self._avg_entry_price) if quantity > 0 else "N/A",
                        "total_orders": len(self._trades),
                        "max_orders": 100  # Default max for universal bot
                    },
                    "financial": {
                        "current_profit": round(total_pnl, 2),
                        "profitFormatted": format_currency(total_pnl),
                        "salesFormatted": format_currency(self._realized_pnl)
                    },
                    "unrealized_pnl": round(self._unrealized_pnl, 2),
                    "lastRun": {
                        "status": "success" if self._current_state() in ["running", "paused"] else "error",
                        "timestamp": self._last_snapshot_at.strftime('%Y-%m-%d %H:%M:%S') if self._last_snapshot_at else datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        "timeAgo": time_ago(self._last_snapshot_at),
                        "error": None
                    },
                    "currency": {
                        "display": quote_currency,
                        "userPreference": quote_currency
                    }
                },
                # Keep some original fields for backwards compatibility
                "botInstanceId": self.config.bot_instance_id,
                "riskLevel": "MEDIUM",  # Default risk level
                "maxDrawdown": 0.0,     # TODO: Calculate actual max drawdown
                "sharpeRatio": 0.0      # TODO: Calculate actual Sharpe ratio
            }

    def get_logs(self) -> Dict[str, Any]:
        """Return recent log messages for dashboard logs page"""
        with self._lock:
            # Collect recent log messages from various sources
            log_lines = []

            # Add startup information
            log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Bot started: {self.config.strategy} on {self.config.symbol}")

            # Add trading cycle logs
            # For DCA bots, show purchase count instead of misleading loop cycles
            if self.config.strategy == 'dca':
                # Get purchase count directly from database
                purchase_count = 0
                if self._db_client:
                    try:
                        purchase_count = self._db_client.get_buy_trades_count()
                    except:
                        purchase_count = len([t for t in self._trades if t.get('side') == 'buy'])
                else:
                    purchase_count = len([t for t in self._trades if t.get('side') == 'buy'])
                log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | DCA purchases made: {purchase_count}")
            else:
                log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Current cycle: {self._cycle}")
            log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Bot state: {self._current_state()}")

            # Add recent trade logs
            for trade in list(self._trades)[-10:]:  # Last 10 trades
                timestamp = trade.get('timestamp', datetime.utcnow().isoformat())
                side = trade.get('side', 'unknown')
                size = trade.get('size', 0)
                price = trade.get('price', 0)
                reason = trade.get('reason', '')

                log_lines.append(f"{timestamp} | INFO | {side.upper()} {size:.6f} {self.config.symbol.split('-')[0]} at {price:.2f} - {reason}")

                if 'realized_pnl' in trade:
                    pnl = trade['realized_pnl']
                    pnl_text = f"profit: €{pnl:.2f}" if pnl >= 0 else f"loss: €{pnl:.2f}"
                    log_lines.append(f"{timestamp} | INFO | Trade result: {pnl_text}")

            # Add portfolio status
            if self._last_price:
                if self.config.strategy == 'dca':
                    # For DCA bots, calculate directly from database
                    if self._db_client:
                        try:
                            total_invested = self._db_client.get_total_invested()
                            portfolio_value = self.portfolio.quantity * self._last_price
                            unrealized_profit = portfolio_value - total_invested

                            # Get currency from symbol (BTC-USD -> USD, BTC-EUR -> EUR)
                            currency_symbol = self._get_currency_symbol()

                            log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Portfolio: {currency_symbol}{portfolio_value:.2f} | Invested: {currency_symbol}{total_invested:.2f} | Unrealized: {currency_symbol}{unrealized_profit:.2f}")
                        except Exception as e:
                            print(f"❌ Error calculating portfolio data: {e}")
                            portfolio_value = self.portfolio.value(self._last_price)
                            currency_symbol = self._get_currency_symbol()
                            log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Portfolio: {currency_symbol}{portfolio_value:.2f}")
                    else:
                        portfolio_value = self.portfolio.value(self._last_price)
                        currency_symbol = self._get_currency_symbol()
                        log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Portfolio: {currency_symbol}{portfolio_value:.2f}")
                else:
                    portfolio_value = self.portfolio.value(self._last_price)
                    total_pnl = self._realized_pnl + self._unrealized_pnl
                    log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Portfolio value: €{portfolio_value:.2f}, P&L: €{total_pnl:.2f}")

            # Add current position info
            if self.portfolio.quantity > 0:
                # Calculate weighted average price from database trades
                avg_price = self._calculate_weighted_average_price()
                currency_symbol = self._get_currency_symbol()
                if avg_price > 0:
                    log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Active position: {self.portfolio.quantity:.6f} {self.config.symbol.split('-')[0]} at avg price {currency_symbol}{avg_price:.2f}")
                else:
                    log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Active position: {self.portfolio.quantity:.6f} {self.config.symbol.split('-')[0]} (calculating avg price...)")

            # Add any error conditions
            if self._current_state() == "error":
                log_lines.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | Bot encountered an error")

            # Join all log lines
            logs_text = "\n".join(log_lines)

            return {
                "logs": logs_text,
                "timestamp": datetime.utcnow().isoformat(),
                "lines_count": len(log_lines)
            }


    def _configure_logging(self) -> None:
        """Setup enhanced logging - actual configuration done in __init__."""
        pass  # Configuration now handled by setup_enhanced_logging

    def _calculate_weighted_average_price(self) -> float:
        """Calculate weighted average entry price directly from database."""
        if not self._db_client:
            return self._avg_entry_price

        try:
            return self._db_client.get_weighted_average_price()
        except Exception as e:
            print(f"❌ Error calculating weighted average price: {e}")
            return self._avg_entry_price

    def _get_currency_symbol(self) -> str:
        """Get currency symbol from trading pair (BTC-USD -> $, BTC-EUR -> €)."""
        if self._db_client:
            try:
                currency = self._db_client.get_currency_from_trades()
                if currency:
                    if 'USD' in currency.upper():
                        return '$'
                    elif 'EUR' in currency.upper():
                        return '€'
            except Exception as e:
                print(f"❌ Error getting currency from trades: {e}")

        # Fallback to config symbol
        symbol = getattr(self.config, 'symbol', 'BTC-USD')
        if 'USD' in symbol.upper():
            return '$'
        elif 'EUR' in symbol.upper():
            return '€'
        else:
            return '$'  # Default to USD

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from completed trades with realized P&L."""
        profitable_trades = 0
        total_completed_trades = 0

        for trade in self._trades:
            if 'realized_pnl' in trade:
                total_completed_trades += 1
                if trade['realized_pnl'] > 0:
                    profitable_trades += 1

        if total_completed_trades == 0:
            return 0.0

        return (profitable_trades / total_completed_trades) * 100


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    bot = UniversalBot(config_path)
    bot.run()


if __name__ == "__main__":
    main()
