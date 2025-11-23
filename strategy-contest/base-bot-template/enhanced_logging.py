#!/usr/bin/env python3
"""Enhanced logging system with UTF-8 support and detailed formatting."""

import sys
import logging
from typing import Optional


class Utf8StreamHandler(logging.StreamHandler):
    """Custom StreamHandler με UTF-8 encoding για Windows compatibility."""

    def __init__(self, stream=None):
        super().__init__(stream)
        self.stream = stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream

            # Safe UTF-8 handling without reopening stdout
            if hasattr(stream, 'buffer') and hasattr(stream.buffer, 'write'):
                # Use buffer.write for binary UTF-8 output (safer than reopening)
                stream.buffer.write((msg + self.terminator).encode('utf-8', 'replace'))
                stream.buffer.flush()
            elif hasattr(stream, 'write'):
                # Fallback to normal write with error handling
                try:
                    stream.write(msg + self.terminator)
                    stream.flush()
                except UnicodeEncodeError:
                    # Replace problematic characters
                    safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                    stream.flush()
            else:
                # Last resort fallback to plain stdout
                print(msg)
        except Exception:
            self.handleError(record)


def setup_enhanced_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    detail_logging: bool = False,
    logger_name: Optional[str] = None,
    structured: bool = False
) -> logging.Logger:
    """
    Setup enhanced logging with UTF-8 support and configurable detail level.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path, defaults to /app/logs/trading.log if not specified
        detail_logging: If True, includes filename, function, and line number
        logger_name: Optional specific logger name, defaults to root logger
        structured: Future opt-in for structured logging (Phase 3 scaffolding only)

    Returns:
        Configured logger instance
    """
    # Determine logging format based on detailed logging setting
    if detail_logging:
        log_format = "%(asctime)s - %(levelname)s - [%(filename)s - %(funcName)s:%(lineno)d] - %(message)s"
    else:
        log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Setup default log file if none specified
    if log_file is None:
        log_file = "/app/logs/trading.log"

    # Phase 3 scaffolding: structured flag documented but not implemented yet
    if structured:
        # Future: implement structured/JSON logging format
        pass

    # Setup handlers
    handlers = [Utf8StreamHandler(sys.stdout)]

    if log_file:
        # Ensure directory exists for log file
        import os
        from logging.handlers import RotatingFileHandler

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Add rotating file handler with UTF-8 encoding
        # Max 10MB per file, keep 5 backup files
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handlers.append(file_handler)

    # Clear existing handlers if configuring root logger
    if not logger_name:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True
    )

    # Return specific logger or root logger
    return logging.getLogger(logger_name) if logger_name else logging.getLogger()


def get_trade_logger(logger_name: str = "trade") -> logging.Logger:
    """Get a specialized logger for trade-specific messages."""
    return logging.getLogger(logger_name)


def get_performance_logger(logger_name: str = "performance") -> logging.Logger:
    """Get a specialized logger for performance metrics."""
    return logging.getLogger(logger_name)


def get_currency_symbol(symbol: str) -> str:
    """Extract quote currency symbol from trading pair."""
    if '-' in symbol:
        quote_currency = symbol.split('-')[1]
        return '€' if quote_currency == 'EUR' else '$' if quote_currency == 'USD' else quote_currency
    return '$'  # Default fallback


def log_trade_execution(
    logger: logging.Logger,
    action: str,
    symbol: str,
    size: float,
    price: float,
    reason: str,
    portfolio_value: float,
    pnl: float = 0.0
) -> None:
    """
    Log trade execution with standardized format.

    Args:
        logger: Logger instance to use
        action: Trade action (BUY/SELL)
        symbol: Trading symbol
        size: Trade size
        price: Execution price
        reason: Strategy reason for trade
        portfolio_value: Current portfolio value
        pnl: Realized P&L if applicable
    """
    # Get the appropriate currency symbol from the trading pair
    curr_symbol = get_currency_symbol(symbol)

    pnl_str = f" | PnL: {curr_symbol}{pnl:.2f}" if pnl != 0.0 else ""
    # Calculate total cost/proceeds
    total_value = size * price
    cost_info = f"Total: {curr_symbol}{total_value:.2f}" if action == "BUY" else f"Proceeds: {curr_symbol}{total_value:.2f}"

    logger.info(
        f"TRADE: {action} {size:.6f} {symbol} @ {curr_symbol}{price:.2f} | "
        f"{cost_info} | Reason: {reason} | Portfolio: {curr_symbol}{portfolio_value:.2f}{pnl_str}"
    )


def log_strategy_signal(
    logger: logging.Logger,
    strategy_name: str,
    signal_action: str,
    signal_reason: str,
    market_price: float,
    technical_data: Optional[dict] = None,
    symbol: str = "BTC-USD",
    detailed: bool = False,
    scalping_data: Optional[dict] = None
) -> None:
    """
    Log strategy signal generation with technical indicators.

    Args:
        logger: Logger instance to use
        strategy_name: Name of the strategy
        signal_action: Signal action (BUY/SELL/HOLD)
        signal_reason: Strategy reasoning
        market_price: Current market price
        technical_data: Optional technical indicator values
        symbol: Trading pair for currency formatting
        detailed: If True, provides enhanced human-readable format (backwards compatible)
        scalping_data: Optional scalping score breakdown (score, reasoning, etc.)
    """
    # Get the appropriate currency symbol
    curr_symbol = get_currency_symbol(symbol)

    # Special enhanced format for scalping strategies
    if strategy_name.lower() in ['scalping', 'advanced_scalping'] and scalping_data:
        score = scalping_data.get('score', 0)
        reasoning = scalping_data.get('reasoning', [])

        # Build score breakdown
        score_parts = []
        for reason in reasoning:
            if "Uptrend" in reason and "(+1)" in reason:
                score_parts.append("trend:+1")
            elif "Downtrend" in reason and "(-1)" in reason:
                score_parts.append("trend:-1")
            elif "Oversold" in reason and "(+1)" in reason:
                score_parts.append("rsi:+1")
            elif "Overbought" in reason and "(-1)" in reason:
                score_parts.append("rsi:-1")
            # Add more indicators as needed

        # If no detailed breakdown available, show neutral indicators
        if not score_parts:
            score_parts = ["trend:0", "rsi:0"]

        score_breakdown = " • ".join(score_parts)

        logger.info(
            f"SIGNAL [{strategy_name}]: {signal_action.upper()} | total score: {score:+.1f} → {score_breakdown}"
        )

        # Add detailed explanation as secondary line if reasoning available
        if reasoning:
            # Clean up the reasoning text - remove redundant (+1)/(-1) since it's already in breakdown
            clean_reason = reasoning[0] if reasoning else 'No clear signal'
            # Remove the score indicators like "(+1)" since they're shown in breakdown
            import re
            clean_reason = re.sub(r'\s*\([+-]\d+\)', '', clean_reason)
            # Replace "Price(X.XX)" format with cleaner version
            clean_reason = re.sub(r'Price\(([0-9.]+)\)', rf'{curr_symbol}\1', clean_reason)

            logger.info(f"   ↳ {clean_reason}")

        return

    # Original logging format for non-scalping strategies
    tech_str = ""
    if technical_data:
        if detailed:
            # Enhanced precision formatting for detailed mode
            tech_parts = []
            for k, v in technical_data.items():
                if v is not None:
                    if isinstance(v, float):
                        if 'rsi' in k.lower():
                            tech_parts.append(f"{k}={v:.1f}")
                        elif 'pct' in k.lower() or 'momentum' in k.lower():
                            tech_parts.append(f"{k}={v:+.3f}%")  # Always show sign for momentum
                        else:
                            tech_parts.append(f"{k}={v:.2f}")
                    else:
                        tech_parts.append(f"{k}={v}")
            tech_str = f" | Tech: {', '.join(tech_parts)}" if tech_parts else ""
        else:
            # Original compact format (backwards compatible)
            tech_parts = [f"{k}={v}" for k, v in technical_data.items() if v is not None]
            tech_str = f" | Tech: {', '.join(tech_parts)}" if tech_parts else ""

    # Consistent action formatting in detailed mode
    action_display = signal_action.upper() if detailed else signal_action

    logger.info(
        f"SIGNAL [{strategy_name}]: {action_display} @ {curr_symbol}{market_price:.2f} | "
        f"Reason: {signal_reason}{tech_str}"
    )


def log_bot_status(
    logger: logging.Logger,
    status: str,
    portfolio_cash: float,
    portfolio_quantity: float,
    portfolio_value: float,
    symbol: str,
    current_price: float,
    cycle: int,
    bot_type: str = None
) -> None:
    """
    Log bot status with portfolio summary.

    Args:
        logger: Logger instance to use
        status: Bot status (RUNNING/PAUSED/STOPPED)
        portfolio_cash: Available cash
        portfolio_quantity: Current position size
        portfolio_value: Total portfolio value
        symbol: Trading symbol
        current_price: Current market price
        cycle: Current cycle number
        bot_type: Bot type (scalping bots hide misleading cash/value info)
    """
    # Get the appropriate currency symbol
    curr_symbol = get_currency_symbol(symbol)

    # For scalping bots, hide misleading cash/total value info
    if bot_type and bot_type.lower() in ['scalping', 'advanced_scalping']:
        logger.info(
            f"STATUS: {status} | Cycle #{cycle} | {symbol} @ {curr_symbol}{current_price:.2f} | "
            f"Position: {portfolio_quantity:.6f}"
        )
    else:
        logger.info(
            f"STATUS: {status} | Cycle #{cycle} | {symbol} @ {curr_symbol}{current_price:.2f} | "
            f"Cash: {curr_symbol}{portfolio_cash:.2f} | Position: {portfolio_quantity:.6f} | "
            f"Total Value: {curr_symbol}{portfolio_value:.2f}"
        )


def log_performance_metrics(
    logger: logging.Logger,
    realized_pnl: float,
    unrealized_pnl: float,
    total_pnl: float,
    win_rate: float,
    total_trades: int,
    avg_entry_price: float = 0.0,
    symbol: str = "BTC-USD"
) -> None:
    """
    Log performance metrics summary.

    Args:
        logger: Logger instance to use
        realized_pnl: Realized profit/loss
        unrealized_pnl: Unrealized profit/loss
        total_pnl: Total profit/loss
        win_rate: Win rate percentage
        total_trades: Total number of trades
        avg_entry_price: Average entry price
        symbol: Trading symbol for currency formatting (backwards compatible)
    """
    # Get the appropriate currency symbol from trading pair
    curr_symbol = get_currency_symbol(symbol)

    entry_str = f" | Avg Entry: {curr_symbol}{avg_entry_price:.2f}" if avg_entry_price > 0 else ""
    logger.info(
        f"PERFORMANCE: Realized PnL: {curr_symbol}{realized_pnl:.2f} | "
        f"Unrealized PnL: {curr_symbol}{unrealized_pnl:.2f} | Total PnL: {curr_symbol}{total_pnl:.2f} | "
        f"Win Rate: {win_rate:.1f}% | Trades: {total_trades}{entry_str}"
    )