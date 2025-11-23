#!/usr/bin/env python3
"""
Generated ENV Schema for Python Bots
DO NOT EDIT MANUALLY - Generated from env-registry/registry.yml
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Schema version
SCHEMA_VERSION = "1.0.0"

# Strategy names
STRATEGY_NAMES = ["scalping", "dca", "momentum", "grid", "swing"]

class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, field: str, message: str, code: str = "VALIDATION_ERROR"):
        self.field = field
        self.message = message
        self.code = code
        super().__init__(f"{field}: {message}")

def validate_dashboard_settings(strategy: str, dashboard_settings: Dict[str, Any]) -> bool:
    """Validate incoming dashboard settings based on strategy requirements."""

    if strategy not in STRATEGY_NAMES:
        raise ValidationError("strategy", f"Unsupported strategy: {strategy}", "UNKNOWN_STRATEGY")

    # Get required fields
    required_fields = get_strategy_required_fields(strategy)

    # Check for missing fields
    missing = [field for field in required_fields if field not in dashboard_settings or dashboard_settings[field] is None]

    if missing:
        raise ValidationError("validation", f"Missing required fields for {strategy}: {', '.join(missing)}", "MISSING_REQUIRED")

    # Basic type validation
    for field, value in dashboard_settings.items():
        if field.endswith('Amount') or field.endswith('Target') or field.endswith('Threshold'):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValidationError(field, f"{field} must be a positive number", "INVALID_TYPE")

    return True

def map_dashboard_to_env_vars(strategy: str, dashboard_settings: Dict[str, Any]) -> Dict[str, str]:
    """Convert dashboard settings to environment variables."""

    # Validate first
    validate_dashboard_settings(strategy, dashboard_settings)

    env_vars: Dict[str, str] = {}

    # Common field mappings
    field_mappings = {
        'botSymbol': 'BOT_SYMBOL',
        'botExchange': 'BOT_EXCHANGE',
        'botSleep': 'BOT_SLEEP',
        'botStartingCash': 'BOT_STARTING_CASH',
        'tradeAmount': 'TRADE_AMOUNT',
        'scalpTarget': 'SCALP_TARGET',
        'baseAmount': 'BASE_AMOUNT',
        'intervalMinutes': 'INTERVAL_MINUTES',
        'gridSize': 'GRID_SIZE',
        'gridCount': 'GRID_COUNT',
        'maxOrders': 'MAX_ORDERS',
        'amount': 'AMOUNT',
        'momentumThreshold': 'MOMENTUM_THRESHOLD',
        'momentumPeriod': 'MOMENTUM_PERIOD',
        'volumeThreshold': 'VOLUME_THRESHOLD',
        'tradingIntervalMinutes': 'TRADING_INTERVAL_MINUTES',
        # Scalping filter parameters
        'buyThreshold': 'BUY_THRESHOLD',
        'shortMaPeriod': 'SHORT_MA_PERIOD',
        'longMaPeriod': 'LONG_MA_PERIOD',
        'rsiThreshold': 'RSI_THRESHOLD',
        'rsiMin': 'RSI_MIN',
        'rsiMax': 'RSI_MAX',
        'enableVolumeConfirmation': 'ENABLE_VOLUME_CONFIRMATION'
    }

    # Map dashboard fields to ENV vars
    for dashboard_key, env_key in field_mappings.items():
        value = dashboard_settings.get(dashboard_key)
        if value is not None:
            env_vars[env_key] = str(value)

    # Add strategy
    env_vars['BOT_STRATEGY'] = strategy

    # Add API keys if provided (secrets)
    if dashboard_settings.get('coinbaseApiKey'):
        env_vars['COINBASE_API_KEY'] = str(dashboard_settings['coinbaseApiKey'])
    if dashboard_settings.get('coinbaseSecret'):
        env_vars['COINBASE_SECRET'] = str(dashboard_settings['coinbaseSecret'])

    return env_vars

def apply_settings_with_scope_check(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply settings with scope validation."""

    runtime_settings = {}
    secrets_found = []

    # Secret keys (should not be in runtime settings)
    secret_keys = {'COINBASE_API_KEY', 'COINBASE_SECRET', 'BOT_SECRET'}

    for key, value in settings.items():
        if key in secret_keys:
            secrets_found.append(key)
            continue
        else:
            runtime_settings[key] = value

    if secrets_found:
        print(f"🔒 Secret keys filtered from settings: {', '.join(secrets_found)}")

    print(f"✅ Applied {len(runtime_settings)} runtime settings")
    return runtime_settings

def get_strategy_required_fields(strategy: str) -> List[str]:
    """Get required fields for a strategy."""

    strategy_requirements = {
        "scalping": ["botSymbol", "tradeAmount", "scalpTarget"],
        "dca": ["botSymbol", "botStartingCash", "baseAmount", "intervalMinutes"],
        "momentum": ["botSymbol", "botStartingCash", "baseAmount"],
        "grid": ["botSymbol", "amount", "gridSize", "gridCount", "maxOrders"],
        "swing": ["botSymbol", "botStartingCash", "baseAmount"]
    }

    return strategy_requirements.get(strategy, [])

# Legacy compatibility - replaces settings_mapping.py
STRATEGY_MAPPINGS = {
    "scalping": {
        "required_fields": ["botSymbol", "tradeAmount", "scalpTarget"],
        "env_mapping": {
            "BOT_SYMBOL": "botSymbol",
            "TRADE_AMOUNT": "tradeAmount",
            "SCALP_TARGET": "scalpTarget",
            "BOT_SLEEP": "botSleep",
            "BOT_EXCHANGE": "botExchange",
        },
    },
    "dca": {
        "required_fields": ["botSymbol", "botStartingCash", "baseAmount", "intervalMinutes"],
        "env_mapping": {
            "BOT_SYMBOL": "botSymbol",
            "BOT_STARTING_CASH": "botStartingCash",
            "BASE_AMOUNT": "baseAmount",
            "INTERVAL_MINUTES": "intervalMinutes",
            "BOT_SLEEP": "botSleep",
            "BOT_EXCHANGE": "botExchange",
        },
    },
    "momentum": {
        "required_fields": ["botSymbol", "botStartingCash", "baseAmount"],
        "env_mapping": {
            "BOT_SYMBOL": "botSymbol",
            "BOT_STARTING_CASH": "botStartingCash",
            "BASE_AMOUNT": "baseAmount",
            "MOMENTUM_THRESHOLD": "momentumThreshold",
            "MOMENTUM_PERIOD": "momentumPeriod",
            "VOLUME_THRESHOLD": "volumeThreshold",
            "BOT_SLEEP": "botSleep",
            "BOT_EXCHANGE": "botExchange",
        },
    },
    "grid": {
        "required_fields": ["botSymbol", "amount", "gridSize", "gridCount", "maxOrders"],
        "env_mapping": {
            "BOT_SYMBOL": "botSymbol",
            "AMOUNT": "amount",
            "GRID_SIZE": "gridSize",
            "GRID_COUNT": "gridCount",
            "MAX_ORDERS": "maxOrders",
            "BOT_SLEEP": "botSleep",
            "BOT_EXCHANGE": "botExchange",
        },
    },
    "swing": {
        "required_fields": ["botSymbol", "botStartingCash", "baseAmount"],
        "env_mapping": {
            "BOT_SYMBOL": "botSymbol",
            "BOT_STARTING_CASH": "botStartingCash",
            "BASE_AMOUNT": "baseAmount",
            "TRADING_INTERVAL_MINUTES": "tradingIntervalMinutes",
            "BOT_SLEEP": "botSleep",
            "BOT_EXCHANGE": "botExchange",
        },
    },
}
