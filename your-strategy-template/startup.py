#!/usr/bin/env python3
"""Adaptive Momentum Bot entry point."""

from __future__ import annotations

import os
import sys

# Support both local development and Docker execution paths.
base_path = os.path.join(os.path.dirname(__file__), "..", "base-bot-template")
if not os.path.exists(base_path):
    base_path = "/app/base"

sys.path.insert(0, base_path)

# Register strategies on import.
import your_strategy  # noqa: F401

from universal_bot import UniversalBot  # noqa: E402


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    bot = UniversalBot(config_path)

    print(" Adaptive Momentum Bot")
    print(f" Bot ID: {bot.config.bot_instance_id}")
    print(f" User ID: {bot.config.user_id}")
    print(f" Strategy: {bot.config.strategy}")
    print(f" Symbol: {bot.config.symbol}")
    print(f" Exchange: {bot.config.exchange}")
    print(f" Starting Cash: ${bot.config.starting_cash}")
    available = "adaptive_momentum"
    print(f" Available strategy handles: {available}")
    print("-" * 60)

    bot.run()


if __name__ == "__main__":
    main()
