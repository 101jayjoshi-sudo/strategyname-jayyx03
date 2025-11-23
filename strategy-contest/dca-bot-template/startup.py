#!/usr/bin/env python3
"""DCA Bot Template - Dollar Cost Averaging Trading Bot with Dashboard Integration."""

from __future__ import annotations

import sys
import os

# Import base infrastructure from base-bot-template
# Handle both local development and Docker container paths
base_path = os.path.join(os.path.dirname(__file__), '..', 'base-bot-template')
if not os.path.exists(base_path):
    # In Docker container, base template is at /app/base/
    base_path = '/app/base'

sys.path.insert(0, base_path)

# Import DCA strategies
import dca_strategy  # This registers the DCA strategies

# Import base bot infrastructure
from universal_bot import UniversalBot


def main() -> None:
    """Main entry point for DCA Bot."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    bot = UniversalBot(config_path)

    # Print startup info with unique identifiers
    print(f"🤖 DCA Trading Bot")
    print(f"🆔 Bot ID: {bot.config.bot_instance_id}")
    print(f"👤 User ID: {bot.config.user_id}")
    print(f"📈 Strategy: {bot.config.strategy}")
    print(f"💰 Symbol: {bot.config.symbol}")
    print(f"🏦 Exchange: {bot.config.exchange}")
    print(f"💵 Starting Cash: ${bot.config.starting_cash}")
    print("🎯 Available strategies: DCA, Advanced DCA (Enterprise)")
    print("-" * 60)

    bot.run()


if __name__ == "__main__":
    main()