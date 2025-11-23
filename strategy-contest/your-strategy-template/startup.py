#!/usr/bin/env python3
"""Startup script for the Adaptive Momentum bot template."""

from __future__ import annotations

import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "base-bot-template")
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/app/base"
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

import your_strategy  # noqa: F401  (registers adaptive_momentum)
from universal_bot import UniversalBot  # noqa: E402


def main() -> None:
    print("🚀 Launching Adaptive Momentum Bot...")
    bot = UniversalBot()
    bot.run()


if __name__ == "__main__":
    main()
