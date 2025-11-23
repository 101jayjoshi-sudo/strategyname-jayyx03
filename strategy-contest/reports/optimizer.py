#!/usr/bin/env python3
"""Random search optimizer for Adaptive Momentum parameters."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

import pandas as pd

import backtest_runner

SYMBOLS = ["BTC-USD", "ETH-USD"]
START = "2024-01-01"
END = "2024-06-30"
INTERVAL = "1h"
STARTING_CASH = 10000.0
SAMPLES = 150
OUTPUT_PATH = Path(__file__).resolve().parent / "optimizer_results.json"

PARAM_GRID = {
    "fast_period": [8, 10, 12, 16, 20],
    "slow_period": [50, 80, 120, 180],
    "momentum_period": [24, 36, 48, 72],
    "atr_period": [14, 24, 36],
    "momentum_threshold": [0.5, 0.8, 1.0, 1.5],
    "min_slope_pct": [0.1, 0.2, 0.35],
    "exit_momentum_floor": [-2.0, -1.0, 0.0, 0.5],
    "stop_pct": [0.04, 0.06, 0.08, 0.1],
    "take_profit_pct": [0.05, 0.1, 0.15, 0.25],
    "reward_risk": [2.0, 2.5, 3.0, 4.0],
    "cooldown_hours": [4, 8, 12, 24],
    "min_hold_hours": [2, 6, 10, 14],
    "trend_down_confirm": [3, 5, 8, 12],
    "risk_per_trade": [0.04, 0.06, 0.08, 0.1],
    "trailing_atr_multiplier": [1.0, 1.5, 2.0],
    "stop_atr_multiplier": [1.0, 1.5, 2.5],
    "max_position_pct": [0.45, 0.5, 0.55],
    "breakout_lookback": [24, 48, 72, 96],
    "breakout_buffer_pct": [0.1, 0.25, 0.5, 1.0],
    "min_atr_pct": [0.002, 0.003, 0.004, 0.006],
    "max_layers": [1, 2, 3],
    "layer_step_pct": [0.01, 0.02, 0.03],
    "rsi_period": [8, 12, 14, 21],
    "rsi_entry": [52.0, 55.0, 58.0, 61.0],
    "rsi_exit": [70.0, 74.0, 78.0, 82.0],
    "rsi_reset": [3.0, 5.0, 7.0, 9.0],
}


def sample_params() -> Dict[str, float]:
    return {k: random.choice(v) for k, v in PARAM_GRID.items()}


def evaluate(params: Dict[str, float], data_cache: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    combined = []
    for symbol in SYMBOLS:
        df = data_cache[symbol]
        result = backtest_runner.run_strategy(symbol, df, params, starting_cash=STARTING_CASH)
        combined.append(result.return_pct)
    return {
        "params": params,
        "avg_return": mean(combined),
        "min_return": min(combined),
        "max_return": max(combined),
    }


def main() -> None:
    data_cache = {}
    for symbol in SYMBOLS:
        data_cache[symbol] = backtest_runner.fetch_data(symbol, START, END, INTERVAL, use_cache=True)

    best_record = None
    evaluations: List[Dict[str, float]] = []

    for _ in range(SAMPLES):
        params = sample_params()
        record = evaluate(params, data_cache)
        evaluations.append(record)
        if best_record is None or record["avg_return"] > best_record["avg_return"]:
            best_record = record
            print(f"New best avg return {record['avg_return']:.2f}% with params {record['params']}")

    OUTPUT_PATH.write_text(json.dumps({
        "best": best_record,
        "evaluations": evaluations,
    }, indent=2))
    if best_record:
        print("Best params:", json.dumps(best_record, indent=2))


if __name__ == "__main__":
    random.seed(42)
    main()
