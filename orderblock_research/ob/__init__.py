"""Order-block (ICT / Smart Money Concepts) detector, backtester and bias tests."""

from .detect import DetectConfig, atr, detect_order_blocks, pivot_points
from .backtest import TradeConfig, simulate
from .stats import summarize

__all__ = [
    "DetectConfig",
    "TradeConfig",
    "atr",
    "detect_order_blocks",
    "pivot_points",
    "simulate",
    "summarize",
]
