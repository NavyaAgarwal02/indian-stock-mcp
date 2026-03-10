"""
utils/helpers.py — Shared helper functions used across all modules.
"""

import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── NSE Symbol normalisation ──────────────────────────────────────────────────

def to_yf_symbol(symbol: str) -> str:
    """
    Convert a bare NSE ticker (e.g. 'RELIANCE') to a yfinance-compatible
    symbol (e.g. 'RELIANCE.NS').  BSE symbols can end with '.BO'.
    Already-qualified symbols are returned unchanged.
    """
    symbol = symbol.upper().strip()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def clean_symbol(symbol: str) -> str:
    """Return bare ticker without exchange suffix."""
    return symbol.upper().strip().replace(".NS", "").replace(".BO", "")


# ── Safe numeric extraction ───────────────────────────────────────────────────

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# ── Response builders ─────────────────────────────────────────────────────────

def success_response(data: dict) -> dict:
    return {"status": "success", "data": data}


def error_response(message: str) -> dict:
    logger.error("Tool error: %s", message)
    return {"status": "error", "message": message}


# ── Environment helpers ───────────────────────────────────────────────────────

def get_env(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    if not val:
        logger.warning("Environment variable '%s' is not set.", key)
    return val


# ── Indian market hours check ─────────────────────────────────────────────────

def is_market_open() -> bool:
    """
    Returns True if current IST time falls within NSE trading hours
    (Mon–Fri, 09:15–15:30 IST).
    """
    import pytz
    from datetime import datetime as dt

    ist = pytz.timezone("Asia/Kolkata")
    now = dt.now(ist)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close
