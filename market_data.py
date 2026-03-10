"""
modules/market_data.py — Module 1: Market Data Engine

Fetches live stock prices, OHLCV history, and market summary for
Indian equities via yfinance (Yahoo Finance).  All data is real-time
and sourced from Yahoo Finance's free endpoints.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils.helpers import (
    clean_symbol,
    error_response,
    safe_float,
    safe_int,
    success_response,
    to_yf_symbol,
)

logger = logging.getLogger(__name__)

# Popular NSE indices available via yfinance
NIFTY_INDICES = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "NIFTYMIDCAP": "^NSEMDCP50",
    "SENSEX": "^BSESN",
}


def get_live_quote(symbol: str) -> dict:
    """
    Fetch real-time quote for a single NSE stock.

    Parameters
    ----------
    symbol : str  e.g. 'RELIANCE' or 'RELIANCE.NS'

    Returns
    -------
    dict with price, change, volume, market-cap, 52-week hi/lo, etc.
    """
    yf_sym = to_yf_symbol(symbol)
    ticker = yf.Ticker(yf_sym)

    try:
        info = ticker.info
        fast = ticker.fast_info

        # fast_info gives the freshest price
        price = safe_float(getattr(fast, "last_price", None) or info.get("currentPrice"))
        prev_close = safe_float(
            getattr(fast, "previous_close", None) or info.get("previousClose") or info.get("regularMarketPreviousClose")
        )
        change = round(price - prev_close, 2) if price and prev_close else 0.0
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

        return success_response({
            "symbol": clean_symbol(symbol),
            "exchange": "NSE",
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "open": safe_float(getattr(fast, "open", None) or info.get("open")),
            "high": safe_float(getattr(fast, "day_high", None) or info.get("dayHigh")),
            "low": safe_float(getattr(fast, "day_low", None) or info.get("dayLow")),
            "volume": safe_int(getattr(fast, "three_month_average_volume", None) or info.get("volume")),
            "market_cap": safe_float(info.get("marketCap")),
            "pe_ratio": safe_float(info.get("trailingPE")),
            "week_52_high": safe_float(info.get("fiftyTwoWeekHigh")),
            "week_52_low": safe_float(info.get("fiftyTwoWeekLow")),
            "company_name": info.get("longName", clean_symbol(symbol)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "INR"),
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as exc:
        logger.exception("Failed to fetch quote for %s", yf_sym)
        return error_response(f"Could not fetch quote for '{symbol}': {exc}")


def get_historical_data(symbol: str, period: str = "3mo", interval: str = "1d") -> dict:
    """
    Download OHLCV history for a symbol.

    Parameters
    ----------
    symbol   : NSE ticker, e.g. 'TCS'
    period   : yfinance period string — '1d','5d','1mo','3mo','6mo','1y','2y','5y','max'
    interval : bar size  — '1m','5m','15m','30m','1h','1d','1wk','1mo'

    Returns
    -------
    dict with list of OHLCV bars and basic stats.
    """
    yf_sym = to_yf_symbol(symbol)
    try:
        df: pd.DataFrame = yf.download(
            yf_sym,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return error_response(f"No historical data found for '{symbol}' (period={period}, interval={interval})")

        # Flatten MultiIndex columns if present (yfinance >=0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(subset=["Close"], inplace=True)

        records = []
        for ts, row in df.iterrows():
            records.append({
                "date": str(ts.date() if hasattr(ts, "date") else ts),
                "open": round(safe_float(row.get("Open")), 2),
                "high": round(safe_float(row.get("High")), 2),
                "low": round(safe_float(row.get("Low")), 2),
                "close": round(safe_float(row.get("Close")), 2),
                "volume": safe_int(row.get("Volume")),
            })

        closes = df["Close"].dropna()
        return success_response({
            "symbol": clean_symbol(symbol),
            "period": period,
            "interval": interval,
            "bars_count": len(records),
            "start_date": records[0]["date"] if records else None,
            "end_date": records[-1]["date"] if records else None,
            "latest_close": round(safe_float(closes.iloc[-1]), 2) if not closes.empty else None,
            "period_return_pct": round(
                ((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]) * 100, 2
            ) if len(closes) > 1 else 0.0,
            "ohlcv": records,
        })
    except Exception as exc:
        logger.exception("Historical data fetch failed for %s", yf_sym)
        return error_response(f"Historical data error for '{symbol}': {exc}")


def get_market_overview() -> dict:
    """
    Fetch current levels of major Indian indices (Nifty 50, Bank Nifty, Sensex, etc.)
    and return a consolidated market snapshot.
    """
    results = {}
    for name, yf_sym in NIFTY_INDICES.items():
        try:
            ticker = yf.Ticker(yf_sym)
            fast = ticker.fast_info
            info = ticker.info
            price = safe_float(getattr(fast, "last_price", None) or info.get("regularMarketPrice"))
            prev = safe_float(getattr(fast, "previous_close", None) or info.get("regularMarketPreviousClose"))
            change = round(price - prev, 2) if price and prev else 0.0
            results[name] = {
                "level": price,
                "change": change,
                "change_pct": round((change / prev) * 100, 2) if prev else 0.0,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return success_response({
        "indices": results,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    })


def get_multiple_quotes(symbols: list[str]) -> dict:
    """Batch fetch quotes for a list of symbols."""
    results = {}
    for sym in symbols:
        resp = get_live_quote(sym)
        results[clean_symbol(sym)] = resp.get("data") if resp["status"] == "success" else {"error": resp["message"]}
    return success_response({"quotes": results, "count": len(results)})
