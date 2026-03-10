"""
tools/mcp_tools.py — Module 5: MCP Tools Layer

Registers all 10 MCP tools that Claude can invoke.  Each tool is a thin
adapter that validates inputs, delegates to the appropriate domain module,
and returns a structured JSON-serialisable response.

Tool roster
-----------
 1. get_live_quote          — Real-time stock price & fundamentals
 2. get_historical_data     — OHLCV bars for any period / interval
 3. get_market_overview     — Index dashboard (Nifty, Sensex, etc.)
 4. generate_trade_signal   — RSI + MACD + BB + EMA + sentiment signal
 5. analyze_options_chain   — Full options chain with BS Greeks
 6. get_portfolio_summary   — Virtual portfolio snapshot & risk metrics
 7. buy_stock               — Execute a virtual BUY order
 8. sell_stock              — Execute a virtual SELL order
 9. get_stock_news          — Latest news headlines for a ticker
10. get_technical_indicators — Detailed indicator report (RSI, MACD, BB, EMA, ADX)
"""

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.modules.market_data import (
    get_historical_data,
    get_live_quote,
    get_market_overview,
)
from src.modules.options_analyzer import analyze_options_chain
from src.modules.portfolio_manager import (
    buy_stock,
    get_portfolio_summary,
    sell_stock,
)
from src.modules.trade_signals import (
    _download_ohlcv,
    _fetch_news_sentiment,
    compute_adx,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_rsi,
    generate_trade_signal,
)
from src.utils.helpers import clean_symbol, error_response, get_env, safe_float

logger = logging.getLogger(__name__)

mcp = FastMCP("indian-stock-assistant")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Live Quote
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_live_quote_tool(symbol: str) -> str:
    """
    Fetch a real-time stock quote for an NSE/BSE listed company.

    Returns current price, change, volume, PE ratio, 52-week hi/lo,
    market-cap, sector, and company name.  Data is sourced live from
    Yahoo Finance.

    Parameters
    ----------
    symbol : NSE ticker symbol, e.g. 'RELIANCE', 'TCS', 'INFY', 'HDFC'
    """
    result = get_live_quote(symbol)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Historical Data
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_historical_data_tool(symbol: str, period: str = "3mo", interval: str = "1d") -> str:
    """
    Download OHLCV (Open, High, Low, Close, Volume) historical data.

    Parameters
    ----------
    symbol   : NSE ticker, e.g. 'WIPRO'
    period   : Duration — '1d','5d','1mo','3mo','6mo','1y','2y','5y','max'
    interval : Bar size — '1m','5m','15m','30m','1h','1d','1wk','1mo'
               (intraday intervals only work for period ≤ 60 days)

    Returns list of OHLCV bars + period return %.
    """
    result = get_historical_data(symbol, period, interval)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Market Overview
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_market_overview_tool() -> str:
    """
    Fetch current levels of major Indian market indices:
    Nifty 50, Bank Nifty, Nifty IT, Nifty Midcap 50, and BSE Sensex.

    Returns index level, absolute change, and percentage change for each.
    Useful for a quick morning market briefing.
    """
    result = get_market_overview()
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Trade Signal
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def generate_trade_signal_tool(symbol: str) -> str:
    """
    Generate an AI-powered BUY / SELL / HOLD trade signal for a stock.

    Combines four technical indicators (RSI-14, MACD-12/26/9,
    Bollinger Bands-20, EMA-20/50 crossover) with news sentiment
    from NewsAPI into a weighted composite score.

    Returns signal, confidence %, individual indicator readings, and
    recent relevant headlines.

    Parameters
    ----------
    symbol : NSE ticker, e.g. 'SBIN', 'TATAMOTORS', 'BAJFINANCE'

    Disclaimer: Educational purposes only. Not financial advice.
    """
    result = generate_trade_signal(symbol)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Options Chain
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_options_chain_tool(symbol: str, expiry_index: int = 0) -> str:
    """
    Fetch and analyse the live options chain for an NSE derivative.

    For each contract (calls & puts) computes:
    • Implied Volatility (IV)
    • Black-Scholes Greeks — Delta, Gamma, Theta (per day), Vega (per 1%), Rho
    • Open Interest (OI) and Volume

    Also derives:
    • Put-Call Ratio (PCR) with bullish/bearish interpretation
    • Max Pain strike price
    • Top 3 unusual OI activity contracts per side

    Parameters
    ----------
    symbol       : NSE derivative ticker — 'NIFTY', 'BANKNIFTY', 'RELIANCE', etc.
    expiry_index : 0 = nearest weekly/monthly expiry, 1 = next, 2 = far month, …

    Disclaimer: Options trading involves substantial risk. Educational only.
    """
    result = analyze_options_chain(symbol, expiry_index)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 — Portfolio Summary
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_portfolio_summary_tool() -> str:
    """
    Return a full snapshot of the virtual trading portfolio.

    Includes:
    • Per-position: quantity, avg buy price, live price, current value,
      unrealised P&L (INR & %), and allocation %
    • Portfolio totals: equity value, cash balance, total P&L
    • Risk metrics: 95% Value-at-Risk (VaR), Sharpe ratio
    • Last 20 trades

    Virtual portfolio starts with ₹10,00,000 (ten lakh) capital.
    All prices are fetched live from Yahoo Finance.
    """
    result = get_portfolio_summary()
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 7 — Buy Stock
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def buy_stock_tool(symbol: str, quantity: int) -> str:
    """
    Execute a virtual BUY order at the current live market price.

    Deducts the cost from the virtual cash balance (₹10L starting capital).
    Uses a weighted-average price if the stock is already held.

    Parameters
    ----------
    symbol   : NSE ticker, e.g. 'RELIANCE'
    quantity : Number of shares to purchase (must be ≥ 1)

    Returns execution price, total cost, and remaining cash.
    """
    if quantity < 1:
        return json.dumps(error_response("Quantity must be at least 1"), indent=2)
    result = buy_stock(symbol, quantity)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 8 — Sell Stock
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def sell_stock_tool(symbol: str, quantity: int) -> str:
    """
    Execute a virtual SELL order at the current live market price.

    Calculates and logs realised P&L.  Fails gracefully if you try to sell
    more shares than you hold.

    Parameters
    ----------
    symbol   : NSE ticker, e.g. 'INFY'
    quantity : Number of shares to sell (must be ≤ quantity held)

    Returns execution price, proceeds, realised P&L, and updated cash balance.
    """
    if quantity < 1:
        return json.dumps(error_response("Quantity must be at least 1"), indent=2)
    result = sell_stock(symbol, quantity)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 9 — Stock News
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_stock_news_tool(symbol: str, max_articles: int = 10) -> str:
    """
    Retrieve the latest news articles related to a stock from NewsAPI.

    Returns headlines, sources, publication timestamps, URLs, and an
    aggregate sentiment score (ranging from -1 = very negative to +1 = very positive).

    Parameters
    ----------
    symbol       : NSE ticker, e.g. 'ZOMATO'
    max_articles : Maximum headlines to return (1–20, default 10)

    Requires NEWS_API_KEY environment variable (free at newsapi.org).
    """
    import requests

    api_key = get_env("NEWS_API_KEY")
    if not api_key:
        return json.dumps(error_response(
            "NEWS_API_KEY not set. Get a free key at https://newsapi.org and add it to .env"
        ), indent=2)

    bare = clean_symbol(symbol)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{bare} NSE BSE India stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max(1, max_articles), 20),
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles_raw = resp.json().get("articles", [])
    except Exception as e:
        return json.dumps(error_response(f"NewsAPI request failed: {e}"), indent=2)

    positive_words = {
        "surge", "rally", "gain", "profit", "beat", "growth", "strong",
        "bullish", "outperform", "upgrade", "record", "buy", "soar", "jump",
    }
    negative_words = {
        "fall", "drop", "loss", "miss", "decline", "weak", "bearish",
        "downgrade", "sell", "crash", "plunge", "risk", "concern", "debt",
    }

    articles = []
    for art in articles_raw:
        title = (art.get("title") or "").lower()
        desc = (art.get("description") or "").lower()
        txt = f"{title} {desc}"
        pos = sum(1 for w in positive_words if w in txt)
        neg = sum(1 for w in negative_words if w in txt)
        total = pos + neg
        sent = round((pos - neg) / total, 2) if total > 0 else 0.0

        articles.append({
            "headline": art.get("title"),
            "source": art.get("source", {}).get("name"),
            "published_at": art.get("publishedAt"),
            "url": art.get("url"),
            "sentiment": sent,
        })

    sentiment_score, _ = _fetch_news_sentiment(symbol)

    return json.dumps({
        "status": "success",
        "data": {
            "symbol": bare,
            "article_count": len(articles),
            "aggregate_sentiment": sentiment_score,
            "sentiment_label": (
                "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
            ),
            "articles": articles,
        }
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 10 — Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_technical_indicators_tool(symbol: str, period: str = "6mo") -> str:
    """
    Return a comprehensive technical indicator report for a stock.

    Computes and returns latest values (plus 5-bar history) for:
    • RSI-14 — momentum oscillator
    • MACD (12,26,9) — trend & momentum
    • Bollinger Bands (20, 2σ) — volatility envelope
    • EMA-20 and EMA-50 — trend direction
    • ADX-14 — trend strength
    • Volume analysis — average and latest vs average

    Parameters
    ----------
    symbol : NSE ticker, e.g. 'HDFCBANK'
    period : Lookback for download — '3mo','6mo','1y' (default '6mo')
    """
    try:
        df = _download_ohlcv(symbol, period=period)
        if len(df) < 60:
            return json.dumps(error_response("Need at least 60 trading days of data"), indent=2)

        rsi = compute_rsi(df)
        macd_df = compute_macd(df)
        bb_df = compute_bollinger_bands(df)
        ema20, ema50 = compute_ema(df)
        adx_df = compute_adx(df)

        def tail(series, n=5):
            return [round(safe_float(v), 4) for v in series.dropna().tail(n).tolist()]

        # Identify column names
        macd_col = next((c for c in macd_df.columns if c.startswith("MACD_") and "h" not in c.lower() and "s" not in c.lower()), None)
        sig_col = next((c for c in macd_df.columns if "MACDs" in c), None)
        hist_col = next((c for c in macd_df.columns if "MACDh" in c), None)
        bb_u = next((c for c in bb_df.columns if "BBU" in c), None)
        bb_l = next((c for c in bb_df.columns if "BBL" in c), None)
        bb_m = next((c for c in bb_df.columns if "BBM" in c), None)
        bb_b = next((c for c in bb_df.columns if "BBB" in c), None)  # bandwidth
        adx_col = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
        dmp_col = next((c for c in adx_df.columns if "DMP" in c), None)
        dmn_col = next((c for c in adx_df.columns if "DMN" in c), None)

        # Volume
        vol_avg = float(df["Volume"].tail(20).mean())
        vol_latest = float(df["Volume"].iloc[-1])
        vol_ratio = round(vol_latest / vol_avg, 2) if vol_avg > 0 else 1.0

        return json.dumps({
            "status": "success",
            "data": {
                "symbol": clean_symbol(symbol),
                "period": period,
                "close_price": round(safe_float(df["Close"].iloc[-1]), 2),
                "rsi_14": {
                    "current": round(safe_float(rsi.iloc[-1]), 2),
                    "history_5d": tail(rsi),
                    "interpretation": (
                        "Oversold (<30)" if safe_float(rsi.iloc[-1]) < 30 else
                        "Overbought (>70)" if safe_float(rsi.iloc[-1]) > 70 else "Neutral"
                    ),
                },
                "macd_12_26_9": {
                    "macd": round(safe_float(macd_df[macd_col].iloc[-1]), 4) if macd_col else None,
                    "signal": round(safe_float(macd_df[sig_col].iloc[-1]), 4) if sig_col else None,
                    "histogram": round(safe_float(macd_df[hist_col].iloc[-1]), 4) if hist_col else None,
                    "history_5d": {
                        "macd": tail(macd_df[macd_col]) if macd_col else [],
                        "signal": tail(macd_df[sig_col]) if sig_col else [],
                        "histogram": tail(macd_df[hist_col]) if hist_col else [],
                    },
                },
                "bollinger_bands_20_2": {
                    "upper": round(safe_float(bb_df[bb_u].iloc[-1]), 2) if bb_u else None,
                    "middle": round(safe_float(bb_df[bb_m].iloc[-1]), 2) if bb_m else None,
                    "lower": round(safe_float(bb_df[bb_l].iloc[-1]), 2) if bb_l else None,
                    "bandwidth_pct": round(safe_float(bb_df[bb_b].iloc[-1]) * 100, 2) if bb_b else None,
                },
                "ema_crossover": {
                    "ema20": round(safe_float(ema20.iloc[-1]), 2),
                    "ema50": round(safe_float(ema50.iloc[-1]), 2),
                    "golden_cross": bool(safe_float(ema20.iloc[-1]) > safe_float(ema50.iloc[-1])),
                    "history_5d": {
                        "ema20": tail(ema20),
                        "ema50": tail(ema50),
                    },
                },
                "adx_14": {
                    "adx": round(safe_float(adx_df[adx_col].iloc[-1]), 2) if adx_col else None,
                    "di_plus": round(safe_float(adx_df[dmp_col].iloc[-1]), 2) if dmp_col else None,
                    "di_minus": round(safe_float(adx_df[dmn_col].iloc[-1]), 2) if dmn_col else None,
                    "trend_strength": (
                        "Very Strong (ADX>40)" if (adx_col and safe_float(adx_df[adx_col].iloc[-1]) > 40) else
                        "Strong (ADX>25)" if (adx_col and safe_float(adx_df[adx_col].iloc[-1]) > 25) else
                        "Weak/Ranging"
                    ),
                },
                "volume_analysis": {
                    "latest_volume": int(vol_latest),
                    "avg_20d_volume": int(vol_avg),
                    "volume_ratio": vol_ratio,
                    "interpretation": (
                        "High volume (breakout signal)" if vol_ratio > 1.5 else
                        "Low volume (weak conviction)" if vol_ratio < 0.5 else "Normal"
                    ),
                },
            }
        }, indent=2)

    except Exception as exc:
        logger.exception("Technical indicator error for %s", symbol)
        return json.dumps(error_response(f"Technical indicator error: {exc}"), indent=2)
