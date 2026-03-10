"""
modules/trade_signals.py — Module 2: AI Trade Signal Generator

Computes RSI, MACD, Bollinger Bands, EMA crossovers, ADX and combines
them with news sentiment to produce a composite BUY / SELL / HOLD signal.
Uses pandas-ta for all indicator calculations.
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from src.utils.helpers import (
    clean_symbol,
    error_response,
    get_env,
    safe_float,
    success_response,
    to_yf_symbol,
)

logger = logging.getLogger(__name__)


# ── Indicator computation ──────────────────────────────────────────────────────

def _download_ohlcv(symbol: str, period: str = "6mo") -> pd.DataFrame:
    yf_sym = to_yf_symbol(symbol)
    df = yf.download(yf_sym, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(subset=["Close"], inplace=True)
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.rsi(df["Close"], length=period)


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    return ta.macd(df["Close"], fast=12, slow=26, signal=9)


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    return ta.bbands(df["Close"], length=period, std=std)


def compute_ema(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> tuple[pd.Series, pd.Series]:
    return ta.ema(df["Close"], length=fast), ta.ema(df["Close"], length=slow)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    return ta.adx(df["High"], df["Low"], df["Close"], length=period)


# ── Sentiment (NewsAPI) ───────────────────────────────────────────────────────

def _fetch_news_sentiment(symbol: str) -> tuple[float, list[str]]:
    """
    Pull recent headlines via NewsAPI and score sentiment using a simple
    keyword lexicon.  Returns (score [-1,+1], headline_list).
    """
    import requests

    api_key = get_env("NEWS_API_KEY")
    if not api_key:
        return 0.0, []

    bare = clean_symbol(symbol)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{bare} NSE OR BSE stock India",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception as e:
        logger.warning("NewsAPI error for %s: %s", symbol, e)
        return 0.0, []

    positive_words = {
        "surge", "rally", "gain", "profit", "beat", "growth", "strong",
        "bullish", "outperform", "upgrade", "record", "buy", "soar", "jump",
        "rise", "positive", "expansion", "revenue", "dividend",
    }
    negative_words = {
        "fall", "drop", "loss", "miss", "decline", "weak", "bearish",
        "downgrade", "sell", "crash", "plunge", "risk", "concern", "debt",
        "lawsuit", "fraud", "cut", "layoff", "recession", "default",
    }

    headlines = []
    scores = []
    for art in articles:
        title = (art.get("title") or "").lower()
        desc = (art.get("description") or "").lower()
        text = f"{title} {desc}"
        headlines.append(art.get("title", ""))

        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)
        total = pos + neg
        if total > 0:
            scores.append((pos - neg) / total)

    avg_score = float(np.mean(scores)) if scores else 0.0
    return round(avg_score, 3), headlines[:5]


# ── Signal logic ───────────────────────────────────────────────────────────────

def _score_rsi(rsi_val: float) -> tuple[int, str]:
    if rsi_val < 30:
        return 2, "Oversold → BUY"
    if rsi_val < 45:
        return 1, "Mildly oversold → Lean BUY"
    if rsi_val > 70:
        return -2, "Overbought → SELL"
    if rsi_val > 55:
        return -1, "Mildly overbought → Lean SELL"
    return 0, "Neutral"


def _score_macd(macd_df: pd.DataFrame) -> tuple[int, str]:
    macd_col = [c for c in macd_df.columns if c.startswith("MACD_") and "h" not in c.lower() and "s" not in c.lower()]
    sig_col = [c for c in macd_df.columns if "MACDs" in c]
    hist_col = [c for c in macd_df.columns if "MACDh" in c]

    if not (macd_col and sig_col and hist_col):
        return 0, "MACD data unavailable"

    macd_val = safe_float(macd_df[macd_col[0]].iloc[-1])
    sig_val = safe_float(macd_df[sig_col[0]].iloc[-1])
    hist_val = safe_float(macd_df[hist_col[0]].iloc[-1])
    prev_hist = safe_float(macd_df[hist_col[0]].iloc[-2]) if len(macd_df) > 1 else hist_val

    if macd_val > sig_val and hist_val > 0 and hist_val > prev_hist:
        return 2, "MACD bullish crossover & rising histogram → BUY"
    if macd_val > sig_val:
        return 1, "MACD above signal → Lean BUY"
    if macd_val < sig_val and hist_val < 0 and hist_val < prev_hist:
        return -2, "MACD bearish crossover & falling histogram → SELL"
    if macd_val < sig_val:
        return -1, "MACD below signal → Lean SELL"
    return 0, "MACD neutral"


def _score_bollinger(df: pd.DataFrame, bb_df: pd.DataFrame) -> tuple[int, str]:
    close = safe_float(df["Close"].iloc[-1])
    upper_col = [c for c in bb_df.columns if "BBU" in c]
    lower_col = [c for c in bb_df.columns if "BBL" in c]
    mid_col = [c for c in bb_df.columns if "BBM" in c]

    if not (upper_col and lower_col):
        return 0, "BB data unavailable"

    upper = safe_float(bb_df[upper_col[0]].iloc[-1])
    lower = safe_float(bb_df[lower_col[0]].iloc[-1])
    mid = safe_float(bb_df[mid_col[0]].iloc[-1]) if mid_col else (upper + lower) / 2

    if close <= lower:
        return 2, f"Price at lower BB band ({lower:.2f}) → BUY"
    if close >= upper:
        return -2, f"Price at upper BB band ({upper:.2f}) → SELL"
    if close < mid:
        return 1, "Price below BB midline → Lean BUY"
    return -1, "Price above BB midline → Lean SELL"


def _score_ema(ema_fast: pd.Series, ema_slow: pd.Series) -> tuple[int, str]:
    f = safe_float(ema_fast.iloc[-1])
    s = safe_float(ema_slow.iloc[-1])
    prev_f = safe_float(ema_fast.iloc[-2]) if len(ema_fast) > 1 else f
    prev_s = safe_float(ema_slow.iloc[-2]) if len(ema_slow) > 1 else s

    if f > s and prev_f <= prev_s:
        return 2, "Golden cross (EMA20 crossed above EMA50) → Strong BUY"
    if f < s and prev_f >= prev_s:
        return -2, "Death cross (EMA20 crossed below EMA50) → Strong SELL"
    if f > s:
        return 1, "EMA20 above EMA50 → Bullish"
    return -1, "EMA20 below EMA50 → Bearish"


def _score_sentiment(sent_score: float) -> tuple[int, str]:
    if sent_score >= 0.3:
        return 1, f"Positive news sentiment ({sent_score:.2f}) → Lean BUY"
    if sent_score <= -0.3:
        return -1, f"Negative news sentiment ({sent_score:.2f}) → Lean SELL"
    return 0, f"Neutral news sentiment ({sent_score:.2f})"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_trade_signal(symbol: str) -> dict:
    """
    Compute a composite trade signal for *symbol* using technical indicators
    (RSI, MACD, Bollinger Bands, EMA crossover) + news sentiment.

    Returns signal (BUY/SELL/HOLD), confidence score, and full breakdown.
    """
    try:
        df = _download_ohlcv(symbol, period="6mo")
        if len(df) < 60:
            return error_response(f"Insufficient historical data for '{symbol}' (need ≥ 60 bars)")

        # Compute indicators
        rsi = compute_rsi(df)
        macd_df = compute_macd(df)
        bb_df = compute_bollinger_bands(df)
        ema_fast, ema_slow = compute_ema(df)
        adx_df = compute_adx(df)

        # Individual scores
        rsi_score, rsi_reason = _score_rsi(safe_float(rsi.iloc[-1]))
        macd_score, macd_reason = _score_macd(macd_df)
        bb_score, bb_reason = _score_bollinger(df, bb_df)
        ema_score, ema_reason = _score_ema(ema_fast, ema_slow)

        # Sentiment
        sent_score, headlines = _fetch_news_sentiment(symbol)
        sent_signal, sent_reason = _score_sentiment(sent_score)

        # Weighted composite: technicals 80%, sentiment 20%
        raw = (rsi_score * 0.25 + macd_score * 0.25 + bb_score * 0.15 + ema_score * 0.15 + sent_signal * 0.20)
        # raw in range roughly [-2, +2]
        confidence = round(abs(raw) / 2.0 * 100, 1)

        if raw >= 0.6:
            signal = "BUY"
        elif raw <= -0.6:
            signal = "SELL"
        else:
            signal = "HOLD"

        # ADX for trend strength
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
        adx_val = safe_float(adx_df[adx_col[0]].iloc[-1]) if adx_col else 0.0
        trend_strength = "Strong" if adx_val > 25 else "Weak/Ranging"

        # BB values for context
        bb_upper_col = [c for c in bb_df.columns if "BBU" in c]
        bb_lower_col = [c for c in bb_df.columns if "BBL" in c]

        return success_response({
            "symbol": clean_symbol(symbol),
            "signal": signal,
            "confidence_pct": confidence,
            "composite_score": round(raw, 3),
            "current_price": round(safe_float(df["Close"].iloc[-1]), 2),
            "trend_strength": f"{trend_strength} (ADX={adx_val:.1f})",
            "indicators": {
                "rsi": {
                    "value": round(safe_float(rsi.iloc[-1]), 2),
                    "score": rsi_score,
                    "reason": rsi_reason,
                },
                "macd": {
                    "score": macd_score,
                    "reason": macd_reason,
                },
                "bollinger_bands": {
                    "upper": round(safe_float(bb_df[bb_upper_col[0]].iloc[-1]), 2) if bb_upper_col else None,
                    "lower": round(safe_float(bb_df[bb_lower_col[0]].iloc[-1]), 2) if bb_lower_col else None,
                    "score": bb_score,
                    "reason": bb_reason,
                },
                "ema_crossover": {
                    "ema20": round(safe_float(ema_fast.iloc[-1]), 2),
                    "ema50": round(safe_float(ema_slow.iloc[-1]), 2),
                    "score": ema_score,
                    "reason": ema_reason,
                },
            },
            "sentiment": {
                "score": sent_score,
                "signal": sent_signal,
                "reason": sent_reason,
                "recent_headlines": headlines,
            },
            "disclaimer": "Educational purposes only. Not financial advice.",
        })
    except Exception as exc:
        logger.exception("Signal generation failed for %s", symbol)
        return error_response(f"Signal generation error for '{symbol}': {exc}")
