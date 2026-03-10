"""
modules/options_analyzer.py — Module 3: Options Chain Analyzer

Fetches live options chain data from Yahoo Finance (via yfinance) and
computes Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho) for
each contract.  Also highlights unusual OI / volume activity.
"""

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from src.utils.helpers import (
    clean_symbol,
    error_response,
    safe_float,
    success_response,
    to_yf_symbol,
)

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.0685  # ~RBI repo rate (approximate free-rate for India)


# ── Black-Scholes implementation ──────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Theoretical BS price for a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """
    Compute Delta, Gamma, Theta (per day), Vega (per 1% vol), Rho.
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    nd1_pdf = norm.pdf(d1)

    gamma = nd1_pdf / (S * sigma * sqrt_T)
    vega = S * nd1_pdf * sqrt_T / 100        # per 1% change in vol

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * nd1_pdf * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -S * nd1_pdf * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "rho": round(rho, 4),
    }


def _implied_volatility_newton(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str
) -> float:
    """Newton-Raphson IV solver. Returns annualised IV or 0 if it fails."""
    if market_price <= 0 or T <= 0:
        return 0.0
    sigma = 0.3  # initial guess
    for _ in range(100):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        d1, _ = _d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            break
        diff = price - market_price
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))
        if abs(diff) < 1e-5:
            break
    return round(sigma, 4)


# ── Options chain fetch ────────────────────────────────────────────────────────

def analyze_options_chain(symbol: str, expiry_index: int = 0) -> dict:
    """
    Fetch and enrich the full options chain for *symbol*.

    Parameters
    ----------
    symbol       : NSE ticker e.g. 'NIFTY' / 'RELIANCE'
    expiry_index : 0 = nearest expiry, 1 = next, etc.

    Returns
    -------
    dict with calls, puts, Greeks, IV, PCR, Max-Pain, unusual activity.
    """
    yf_sym = to_yf_symbol(symbol)
    ticker = yf.Ticker(yf_sym)

    try:
        expirations = ticker.options
        if not expirations:
            return error_response(f"No options data found for '{symbol}'. Ensure it trades derivatives on NSE.")

        expiry_index = max(0, min(expiry_index, len(expirations) - 1))
        expiry_date = expirations[expiry_index]

        # Spot price
        info = ticker.fast_info
        spot = safe_float(getattr(info, "last_price", None))
        if spot <= 0:
            spot = safe_float(ticker.info.get("currentPrice", 0))

        # Time to expiry in years
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        T = max((exp_dt - datetime.utcnow()).days / 365.0, 0.001)

        chain = ticker.option_chain(expiry_date)
        calls_df: pd.DataFrame = chain.calls.copy()
        puts_df: pd.DataFrame = chain.puts.copy()

        def _enrich(df: pd.DataFrame, opt_type: str) -> list[dict]:
            rows = []
            for _, row in df.iterrows():
                K = safe_float(row.get("strike"))
                mkt_price = safe_float(row.get("lastPrice"))
                iv_raw = safe_float(row.get("impliedVolatility"))

                # Use market IV if available, else compute
                sigma = iv_raw if iv_raw > 0.001 else _implied_volatility_newton(mkt_price, spot, K, T, RISK_FREE_RATE, opt_type)

                greeks = compute_greeks(spot, K, T, RISK_FREE_RATE, sigma, opt_type)

                rows.append({
                    "strike": K,
                    "last_price": mkt_price,
                    "bid": safe_float(row.get("bid")),
                    "ask": safe_float(row.get("ask")),
                    "volume": safe_float(row.get("volume")),
                    "open_interest": safe_float(row.get("openInterest")),
                    "implied_volatility": round(iv_raw * 100, 2) if iv_raw else round(sigma * 100, 2),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                    "greeks": greeks,
                })
            # Sort by strike
            rows.sort(key=lambda x: x["strike"])
            return rows

        calls = _enrich(calls_df, "call")
        puts = _enrich(puts_df, "put")

        # ── Max Pain ──────────────────────────────────────────────────────────
        all_strikes = sorted(set(
            [c["strike"] for c in calls] + [p["strike"] for p in puts]
        ))
        pain_map = {}
        for s in all_strikes:
            call_pain = sum(max(0, s - c["strike"]) * c["open_interest"] for c in calls)
            put_pain = sum(max(0, p["strike"] - s) * p["open_interest"] for p in puts)
            pain_map[s] = call_pain + put_pain
        max_pain_strike = min(pain_map, key=pain_map.get) if pain_map else spot

        # ── Put-Call Ratio (OI based) ─────────────────────────────────────────
        total_call_oi = sum(c["open_interest"] for c in calls)
        total_put_oi = sum(p["open_interest"] for p in puts)
        pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0.0

        # ── Unusual OI (top 3 by OI × volume) ────────────────────────────────
        def _top_unusual(contracts: list[dict], label: str) -> list[dict]:
            scored = sorted(
                contracts,
                key=lambda x: x["open_interest"] * (x["volume"] or 1),
                reverse=True,
            )[:3]
            return [{"type": label, **c} for c in scored]

        unusual = _top_unusual(calls, "CALL") + _top_unusual(puts, "PUT")

        # ATM IV (closest strike to spot)
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot)) if all_strikes else spot
        atm_call = next((c for c in calls if c["strike"] == atm_strike), None)
        atm_iv = atm_call["implied_volatility"] if atm_call else 0.0

        return success_response({
            "symbol": clean_symbol(symbol),
            "expiry_date": expiry_date,
            "days_to_expiry": round(T * 365),
            "spot_price": spot,
            "available_expiries": list(expirations),
            "atm_iv_pct": atm_iv,
            "put_call_ratio": pcr,
            "pcr_sentiment": (
                "Bearish" if pcr > 1.2 else "Bullish" if pcr < 0.8 else "Neutral"
            ),
            "max_pain_strike": max_pain_strike,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "calls": calls,
            "puts": puts,
            "unusual_activity": unusual,
            "disclaimer": "Options trading involves substantial risk. Educational only.",
        })

    except Exception as exc:
        logger.exception("Options chain analysis failed for %s", symbol)
        return error_response(f"Options analysis error for '{symbol}': {exc}")
