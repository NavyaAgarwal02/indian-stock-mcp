"""
tests/test_tools.py — Smoke tests for all 10 MCP tools.

Run with:  pytest tests/ -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.database import init_db

# Initialise DB before any test
@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()


# ── Market Data ────────────────────────────────────────────────────────────────

class TestMarketData:
    def test_live_quote_returns_price(self):
        from src.modules.market_data import get_live_quote
        result = get_live_quote("RELIANCE")
        assert result["status"] == "success"
        assert result["data"]["price"] > 0, "Price should be positive"
        assert result["data"]["symbol"] == "RELIANCE"

    def test_historical_data_structure(self):
        from src.modules.market_data import get_historical_data
        result = get_historical_data("TCS", period="1mo", interval="1d")
        assert result["status"] == "success"
        ohlcv = result["data"]["ohlcv"]
        assert len(ohlcv) > 0
        assert all(k in ohlcv[0] for k in ["date", "open", "high", "low", "close", "volume"])

    def test_market_overview_has_nifty(self):
        from src.modules.market_data import get_market_overview
        result = get_market_overview()
        assert result["status"] == "success"
        assert "NIFTY50" in result["data"]["indices"]
        assert result["data"]["indices"]["NIFTY50"]["level"] > 0

    def test_invalid_symbol_returns_error(self):
        from src.modules.market_data import get_live_quote
        result = get_live_quote("FAKESYMBOL99999")
        # Should return error or zero price — not crash
        assert "status" in result


# ── Trade Signals ──────────────────────────────────────────────────────────────

class TestTradeSignals:
    def test_signal_is_valid(self):
        from src.modules.trade_signals import generate_trade_signal
        result = generate_trade_signal("INFY")
        assert result["status"] == "success"
        assert result["data"]["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["data"]["confidence_pct"] <= 100.0

    def test_signal_contains_indicators(self):
        from src.modules.trade_signals import generate_trade_signal
        result = generate_trade_signal("SBIN")
        assert result["status"] == "success"
        indicators = result["data"]["indicators"]
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_bands" in indicators
        assert "ema_crossover" in indicators


# ── Options Chain ──────────────────────────────────────────────────────────────

class TestOptionsChain:
    def test_options_chain_fields(self):
        from src.modules.options_analyzer import analyze_options_chain
        result = analyze_options_chain("NIFTY")
        if result["status"] == "error":
            pytest.skip("Options data unavailable (market closed or API limit)")
        assert "calls" in result["data"]
        assert "puts" in result["data"]
        assert "put_call_ratio" in result["data"]
        assert "max_pain_strike" in result["data"]

    def test_greeks_present(self):
        from src.modules.options_analyzer import analyze_options_chain
        result = analyze_options_chain("NIFTY")
        if result["status"] == "error":
            pytest.skip("Options data unavailable")
        if result["data"]["calls"]:
            greeks = result["data"]["calls"][0]["greeks"]
            assert "delta" in greeks
            assert "gamma" in greeks
            assert "theta" in greeks
            assert "vega" in greeks


# ── Portfolio Manager ──────────────────────────────────────────────────────────

class TestPortfolioManager:
    def test_initial_cash_balance(self):
        from src.modules.portfolio_manager import get_portfolio_summary
        result = get_portfolio_summary()
        assert result["status"] == "success"
        assert result["data"]["portfolio_value"] > 0

    def test_buy_and_sell_cycle(self):
        from src.modules.portfolio_manager import buy_stock, sell_stock, get_portfolio_summary

        before = get_portfolio_summary()["data"]["cash_balance"]

        buy_result = buy_stock("WIPRO", 1)
        assert buy_result["status"] == "success"
        assert buy_result["data"]["action"] == "BUY"

        sell_result = sell_stock("WIPRO", 1)
        assert sell_result["status"] == "success"
        assert sell_result["data"]["action"] == "SELL"

        after = get_portfolio_summary()["data"]["cash_balance"]
        # Cash should be roughly the same (small diff due to bid/ask spread)
        assert abs(after - before) < before * 0.05  # within 5%

    def test_insufficient_funds(self):
        from src.modules.portfolio_manager import buy_stock
        result = buy_stock("RELIANCE", 999_999)
        assert result["status"] == "error"
        assert "Insufficient" in result["message"]

    def test_sell_more_than_held(self):
        from src.modules.portfolio_manager import sell_stock
        result = sell_stock("TATAMOTORS", 999_999)
        assert result["status"] == "error"


# ── Black-Scholes ──────────────────────────────────────────────────────────────

class TestBlackScholes:
    def test_call_price_positive(self):
        from src.modules.options_analyzer import black_scholes_price
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.07, sigma=0.2, option_type="call")
        assert price > 0

    def test_put_call_parity(self):
        from src.modules.options_analyzer import black_scholes_price
        S, K, T, r, sigma = 18500, 18500, 0.083, 0.0685, 0.15
        call = black_scholes_price(S, K, T, r, sigma, "call")
        put = black_scholes_price(S, K, T, r, sigma, "put")
        # C - P = S - K*e^{-rT}
        import math
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1.0, f"Put-Call parity violated: {lhs:.4f} != {rhs:.4f}"

    def test_greeks_delta_bounds(self):
        from src.modules.options_analyzer import compute_greeks
        g = compute_greeks(S=100, K=100, T=0.5, r=0.07, sigma=0.2, option_type="call")
        assert 0 <= g["delta"] <= 1.0
        assert g["gamma"] >= 0
        assert g["vega"] >= 0
