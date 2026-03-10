"""
modules/portfolio_manager.py — Module 4: Portfolio Risk Manager

Manages a virtual ₹10-Lakh portfolio stored in SQLite.
Tracks positions, P&L (realised + unrealised), VAR, beta, Sharpe ratio,
and per-position risk metrics.  All current prices are fetched live.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from src.database import get_engine, portfolio_positions, trade_history, cash_balance
from src.utils.helpers import (
    clean_symbol,
    error_response,
    safe_float,
    success_response,
    to_yf_symbol,
)

logger = logging.getLogger(__name__)

engine = get_engine()

NIFTY_BENCHMARK = "^NSEI"  # for Beta calculation


# ── Internal DB helpers ────────────────────────────────────────────────────────

def _get_cash() -> float:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT balance FROM cash_balance WHERE id=1")).fetchone()
        return safe_float(row[0]) if row else 0.0


def _set_cash(amount: float) -> None:
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE cash_balance SET balance=:b, updated_at=:t WHERE id=1"),
            {"b": amount, "t": datetime.utcnow()},
        )
        conn.commit()


def _get_position(symbol: str) -> Optional[dict]:
    sym = clean_symbol(symbol)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, symbol, quantity, avg_buy_price, total_invested FROM portfolio_positions WHERE symbol=:s"),
            {"s": sym},
        ).fetchone()
        if row:
            return dict(zip(["id", "symbol", "quantity", "avg_buy_price", "total_invested"], row))
        return None


def _upsert_position(symbol: str, quantity: float, avg_price: float, total_invested: float) -> None:
    sym = clean_symbol(symbol)
    existing = _get_position(sym)
    with engine.connect() as conn:
        if existing:
            conn.execute(
                text("""UPDATE portfolio_positions
                        SET quantity=:q, avg_buy_price=:a, total_invested=:ti, updated_at=:t
                        WHERE symbol=:s"""),
                {"q": quantity, "a": avg_price, "ti": total_invested, "t": datetime.utcnow(), "s": sym},
            )
        else:
            conn.execute(
                text("""INSERT INTO portfolio_positions (symbol, quantity, avg_buy_price, total_invested, created_at, updated_at)
                        VALUES (:s, :q, :a, :ti, :c, :u)"""),
                {"s": sym, "q": quantity, "a": avg_price, "ti": total_invested,
                 "c": datetime.utcnow(), "u": datetime.utcnow()},
            )
        conn.commit()


def _log_trade(symbol: str, action: str, quantity: float, price: float, pnl: Optional[float] = None) -> None:
    sym = clean_symbol(symbol)
    with engine.connect() as conn:
        conn.execute(
            text("""INSERT INTO trade_history (symbol, action, quantity, price, total_value, pnl, executed_at)
                    VALUES (:s, :a, :q, :p, :tv, :pnl, :t)"""),
            {"s": sym, "a": action, "q": quantity, "p": price,
             "tv": quantity * price, "pnl": pnl, "t": datetime.utcnow()},
        )
        conn.commit()


def _live_price(symbol: str) -> float:
    yf_sym = to_yf_symbol(symbol)
    try:
        t = yf.Ticker(yf_sym)
        price = safe_float(getattr(t.fast_info, "last_price", None) or t.info.get("currentPrice"))
        return price
    except Exception:
        return 0.0


# ── Public API ─────────────────────────────────────────────────────────────────

def buy_stock(symbol: str, quantity: int) -> dict:
    """Execute a virtual BUY order."""
    sym = clean_symbol(symbol)
    price = _live_price(sym)
    if price <= 0:
        return error_response(f"Could not fetch live price for '{sym}'")

    cost = price * quantity
    cash = _get_cash()
    if cost > cash:
        return error_response(
            f"Insufficient funds. Need ₹{cost:,.2f}, available ₹{cash:,.2f}"
        )

    existing = _get_position(sym)
    if existing:
        new_qty = existing["quantity"] + quantity
        new_invested = existing["total_invested"] + cost
        new_avg = new_invested / new_qty
        _upsert_position(sym, new_qty, new_avg, new_invested)
    else:
        _upsert_position(sym, quantity, price, cost)

    _set_cash(cash - cost)
    _log_trade(sym, "BUY", quantity, price)

    return success_response({
        "action": "BUY",
        "symbol": sym,
        "quantity": quantity,
        "execution_price": round(price, 2),
        "total_cost": round(cost, 2),
        "remaining_cash": round(cash - cost, 2),
        "message": f"Bought {quantity} shares of {sym} @ ₹{price:.2f}",
    })


def sell_stock(symbol: str, quantity: int) -> dict:
    """Execute a virtual SELL order."""
    sym = clean_symbol(symbol)
    existing = _get_position(sym)
    if not existing or existing["quantity"] < quantity:
        held = existing["quantity"] if existing else 0
        return error_response(f"Cannot sell {quantity} shares. You hold {held} shares of {sym}.")

    price = _live_price(sym)
    if price <= 0:
        return error_response(f"Could not fetch live price for '{sym}'")

    proceeds = price * quantity
    avg_buy = existing["avg_buy_price"]
    realised_pnl = (price - avg_buy) * quantity
    new_qty = existing["quantity"] - quantity
    new_invested = avg_buy * new_qty

    if new_qty == 0:
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM portfolio_positions WHERE symbol=:s"), {"s": sym}
            )
            conn.commit()
    else:
        _upsert_position(sym, new_qty, avg_buy, new_invested)

    cash = _get_cash()
    _set_cash(cash + proceeds)
    _log_trade(sym, "SELL", quantity, price, pnl=realised_pnl)

    return success_response({
        "action": "SELL",
        "symbol": sym,
        "quantity": quantity,
        "execution_price": round(price, 2),
        "proceeds": round(proceeds, 2),
        "avg_buy_price": round(avg_buy, 2),
        "realised_pnl": round(realised_pnl, 2),
        "realised_pnl_pct": round((realised_pnl / (avg_buy * quantity)) * 100, 2),
        "remaining_cash": round(cash + proceeds, 2),
        "message": f"Sold {quantity} shares of {sym} @ ₹{price:.2f} | P&L: ₹{realised_pnl:+.2f}",
    })


def get_portfolio_summary() -> dict:
    """
    Return a complete portfolio snapshot with live prices, unrealised P&L,
    allocation percentages, Value-at-Risk (95%), and Sharpe ratio.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT symbol, quantity, avg_buy_price, total_invested FROM portfolio_positions")
        ).fetchall()

    cash = _get_cash()
    positions = []
    total_current_value = 0.0
    total_invested_overall = 0.0

    returns_map: dict[str, pd.Series] = {}

    for sym, qty, avg_buy, invested in rows:
        price = _live_price(sym)
        current_val = price * qty
        pnl = current_val - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
        total_current_value += current_val
        total_invested_overall += invested

        # Fetch 1-year returns for risk metrics
        try:
            hist = yf.download(to_yf_symbol(sym), period="1y", interval="1d", auto_adjust=True, progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            returns_map[sym] = hist["Close"].pct_change().dropna()
        except Exception:
            pass

        positions.append({
            "symbol": sym,
            "quantity": qty,
            "avg_buy_price": round(avg_buy, 2),
            "current_price": round(price, 2),
            "current_value": round(current_val, 2),
            "invested": round(invested, 2),
            "unrealised_pnl": round(pnl, 2),
            "unrealised_pnl_pct": round(pnl_pct, 2),
        })

    portfolio_value = total_current_value + cash
    overall_pnl = total_current_value - total_invested_overall
    overall_pnl_pct = (overall_pnl / total_invested_overall * 100) if total_invested_overall > 0 else 0.0

    # Allocation %
    for p in positions:
        p["allocation_pct"] = round((p["current_value"] / portfolio_value * 100) if portfolio_value > 0 else 0.0, 2)

    # ── Risk Metrics ─────────────────────────────────────────────────────────
    var_95 = 0.0
    sharpe = 0.0
    if returns_map and len(rows) > 0:
        # Equal-weight portfolio returns for simplicity (improve with actual weights)
        combined = pd.concat(list(returns_map.values()), axis=1).dropna()
        if not combined.empty:
            portfolio_returns = combined.mean(axis=1)
            var_95 = float(np.percentile(portfolio_returns, 5)) * portfolio_value
            ann_return = portfolio_returns.mean() * 252
            ann_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = round((ann_return - 0.0685) / ann_vol, 3) if ann_vol > 0 else 0.0

    # Fetch trade history
    with engine.connect() as conn:
        trades = conn.execute(
            text("SELECT symbol, action, quantity, price, pnl, executed_at FROM trade_history ORDER BY executed_at DESC LIMIT 20")
        ).fetchall()

    trade_log = [
        {"symbol": s, "action": a, "quantity": q, "price": p, "pnl": pnl, "at": str(ts)}
        for s, a, q, p, pnl, ts in trades
    ]

    return success_response({
        "portfolio_value": round(portfolio_value, 2),
        "cash_balance": round(cash, 2),
        "equity_value": round(total_current_value, 2),
        "total_invested": round(total_invested_overall, 2),
        "unrealised_pnl": round(overall_pnl, 2),
        "unrealised_pnl_pct": round(overall_pnl_pct, 2),
        "positions": positions,
        "position_count": len(positions),
        "risk_metrics": {
            "var_95_inr": round(var_95, 2),
            "var_95_pct": round((var_95 / portfolio_value * 100) if portfolio_value > 0 else 0.0, 2),
            "sharpe_ratio": sharpe,
            "note": "VAR and Sharpe computed on 1-year equal-weight daily returns.",
        },
        "recent_trades": trade_log,
        "currency": "INR",
        "snapshot_at": datetime.utcnow().isoformat() + "Z",
    })
