"""
database.py — SQLite persistence layer using SQLAlchemy Core.
Stores portfolio positions, trade history, and cached price snapshots.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./indian_stock_mcp.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

metadata = MetaData()

# ── Table Definitions ─────────────────────────────────────────────────────────

portfolio_positions = Table(
    "portfolio_positions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False, index=True),
    Column("quantity", Float, nullable=False, default=0.0),
    Column("avg_buy_price", Float, nullable=False, default=0.0),
    Column("total_invested", Float, nullable=False, default=0.0),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

trade_history = Table(
    "trade_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False, index=True),
    Column("action", String(10), nullable=False),          # BUY / SELL
    Column("quantity", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("total_value", Float, nullable=False),
    Column("pnl", Float, nullable=True),                   # realized P&L on SELL
    Column("executed_at", DateTime, default=datetime.utcnow),
)

price_cache = Table(
    "price_cache",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False, index=True),
    Column("price", Float, nullable=False),
    Column("change_pct", Float, nullable=True),
    Column("volume", Float, nullable=True),
    Column("cached_at", DateTime, default=datetime.utcnow),
)

cash_balance = Table(
    "cash_balance",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("balance", Float, nullable=False, default=1_000_000.0),  # ₹10 Lakh virtual capital
    Column("updated_at", DateTime, default=datetime.utcnow),
)


def init_db() -> None:
    """Create all tables and seed initial cash balance if needed."""
    metadata.create_all(engine)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM cash_balance")).scalar()
        if row == 0:
            conn.execute(
                cash_balance.insert().values(id=1, balance=1_000_000.0, updated_at=datetime.utcnow())
            )
            conn.commit()


def get_engine():
    return engine
