# 🇮🇳 Indian Stock Market AI Assistant — MCP Server

A **Model Context Protocol (MCP) server** that connects Claude to live Indian stock market data.  Claude can call 10 specialised tools covering real-time quotes, technical analysis, options pricing with Black-Scholes Greeks, virtual portfolio management, and news sentiment — all powered by free APIs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Claude (AI Agent)                           │
│                    (Claude Desktop / Claude API)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │  MCP stdio transport
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MCP Server  (server.py)                        │
│                   FastMCP — 10 registered tools                     │
├──────────────────┬──────────────────┬───────────────────────────────┤
│  Module 1        │  Module 2        │  Module 3                     │
│  Market Data     │  Trade Signal    │  Options Analyzer             │
│  Engine          │  Generator       │  (Black-Scholes Greeks)       │
├──────────────────┼──────────────────┼───────────────────────────────┤
│  Module 4        │  Module 5        │                               │
│  Portfolio       │  MCP Tools Layer │                               │
│  Risk Manager    │  (adapter)       │                               │
└──────────────────┴──────────────────┴───────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
   Yahoo Finance         NewsAPI              Alpha Vantage
   (yfinance)           (sentiment)          (supplemental)
         │
         ▼
    SQLite DB
  (portfolio state,
   trade history,
   price cache)
```

---

## 10 MCP Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | `get_live_quote_tool` | Real-time price, change, PE, market-cap, 52-week range |
| 2 | `get_historical_data_tool` | OHLCV bars for any period & interval |
| 3 | `get_market_overview_tool` | Nifty 50, Bank Nifty, Sensex, Nifty IT dashboard |
| 4 | `generate_trade_signal_tool` | BUY/SELL/HOLD with RSI+MACD+BB+EMA+sentiment |
| 5 | `analyze_options_chain_tool` | Full chain with BS Greeks, PCR, Max Pain, unusual OI |
| 6 | `get_portfolio_summary_tool` | Virtual portfolio snapshot, P&L, VaR, Sharpe ratio |
| 7 | `buy_stock_tool` | Execute virtual BUY at live market price |
| 8 | `sell_stock_tool` | Execute virtual SELL with realised P&L calculation |
| 9 | `get_stock_news_tool` | Latest headlines + aggregate sentiment score |
| 10 | `get_technical_indicators_tool` | RSI, MACD, BB, EMA, ADX full indicator report |

---

## Tech Stack

| Component | Library / Service |
|-----------|------------------|
| Language | Python 3.11+ |
| MCP framework | `mcp` (FastMCP) |
| Market data | `yfinance` (Yahoo Finance) |
| Technical indicators | `pandas-ta` |
| Options & Greeks | `scipy` (Black-Scholes) |
| News sentiment | NewsAPI (free tier) |
| Supplemental data | Alpha Vantage (free tier) |
| Database | SQLite via SQLAlchemy |
| HTTP client | `requests` / `aiohttp` |

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/NavyaAgarwal02/indian-stock-mcp.git
cd indian-stock-mcp
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your free API keys:

```env
NEWS_API_KEY=your_key_here          # https://newsapi.org  (free)
ALPHA_VANTAGE_KEY=your_key_here     # https://alphavantage.co (free)
```

> **yfinance** requires no API key — it accesses Yahoo Finance directly.

### 3. Run the server

```bash
python server.py
```

---

## Connecting to Claude Desktop

Add the following block to your Claude Desktop `claude_desktop_config.json`:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "indian-stock-assistant": {
      "command": "python",
      "args": ["/absolute/path/to/indian-stock-mcp/server.py"],
      "env": {
        "NEWS_API_KEY": "your_newsapi_key",
        "ALPHA_VANTAGE_KEY": "your_alpha_vantage_key"
      }
    }
  }
}
```

Restart Claude Desktop.  You will see the 🔨 tools icon confirming the server is connected.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Live quote accuracy (price > 0)
- Historical OHLCV structure validation
- Trade signal validity (BUY/SELL/HOLD enum)
- Black-Scholes put-call parity
- Greeks bounds (0 ≤ Δ ≤ 1 for calls)
- Portfolio buy/sell cycle
- Edge cases: invalid symbols, insufficient funds, overselling

---

## Example Claude Conversations

**Market overview:**
> "What are Indian markets doing today?"

**Trade analysis:**
> "Should I buy Reliance Industries? Give me a detailed technical analysis."

**Options:**
> "Show me the Nifty options chain for this week's expiry and highlight unusual activity."

**Portfolio:**
> "Buy 10 shares of HDFC Bank, then show me my portfolio P&L."

**News:**
> "What's the latest news on Zomato and what's the market sentiment?"

---

## Project Structure

```
indian-stock-mcp/
├── server.py                    # MCP server entry point
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── database.py              # SQLite schema & engine
│   ├── modules/
│   │   ├── market_data.py       # Module 1: Live quotes & history
│   │   ├── trade_signals.py     # Module 2: RSI/MACD/BB signals
│   │   ├── options_analyzer.py  # Module 3: Options + Black-Scholes
│   │   └── portfolio_manager.py # Module 4: Virtual portfolio
│   ├── tools/
│   │   └── mcp_tools.py         # Module 5: All 10 MCP tool definitions
│   └── utils/
│       └── helpers.py           # Shared utilities
└── tests/
    └── test_tools.py            # pytest smoke tests
```

---

## Design Decisions

### Why yfinance?
Yahoo Finance provides free, reliable real-time data for all NSE/BSE stocks.  Adding `.NS` suffix gives NSE quotes; `.BO` gives BSE quotes.  No API key required, making it the ideal primary data source.

### Why pandas-ta over TA-Lib?
`pandas-ta` is a pure-Python indicator library with zero C compilation requirements, making it trivially installable in any environment.  It covers all required indicators (RSI, MACD, Bollinger Bands, EMA, ADX) and returns pandas Series/DataFrames that compose naturally with yfinance output.

### Black-Scholes implementation
Rather than a third-party library, the Greeks are computed directly using the closed-form formulae with `scipy.stats.norm` for the standard normal CDF/PDF.  Implied volatility is solved via Newton-Raphson iteration (convergence within ~100 iterations for typical market conditions).

### Signal weighting
| Indicator | Weight |
|-----------|--------|
| RSI-14 | 25% |
| MACD (12,26,9) | 25% |
| Bollinger Bands (20) | 15% |
| EMA crossover (20/50) | 15% |
| News sentiment | 20% |

Scores are normalised to [-2, +2]; composite score ≥ 0.6 → BUY, ≤ -0.6 → SELL, else HOLD.

### Virtual portfolio
Starts with ₹10,00,000 (ten lakh) virtual capital.  All buy/sell operations use live Yahoo Finance prices.  Trade history, positions, and cash balance are persisted in SQLite so the portfolio survives server restarts.

---

## Disclaimer

This tool is built for **educational and research purposes only**.  Nothing produced by this MCP server constitutes financial advice.  Options trading and equity investment carry substantial risk.  Always consult a SEBI-registered financial advisor before making investment decisions.

---

## Free API Resources

- **Yahoo Finance** via yfinance: https://pypi.org/project/yfinance/
- **NewsAPI** (free tier — 100 req/day): https://newsapi.org/docs
- **Alpha Vantage** (free tier — 25 req/day): https://www.alphavantage.co/documentation/
