"""
server.py — MCP Server entry point.

Run with:
    python server.py

Or via Claude Desktop config (see README).
"""

import logging
import os
import sys

# Load .env before any module import
from dotenv import load_dotenv
load_dotenv()

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.database import init_db
from src.tools.mcp_tools import mcp  # registers all 10 tools via decorators

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server")


def main() -> None:
    logger.info("Initialising SQLite database …")
    init_db()
    logger.info("Database ready.")
    logger.info("Starting Indian Stock Market MCP Server …")
    logger.info("10 tools registered and available to Claude.")
    # FastMCP.run() uses stdio transport by default — perfect for Claude Desktop
    mcp.run()


if __name__ == "__main__":
    main()
