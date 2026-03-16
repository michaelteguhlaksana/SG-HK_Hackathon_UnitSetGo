'''
Classes for managing the interface to the database
Supports Orders, Real-time Tickers, and Historical OHLCV Candles.

@ MTL 16 March 2026
'''
import aiosqlite
import time
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger("BotDB")

class DatabaseManager:
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path

    async def init_db(self):
        """Initializes tables, indexes, and enables WAL mode for concurrency."""
        async with aiosqlite.connect(self.db_path) as db:
            # 1. Performance Optimizations
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            # Allocate ~128MB of RAM for SQLite cache to speed up historical queries
            await db.execute("PRAGMA cache_size = -128000;") 
            
            # 2. Create Orders Table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    pair TEXT,
                    side TEXT,
                    type TEXT,
                    quantity REAL,
                    price REAL,
                    status TEXT,
                    timestamp INTEGER
                )
            """)
            
            # 3. Create Tickers Table (Latest Price Only)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    pair TEXT PRIMARY KEY,
                    last_price REAL,
                    timestamp INTEGER
                )
            """)

            # 4. Create Candles Table (Historical OHLCV)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    pair TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (pair, timestamp)
                )
            """)
            
            # 5. Index for fast strategy lookback (Descending time)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_candle_lookup ON candles (pair, timestamp DESC);")
            
            await db.commit()
            logger.info("Database initialized: WAL mode enabled, Indexes created.")

    # --- Order Methods ---
    async def save_order(self, order: Dict[str, Any]):
        """Saves or updates an order. Handles the Roostoo response structure."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO orders (order_id, pair, side, type, quantity, price, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET status=excluded.status
            """, (
                str(order['OrderID']), order['Pair'], order['Side'], 
                order['Type'], order['Quantity'], order['Price'], 
                order['Status'], order['CreateTimestamp']
            ))
            await db.commit()

    # --- Price Methods (Real-time Feed) ---
    async def update_ticker(self, pair: str, price: float):
        """Updates the latest price for a pair (used for quick RAM-like access)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO tickers (pair, last_price, timestamp)
                VALUES (?, ?, ?)
                ON CONFLICT(pair) DO UPDATE SET 
                    last_price=excluded.last_price,
                    timestamp=excluded.timestamp
            """, (pair, price, int(time.time() * 1000)))
            await db.commit()

    async def get_latest_price(self, pair: str) -> float:
        """Fetch the single most recent price for a pair."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT last_price FROM tickers WHERE pair = ?", (pair,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0.0

    # --- Historical Methods (Strategy Lookback) ---
    async def save_candle(self, pair: str, candle_data: Dict[str, Any]):
        """
        Saves a 1-minute OHLCV candle. 
        candle_data should have keys: timestamp, open, high, low, close, volume.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO candles (pair, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair, timestamp) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume
            """, (
                pair, candle_data['timestamp'], candle_data['open'],
                candle_data['high'], candle_data['low'], candle_data['close'],
                candle_data['volume']
            ))
            await db.commit()

    async def get_history(self, pair: str, limit: int = 100) -> List[Dict]:
        """
        Returns the last N candles for a specific pair.
        Sorted from OLDEST to NEWEST (ready for technical indicator calculations).
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row # Allows access by column name
            async with db.execute("""
                SELECT timestamp, open, high, low, close, volume 
                FROM candles 
                WHERE pair = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (pair, limit)) as cursor:
                rows = await cursor.fetchall()
                # Reverse the results to return chronological order (oldest -> newest)
                return [dict(row) for row in rows][::-1]

    async def prune_history(self, days_to_keep: int = 7):
        """Optional: Keeps the 30GB disk from filling up over long periods."""
        cutoff = int((time.time() - (days_to_keep * 86400)) * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM candles WHERE timestamp < ?", (cutoff,))
            await db.commit()
            logger.info(f"Pruned historical data older than {days_to_keep} days.")