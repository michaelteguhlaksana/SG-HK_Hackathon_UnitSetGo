'''
Classes for managing the interface to the database
Supports Orders, Real-time Tickers, and Historical Tick Data.

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
            
            # 3. Create Tickers Table (Latest State - The "Data Bus")
            # This allows strategies to get the current price without scanning history.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    pair TEXT PRIMARY KEY,
                    last_price REAL,
                    last_volume REAL,
                    timestamp INTEGER
                )
            """)

            # 4. Create Ticks Table (Historical Log)
            # We use a composite primary key to prevent duplicate entries if the API is polled twice in 1ms.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    pair TEXT,
                    price REAL,
                    volume REAL,
                    timestamp INTEGER,
                    PRIMARY KEY (pair, timestamp)
                )
            """)
            
            # 5. Index for fast strategy lookback
            await db.execute("CREATE INDEX IF NOT EXISTS idx_tick_lookup ON ticks (pair, timestamp DESC);")
            
            await db.commit()
            logger.info("Database initialized: Tick history enabled.")

    # --- Order Methods ---
    async def save_order(self, order: Dict[str, Any]):
        """Saves or updates an order status."""
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

    # --- Price Methods (Real-time & History) ---
    async def update_ticker(self, pair: str, price: float, volume: float = 0.0):
        """
        Updates the current price and logs it to history.
        Called by _parse_ticker_price.
        """
        ts = int(time.time() * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            # We do both in one transaction for speed
            async with db.cursor() as cursor:
                # 1. Update the "Live" state
                await cursor.execute("""
                    INSERT INTO tickers (pair, last_price, last_volume, timestamp)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(pair) DO UPDATE SET 
                        last_price=excluded.last_price,
                        last_volume=excluded.last_volume,
                        timestamp=excluded.timestamp
                """, (pair, price, volume, ts))

                # 2. Append to the "Historical" log
                await cursor.execute("""
                    INSERT INTO ticks (pair, price, volume, timestamp)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(pair, timestamp) DO NOTHING
                """, (pair, price, volume, ts))
            
            await db.commit()

    async def get_latest_price(self, pair: str) -> Dict[str, Any]:
        """Fetch the current market state for a pair."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM tickers WHERE pair = ?", (pair,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else {"last_price": 0.0, "last_volume": 0.0, "timestamp": 0}

    # --- Historical Methods ---
    async def get_tick_history(self, pair: str, limit: int = 100) -> List[Dict]:
        """
        Returns the last N historical ticks for a pair.
        Sorted from OLDEST to NEWEST.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT timestamp, price, volume 
                FROM ticks 
                WHERE pair = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (pair, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows][::-1]

    async def prune_ticks(self, hours_to_keep: int = 48):
        """Cleans up old ticks to save space and maintain query speed."""
        cutoff = int((time.time() - (hours_to_keep * 3600)) * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM ticks WHERE timestamp < ?", (cutoff,))
            await db.commit()
            logger.info(f"Pruned ticks older than {hours_to_keep} hours.")