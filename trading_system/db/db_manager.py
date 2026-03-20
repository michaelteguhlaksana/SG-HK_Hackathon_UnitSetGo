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
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    pair TEXT PRIMARY KEY,
                    last_price REAL,
                    last_volume REAL,
                    timestamp INTEGER
                )
            """)

            # 4. Create Ticks Table (Historical Log)
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
            
            # 6. Order Intents Table
            # NOTE: 'conviction' is a float in [-1.0, 1.0].
            #   +1.0 = maximum bullish, -1.0 = maximum bearish, 0.0 = no view.
            # Strategies must submit a conviction score, NOT a raw quantity.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS order_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT,
                    symbol TEXT,
                    conviction REAL,
                    status TEXT DEFAULT 'PENDING',
                    timestamp INTEGER
                )
            """)
            await db.commit()
            logger.info("Database initialized.")

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

    async def cancel_order_by_id(self, order_id: str):
        """Updates an existing order's status to CANCELED."""
        ts = int(time.time() * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE orders 
                SET status = 'CANCELED', timestamp = ? 
                WHERE order_id = ?
            """, (ts, str(order_id)))
            await db.commit()
            logger.info(f"DB: Order {order_id} marked as CANCELED.")

    async def update_tickers_batch(self, ticker_data_list: List[Dict[str, Any]]):
        """
        Updates multiple tickers in a single transaction.
        ticker_data_list: [{'pair': 'BTC/USD', 'price': 9000.0, 'volume': 1000.0}, ...]
        """
        ts = int(time.time() * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            # Update Live Bus
            await db.executemany("""
                INSERT INTO tickers (pair, last_price, last_volume, timestamp)
                VALUES (:pair, :price, :volume, :ts)
                ON CONFLICT(pair) DO UPDATE SET 
                    last_price=excluded.last_price,
                    last_volume=excluded.last_volume,
                    timestamp=excluded.timestamp
            """, [{**d, 'ts': ts} for d in ticker_data_list])

            # Log History
            await db.executemany("""
                INSERT INTO ticks (pair, price, volume, timestamp)
                VALUES (:pair, :price, :volume, :ts)
                ON CONFLICT(pair, timestamp) DO NOTHING
            """, [{**d, 'ts': ts} for d in ticker_data_list])
            
            await db.commit()

    # --- Intent Methods ---

    async def save_order_intent(self, intent: Dict[str, Any]):
        """
        Called by a Strategy to submit a conviction score.
        intent must contain: 'name', 'symbol', 'conviction' (float in [-1, 1])
        """
        ts = int(time.time() * 1000)
        # Hard clamp at DB layer as a safety net
        conviction = max(-1.0, min(1.0, float(intent['conviction'])))
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO order_intents (strategy_name, symbol, conviction, timestamp)
                VALUES (?, ?, ?, ?)
            """, (intent['name'], intent['symbol'].upper(), conviction, ts))
            await db.commit()

    async def get_pending_intents(self) -> List[Dict]:
        """Called by the Master Allocator to collect unprocessed signals."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM order_intents WHERE status = 'PENDING'") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def update_intent_status(self, intent_id: int, status: str):
        """Marks an intent as 'EXECUTED', 'REJECTED', or 'CANCELLED'."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE order_intents SET status = ? WHERE id = ?", 
                (status, intent_id)
            )
            await db.commit()

    # --- Price Methods ---

    async def get_latest_price(self, pair: str) -> Dict[str, Any]:
        """Fetch the current market state for a pair."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tickers WHERE pair = ?", (pair,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else {"last_price": 0.0, "last_volume": 0.0, "timestamp": 0}
            
    async def get_latest_price_batch(self, pairs: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fetches the latest state for multiple pairs.
        If pairs is None, returns all tickers in the database.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if pairs:
                placeholders = ','.join(['?'] * len(pairs))
                query = f"SELECT * FROM tickers WHERE pair IN ({placeholders})"
                params = pairs
            else:
                query = "SELECT * FROM tickers"
                params = ()

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return {row['pair']: dict(row) for row in rows}

    async def get_tick_history(self, pair: str, limit: int = 100) -> List[Dict]:
        """
        Returns the last N historical ticks for a single pair.
        Sorted OLDEST to NEWEST.
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

    async def get_tick_history_batch(
        self, 
        pairs: Optional[List[str]] = None, 
        limit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Returns the last N historical ticks for multiple pairs in one DB round-trip.
        Returns: { 'BTC/USD': [ {timestamp, price, volume}, ... oldest→newest ], ... }

        Strategy: fetch limit*len(pairs) rows with ROW_NUMBER filtering.
        Falls back to fetching all pairs if pairs=None.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if pairs:
                placeholders = ','.join(['?'] * len(pairs))
                # Use a subquery with ROW_NUMBER to get the last N per pair efficiently
                query = f"""
                    SELECT pair, timestamp, price, volume
                    FROM (
                        SELECT 
                            pair, timestamp, price, volume,
                            ROW_NUMBER() OVER (PARTITION BY pair ORDER BY timestamp DESC) as rn
                        FROM ticks
                        WHERE pair IN ({placeholders})
                    )
                    WHERE rn <= ?
                    ORDER BY pair, timestamp ASC
                """
                params = pairs + [limit]
            else:
                query = """
                    SELECT pair, timestamp, price, volume
                    FROM (
                        SELECT 
                            pair, timestamp, price, volume,
                            ROW_NUMBER() OVER (PARTITION BY pair ORDER BY timestamp DESC) as rn
                        FROM ticks
                    )
                    WHERE rn <= ?
                    ORDER BY pair, timestamp ASC
                """
                params = [limit]

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Group into dict, already sorted ASC (oldest → newest) by query
            result: Dict[str, List[Dict]] = {}
            for row in rows:
                p = row['pair']
                if p not in result:
                    result[p] = []
                result[p].append({
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'volume': row['volume']
                })
            return result

    async def prune_ticks(self, hours_to_keep: int = 48):
        """Cleans up old ticks to save space and maintain query speed."""
        cutoff = int((time.time() - (hours_to_keep * 3600)) * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM ticks WHERE timestamp < ?", (cutoff,))
            await db.commit()
            logger.info(f"Pruned ticks older than {hours_to_keep} hours.")