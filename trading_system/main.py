import asyncio
import logging
import signal
import sys
import json
import time
from db.db_manager import DatabaseManager
from gateway.Roostoo import RoostooClientV3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainLoop")

class TradingBot:
    def __init__(self, cred_path="./config/credentials.json", db_path="./config/db.json"):
        with open(cred_path, 'r') as f:
            cred = json.load(f)

        self.API_KEY = cred["API_KEY"]
        self.API_SECRET = cred["API_SECRET"]

        with open(db_path, 'r') as f:
            db_filename = json.load(f)["PATH_TO_DB"]
        
        self.db = DatabaseManager(db_filename)
        self.client = RoostooClientV3(
            api_key=self.API_KEY, 
            api_secret=self.API_SECRET, 
            db_manager=self.db
        )
        self.is_running = True
        self.tasks = []

    async def initialize(self):
        """Warm up the bot: Init DB and check exchange status."""
        logger.info("Initializing Database...")
        await self.db.init_db()
        
        logger.info("Checking Exchange Connection...")
        is_live, _ = await self.client.handle_get_exchange_info()
        if not is_live:
            logger.warning("Exchange is currently NOT running (IsRunning=False).")

    async def run_allocator(self):
        """Reads pending intents from strategies and executes them."""
        pending_intents = await self.db.get_pending_intents()
        
        for intent in pending_intents:
            intent_id = intent['id']
            symbol = intent['symbol']
            
            try:
                await self.db.update_intent_status(intent_id, "PROCESSING")
                logger.info(f"Master executing intent from {intent['strategy_name']}: {intent['side']} {intent['quantity']} {symbol}")
                
                await self.client.handle_place_order(
                    symbol=symbol,
                    side=intent['side'],
                    quantity=intent['quantity'],
                    price=intent.get('price')
                )
                
                # Mark as processed on success
                await self.db.update_intent_status(intent_id, "EXECUTED")
                
            except Exception as e:
                logger.error(f"Failed to execute intent {intent_id}: {e}")
                # Mark as FAILED so we can audit later without infinite retries
                await self.db.update_intent_status(intent_id, "FAILED")

    # ---------------------------------------------------------
    # --- CONCURRENT EVENT LOOPS ---
    # ---------------------------------------------------------

    async def market_data_cycle(self):
        """Fetches prices synced precisely to 5-minute clock boundaries."""
        while self.is_running:
            try:
                now = time.time()
                # Calculate exactly how many seconds until the next 300s (5m) interval
                sleep_duration = 300 - (now % 300)
                
                logger.info(f"Data Cycle: Sleeping {sleep_duration:.1f}s until next 5-min close...")
                await asyncio.sleep(sleep_duration)
                
                if not self.is_running:
                    break
                    
                # Fetch immediately at the exact boundary
                await self.client.handle_get_ticker()
                logger.info("5-min closing prices updated on Data Bus.")
                
            except Exception as e:
                logger.error(f"Market data cycle error: {e}")
                await asyncio.sleep(5) # Brief pause on error before retrying

    async def execution_cycle(self):
        """High-frequency loop checking for strategy intents."""
        while self.is_running:
            try:
                await self.run_allocator()
                await asyncio.sleep(5.0) # Check for new intents every 1 second
            except Exception as e:
                logger.error(f"Execution cycle error: {e}")
                await asyncio.sleep(5.0)

    async def sync_cycle(self):
        """Lower-frequency loop to keep local balance state accurate."""
        while self.is_running:
            try:
                await asyncio.sleep(60.0) # Sync every 1 minute
                if not self.is_running:
                    break
                await self.client.handle_get_balance()
                await self.client.handle_query_order(pending_only=True)
            except Exception as e:
                logger.error(f"Sync cycle error: {e}")
                await asyncio.sleep(5)

    async def start_processes(self):
        """Kicks off all concurrent tasks."""
        # Do one initial fetch so strategies aren't flying blind for the first 5 minutes
        logger.info("Running initial data fetch...")
        await self.client.handle_get_ticker()
        await self.client.handle_get_balance()

        # Launch the independent loops
        self.tasks = [
            asyncio.create_task(self.market_data_cycle()),
            asyncio.create_task(self.execution_cycle()),
            asyncio.create_task(self.sync_cycle())
        ]
        
        # Wait for them to finish (which is never, until shutdown)
        await asyncio.gather(*self.tasks)

    async def shutdown(self):
        logger.info("Shutting down bot...")
        self.is_running = False
        
        # Cancel pending tasks to prevent them from hanging
        for task in self.tasks:
            task.cancel()
            
        await self.client.close()
        logger.info("Cleanup complete.")

async def main():
    bot = TradingBot()
    
    # Handle OS signals for graceful exit (Linux/EC2)
    if sys.platform != "win32":
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))
        except NotImplementedError:
            pass

    try:
        await bot.initialize()
        await bot.start_processes()
    except asyncio.CancelledError:
        # Expected behavior during shutdown
        pass
    except KeyboardInterrupt:
        # Fallback for local Windows testing
        logger.info("KeyboardInterrupt received.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())