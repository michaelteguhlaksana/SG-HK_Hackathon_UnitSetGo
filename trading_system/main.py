import asyncio
import logging
import signal
import sys
import json
from db.db_manager import DatabaseManager
from gateway.Roostoo import RoostooClientV3  # Assuming your client file is named api_client.py

# --- Configuration ---

POLL_INTERVAL = 1.0  # Seconds between ticker updates
SYNC_INTERVAL = 10   # Seconds between balance/order syncs

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainLoop")

class TradingBot:
    def __init__(self, cred_path = "./config/credentials.json", db_path = "./config/db.json"):
        
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

    async def initialize(self):
        """Warm up the bot: Init DB and check exchange status."""
        logger.info("Initializing Database...")
        await self.db.init_db()
        
        logger.info("Checking Exchange Connection...")
        # Check exchange info to see if the competition is live
        is_live, _ = await self.client.handle_get_exchange_info()
        if not is_live:
            logger.warning("Exchange is currently NOT running (IsRunning=False).")

    async def run_allocator(self):
        """
        Reads pending intents from strategies and executes them.
        """
        pending_intents = await self.db.get_pending_intents()

        #TODO: Handle risk maangement and sizing here
        
        for intent in pending_intents:
            intent_id = intent['id']
            symbol = intent['symbol']
            
            try:
                # 1. Execute on Exchange
                logger.info(f"Master executing intent from {intent['strategy_name']}: {intent['side']} {intent['quantity']} {symbol}")
                
                await self.client.handle_place_order(
                    symbol=symbol,
                    side=intent['side'],
                    quantity=intent['quantity'],
                    price=intent.get('price')
                )
                
                # 2. Mark as processed so we don't buy it again next loop
                await self.db.update_intent_status(intent_id, "EXECUTED")
                
            except Exception as e:
                logger.error(f"Failed to execute intent {intent_id}: {e}")
                # Optional: Mark as FAILED so it doesn't get stuck in an infinite retry loop
                await self.db.update_intent_status(intent_id, "FAILED")

    async def data_cycle(self):
        """Main loop for fetching tickers and running strategy."""
        counter = 0
        while self.is_running:
            try:
                # 1. Update Tick Data (Saves to DB)
                await self.client.handle_get_ticker()
                
                # 2. Periodically sync account state (Balance & Open Orders)
                if counter % SYNC_INTERVAL == 0:
                    await self.client.handle_get_balance()
                    await self.client.handle_query_order(pending_only=True)
                
                # 3. Trigger Strategy
                await self.run_allocator()
                
                counter += 1
                await asyncio.sleep(POLL_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5) # Wait before retrying on crash

    async def shutdown(self):
        logger.info("Shutting down bot...")
        self.is_running = False
        await self.client.close()
        logger.info("Cleanup complete.")

async def main():
    bot = TradingBot()
    
    # Handle OS signals for graceful exit (Ctrl+C)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))

    try:
        await bot.initialize()
        await bot.data_cycle()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())