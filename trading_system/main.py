import asyncio
import logging
import signal
import sys
import json
import time
import pandas as pd
import numpy as np
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
        """
        Portfolio Allocator: 
        1. Pools pending intents (Convictions).
        2. Calculates volatility for Risk-Weights.
        3. Calculates target weights and applies sticky logic.
        4. Executes required rebalancing trades.
        """
        pending_intents = await self.db.get_pending_intents()
        if not pending_intents:
            return  # No new votes to process

        # --- STEP 1: Pool Intents (Average Conviction per Coin) ---
        convictions = {}
        intent_ids = []
        for intent in pending_intents:
            sym = intent['symbol']
            if sym not in convictions:
                convictions[sym] = []
            
            # Assuming strategies pass a score [-1, 1] in the 'quantity' or 'price' field.
            # We will use 'quantity' as the conviction score for this architecture.
            conviction_score = float(intent['quantity'])
            
            # If the strategy sent a SELL intent, make the score negative
            if intent['side'].upper() == 'SELL':
                conviction_score = -abs(conviction_score)
                
            convictions[sym].append(conviction_score)
            intent_ids.append(intent['id'])

        # Average the pooled intents
        avg_convictions = {sym: sum(scores)/len(scores) for sym, scores in convictions.items()}

        # Lock intents as processing
        for i_id in intent_ids:
            await self.db.update_intent_status(i_id, "PROCESSING")

        # --- STEP 2: Fetch Environment Data ---
        pairs_to_fetch = [f"{sym}" for sym in avg_convictions.keys()]
        history = await self.db.get_tick_history_batch(pairs=pairs_to_fetch, limit=288)
        
        # Build price DataFrame for Volatility calculation
        price_data = {}
        for pair, ticks in history.items():
            price_data[sym] = [t['price'] for t in ticks]
        
        df_prices = pd.DataFrame(price_data)

        # Get latest prices for valuation
        latest_prices = await self.db.get_latest_price_batch()

        # Get Current Portfolio Weights
        current_portfolio_value = 0.0
        current_holdings_usd = {}
        
        for asset, qty in self.client.balance.items():
            qty = float(qty)
            if asset == 'USD':
                val = qty
            else:
                price = float(latest_prices.get(f"{asset}/USD", {}).get('last_price', 0))
                val = qty * price
                
            current_holdings_usd[asset] = val
            current_portfolio_value += val
            
        # Avoid division by zero on fresh accounts
        if current_portfolio_value == 0:
            current_portfolio_value = 1.0 
            
        current_weights = {k: (v / current_portfolio_value) for k, v in current_holdings_usd.items()}

        # --- STEP 3: Calculate Target Portfolio ---
        target_weights = self._calculate_target_weights(avg_convictions, df_prices)
        trades_to_execute = self._apply_sticky_logic(target_weights, current_weights)

        # --- STEP 4: Execute Rebalancing Trades ---
        for coin, target_w in trades_to_execute.items():
            target_usd = target_w * current_portfolio_value
            current_usd = current_holdings_usd.get(coin, 0.0)
            diff_usd = target_usd - current_usd
            
            latest_price = float(latest_prices.get(f"{coin}/USD", {}).get('last_price', 0))
            if latest_price <= 0:
                continue

            raw_qty = abs(diff_usd) / latest_price
            
            # Apply Exchange Precision & Minimums (Reduces unprofitable micro-trades)
            rules = self.client.market_rules.get(f"{coin}/USD", {})
            qty_precision = rules.get("qty_precision", 4)
            min_qty = rules.get("min_qty", 0.0)
            
            # Round down to valid precision
            trade_qty = np.floor(raw_qty * (10**qty_precision)) / (10**qty_precision)

            if trade_qty < min_qty:
                logger.info(f"Skipping {coin} rebalance: Qty {trade_qty} below min {min_qty}")
                continue

            side = "BUY" if diff_usd > 0 else "SELL"
            
            try:
                logger.info(f"Rebalancing {coin}: {side} {trade_qty} (Target Weight: {target_w:.1%})")
                await self.client.handle_place_order(
                    symbol=coin,
                    side=side,
                    quantity=trade_qty,
                    price=None # Market order for rebalancing
                )
            except Exception as e:
                logger.error(f"Rebalance execution failed for {coin}: {e}")

        # Mark intents as executed
        for i_id in intent_ids:
            await self.db.update_intent_status(i_id, "EXECUTED")

    # --- ALLOCATOR HELPER FUNCTIONS ---

    def _calculate_target_weights(self, convictions_dict, df_prices, max_single_coin=0.15):
        if df_prices.empty:
            return {}

        #If only bearsih signal is there don't trade, hold cash since its long only
        conviction_series = pd.Series(convictions_dict).clip(lower=0)
        
        # 2. Risk Parity Multiplier
        returns = df_prices.pct_change().dropna()
        if returns.empty:
            risk_scalar = pd.Series(1.0, index=df_prices.columns) # Fallback
        else:
            volatility = returns.std()
            inverse_vol = 1 / (volatility + 1e-9)
            # Scale inverse_vol so the "average" coin has a multiplier of exactly 1.0
            # Low vol coins get > 1.0 multiplier, High vol coins get < 1.0
            risk_scalar = inverse_vol / inverse_vol.mean()
        
        risk_scalar = risk_scalar.reindex(conviction_series.index).fillna(1.0)

        # A conviction of 1.0 * avg risk * 15% cap = Exactly 15% portfolio weight
        #ans scale accordingly
        target_weights = conviction_series * risk_scalar * max_single_coin
        
        target_weights = target_weights.clip(upper=max_single_coin)
        total_weight = target_weights.sum()
        if total_weight > 1.0:
            target_weights = target_weights / total_weight
            
        return target_weights.fillna(0).to_dict()

    def _apply_sticky_logic(self, target_weights, current_weights, threshold=0.02):
        trades_to_execute = {}
        
        # Combine all known keys from both target and current portfolios
        all_coins = set(target_weights.keys()).union(set(current_weights.keys()))
        all_coins.discard('USD') # Don't trade USD against itself
        
        for coin in all_coins:
            target = target_weights.get(coin, 0.0)
            current = current_weights.get(coin, 0.0)
            diff = target - current
            
            #Rduce turnover
            if abs(diff) > threshold:
                trades_to_execute[coin] = target
                
        return trades_to_execute

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