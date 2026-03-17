'''
Parent class of strategies

@Michael Teguh Laksana 17 March 2026 00:05
'''
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from db.db_manager import DatabaseManager

class BaseStrategy(ABC):
    def __init__(self, strategy_name: str, symbols: List[str], db_path: str = "./trading_bot.db"):
        """
        symbols: List of coins this strategy monitors, e.g., ['BTC', 'ETH']
        """
        self.name = strategy_name
        # Convert ['BTC'] to ['BTC/USD'] for DB lookups
        self.pairs = [f"{s.upper()}/USD" if "/" not in s else s.upper() for s in symbols]
        
        self.db = DatabaseManager(db_path)
        self.logger = logging.getLogger(self.name)
        self.is_running = False

    async def run(self, poll_interval: float = 0.5):
        self.is_running = True
        self.logger.info(f"Strategy {self.name} monitoring {self.pairs}...")
        
        while self.is_running:
            try:
                # 1. Bulk fetch all relevant market data in one DB hit
                market_state = await self.db.get_latest_price_batch(self.pairs)
                
                if market_state:
                    # 2. Pass the full dict of prices to the strategy logic
                    await self.on_tick(market_state)
                
                await asyncio.sleep(poll_interval)
            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}")
                await asyncio.sleep(2)

    @abstractmethod
    async def on_tick(self, market_data: Dict[str, Dict]):
        """
        Example usage in child class:
        btc_price = market_data['BTC/USD']['last_price']
        """
        pass

    async def submit_order_intent(self, symbol: str, side: str, quantity: float, price: Optional[float] = None):
        """Standardized way to send a trade request to the Master Allocator."""
        intent = {
            "name": self.name,
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "price": price
        }
        await self.db.save_order_intent(intent)