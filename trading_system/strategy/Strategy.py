'''
Parent class of strategies

@Michael Teguh Laksana 17 March 2026 00:05
'''
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional
from db.db_manager import DatabaseManager

class BaseStrategy(ABC):
    def __init__(self, strategy_name: str, symbol: str):
        self.name = strategy_name
        self.symbol = symbol.upper()
        self.db = DatabaseManager() # Strategy only talks to DB
        self.logger = logging.getLogger(self.name)
        self.is_running = False

    async def run(self, poll_interval: float = 0.5):
        self.is_running = True
        self.logger.info(f"Strategy {self.name} started.")
        
        while self.is_running:
            # 1. Pull latest market data from DB (written by Master)
            market_state = await self.db.get_latest_ticker(f"{self.symbol}")
            
            if market_state:
                await self.on_tick(market_state)
            
            await asyncio.sleep(poll_interval)

    @abstractmethod
    async def on_tick(self, market_data: dict):
        """Logic goes here. Use self.submit_order_intent() to trade."""
        pass

    async def submit_order_intent(self, side: str, quantity: float, price: Optional[float] = None):
        """Writes an 'Intent' to the DB for the Master to pick up."""
        intent = {
            "strategy_name": self.name,
            "symbol": self.symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": price,
            "status": "PENDING"
        }
        await self.db.save_order_intent(intent)
        self.logger.info(f"Intent submitted: {side} {quantity} {self.symbol}")