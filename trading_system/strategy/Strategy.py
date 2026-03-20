'''
Parent class of strategies.
Strategies communicate with the allocator via CONVICTION SCORES, not raw quantities.

Conviction score semantics:
  +1.0  = Maximum bullish — allocator will size up to max_single_coin weight
   0.0  = No view — allocator will not open a new position
  -1.0  = Maximum bearish — allocator will exit or avoid the coin (long-only system)

@ MTL 21 March 2026
'''
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from db.db_manager import DatabaseManager


class BaseStrategy(ABC):
    def __init__(
        self, 
        strategy_name: str, 
        symbols: List[str], 
        db_path: str = "./trading_bot.db",
        lookback: int = 100
    ):
        """
        Args:
            strategy_name: Unique name for this strategy (used in logs and DB).
            symbols:        List of coins to monitor, e.g. ['BTC', 'ETH'].
            db_path:        Path to the SQLite database.
            lookback:       Number of historical bars to pass to on_tick.
                            Set to 0 to skip history fetching (faster, tick-only).
        """
        self.name = strategy_name
        self.pairs = [
            f"{s.upper()}/USD" if "/" not in s else s.upper() 
            for s in symbols
        ]
        self.lookback = lookback
        self.db = DatabaseManager(db_path)
        self.logger = logging.getLogger(self.name)
        self.is_running = False

    async def run(self, poll_interval: float = 5.0):
        """
        Main strategy loop. Runs on_tick every poll_interval seconds.

        Passes two arguments to on_tick:
          market_state:  { 'BTC/USD': {'last_price': ..., 'last_volume': ..., ...} }
          history:       { 'BTC/USD': [ {timestamp, price, volume}, ... ] }  (oldest→newest)
                         Empty dict if lookback=0.
        """
        self.is_running = True
        self.logger.info(f"{self.name} started | pairs={self.pairs} | lookback={self.lookback}")

        while self.is_running:
            try:
                market_state = await self.db.get_latest_price_batch(self.pairs)

                history = {}
                if self.lookback > 0 and market_state:
                    history = await self.db.get_tick_history_batch(
                        pairs=self.pairs, 
                        limit=self.lookback
                    )

                if market_state:
                    await self.on_tick(market_state, history)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}", exc_info=True)
                await asyncio.sleep(2)

    @abstractmethod
    async def on_tick(
        self, 
        market_state: Dict[str, Dict],
        history: Dict[str, List[Dict]]
    ):
        """
        Implement your strategy logic here.

        Args:
            market_state:  Latest price snapshot.
                           e.g. market_state['BTC/USD']['last_price']
            history:       Historical ticks (oldest → newest).
                           e.g. [t['price'] for t in history['BTC/USD']]
                           Empty dict if lookback=0.

        Call self.submit_conviction(symbol, score) to emit a signal.
        """
        pass

    async def submit_conviction(self, symbol: str, score: float):
        """
        Submit a conviction score for a coin to the Master Allocator.

        Args:
            symbol:  Coin ticker, e.g. 'BTC'.
            score:   Float in [-1.0, 1.0].
                     Positive = bullish, negative = bearish/exit, 0 = no view.

        The allocator clamps this at the DB layer, but you should still
        pass sensible values.
        """
        if not (-1.0 <= score <= 1.0):
            self.logger.warning(
                f"submit_conviction: score {score:.3f} outside [-1, 1], will be clamped."
            )

        intent = {
            "name": self.name,
            "symbol": symbol.upper().replace("/USD", ""),  # Normalise to bare coin
            "conviction": score,
        }
        await self.db.save_order_intent(intent)
        self.logger.debug(f"Conviction submitted: {symbol} = {score:+.3f}")

    def stop(self):
        """Signals the strategy loop to exit cleanly."""
        self.is_running = False