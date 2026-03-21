'''
Parent class of strategies.
Strategies communicate with the allocator via CONVICTION SCORES, not raw quantities.

Conviction score semantics:
  +1.0  = Maximum bullish — allocator will size up to max_single_coin weight
   0.0  = No view — allocator will not open a new position
  -1.0  = Maximum bearish / exit signal (long-only system will close position)

Timeframe resampling:
  Pass resample_tf='15min' or '1h' to make a strategy operate on a coarser
  timeframe than the underlying 5-min DB ticks. The base class resamples the
  history dict before passing it to on_tick — subclass logic is unchanged.

  The lookback parameter is always in NATIVE DB bars (5-min ticks), so the
  base class fetches enough raw bars to cover the requested window after
  resampling. E.g. window=6 on '1h' bars needs 6×12 = 72 raw 5-min bars.

@ MTL 21 March 2026
'''
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
from trading_system.db.db_manager import DatabaseManager

# Map resample string → number of 5-min bars per resampled bar
_TF_TO_BARS = {
    '5min':  1,
    '15min': 3,
    '30min': 6,
    '1h':    12,
    '2h':    24,
    '4h':    48,
    '6h':    72,
    '1d':    288,
}


class BaseStrategy(ABC):
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        db_path: str = "./trading_bot.db",
        lookback: int = 100,
        resample_tf: Optional[str] = None,
    ):
        """
        Args:
            strategy_name: Unique name for this strategy (used in logs and DB).
            symbols:        List of coins to monitor, e.g. ['BTC', 'ETH'].
            db_path:        Path to the SQLite database.
            lookback:       Number of bars in the RESAMPLED timeframe to pass
                            to on_tick. The base class fetches enough raw 5-min
                            ticks to cover this after resampling.
            resample_tf:    Resample DB ticks to this timeframe before calling
                            on_tick. Must be one of: '5min', '15min', '30min',
                            '1h', '2h', '4h', '1d'. None = use raw 5-min bars.
        """
        if resample_tf is not None and resample_tf not in _TF_TO_BARS:
            raise ValueError(
                f"Invalid resample_tf '{resample_tf}'. "
                f"Must be one of: {list(_TF_TO_BARS.keys())}"
            )

        self.name        = strategy_name
        self.pairs       = [
            f"{s.upper()}/USD" if "/" not in s else s.upper()
            for s in symbols
        ]
        self.resample_tf = resample_tf
        self.db          = DatabaseManager(db_path)
        self.logger      = logging.getLogger(self.name)
        self.is_running  = False

        # How many 5-min bars per resampled bar
        self._bars_per_tf = _TF_TO_BARS.get(resample_tf or '5min', 1)

        # lookback is in resampled bars — convert to raw 5-min bars for the DB fetch.
        # Add 2× buffer so rolling indicators have stable values from the first bar.
        self._lookback_resampled = lookback
        self.lookback            = lookback * self._bars_per_tf * 2

    # --------------------------------------------------------------------------
    # Resampling helper
    # --------------------------------------------------------------------------

    def _resample_history(
        self, history: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Resample raw 5-min tick history to self.resample_tf.
        Returns the last self._lookback_resampled bars per pair.
        Input/output format: { 'BTC/USD': [{timestamp, price, volume}, ...] }
        Timestamps in the output are Unix ms of the resampled bar's close.
        """
        if self.resample_tf is None or self._bars_per_tf == 1:
            # No resampling needed — just trim to lookback
            return {
                pair: ticks[-self._lookback_resampled:]
                for pair, ticks in history.items()
            }

        resampled = {}
        for pair, ticks in history.items():
            if not ticks:
                continue

            # Build a mini DataFrame from the raw ticks
            df = pd.DataFrame(ticks)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp').sort_index()

            # Resample: last close price and summed volume per bar
            df_r = pd.DataFrame({
                'price':  df['price'].resample(self.resample_tf).last(),
                'volume': df['volume'].resample(self.resample_tf).sum(),
            }).dropna(subset=['price'])

            # Convert back to list-of-dicts format, keeping last N bars
            df_r = df_r.tail(self._lookback_resampled)
            df_r.index = (df_r.index.astype('int64') // 1_000_000)  # back to Unix ms

            resampled[pair] = [
                {'timestamp': ts, 'price': row['price'], 'volume': row['volume']}
                for ts, row in df_r.iterrows()
            ]

        return resampled

    # --------------------------------------------------------------------------
    # Live trading loop
    # --------------------------------------------------------------------------

    async def run(self, poll_interval: float = 5.0):
        """
        Main strategy loop. Fetches raw ticks from DB, resamples if configured,
        then calls on_tick every poll_interval seconds.
        """
        self.is_running = True
        tf_label = self.resample_tf or '5min'
        self.logger.info(
            f"{self.name} started | pairs={self.pairs} | "
            f"tf={tf_label} | lookback={self._lookback_resampled} {tf_label} bars"
        )

        while self.is_running:
            try:
                market_state = await self.db.get_latest_price_batch(self.pairs)

                history = {}
                if self.lookback > 0 and market_state:
                    raw_history = await self.db.get_tick_history_batch(
                        pairs=self.pairs,
                        limit=self.lookback
                    )
                    history = self._resample_history(raw_history)

                if market_state:
                    await self.on_tick(market_state, history)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}", exc_info=True)
                await asyncio.sleep(2)

    # --------------------------------------------------------------------------
    # Abstract interface
    # --------------------------------------------------------------------------

    @abstractmethod
    async def on_tick(
        self,
        market_state: Dict[str, Dict],
        history: Dict[str, List[Dict]]
    ):
        """
        Implement your strategy logic here.

        Args:
            market_state:  Latest price snapshot (always 5-min resolution).
                           e.g. market_state['BTC/USD']['last_price']
            history:       Resampled tick history (oldest → newest).
                           If resample_tf='1h', each entry is one hourly bar.
                           e.g. [t['price'] for t in history['BTC/USD']]

        Call self.submit_conviction(symbol, score) to emit a signal.
        """
        pass

    # --------------------------------------------------------------------------
    # Signal submission
    # --------------------------------------------------------------------------

    async def submit_conviction(self, symbol: str, score: float):
        """
        Submit a conviction score for a coin to the Master Allocator.

        Args:
            symbol:  Coin ticker, e.g. 'BTC'.
            score:   Float in [-1.0, 1.0].
                     Positive = bullish, negative = bearish/exit, 0 = no view.
        """
        if not (-1.0 <= score <= 1.0):
            self.logger.warning(
                f"submit_conviction: score {score:.3f} outside [-1, 1], will be clamped."
            )

        intent = {
            "name":       self.name,
            "symbol":     symbol.upper().replace("/USD", ""),
            "conviction": score,
        }
        await self.db.save_order_intent(intent)
        self.logger.debug(f"Conviction: {symbol} = {score:+.3f}")

    def stop(self):
        self.is_running = False