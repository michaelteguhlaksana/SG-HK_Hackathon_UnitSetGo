'''
Strategy implementations.
All strategies extend BaseStrategy and use submit_conviction(symbol, score).

Conviction score semantics:
  +1.0  = Maximum bullish
   0.0  = No view (hold cash for this slot)
  -1.0  = Maximum bearish / exit signal (long-only system will close position)

Strategies receive history pre-fetched by the base class run() loop —
do NOT make additional DB calls for tick history inside on_tick().

@ MTL 17 March 2026
'''
import pandas as pd
import numpy as np
from typing import Dict, List
from trading_system.strategy.Strategy import BaseStrategy


# ==============================================================================
# 1. BOLLINGER BAND REVERSION
# ==============================================================================

class BollingerReversion(BaseStrategy):
    """
    Mean-reversion strategy using Bollinger Bands.

    Logic:
      - Price far below the rolling mean  → strong buy conviction
      - Price far above the rolling mean  → strong sell / exit conviction

    Score = (mean - price) / (2.5 * std), clipped to [-1, 1].
    A band width of 2.5 (vs classic 2.0) reduces false signals on
    volatile crypto assets.

    Sensitivity: Short-term (default window=20 bars = ~100 min at 5m bars).
    Character:   Mean-reversion. Will underperform in sustained trends.
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 20,
        db_path: str = "./trading_bot.db"
    ):
        # lookback = 2x window so rolling stats are stable from the first valid bar
        super().__init__(strategy_name, symbols, db_path, lookback=window * 2)
        self.window = window

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {
            pair: [t['price'] for t in ticks]
            for pair, ticks in history.items()
        }
        df = pd.DataFrame(price_dict)

        if len(df) < self.window:
            return

        rolling_mean = df.rolling(window=self.window).mean()
        rolling_std  = df.rolling(window=self.window).std()

        scores = ((rolling_mean - df) / (rolling_std * 2.5 + 1e-9)).clip(-1, 1)
        latest = scores.iloc[-1]

        for pair, score in latest.items():
            if pd.isna(score):
                continue
            symbol = pair.replace('/USD', '')
            await self.submit_conviction(symbol, float(score))


# ==============================================================================
# 2. MACD MOMENTUM
# ==============================================================================

class MACDStrategy(BaseStrategy):
    """
    Trend-following strategy using MACD.

    Logic:
      - MACD line crossing above signal line → bullish conviction
      - MACD line crossing below signal line → bearish / exit conviction

    Score = tanh(2 * (macd - signal) / rolling_std), naturally bounded to (-1, 1).
    The factor of 2 inside tanh ensures the strategy emits non-trivial conviction
    scores rather than clustering near zero.

    Default spans approximate standard MACD (12 / 26 / 9) at window=26:
      fast  = window // 2  = 13
      slow  = window       = 26
      signal= window // 3  = ~9

    Sensitivity: Medium-term trend.
    Character:   Momentum. Complements mean-reversion strategies.
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 26,
        db_path: str = "./trading_bot.db"
    ):
        self.window = max(window, 3)
        # 3x window gives the EMA enough bars to "warm up" before signals are used
        super().__init__(strategy_name, symbols, db_path, lookback=self.window * 3)

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {
            pair: [t['price'] for t in ticks]
            for pair, ticks in history.items()
        }
        df = pd.DataFrame(price_dict)

        if len(df) < self.window:
            return

        ema_fast    = df.ewm(span=self.window // 2, adjust=False).mean()
        ema_slow    = df.ewm(span=self.window,       adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.window // 3, adjust=False).mean()

        rolling_std = df.rolling(self.window).std() + 1e-9
        # Factor of 2 widens the tanh input so scores spread across (-1, 1)
        scores = np.tanh(2 * (macd_line - signal_line) / rolling_std)
        latest = scores.iloc[-1]

        for pair, score in latest.items():
            if pd.isna(score):
                continue
            symbol = pair.replace('/USD', '')
            await self.submit_conviction(symbol, float(score))


# ==============================================================================
# 3. VWAP REVERSION
# ==============================================================================

class VWAPReversion(BaseStrategy):
    """
    Mean-reversion strategy using rolling VWAP as the fair-value anchor.

    NOTE: Roostoo's 'UnitTradeValue' is a 24h rolling USD volume figure,
    not a per-bar volume. Verify it actually varies bar-to-bar before
    relying on this strategy — if volume is flat it degrades to a
    price-only moving average reversion (similar to Bollinger).

    Score = (vwap - price) / rolling_std, clipped to [-1, 1].

    Sensitivity: Short-to-medium term.
    Character:   Mean-reversion with volume weighting.
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 14,
        db_path: str = "./trading_bot.db"
    ):
        super().__init__(strategy_name, symbols, db_path, lookback=window * 2)
        self.window = window

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {}
        vol_dict   = {}
        for pair, ticks in history.items():
            price_dict[pair] = [t['price']  for t in ticks]
            vol_dict[pair]   = [t['volume'] for t in ticks]

        df  = pd.DataFrame(price_dict)
        vol = pd.DataFrame(vol_dict)

        if len(df) < self.window:
            return

        pv   = (df * vol).rolling(window=self.window).sum()
        v    = vol.rolling(window=self.window).sum()
        vwap = pv / (v + 1e-9)

        rolling_std = df.rolling(self.window).std() + 1e-9
        scores = ((vwap - df) / rolling_std).clip(-1, 1)
        latest = scores.iloc[-1]

        for pair, score in latest.items():
            if pd.isna(score):
                continue
            symbol = pair.replace('/USD', '')
            await self.submit_conviction(symbol, float(score))


# ==============================================================================
# 4. CROSS-SECTIONAL MOMENTUM
# ==============================================================================

class CrossSectionalMomentum(BaseStrategy):
    """
    Trend-following strategy that ranks coins by recent return and goes
    long the strongest, avoiding (or exiting) the weakest.

    Logic:
      1. Compute each coin's N-bar return.
      2. Rank coins by return (cross-sectionally, percentile 0→1).
      3. Rescale rank to conviction: top coin → +1.0, bottom coin → -1.0.

    This produces naturally sticky, low-turnover signals because rankings
    don't flip every bar — directly addressing the turnover friction problem.

    Character:   Trend / momentum. Exploits the crypto "winner keeps winning"
                 effect. Complements all three mean-reversion strategies above.
    Sensitivity: Determined by window (default=24 bars = 2h at 5m bars).
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 24,
        db_path: str = "./trading_bot.db"
    ):
        # Need window + a little buffer for a stable return calculation
        super().__init__(strategy_name, symbols, db_path, lookback=window + 5)
        self.window = window

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        # Need at least 2 bars per coin to compute a return
        valid = {
            pair: ticks
            for pair, ticks in history.items()
            if len(ticks) >= 2
        }
        if len(valid) < 2:
            # Cross-sectional ranking is meaningless with only 1 coin
            return

        # N-bar return: (latest price / price N bars ago) - 1
        returns = {}
        for pair, ticks in valid.items():
            start_price = ticks[max(0, len(ticks) - self.window)]['price']
            end_price   = ticks[-1]['price']
            if start_price > 0:
                returns[pair] = (end_price / start_price) - 1.0

        if len(returns) < 2:
            return

        # Percentile rank across coins (0 = worst, 1 = best)
        ret_series = pd.Series(returns)
        ranked     = ret_series.rank(pct=True)          # 0.0 → 1.0
        conviction = (ranked - 0.5) * 2.0               # rescale to [-1.0, +1.0]

        for pair, score in conviction.items():
            symbol = pair.replace('/USD', '')
            await self.submit_conviction(symbol, float(score))


# ==============================================================================
# 5. ADAPTIVE RSI (ROLLING PERCENTILE)
# ==============================================================================

class AdaptiveRSI(BaseStrategy):
    """
    Mean-reversion strategy using RSI with dynamic thresholds.

    Unlike standard RSI (fixed 30/70 levels tuned for equities), this version
    measures where today's RSI sits within its own recent distribution.
    This makes the signal adaptive to the current volatility regime.

    Logic:
      1. Compute RSI over rsi_window bars.
      2. Rank current RSI against its own history (rolling percentile).
      3. conviction = (0.5 - percentile) * 2
           → RSI at historical low  (percentile≈0) → conviction ≈ +1.0 (oversold, buy)
           → RSI at historical high (percentile≈1) → conviction ≈ -1.0 (overbought, exit)

    Character:   Mean-reversion. More adaptive than standard RSI to different
                 market regimes — particularly useful when volatility shifts.
    Sensitivity: rsi_window controls RSI responsiveness (default=14).
                 percentile_window controls how far back we look for context (default=100).
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        rsi_window: int = 14,
        percentile_window: int = 100,
        db_path: str = "./trading_bot.db"
    ):
        # Need enough bars to compute RSI and then rank it
        super().__init__(
            strategy_name, symbols, db_path,
            lookback=rsi_window + percentile_window + 5
        )
        self.rsi_window        = rsi_window
        self.percentile_window = percentile_window

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        """Standard Wilder RSI using EWM for smooth rolling gain/loss."""
        delta = prices.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        # Wilder smoothing: com = rsi_window - 1
        avg_gain = gain.ewm(com=self.rsi_window - 1, min_periods=self.rsi_window).mean()
        avg_loss = loss.ewm(com=self.rsi_window - 1, min_periods=self.rsi_window).mean()

        rs  = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {
            pair: [t['price'] for t in ticks]
            for pair, ticks in history.items()
        }
        df = pd.DataFrame(price_dict)

        min_required = self.rsi_window + self.percentile_window
        if len(df) < min_required:
            return

        for pair in df.columns:
            prices = df[pair].dropna()
            if len(prices) < min_required:
                continue

            rsi = self._compute_rsi(prices)

            # Rolling percentile rank of current RSI vs recent history
            rsi_pct = rsi.rolling(self.percentile_window).rank(pct=True)

            latest_pct = rsi_pct.iloc[-1]
            if pd.isna(latest_pct):
                continue

            # oversold (low percentile) → positive conviction
            # overbought (high percentile) → negative conviction
            score  = (0.5 - latest_pct) * 2.0
            symbol = pair.replace('/USD', '')
            await self.submit_conviction(symbol, float(score))