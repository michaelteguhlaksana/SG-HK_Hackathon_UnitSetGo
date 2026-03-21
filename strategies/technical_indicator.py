'''
Strategy implementations.
All strategies extend BaseStrategy and use submit_conviction(symbol, score).

Timeframe parameter:
  Every strategy accepts resample_tf='5min'|'15min'|'30min'|'1h'|'2h'|'4h'|'1d'.
  The window parameter is always in RESAMPLED bars, so the semantics stay
  consistent regardless of timeframe:
    MACDStrategy(window=6, resample_tf='1h')  → 6-hour MACD
    MACDStrategy(window=6, resample_tf='5min') → 30-minute MACD

@ MTL 21 March 2026
'''
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from trading_system.strategy.Strategy import BaseStrategy


# ==============================================================================
# 1. BOLLINGER BAND REVERSION
# ==============================================================================

class BollingerReversion(BaseStrategy):
    """
    Mean-reversion using Bollinger Bands.
    Score = (mean - price) / (2.5 * std), clipped to [-1, 1].
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 20,
        db_path: str = "./trading_bot.db",
        resample_tf: Optional[str] = None,
    ):
        super().__init__(
            strategy_name, symbols, db_path,
            lookback=window * 2,
            resample_tf=resample_tf,
        )
        self.window = window

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {
            pair: [t['price'] for t in ticks]
            for pair, ticks in history.items()
            if ticks
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
            await self.submit_conviction(pair.replace('/USD', ''), float(score))


# ==============================================================================
# 2. MACD MOMENTUM
# ==============================================================================

class MACDStrategy(BaseStrategy):
    """
    Trend-following using MACD.
    Score = tanh(2 * (macd - signal) / rolling_std).

    Spans scale with window:
      fast   = window // 2
      slow   = window
      signal = window // 3
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 26,
        db_path: str = "./trading_bot.db",
        resample_tf: Optional[str] = None,
    ):
        self.window = max(window, 3)
        super().__init__(
            strategy_name, symbols, db_path,
            lookback=self.window * 3,
            resample_tf=resample_tf,
        )

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {
            pair: [t['price'] for t in ticks]
            for pair, ticks in history.items()
            if ticks
        }
        df = pd.DataFrame(price_dict)
        if len(df) < self.window:
            return

        ema_fast    = df.ewm(span=self.window // 2, adjust=False).mean()
        ema_slow    = df.ewm(span=self.window,       adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.window // 3, adjust=False).mean()
        rolling_std = df.rolling(self.window).std() + 1e-9
        scores      = np.tanh(2 * (macd_line - signal_line) / rolling_std)
        latest      = scores.iloc[-1]

        for pair, score in latest.items():
            if pd.isna(score):
                continue
            await self.submit_conviction(pair.replace('/USD', ''), float(score))


# ==============================================================================
# 3. VWAP REVERSION
# ==============================================================================

class VWAPReversion(BaseStrategy):
    """
    Mean-reversion using rolling VWAP as fair-value anchor.
    Score = (vwap - price) / rolling_std, clipped to [-1, 1].
    Note: volume quality depends on data source — verify before relying on this.
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 14,
        db_path: str = "./trading_bot.db",
        resample_tf: Optional[str] = None,
    ):
        super().__init__(
            strategy_name, symbols, db_path,
            lookback=window * 2,
            resample_tf=resample_tf,
        )
        self.window = window

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {p: [t['price']  for t in ticks] for p, ticks in history.items() if ticks}
        vol_dict   = {p: [t['volume'] for t in ticks] for p, ticks in history.items() if ticks}

        df  = pd.DataFrame(price_dict)
        vol = pd.DataFrame(vol_dict)
        if len(df) < self.window:
            return

        pv   = (df * vol).rolling(window=self.window).sum()
        v    = vol.rolling(window=self.window).sum()
        vwap = pv / (v + 1e-9)
        scores = ((vwap - df) / (df.rolling(self.window).std() + 1e-9)).clip(-1, 1)
        latest = scores.iloc[-1]

        for pair, score in latest.items():
            if pd.isna(score):
                continue
            await self.submit_conviction(pair.replace('/USD', ''), float(score))


# ==============================================================================
# 4. CROSS-SECTIONAL MOMENTUM
# ==============================================================================

class CrossSectionalMomentum(BaseStrategy):
    """
    Ranks coins by N-bar return cross-sectionally.
    Top coin → +1.0 conviction, bottom → -1.0.
    Naturally low-turnover since rankings change slowly.
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        window: int = 24,
        db_path: str = "./trading_bot.db",
        resample_tf: Optional[str] = None,
    ):
        super().__init__(
            strategy_name, symbols, db_path,
            lookback=window + 5,
            resample_tf=resample_tf,
        )
        self.window = window

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        valid = {
            pair: ticks
            for pair, ticks in history.items()
            if len(ticks) >= 2 and ticks[0].get('price') is not None
        }
        if len(valid) < 2:
            return

        returns = {}
        for pair, ticks in valid.items():
            start_price = ticks[max(0, len(ticks) - self.window)]['price']
            end_price   = ticks[-1]['price']
            if start_price and start_price > 0:
                returns[pair] = (end_price / start_price) - 1.0

        if len(returns) < 2:
            return

        ret_series = pd.Series(returns)
        ranked     = ret_series.rank(pct=True)
        conviction = (ranked - 0.5) * 2.0

        for pair, score in conviction.items():
            await self.submit_conviction(pair.replace('/USD', ''), float(score))


# ==============================================================================
# 5. ADAPTIVE RSI
# ==============================================================================

class AdaptiveRSI(BaseStrategy):
    """
    RSI with dynamic thresholds via rolling percentile.
    conviction = (0.5 - rsi_percentile) * 2
    Oversold → positive, overbought → negative.
    """
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        rsi_window: int = 14,
        percentile_window: int = 100,
        db_path: str = "./trading_bot.db",
        resample_tf: Optional[str] = None,
    ):
        super().__init__(
            strategy_name, symbols, db_path,
            lookback=rsi_window + percentile_window + 5,
            resample_tf=resample_tf,
        )
        self.rsi_window        = rsi_window
        self.percentile_window = percentile_window

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta    = prices.diff()
        avg_gain = delta.clip(lower=0).ewm(com=self.rsi_window - 1, min_periods=self.rsi_window).mean()
        avg_loss = (-delta).clip(lower=0).ewm(com=self.rsi_window - 1, min_periods=self.rsi_window).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    async def on_tick(self, market_state: Dict[str, Dict], history: Dict[str, List[Dict]]):
        if not history:
            return

        price_dict = {
            pair: [t['price'] for t in ticks]
            for pair, ticks in history.items()
            if ticks
        }
        df          = pd.DataFrame(price_dict)
        min_bars    = self.rsi_window + self.percentile_window

        if len(df) < min_bars:
            return

        for pair in df.columns:
            prices = df[pair].dropna()
            if len(prices) < min_bars:
                continue

            rsi        = self._compute_rsi(prices)
            rsi_pct    = rsi.rolling(self.percentile_window).rank(pct=True)
            latest_pct = rsi_pct.iloc[-1]

            if pd.isna(latest_pct):
                continue

            score = (0.5 - latest_pct) * 2.0
            await self.submit_conviction(pair.replace('/USD', ''), float(score))