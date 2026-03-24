'''
This is for a pseudo stat arb strategy.

Basically the same as sstat arb, but with the system's intent method.
To add, this strategy 'borrows' from the other strategy's long position.
This can also be seen as advising against othe rsrtategy's long, given the architecture.
Long-only adaptation:
  When z < -entry_z  (spread too low, expect reversion up):
      dep   conviction = +|scaled_z|   → allocator buys dep
      indep conviction = -|scaled_z|   → allocator suppresses indep
 
  When z > +entry_z  (spread too high, expect reversion down):
      dep   conviction = -|scaled_z|   → allocator suppresses dep
      indep conviction = +|scaled_z|   → allocator buys indep
 
  Inside band:  both convictions = 0.0  (no view)
 
Recommended pairs (from walk-forward backtest, Jan–Mar 2026):
  PAIRS = [
      {"dep": "AVAX", "indep": "DOGE", "beta": 1.2029, "alpha": 4.9703},
      {"dep": "BTC",  "indep": "BNB",  "beta": 0.8192, "alpha": 5.8611},
      {"dep": "SOL",  "indep": "BTC",  "beta": 1.6665, "alpha": -14.1224},
      {"dep": "LINK", "indep": "SOL",  "beta": 0.8907, "alpha": -1.7813},
  ]
 
Usage (config JSON):
    {
        "strategy":     "PairsTrading",
        "name":         "Pairs_1h",
        "pairs":        [
            {"dep": "AVAX", "indep": "DOGE", "beta": 1.2029, "alpha": 4.9703},
            {"dep": "BTC",  "indep": "BNB",  "beta": 0.8192, "alpha": 5.8611},
            {"dep": "SOL",  "indep": "BTC",  "beta": 1.6665, "alpha": -14.1224},
            {"dep": "LINK", "indep": "SOL",  "beta": 0.8907, "alpha": -1.7813}
        ],
        "resample_tf":  "1h",
        "lookback":     120,
        "zscore_window": 72,
        "entry_z":      2.0,
        "exit_z":       0.5,
        "stop_z":       3.5,
        "poll_interval": 30.0,
        "db_path":      "../trading_system/config/trading_bot.db"
    }
 
@ MTL 24 March 2026
'''
 
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from trading_system.strategy.Strategy import BaseStrategy 
 
class PairsStrategy(BaseStrategy):
    """
    Multi-pair stat arb strategy for the Roostoo voting system.
 
    One instance handles N pairs simultaneously. Each pair contributes
    independent conviction scores per coin — the Master Allocator's
    weighted sum then nets them out naturally.
    """
 
    def __init__(
        self,
        strategy_name: str,
        pairs: List[Dict],              # [{"dep": "AVAX", "indep": "DOGE", "beta": 1.2, "alpha": 4.97}, ...]
        db_path: str = "./trading_bot.db",
        resample_tf: Optional[str] = "1h",
        lookback: int = 120,            # bars in resampled tf (default 120h = 5 days at 1h)
        zscore_window: int = 72,        # rolling window for z-score (bars in resampled tf)
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ):
        # Collect all unique symbols for the BaseStrategy DB subscription
        all_symbols = list({
            sym
            for p in pairs
            for sym in (p["dep"], p["indep"])
        })
 
        super().__init__(
            strategy_name=strategy_name,
            symbols=all_symbols,
            db_path=db_path,
            resample_tf=resample_tf,
            lookback=lookback,
        )
 
        self.pairs         = pairs
        self.zscore_window = zscore_window
        self.entry_z       = entry_z
        self.exit_z        = exit_z
        self.stop_z        = stop_z
 
        # Per-pair state: track last z-score and position for logging
        self._state: Dict[str, Dict] = {
            self._pair_key(p): {"zscore": 0.0, "position": 0}
            for p in pairs
        }
 
        self.logger.info(
            f"PairsStrategy init | pairs={[self._pair_key(p) for p in pairs]} | "
            f"tf={resample_tf} | window={zscore_window} | "
            f"entry_z={entry_z} | exit_z={exit_z} | stop_z={stop_z}"
        )
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _pair_key(pair: Dict) -> str:
        return f"{pair['dep']}/{pair['indep']}"
 
    def _build_price_series(
        self,
        symbol: str,
        history: Dict[str, List[Dict]]
    ) -> Optional[pd.Series]:
        """Extract log-price series for a symbol from the history dict."""
        key = f"{symbol}/USD"
        ticks = history.get(key, [])
        if len(ticks) < self.zscore_window + 5:
            return None
        prices = pd.Series([t["price"] for t in ticks], dtype=float)
        return np.log(prices)                   # work in log-price space
 
    def _compute_zscore(
        self,
        log_dep: pd.Series,
        log_indep: pd.Series,
        beta: float,
        alpha: float,
    ) -> Optional[float]:
        """
        Compute the current z-score of the spread.
        spread = log_dep - beta * log_indep - alpha
        z      = (spread - rolling_mean) / rolling_std
        Returns the latest z value, or None if insufficient data.
        """
        spread = log_dep - beta * log_indep - alpha
 
        if len(spread) < self.zscore_window:
            return None
 
        roll_mean = spread.rolling(self.zscore_window).mean()
        roll_std  = spread.rolling(self.zscore_window).std()
 
        # Guard against flatlines / zero std
        latest_std = roll_std.iloc[-1]
        if pd.isna(latest_std) or latest_std < 1e-9:
            return None
 
        z = (spread.iloc[-1] - roll_mean.iloc[-1]) / latest_std
        return float(z)
 
    def _z_to_convictions(
        self,
        z: float,
        dep: str,
        indep: str,
        state: Dict,
    ) -> Dict[str, float]:
        """
        Convert z-score to per-coin conviction scores, respecting position state.
 
        Position state (stored in self._state per pair):
            0  = flat
           +1  = long spread  (long dep / suppress indep)
           -1  = short spread (suppress dep / long indep)
 
        Returns {dep: conviction, indep: conviction}
        """
        pos = state["position"]
 
        # ── Entry logic ────────────────────────────────────────────────
        if pos == 0:
            if z < -self.entry_z:
                state["position"] = 1
            elif z > self.entry_z:
                state["position"] = -1
 
        # ── Exit / stop-loss logic ─────────────────────────────────────
        elif pos == 1:
            if z > -self.exit_z or z > self.stop_z:
                state["position"] = 0
 
        elif pos == -1:
            if z < self.exit_z or z < -self.stop_z:
                state["position"] = 0
 
        # ── Conviction magnitude: scale linearly from entry_z to stop_z ──
        #    |z| / stop_z → clipped to [0, 1]
        magnitude = float(np.clip(abs(z) / self.stop_z, 0.0, 1.0))
 
        pos = state["position"]     # read updated position
 
        if pos == 0:
            return {dep: 0.0, indep: 0.0}
        elif pos == 1:
            # Long spread: bullish dep, bearish indep
            return {dep: +magnitude, indep: -magnitude}
        else:
            # Short spread: bearish dep, bullish indep
            return {dep: -magnitude, indep: +magnitude}
 
    async def on_tick(
        self,
        market_state: Dict[str, Dict],
        history: Dict[str, List[Dict]],
    ):
        """
        For each pair:
          1. Build log-price series from history.
          2. Compute z-score with static (pre-fitted) beta/alpha.
          3. Convert z to per-coin convictions.
          4. Submit convictions — allocator nets them across all pairs.
 
        Note on beta/alpha: these are fitted once offline on training data
        and stored in the config. For a live Kalman filter update, you could
        maintain state here and update beta each tick — left as an extension.
        """
        # Track per-coin conviction accumulation across all pairs this tick
        # so we can log the net view per coin before submitting
        coin_convictions: Dict[str, List[float]] = {}
 
        for pair in self.pairs:
            dep   = pair["dep"]
            indep = pair["indep"]
            beta  = float(pair["beta"])
            alpha = float(pair["alpha"])
            key   = self._pair_key(pair)
 
            # ── 1. Build price series ──────────────────────────────────
            log_dep   = self._build_price_series(dep,   history)
            log_indep = self._build_price_series(indep, history)
 
            if log_dep is None or log_indep is None:
                self.logger.debug(f"{key}: insufficient history, skipping.")
                continue
 
            # Align lengths (trim to shorter series — should be equal but defensive)
            min_len   = min(len(log_dep), len(log_indep))
            log_dep   = log_dep.iloc[-min_len:]
            log_indep = log_indep.iloc[-min_len:]
 
            # ── 2. Compute z-score ─────────────────────────────────────
            z = self._compute_zscore(log_dep, log_indep, beta, alpha)
            if z is None:
                self.logger.debug(f"{key}: z-score unavailable, skipping.")
                continue
 
            state = self._state[key]
            prev_z = state["zscore"]
            state["zscore"] = z
 
            self.logger.info(
                f"{key} | z={z:+.3f} (prev={prev_z:+.3f}) | "
                f"pos={state['position']:+d}"
            )
 
            # ── 3. Convert to convictions ──────────────────────────────
            convictions = self._z_to_convictions(z, dep, indep, state)
 
            # Accumulate for logging
            for coin, conv in convictions.items():
                coin_convictions.setdefault(coin, []).append(conv)
 
            # ── 4. Submit ─────────────────────────────────────────────
            for coin, conv in convictions.items():
                await self.submit_conviction(coin, conv)
 
        # Summary log — shows what this strategy is net voting for
        if coin_convictions:
            net = {
                coin: round(sum(vals) / len(vals), 3)
                for coin, vals in coin_convictions.items()
            }
            self.logger.info(f"Net conviction this tick: {net}")
 
# ==========================================================================
# Standalone test (no exchange needed — uses DB history only)
# ==========================================================================
if __name__ == "__main__":
    import json, sys
 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
 
    PAIRS = [
        {"dep": "AVAX", "indep": "DOGE", "beta": 1.2029, "alpha": 4.9703},
        {"dep": "BTC",  "indep": "BNB",  "beta": 0.8192, "alpha": 5.8611},
        {"dep": "SOL",  "indep": "BTC",  "beta": 1.6665, "alpha": -14.1224},
        {"dep": "LINK", "indep": "SOL",  "beta": 0.8907, "alpha": -1.7813},
    ]
 
    strategy = PairsStrategy(
        strategy_name="Pairs_test",
        pairs=PAIRS,
        db_path="./trading_bot.db",
        resample_tf="1h",
        lookback=120,
        zscore_window=72,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
    )
 
    asyncio.run(strategy.run(poll_interval=30.0))