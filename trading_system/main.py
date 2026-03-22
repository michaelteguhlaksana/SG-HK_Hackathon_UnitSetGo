import asyncio
import logging
import signal
import sys
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
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

        # --- Allocator Configuration ---
        # Tune these during the competition without changing logic.
        self.cfg = {
            # Maximum weight any single coin can hold (e.g. 0.30 = 30%)
            "max_single_coin":   0.30,

            # Minimum cash (USD) reserve as a fraction of portfolio. 
            # Acts as a drawdown circuit-breaker. Never deploy below this.
            "min_cash_fraction": 0.10,

            # Don't rebalance a coin unless its weight diff exceeds this.
            # Reduces turnover. Raise if fees are hurting you.
            "sticky_threshold":  0.075,

            # Lookback window for volatility calculation (number of 5-min bars).
            # 288 = 24 hours at 5-min intervals.
            "vol_lookback":      288,

            # Minimum USD notional for a trade to be sent.
            # Avoids API rejections and pointless micro-trades.
            "min_trade_usd":     1500.0,

            # Decay factor for exponentially-weighted volatility (0 < λ < 1).
            # Higher = more weight on recent vol. Set to None for simple std.
            "ewm_span":          48,

            # Per-strategy conviction weights.
            # Strategies not listed here default to 1.0.
            #
            "strategy_weights": {
                "MACD_1h_6h":    3.0,
                "XSMom_1h_24h":  3.0,
                "MACD_15m_6h":   1.0, #Curently not used due to high turnover
            },
        }

    async def initialize(self):
        """Warm up: init DB and check exchange status."""
        logger.info("Initializing Database...")
        await self.db.init_db()
        logger.info("Checking Exchange Connection...")
        result = await self.client.handle_get_exchange_info()
        if result:
            is_live, _ = result
            if not is_live:
                logger.warning("Exchange is NOT running (IsRunning=False).")

    # ==========================================================================
    #  ALLOCATOR
    # ==========================================================================

    async def run_allocator(self):
        """
        Portfolio Allocator — called every execution cycle.

        Pipeline:
          1. Collect & pool pending conviction signals from strategies.
          2. Fetch environment: prices, volatility history, current balance.
          3. Compute target weights via conviction-weighted risk parity.
          4. Apply cash floor and single-coin cap.
          5. Apply sticky threshold to suppress low-value rebalances.
          6. Execute required trades (sells first, then buys).
        """
        # ------------------------------------------------------------------
        # STEP 1: Weighted Conviction Pooling
        # ------------------------------------------------------------------
        pending_intents = await self.db.get_pending_intents()
        if not pending_intents:
            return

        strategy_weights = self.cfg.get("strategy_weights", {})

        weighted_sums: Dict[str, float] = {}
        total_weights: Dict[str, float] = {}
        intent_ids = []

        for intent in pending_intents:
            sym        = intent['symbol']
            conviction = float(intent['conviction'])
            strat_name = intent['strategy_name']
            weight     = strategy_weights.get(strat_name, 1.0)

            weighted_sums.setdefault(sym, 0.0)
            total_weights.setdefault(sym, 0.0)

            weighted_sums[sym] += conviction * weight
            total_weights[sym] += weight
            intent_ids.append(intent['id'])

        avg_convictions: Dict[str, float] = {
            sym: weighted_sums[sym] / total_weights[sym]
            for sym in weighted_sums
        }

        logger.info(f"Allocator: pooled convictions = {avg_convictions}")

        for i_id in intent_ids:
            await self.db.update_intent_status(i_id, "PROCESSING")

        # ------------------------------------------------------------------
        # STEP 2: Fetch Environment
        # ------------------------------------------------------------------
        pairs_with_usd = [f"{sym}/USD" for sym in avg_convictions.keys()]
        history = await self.db.get_tick_history_batch(
            pairs=pairs_with_usd,
            limit=self.cfg["vol_lookback"]
        )

        price_series = {}
        for pair, ticks in history.items():
            coin = pair.replace("/USD", "")
            if len(ticks) >= 2:
                price_series[coin] = [t['price'] for t in ticks]

        df_prices     = pd.DataFrame(price_series) if price_series else pd.DataFrame()
        latest_prices = await self.db.get_latest_price_batch()

        balance = self.client.balance
        if not balance:
            logger.warning("Allocator: balance is empty, skipping cycle.")
            for i_id in intent_ids:
                await self.db.update_intent_status(i_id, "REJECTED")
            return

        # ------------------------------------------------------------------
        # STEP 3: Current Portfolio Valuation
        # ------------------------------------------------------------------
        current_holdings_usd: Dict[str, float] = {}
        total_portfolio_usd = 0.0

        for asset, free_qty in balance.items():
            free_qty = float(free_qty)
            if asset == 'USD':
                val = free_qty
            else:
                price = float(latest_prices.get(f"{asset}/USD", {}).get('last_price', 0.0))
                val   = free_qty * price

            current_holdings_usd[asset]  = val
            total_portfolio_usd         += val

        if total_portfolio_usd <= 0:
            logger.warning("Allocator: portfolio value is zero, skipping.")
            for i_id in intent_ids:
                await self.db.update_intent_status(i_id, "REJECTED")
            return

        current_weights: Dict[str, float] = {
            k: v / total_portfolio_usd
            for k, v in current_holdings_usd.items()
        }
        logger.info(
            f"Portfolio: ${total_portfolio_usd:,.0f} | "
            f"Cash: {current_weights.get('USD', 0):.1%}"
        )

        # ------------------------------------------------------------------
        # STEP 4: Target Weights
        # ------------------------------------------------------------------
        target_weights = self._calculate_target_weights(
            avg_convictions, df_prices, latest_prices
        )

        # ------------------------------------------------------------------
        # STEP 5: Sticky Filter
        # ------------------------------------------------------------------
        trades = self._apply_sticky_logic(target_weights, current_weights)

        if not trades:
            logger.info("Allocator: no rebalances exceed sticky threshold.")
            for i_id in intent_ids:
                await self.db.update_intent_status(i_id, "EXECUTED")
            return

        # ------------------------------------------------------------------
        # STEP 6: Execute — Sells First, then Buys
        # ------------------------------------------------------------------
        sells = {c: w for c, w in trades.items() if w < current_weights.get(c, 0.0)}
        buys  = {c: w for c, w in trades.items() if w > current_weights.get(c, 0.0)}

        for coin, target_w in {**sells, **buys}.items():
            current_usd  = current_holdings_usd.get(coin, 0.0)
            target_usd   = target_w * total_portfolio_usd
            diff_usd     = target_usd - current_usd
            latest_price = float(
                latest_prices.get(f"{coin}/USD", {}).get('last_price', 0.0)
            )

            if latest_price <= 0:
                logger.warning(f"Allocator: no price for {coin}, skipping.")
                continue

            if abs(diff_usd) < self.cfg["min_trade_usd"]:
                logger.info(
                    f"Allocator: skipping {coin}, diff ${abs(diff_usd):.0f} "
                    f"< min ${self.cfg['min_trade_usd']:.0f}"
                )
                continue

            rules        = self.client.market_rules.get(f"{coin}/USD", {})
            qty_prec     = rules.get("qty_precision", 6)
            min_notional = rules.get("min_notional", 1.0)

            raw_qty   = abs(diff_usd) / latest_price
            trade_qty = np.floor(raw_qty * (10 ** qty_prec)) / (10 ** qty_prec)

            if trade_qty * latest_price <= min_notional:
                logger.info(
                    f"Allocator: skipping {coin}, notional "
                    f"${trade_qty * latest_price:.2f} <= MiniOrder {min_notional}"
                )
                continue

            side = "BUY" if diff_usd > 0 else "SELL"
            logger.info(
                f"Rebalance {coin}: {side} {trade_qty} @ ~${latest_price:,.2f} "
                f"(cur {current_weights.get(coin, 0):.1%} → tgt {target_w:.1%})"
            )

            try:
                await self.client.handle_place_order(
                    symbol=coin,
                    side=side,
                    quantity=trade_qty,
                    price=None  # Market order for rebalancing
                )
            except Exception as e:
                logger.error(f"Rebalance execution failed for {coin}: {e}")

        for i_id in intent_ids:
            await self.db.update_intent_status(i_id, "EXECUTED")

    # --------------------------------------------------------------------------
    # ALLOCATOR HELPERS
    # --------------------------------------------------------------------------

    def _calculate_target_weights(
        self,
        convictions: Dict[str, float],
        df_prices: pd.DataFrame,
        latest_prices: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Converts raw conviction scores into portfolio target weights.

        Logic:
          - Conviction in (0, 1]  → long position, size proportional to conviction
          - Conviction == 0       → no position (hold cash for this slot)
          - Conviction in [-1, 0) → no position (long-only; bearish = exit/avoid)

          Weight = conviction (clipped to [0,1]) * inverse_vol_scalar * max_single_coin
          Then:
            1. Cap each coin at max_single_coin
            2. Enforce cash floor (min_cash_fraction)
            3. Renormalise so crypto weights + cash floor <= 1.0
        """
        cfg = self.cfg

        # Long-only: negative convictions become zero (exit / don't enter)
        conviction_series = pd.Series(convictions).clip(lower=0.0)

        if not df_prices.empty:
            available_coins = [c for c in conviction_series.index if c in df_prices.columns]
            df_sub  = df_prices[available_coins]
            returns = df_sub.pct_change().dropna()

            if len(returns) >= 5:
                if cfg["ewm_span"]:
                    vol = returns.ewm(span=cfg["ewm_span"], min_periods=5).std().iloc[-1]
                else:
                    vol = returns.std()
                inverse_vol = 1.0 / (vol + 1e-9)
                risk_scalar = inverse_vol / (inverse_vol.mean() + 1e-9)
            else:
                risk_scalar = pd.Series(1.0, index=df_sub.columns)

            risk_scalar = risk_scalar.reindex(conviction_series.index).fillna(1.0)
        else:
            risk_scalar = pd.Series(1.0, index=conviction_series.index)

        raw_weights    = conviction_series * risk_scalar * cfg["max_single_coin"]
        raw_weights    = raw_weights.clip(upper=cfg["max_single_coin"])
        max_investable = 1.0 - cfg["min_cash_fraction"]
        total_raw      = raw_weights.sum()

        if total_raw > max_investable:
            raw_weights = raw_weights * (max_investable / total_raw)

        target = raw_weights.fillna(0.0).to_dict()

        logger.info(
            f"Target weights: { {k: f'{v:.1%}' for k, v in target.items()} } "
            f"| Cash floor: {cfg['min_cash_fraction']:.0%} "
            f"| Investable: {sum(target.values()):.1%}"
        )
        return target

    def _apply_sticky_logic(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        threshold = threshold or self.cfg["sticky_threshold"]
        trades    = {}

        all_coins = (
            set(target_weights.keys()) |
            set(k for k in current_weights.keys() if k != 'USD')
        )

        for coin in all_coins:
            target  = target_weights.get(coin, 0.0)
            current = current_weights.get(coin, 0.0)
            diff    = target - current

            if abs(diff) > threshold:
                trades[coin] = target
                logger.debug(
                    f"Sticky: {coin} queued | cur={current:.2%} tgt={target:.2%} diff={diff:+.2%}"
                )

        return trades

    # ==========================================================================
    # CONCURRENT EVENT LOOPS
    # ==========================================================================

    async def market_data_cycle(self):
        """Fetches prices synced to 5-minute clock boundaries."""
        while self.is_running:
            try:
                now = time.time()
                sleep_duration = 300 - (now % 300)
                logger.info(f"Data Cycle: sleeping {sleep_duration:.1f}s until next 5-min bar...")
                await asyncio.sleep(sleep_duration)
                if not self.is_running:
                    break
                await self.client.handle_get_ticker()
                logger.info("5-min prices updated on Data Bus.")
            except Exception as e:
                logger.error(f"Market data cycle error: {e}")
                await asyncio.sleep(5)

    async def execution_cycle(self):
        while self.is_running:
            try:
                # Sync to 5-min boundaries, but offset by 60s to give
                # strategies time to compute and submit after price update
                now = time.time()
                sleep_duration = 300 - (now % 300) + 60
                if sleep_duration > 300:
                    sleep_duration -= 300
                await asyncio.sleep(sleep_duration)
                if not self.is_running:
                    break
                await self.run_allocator()
            except Exception as e:
                logger.error(f"Execution cycle error: {e}")
                await asyncio.sleep(5)

    async def sync_cycle(self):
        """Keeps local balance and order state accurate."""
        while self.is_running:
            try:
                await asyncio.sleep(60.0)
                if not self.is_running:
                    break
                await self.client.handle_get_balance()
                await self.client.handle_query_order(pending_only=True)
                await self.db.prune_ticks(hours_to_keep=48)
            except Exception as e:
                logger.error(f"Sync cycle error: {e}")
                await asyncio.sleep(5)

    async def start_processes(self):
        """Kicks off all concurrent tasks."""
        logger.info("Running initial data fetch...")
        await self.client.handle_get_ticker()
        await self.client.handle_get_balance()
        self.tasks = [
            asyncio.create_task(self.market_data_cycle()),
            asyncio.create_task(self.execution_cycle()),
            asyncio.create_task(self.sync_cycle()),
        ]
        await asyncio.gather(*self.tasks)

    async def shutdown(self):
        logger.info("Shutting down bot...")
        self.is_running = False
        for task in self.tasks:
            task.cancel()
        await self.client.close()
        logger.info("Cleanup complete.")


async def main():
    bot = TradingBot()

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
        pass
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())