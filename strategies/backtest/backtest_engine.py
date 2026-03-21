'''
BacktestEngine — replays historical OHLCV data bar-by-bar into a
real (but isolated) SQLite database, then runs your strategies and
allocator against it tick-by-tick.

The strategies and allocator run completely unmodified.
Only two things are swapped out vs production:
  1. The DB is initialised fresh for each run (in-memory or a temp file).
  2. Order execution is routed to MockBroker instead of RoostooClientV3.

Usage:
    python backtest/run_backtest.py

@ MTL 21 March 2026
'''
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from trading_system.db.db_manager import DatabaseManager
from backtest.mock_broker import MockBroker

logger = logging.getLogger("Backtest")


class BacktestEngine:
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        initial_cash: float = 1_000_000.0,
        db_path: str = ":memory:",
        cfg: Optional[Dict] = None
    ):
        """
        Args:
            price_data:    { 'BTC/USD': DataFrame(columns=['timestamp','open','high','low','close','volume']) }
                           Timestamps should be Unix ms integers, sorted ascending.
            initial_cash:  Starting USD balance (matches competition: $1,000,000).
            db_path:       ':memory:' for isolated runs, or a file path to inspect
                           the DB after the run.
            cfg:           Allocator config dict (same keys as TradingBot.cfg).
                           Defaults match the live system if not provided.
        """
        self.price_data    = price_data
        self.initial_cash  = initial_cash
        self.db            = DatabaseManager(db_path)
        self.broker        = MockBroker(initial_cash, price_data)
        self.cfg           = cfg or {
            "max_single_coin":   0.30,
            "min_cash_fraction": 0.20,
            "sticky_threshold":  0.05,
            "vol_lookback":      288,
            "min_trade_usd":     200.0,
            "ewm_span":          48,
        }
        # Align all pairs to a common timeline
        self._bars = self._build_bar_index()

    def _build_bar_index(self) -> List[int]:
        """Returns sorted list of all unique timestamps across all pairs."""
        all_ts = set()
        for df in self.price_data.values():
            all_ts.update(df['timestamp'].tolist())
        return sorted(all_ts)

    async def _inject_tick(self, ts: int):
        """Write one bar's closing price into the DB for all pairs at timestamp ts."""
        batch = []
        for pair, df in self.price_data.items():
            row = df[df['timestamp'] == ts]
            if row.empty:
                continue
            batch.append({
                "pair":   pair,
                "price":  float(row['close'].iloc[0]),
                "volume": float(row['volume'].iloc[0]) if 'volume' in row.columns else 0.0,
            })
        if batch:
            await self.db.update_tickers_batch(batch)

    async def run(self, strategies: list, warmup_bars: int = 300) -> Dict:
        """
        Run the full backtest.

        Args:
            strategies:   List of instantiated BaseStrategy subclasses.
                          Pass them already constructed — the engine just calls on_tick.
            warmup_bars:  Number of bars to feed into the DB before strategies
                          start submitting signals. Needed so rolling indicators
                          have enough history to be valid.

        Returns:
            results dict with equity curve and risk metrics.
        """
        logger.info(f"Backtest starting: {len(self._bars)} bars, {len(strategies)} strategies")
        logger.info(f"Pairs: {list(self.price_data.keys())}")
        logger.info(f"Warmup: {warmup_bars} bars")

        await self.db.init_db()

        equity_curve = []
        bar_count    = 0

        for ts in self._bars:
            bar_count += 1

            # 1. Inject this bar's prices into the DB
            await self._inject_tick(ts)

            # 2. Skip strategy execution during warmup
            if bar_count <= warmup_bars:
                continue

            # 3. Run all strategies (they write conviction intents to DB)
            market_state = await self.db.get_latest_price_batch()
            history      = await self.db.get_tick_history_batch(
                pairs=list(self.price_data.keys()),
                limit=self.cfg["vol_lookback"]
            )
            for strat in strategies:
                try:
                    await strat.on_tick(market_state, history)
                except Exception as e:
                    logger.error(f"Strategy {strat.name} error at bar {bar_count}: {e}")

            # 4. Run allocator (reads intents from DB, calls broker instead of real API)
            await self._run_allocator(ts)

            # 5. Record portfolio value
            portfolio_value = self.broker.get_portfolio_value(market_state)
            equity_curve.append({
                "timestamp": ts,
                "value":     portfolio_value,
                "bar":       bar_count,
            })

        results = self._compute_metrics(equity_curve)
        results["equity_curve"] = equity_curve
        return results

    # --------------------------------------------------------------------------
    # ALLOCATOR (copied from TradingBot, broker calls substituted)
    # --------------------------------------------------------------------------

    async def _run_allocator(self, current_ts: int):
        pending_intents = await self.db.get_pending_intents()
        if not pending_intents:
            return

        pooled: Dict[str, list] = {}
        intent_ids = []
        for intent in pending_intents:
            sym = intent['symbol']
            pooled.setdefault(sym, []).append(float(intent['conviction']))
            intent_ids.append(intent['id'])

        avg_convictions = {
            sym: sum(scores) / len(scores)
            for sym, scores in pooled.items()
        }

        for i_id in intent_ids:
            await self.db.update_intent_status(i_id, "PROCESSING")

        pairs_with_usd = [f"{sym}/USD" for sym in avg_convictions.keys()]
        history        = await self.db.get_tick_history_batch(
            pairs=pairs_with_usd, limit=self.cfg["vol_lookback"]
        )
        latest_prices  = await self.db.get_latest_price_batch()

        # Current portfolio state from broker (not from Roostoo)
        balance          = self.broker.balance   # { 'USD': float, 'BTC': float, ... }
        holdings_usd     = {}
        total_portfolio  = 0.0
        for asset, qty in balance.items():
            if asset == 'USD':
                val = qty
            else:
                price = float(latest_prices.get(f"{asset}/USD", {}).get('last_price', 0.0))
                val   = qty * price
            holdings_usd[asset]  = val
            total_portfolio     += val

        if total_portfolio <= 0:
            for i_id in intent_ids:
                await self.db.update_intent_status(i_id, "REJECTED")
            return

        current_weights = {k: v / total_portfolio for k, v in holdings_usd.items()}

        # Build price DataFrame (same logic as live allocator)
        price_series = {}
        for pair, ticks in history.items():
            coin = pair.replace("/USD", "")
            if len(ticks) >= 2:
                price_series[coin] = [t['price'] for t in ticks]
        df_prices = pd.DataFrame(price_series) if price_series else pd.DataFrame()

        target_weights = self._calculate_target_weights(avg_convictions, df_prices)
        trades         = self._apply_sticky_logic(target_weights, current_weights)

        # Sells first, then buys
        sells = {c: w for c, w in trades.items() if w < current_weights.get(c, 0.0)}
        buys  = {c: w for c, w in trades.items() if w > current_weights.get(c, 0.0)}

        for coin, target_w in {**sells, **buys}.items():
            current_usd  = holdings_usd.get(coin, 0.0)
            target_usd   = target_w * total_portfolio
            diff_usd     = target_usd - current_usd
            latest_price = float(latest_prices.get(f"{coin}/USD", {}).get('last_price', 0.0))

            if latest_price <= 0 or abs(diff_usd) < self.cfg["min_trade_usd"]:
                continue

            side      = "BUY" if diff_usd > 0 else "SELL"
            trade_qty = abs(diff_usd) / latest_price

            # Route to MockBroker instead of RoostooClientV3
            self.broker.execute_order(
                symbol=coin,
                side=side,
                quantity=trade_qty,
                price=latest_price,
                timestamp=current_ts
            )

        for i_id in intent_ids:
            await self.db.update_intent_status(i_id, "EXECUTED")

    # --------------------------------------------------------------------------
    # ALLOCATOR HELPERS (identical to live system)
    # --------------------------------------------------------------------------

    def _calculate_target_weights(
        self,
        convictions: Dict[str, float],
        df_prices: pd.DataFrame,
    ) -> Dict[str, float]:
        cfg              = self.cfg
        conviction_series = pd.Series(convictions).clip(lower=0.0)

        if not df_prices.empty:
            available = [c for c in conviction_series.index if c in df_prices.columns]
            df_sub    = df_prices[available]
            returns   = df_sub.pct_change().dropna()
            if len(returns) >= 5:
                if cfg["ewm_span"]:
                    vol = returns.ewm(span=cfg["ewm_span"], min_periods=5).std().iloc[-1]
                else:
                    vol = returns.std()
                inverse_vol  = 1.0 / (vol + 1e-9)
                risk_scalar  = inverse_vol / (inverse_vol.mean() + 1e-9)
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

        return raw_weights.fillna(0.0).to_dict()

    def _apply_sticky_logic(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        threshold  = self.cfg["sticky_threshold"]
        trades     = {}
        all_coins  = (
            set(target_weights.keys()) |
            set(k for k in current_weights.keys() if k != 'USD')
        )
        for coin in all_coins:
            target  = target_weights.get(coin, 0.0)
            current = current_weights.get(coin, 0.0)
            if abs(target - current) > threshold:
                trades[coin] = target
        return trades

    # --------------------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------------------

    def _compute_metrics(self, equity_curve: List[Dict]) -> Dict:
        if len(equity_curve) < 2:
            return {"error": "Not enough bars to compute metrics."}

        values = pd.Series([e["value"] for e in equity_curve])
        rets   = values.pct_change().dropna()

        total_return = (values.iloc[-1] / values.iloc[0]) - 1.0
        mean_ret     = rets.mean()
        std_ret      = rets.std()

        # Sharpe (no risk-free rate — matches competition formula)
        sharpe = mean_ret / (std_ret + 1e-9)

        # Sortino (downside deviation only)
        downside = rets[rets < 0]
        sortino  = mean_ret / (downside.std() + 1e-9)

        # Calmar (return / max drawdown)
        rolling_max  = values.cummax()
        drawdown     = (values - rolling_max) / (rolling_max + 1e-9)
        max_drawdown = drawdown.min()  # most negative value
        calmar       = total_return / (abs(max_drawdown) + 1e-9)

        # Competition composite score
        composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

        # Trade stats from broker
        trade_stats = self.broker.get_trade_stats()

        metrics = {
            "total_return":   round(total_return * 100, 3),
            "sharpe":         round(sharpe, 4),
            "sortino":        round(sortino, 4),
            "calmar":         round(calmar, 4),
            "composite":      round(composite, 4),
            "max_drawdown":   round(max_drawdown * 100, 3),
            "final_value":    round(values.iloc[-1], 2),
            "initial_value":  round(values.iloc[0], 2),
            "total_bars":     len(equity_curve),
            **trade_stats,
        }
        return metrics
