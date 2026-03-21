'''
MockBroker — simulates order execution and portfolio state during backtesting.

Mirrors the behaviour of RoostooClientV3 from the strategy/allocator perspective
but never touches the network. Fills market orders immediately at the bar's
closing price, applying the same commission structure as the real exchange.

@ MTL 21 March 2026
'''
import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("MockBroker")

# Match Roostoo fee schedule
TAKER_FEE = 0.001   # 0.10% for market orders
MAKER_FEE = 0.0005  # 0.05% for limit orders (not used in backtest — all market)


class MockBroker:
    def __init__(self, initial_cash: float, price_data: Dict[str, pd.DataFrame]):
        """
        Args:
            initial_cash:  Starting USD.
            price_data:    Same dict passed to BacktestEngine — used to look up
                           prices for portfolio valuation.
        """
        self.price_data   = price_data
        self.initial_cash = initial_cash

        # Flat balance dict: { 'USD': float, 'BTC': float, ... }
        self.balance: Dict[str, float] = {"USD": initial_cash}

        # Full trade log
        self.trades: List[Dict] = []

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: int
    ) -> Optional[Dict]:
        """
        Simulate a market order fill at `price`.

        Returns the fill dict, or None if the order was rejected
        (insufficient balance, zero qty, etc).
        """
        if quantity <= 0 or price <= 0:
            return None

        fee      = quantity * price * TAKER_FEE
        notional = quantity * price

        if side.upper() == "BUY":
            total_cost = notional + fee
            if self.balance.get("USD", 0.0) < total_cost:
                # Partial fill: buy as much as we can afford
                affordable = self.balance.get("USD", 0.0) / (price * (1 + TAKER_FEE))
                if affordable < quantity * 0.01:  # less than 1% of intended — skip
                    logger.debug(f"BUY {symbol} rejected: insufficient USD")
                    return None
                quantity   = affordable
                fee        = quantity * price * TAKER_FEE
                notional   = quantity * price
                total_cost = notional + fee

            self.balance["USD"]               = self.balance.get("USD", 0.0) - total_cost
            self.balance[symbol]              = self.balance.get(symbol, 0.0) + quantity

        elif side.upper() == "SELL":
            available = self.balance.get(symbol, 0.0)
            if available <= 0:
                return None
            # Never sell more than we hold
            quantity  = min(quantity, available)
            fee       = quantity * price * TAKER_FEE
            notional  = quantity * price
            proceeds  = notional - fee

            self.balance[symbol]  = self.balance.get(symbol, 0.0) - quantity
            self.balance["USD"]   = self.balance.get("USD", 0.0) + proceeds

        else:
            return None

        fill = {
            "timestamp": timestamp,
            "symbol":    symbol,
            "side":      side.upper(),
            "quantity":  round(quantity, 8),
            "price":     price,
            "notional":  round(notional, 4),
            "fee":       round(fee, 4),
        }
        self.trades.append(fill)
        logger.debug(
            f"Fill: {side.upper()} {quantity:.6f} {symbol} @ {price:.2f} "
            f"| notional=${notional:.0f} fee=${fee:.2f}"
        )
        return fill

    def get_portfolio_value(self, latest_prices: Dict[str, Dict]) -> float:
        """
        Compute total portfolio value in USD at current prices.
        latest_prices: same format as db.get_latest_price_batch()
        """
        total = self.balance.get("USD", 0.0)
        for asset, qty in self.balance.items():
            if asset == "USD" or qty <= 0:
                continue
            price = float(latest_prices.get(f"{asset}/USD", {}).get("last_price", 0.0))
            total += qty * price
        return total

    def get_trade_stats(self) -> Dict:
        """Summary statistics of all trades executed during the backtest."""
        if not self.trades:
            return {
                "total_trades": 0,
                "total_fees_usd": 0.0,
                "avg_trade_notional": 0.0,
            }

        df = pd.DataFrame(self.trades)
        return {
            "total_trades":        len(df),
            "total_fees_usd":      round(df["fee"].sum(), 2),
            "avg_trade_notional":  round(df["notional"].mean(), 2),
            "total_buy_trades":    int((df["side"] == "BUY").sum()),
            "total_sell_trades":   int((df["side"] == "SELL").sum()),
        }
