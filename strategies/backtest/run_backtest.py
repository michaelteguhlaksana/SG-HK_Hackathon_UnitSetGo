'''
run_backtest.py — entry point for backtesting strategies.

Accepts a single pivoted CSV where:
  - rows    = timestamps
  - columns = coin symbols (e.g. BTCUSDT, ETHUSDT, ...)
  - values  = close prices

@ MTL 21 March 2026
'''
import asyncio
import logging
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import BacktestEngine
from mock_broker import MockBroker
from trading_system.db.db_manager import DatabaseManager
from strategies import (
    BollingerReversion,
    MACDStrategy,
    VWAPReversion,
    CrossSectionalMomentum,
    AdaptiveRSI,
)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Runner")


# ==============================================================================
# DATA LOADING
# ==============================================================================

# Binance symbol -> Roostoo pair format
# Extend this if you have symbols not listed here
SYMBOL_MAP = {
    "BTCUSDT":   "BTC/USD",
    "ETHUSDT":   "ETH/USD",
    "SOLUSDT":   "SOL/USD",
    "XRPUSDT":   "XRP/USD",
    "BNBUSDT":   "BNB/USD",
    "DOGEUSDT":  "DOGE/USD",
    "ZECUSDT":   "ZEC/USD",
    "SUIUSDT":   "SUI/USD",
    "ASTERUSDT": "ASTER/USD",
    "ADAUSDT":   "ADA/USD",
    "PEPEUSDT":  "PEPE/USD",
    "AVAXUSDT":  "AVAX/USD",
    "LINKUSDT":  "LINK/USD",
    "ENAUSDT":   "ENA/USD",
    "PUMPUSDT":  "PUMP/USD",
    "LTCUSDT":   "LTC/USD",
    "TRXUSDT":   "TRX/USD",
    "XPLUSDT":   "XPL/USD",
    "PAXGUSDT":  "PAXG/USD",
    "NEARUSDT":  "NEAR/USD",
}


def load_pivoted_csv(
    csv_path: str,
    symbols: list = None,
    resample_tf: str = None,
) -> dict:
    """
    Load a pivoted close-price CSV into the dict format the engine expects.

    Args:
        csv_path:    Path to CSV. Rows=timestamps, columns=symbols.
        symbols:     Optional list of Binance symbols to include, e.g.
                     ['BTCUSDT', 'ETHUSDT']. If None, loads all columns.
        resample_tf: Optional pandas offset string to resample, e.g. '15min',
                     '1h'. If None, uses the native bar frequency (5min).

    Returns:
        { 'BTC/USD': DataFrame(timestamp, open, high, low, close, volume), ... }
    """
    # --- Load ---
    df = pd.read_csv(csv_path)

    # Handle the index/timestamp column however it arrives
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    elif 'index' in df.columns:
        df['index'] = pd.to_datetime(df['index'], utc=True)
        df = df.set_index('index')
    else:
        # Assume first column is the timestamp
        df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        df = df.iloc[:, 1:]

    df = df.sort_index().dropna(how='all')

    # --- Filter symbols ---
    if symbols:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            logger.warning(f"Symbols not found in CSV, skipping: {missing}")
        df = df[[s for s in symbols if s in df.columns]]

    # --- Resample ---
    if resample_tf:
        df = df.resample(resample_tf).last().dropna(how='all')
        logger.warning(f"Resampled to {resample_tf}: {len(df)} bars")

    # --- Convert to Unix ms integer timestamps ---
    timestamps_ms = (df.index.astype('int64') // 1_000_000).tolist()

    # --- Build per-pair DataFrames ---
    price_data = {}
    for col in df.columns:
        pair = SYMBOL_MAP.get(col)
        if pair is None:
            # Auto-derive: strip USDT/BUSD suffix
            base = col.replace('USDT', '').replace('BUSD', '')
            pair = f"{base}/USD"
            logger.warning(f"'{col}' not in SYMBOL_MAP, mapped to '{pair}'")

        prices = df[col].values.tolist()

        # We only have close prices — use close for open/high/low too.
        # Engine and strategies only use 'close' and 'volume'.
        pair_df = pd.DataFrame({
            "timestamp": timestamps_ms,
            "open":      prices,
            "high":      prices,
            "low":       prices,
            "close":     prices,
            "volume":    [0.0] * len(prices),
        })
        price_data[pair] = pair_df

    logger.warning(
        f"Loaded {len(price_data)} pairs, {len(df)} bars each "
        f"({df.index[0]} -> {df.index[-1]})"
    )
    return price_data


def load_sample_data() -> dict:
    """Synthetic GBM data for smoke-testing when no CSV is available."""
    np.random.seed(42)
    n     = 2000
    start = 1_700_000_000_000
    step  = 300_000

    def make_series(start_price, drift=0.00002, vol=0.002):
        returns = np.random.normal(drift, vol, n)
        prices  = start_price * np.exp(np.cumsum(returns))
        ts      = [start + i * step for i in range(n)]
        return pd.DataFrame({
            "timestamp": ts,
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [0.0] * n,
        })

    return {
        "BTC/USD": make_series(42000),
        "ETH/USD": make_series(2500, drift=0.00003),
        "BNB/USD": make_series(300, drift=-0.00001, vol=0.003),
    }


# ==============================================================================
# RESULTS DISPLAY
# ==============================================================================

def print_results(results: dict, strategy_names: list):
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Strategies:       {', '.join(strategy_names)}")
    print(f"  Initial value:    ${results['initial_value']:>12,.2f}")
    print(f"  Final value:      ${results['final_value']:>12,.2f}")
    print(f"  Total return:     {results['total_return']:>+11.3f}%")
    print(f"  Max drawdown:     {results['max_drawdown']:>+11.3f}%")
    print("-" * 55)
    print(f"  Sharpe ratio:     {results['sharpe']:>12.4f}")
    print(f"  Sortino ratio:    {results['sortino']:>12.4f}")
    print(f"  Calmar ratio:     {results['calmar']:>12.4f}")
    print(f"  Composite score:  {results['composite']:>12.4f}  <- competition metric")
    print("-" * 55)
    print(f"  Total bars:       {results['total_bars']:>12,}")
    print(f"  Total trades:     {results['total_trades']:>12,}")
    print(f"  Buys / Sells:     {results.get('total_buy_trades', 0):>6,} / {results.get('total_sell_trades', 0):<6,}")
    print(f"  Total fees paid:  ${results['total_fees_usd']:>11,.2f}")
    print(f"  Avg trade size:   ${results['avg_trade_notional']:>11,.2f}")
    print("=" * 55 + "\n")


def plot_equity_curve(results: dict):
    try:
        import matplotlib.pyplot as plt
        curve = results["equity_curve"]
        bars  = [e["bar"]   for e in curve]
        vals  = [e["value"] for e in curve]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(bars, vals, linewidth=1.2, color="#534AB7")
        ax.axhline(results["initial_value"], color="#888", linewidth=0.8, linestyle="--")
        ax.set_title("Portfolio equity curve")
        ax.set_xlabel("Bar number")
        ax.set_ylabel("Portfolio value (USD)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()
        os.makedirs("backtest", exist_ok=True)
        plt.savefig("backtest/equity_curve.png", dpi=150)
        print("  Equity curve saved to backtest/equity_curve.png")
        plt.close()
    except ImportError:
        print("  (matplotlib not installed - skipping equity curve plot)")


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    csv_path = "./data/close_5m.csv"

    if os.path.exists(csv_path):

        # All 20 symbols at native 5min resolution
        price_data = load_pivoted_csv(csv_path)

        # Subset to specific symbols:
        # price_data = load_pivoted_csv(csv_path, symbols=[
        #     'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        # ])

        # Resample to a slower timeframe (strategies pick this up automatically):
        # price_data = load_pivoted_csv(csv_path, resample_tf='1h')
        # price_data = load_pivoted_csv(csv_path, resample_tf='15min')

    else:
        print(f"CSV not found at '{csv_path}' - using synthetic data.\n")
        price_data = load_sample_data()

    symbols = [pair.replace("/USD", "") for pair in price_data.keys()]
    print(f"Running on {len(symbols)} pairs: {symbols}\n")

    # ------------------------------------------------------------------
    # 2. Shared in-memory DB
    # ------------------------------------------------------------------
    bt_db = DatabaseManager(":memory:")

    # ------------------------------------------------------------------
    # 3. Strategies
    #    window=6 on 1h-resampled data = 6-bar (6h) lookback,
    #    matching your best backtest result.
    # ------------------------------------------------------------------
    strategies = [
        MACDStrategy("MACD_1h",  symbols, window=6,  db_path=":memory:"),
        CrossSectionalMomentum("XSMom", symbols, window=24, db_path=":memory:"),
        BollingerReversion("Bollinger", symbols, window=12, db_path=":memory:"),
        AdaptiveRSI("AdaptRSI", symbols, rsi_window=14, percentile_window=100, db_path=":memory:"),
        # VWAPReversion("VWAP", symbols, window=14, db_path=":memory:"),
    ]
    for strat in strategies:
        strat.db = bt_db

    # ------------------------------------------------------------------
    # 4. Allocator config
    # ------------------------------------------------------------------
    cfg = {
        "max_single_coin":   0.30,
        "min_cash_fraction": 0.20,
        "sticky_threshold":  0.05,
        "vol_lookback":      288,
        "min_trade_usd":     200.0,
        "ewm_span":          48,
    }

    # ------------------------------------------------------------------
    # 5. Engine
    # ------------------------------------------------------------------
    engine        = BacktestEngine(price_data=price_data, initial_cash=1_000_000.0, cfg=cfg)
    engine.db     = bt_db
    engine.broker = MockBroker(1_000_000.0, price_data)

    warmup_bars = max(strat.lookback for strat in strategies)
    print(f"Warmup: {warmup_bars} bars ({warmup_bars * 5 / 60:.1f} hours at 5min)\n")

    results = await engine.run(strategies, warmup_bars=warmup_bars)

    # ------------------------------------------------------------------
    # 6. Output
    # ------------------------------------------------------------------
    print_results(results, [s.name for s in strategies])
    plot_equity_curve(results)


if __name__ == "__main__":
    asyncio.run(main())