'''
run_backtest.py — parallel grid-search backtester.

Architecture:
  - Main process loads the CSV once and builds the price_data dict.
  - Each worker process receives a single (strategy_config, allocator_cfg) job,
    runs a full independent backtest with its own temp SQLite file, and returns
    the metrics dict.
  - Workers use asyncio internally (each worker has its own event loop).
  - ProcessPoolExecutor parallelises across CPU cores.

Why not thread-based: asyncio + aiosqlite is already async, but the GIL means
threads won't give true parallelism for CPU-bound work. Processes do.

@ MTL 21 March 2026
'''
import asyncio
import logging
import os
import sys
import tempfile
import time
import itertools
import concurrent.futures
from typing import Dict, List, Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works when run from SGHK_Hackathon/ root or from strategies/
# ---------------------------------------------------------------------------
_FILE_DIR    = os.path.dirname(os.path.abspath(__file__))
_STRAT_ROOT  = os.path.join(_FILE_DIR, '..')         # strategies/
_REPO_ROOT   = os.path.join(_FILE_DIR, '..', '..')   # SGHK_Hackathon/
for p in [_REPO_ROOT, _STRAT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, os.path.normpath(p))

from backtest.backtest_engine import BacktestEngine
from backtest.mock_broker import MockBroker
from trading_system.db.db_manager import DatabaseManager
from strategies.technical_indicator import (
    BollingerReversion,
    MACDStrategy,
    VWAPReversion,
    CrossSectionalMomentum,
    AdaptiveRSI,
)

# ---------------------------------------------------------------------------
# Logging — workers log to stdout with their PID so output is distinguishable
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s][%(process)d] %(name)s: %(message)s'
)
logger = logging.getLogger("Runner")


# ==============================================================================
# SYMBOL MAP
# ==============================================================================
SYMBOL_MAP = {
    "BTCUSDT":   "BTC/USD",  "ETHUSDT":   "ETH/USD",  "SOLUSDT":   "SOL/USD",
    "XRPUSDT":   "XRP/USD",  "BNBUSDT":   "BNB/USD",  "DOGEUSDT":  "DOGE/USD",
    "ZECUSDT":   "ZEC/USD",  "SUIUSDT":   "SUI/USD",  "ASTERUSDT": "ASTER/USD",
    "ADAUSDT":   "ADA/USD",  "PEPEUSDT":  "PEPE/USD", "AVAXUSDT":  "AVAX/USD",
    "LINKUSDT":  "LINK/USD", "ENAUSDT":   "ENA/USD",  "PUMPUSDT":  "PUMP/USD",
    "LTCUSDT":   "LTC/USD",  "TRXUSDT":   "TRX/USD",  "XPLUSDT":   "XPL/USD",
    "PAXGUSDT":  "PAXG/USD", "NEARUSDT":  "NEAR/USD",
}


# ==============================================================================
# DATA LOADING  (runs once in main process only)
# ==============================================================================
def load_pivoted_csv(
    csv_path: str,
    symbols: List[str] = None,
    resample_tf: str = None,
    start_date: str = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load pivoted close-price CSV → { 'BTC/USD': DataFrame, ... }
    start_date: e.g. '2023-01-01' to slice for faster iteration.
    """
    df = pd.read_csv(csv_path)

    # Normalise timestamp index
    ts_col = next((c for c in ['timestamp', 'index'] if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        df = df.iloc[:, 1:]

    df = df.sort_index().dropna(how='all')

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz='UTC')]

    if symbols:
        df = df[[s for s in symbols if s in df.columns]]

    if resample_tf:
        df = df.resample(resample_tf).last().dropna(how='all')

    timestamps_ms = (df.index.astype('int64') // 1_000_000).tolist()

    price_data = {}
    for col in df.columns:
        pair   = SYMBOL_MAP.get(col, f"{col.replace('USDT','').replace('BUSD','')}/USD")
        prices = df[col].values.tolist()
        pair_df = pd.DataFrame({
            "timestamp": timestamps_ms,
            "open": prices, "high": prices,
            "low":  prices, "close": prices,
            "volume": [0.0] * len(prices),
        }).dropna(subset=["close"])
        price_data[pair] = pair_df

    logger.warning(
        f"Loaded {len(price_data)} pairs × {len(df)} bars "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )
    return price_data


def load_sample_data() -> Dict[str, pd.DataFrame]:
    np.random.seed(42)
    n, start, step = 2000, 1_700_000_000_000, 300_000
    def gbm(p0, drift=2e-5, vol=2e-3):
        px = p0 * np.exp(np.cumsum(np.random.normal(drift, vol, n)))
        ts = [start + i * step for i in range(n)]
        return pd.DataFrame({"timestamp":ts,"open":px,"high":px,"low":px,"close":px,"volume":0.0})
    return {"BTC/USD": gbm(42000), "ETH/USD": gbm(2500,3e-5), "BNB/USD": gbm(300,-1e-5,3e-3)}


# ==============================================================================
# WORKER  (runs in a subprocess — must be importable at module level)
# ==============================================================================
def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    A single backtest job. Receives everything it needs as a plain dict
    (no lambdas, no DB connections — those can't be pickled across processes).

    job keys:
        price_data      dict of DataFrames (pickled by multiprocessing)
        strategy_cfgs   list of dicts describing which strategies to run
        allocator_cfg   dict
        initial_cash    float
        label           str   human-readable name for this run
    """
    # Each worker needs its own event loop
    return asyncio.run(_worker_async(job))


async def _worker_async(job: Dict[str, Any]) -> Dict[str, Any]:
    price_data    = job["price_data"]
    strategy_cfgs = job["strategy_cfgs"]
    allocator_cfg = job["allocator_cfg"]
    label         = job["label"]

    # Temp file DB — each worker is fully isolated
    tmp      = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path  = tmp.name
    tmp.close()

    try:
        bt_db = DatabaseManager(db_path)
        await bt_db.init_db()

        symbols = [pair.replace("/USD", "") for pair in price_data.keys()]

        # Build strategy instances from config dicts
        strategies = []
        for cfg in strategy_cfgs:
            strat = _build_strategy(cfg, symbols, db_path)
            strat.db = bt_db
            strategies.append(strat)
        print("Starting backtest for:", strategy_cfgs)
        engine        = BacktestEngine(price_data=price_data, initial_cash=job["initial_cash"], cfg=allocator_cfg)
        engine.db     = bt_db
        engine.broker = MockBroker(job["initial_cash"], price_data)

        warmup = max(s.lookback for s in strategies)
        t0     = time.time()
        result = await engine.run(strategies, warmup_bars=warmup)
        elapsed = time.time() - t0
        print("Finished backtest for:", strategy_cfgs)
        result["label"]       = label
        result["elapsed_s"]   = round(elapsed, 1)
        result["strategy_cfgs"] = strategy_cfgs
        return result

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def _build_strategy(cfg: Dict, symbols: List[str], db_path: str):
    """Instantiate a strategy from a plain config dict."""
    kind        = cfg["kind"]
    name        = cfg.get("name", kind)
    resample_tf = cfg.get("resample_tf", None)

    if kind == "MACD":
        return MACDStrategy(
            name, symbols,
            window=cfg.get("window", 26),
            db_path=db_path,
            resample_tf=resample_tf,
        )
    elif kind == "Bollinger":
        return BollingerReversion(
            name, symbols,
            window=cfg.get("window", 20),
            db_path=db_path,
            resample_tf=resample_tf,
        )
    elif kind == "VWAP":
        return VWAPReversion(
            name, symbols,
            window=cfg.get("window", 14),
            db_path=db_path,
            resample_tf=resample_tf,
        )
    elif kind == "XSMom":
        return CrossSectionalMomentum(
            name, symbols,
            window=cfg.get("window", 24),
            db_path=db_path,
            resample_tf=resample_tf,
        )
    elif kind == "AdaptRSI":
        return AdaptiveRSI(
            name, symbols,
            rsi_window=cfg.get("rsi_window", 14),
            percentile_window=cfg.get("percentile_window", 100),
            db_path=db_path,
            resample_tf=resample_tf,
        )
    else:
        raise ValueError(f"Unknown strategy kind: {kind}")


# ==============================================================================
# JOB BUILDER  (define your grid here)
# ==============================================================================
def build_jobs(price_data: Dict, allocator_cfg: Dict, initial_cash: float) -> List[Dict]:
    """
    Returns a list of job dicts to run in parallel.
    Each job is one complete backtest — a specific combination of strategies,
    parameters, and timeframes. Edit this function to define your search space.

    resample_tf in a strategy config resamples the DB ticks before on_tick
    is called. window is always in RESAMPLED bars:
        window=6, resample_tf='1h'   → 6-hour lookback
        window=6, resample_tf='15min' → 90-minute lookback
    """
    jobs = []

    def job(label, *strategy_cfgs, alloc_cfg=None):
        return {
            "price_data":    price_data,
            "allocator_cfg": alloc_cfg or allocator_cfg,
            "initial_cash":  initial_cash,
            "label":         label,
            "strategy_cfgs": list(strategy_cfgs),
        }

    # --------------------------------------------------------------------------
    # Grid 1: MACD — window × timeframe sweep
    # Replicates your standalone backtest results and extends them.
    # --------------------------------------------------------------------------
    for tf, windows in [
        ('5min',  [6, 12, 24, 48]),
        ('15min', [6, 12, 24, 48]),
        ('1h',    [6, 12, 24]),
    ]:
        for w in windows:
            name = f"MACD_w{w}_{tf}"
            jobs.append(job(
                name,
                {"kind": "MACD", "name": name, "window": w, "resample_tf": tf},
            ))

    # --------------------------------------------------------------------------
    # Grid 2: Bollinger — window × timeframe sweep
    # --------------------------------------------------------------------------
    for tf, windows in [
        ('15min', [12, 24, 48]),
        ('1h',    [6, 12, 24]),
    ]:
        for w in windows:
            name = f"Boll_w{w}_{tf}"
            jobs.append(job(
                name,
                {"kind": "Bollinger", "name": name, "window": w, "resample_tf": tf},
            ))

    # --------------------------------------------------------------------------
    # Grid 3: XSMom — window × timeframe sweep
    # --------------------------------------------------------------------------
    for tf, windows in [
        ('15min', [6, 12, 24]),
        ('1h',    [6, 12, 24]),
    ]:
        for w in windows:
            name = f"XSMom_w{w}_{tf}"
            jobs.append(job(
                name,
                {"kind": "XSMom", "name": name, "window": w, "resample_tf": tf},
            ))

    # --------------------------------------------------------------------------
    # Grid 4: Best known combo from your prior backtest + timeframe variants
    # MACD 15min (fast=24, slow=18 → window≈24) was your raw-return winner.
    # --------------------------------------------------------------------------
    for mw, mtf, xw, xtf in [
        (24, '15min', 24, '15min'),   # your prior best
        (6,  '1h',   24, '15min'),    # MACD slower + XSMom faster
        (6,  '1h',   6,  '1h'),       # both on 1h
        (12, '15min', 12, '15min'),
    ]:
        label = f"MACD_w{mw}_{mtf}+XSMom_w{xw}_{xtf}"
        jobs.append(job(
            label,
            {"kind": "MACD",  "name": f"MACD_{mw}_{mtf}",  "window": mw, "resample_tf": mtf},
            {"kind": "XSMom", "name": f"XSMom_{xw}_{xtf}", "window": xw, "resample_tf": xtf},
        ))

    # --------------------------------------------------------------------------
    # Grid 5: Allocator sticky threshold sweep on best single strategy
    # --------------------------------------------------------------------------
    for thresh in [0.02, 0.05, 0.10]:
        cfg = {**allocator_cfg, "sticky_threshold": thresh}
        jobs.append(job(
            f"MACD_w24_15min_sticky{thresh}",
            {"kind": "MACD", "name": "MACD_24_15min", "window": 24, "resample_tf": "15min"},
            alloc_cfg=cfg,
        ))

    return jobs


# ==============================================================================
# RESULTS
# ==============================================================================
def print_summary(results: List[Dict]):
    """Print a ranked summary table sorted by composite score."""
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r.get("composite", -999), reverse=True)

    col = "{:<35} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>6}"
    hdr = col.format("Label", "Return%", "Sharpe", "Sortino", "Calmar", "Composite", "Trades", "Time(s)")
    print("\n" + "=" * len(hdr))
    print("  GRID SEARCH RESULTS  (ranked by composite score)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in valid:
        print(col.format(
            r["label"][:35],
            f"{r['total_return']:+.2f}",
            f"{r['sharpe']:.4f}",
            f"{r['sortino']:.4f}",
            f"{r['calmar']:.4f}",
            f"{r['composite']:.4f}",
            r["total_trades"],
            r["elapsed_s"],
        ))

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n  {len(errors)} jobs failed:")
        for r in errors:
            print(f"    [{r.get('label','?')}] {r['error']}")

    print("=" * len(hdr))

    # Save full results to CSV
    out_path = os.path.join(_FILE_DIR, "grid_results.csv")
    rows = []
    for r in valid:
        rows.append({
            "label":         r["label"],
            "total_return":  r["total_return"],
            "sharpe":        r["sharpe"],
            "sortino":       r["sortino"],
            "calmar":        r["calmar"],
            "composite":     r["composite"],
            "max_drawdown":  r["max_drawdown"],
            "total_trades":  r["total_trades"],
            "total_fees":    r["total_fees_usd"],
            "elapsed_s":     r["elapsed_s"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Full results saved to {out_path}\n")


def plot_top_equity_curves(results: List[Dict], top_n: int = 5):
    try:
        import matplotlib.pyplot as plt
        valid = sorted(
            [r for r in results if "equity_curve" in r],
            key=lambda r: r.get("composite", -999),
            reverse=True
        )[:top_n]

        fig, ax = plt.subplots(figsize=(14, 5))
        colors  = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#D4537E"]
        for i, r in enumerate(valid):
            curve = r["equity_curve"]
            bars  = [e["bar"]   for e in curve]
            vals  = [e["value"] for e in curve]
            ax.plot(bars, vals, linewidth=1.2, color=colors[i % len(colors)],
                    label=f"{r['label']} (composite={r['composite']:.3f})")

        ax.axhline(1_000_000, color="#aaa", linewidth=0.8, linestyle="--")
        ax.set_title(f"Top {top_n} configurations by composite score")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Portfolio value (USD)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = os.path.join(_FILE_DIR, "top_equity_curves.png")
        plt.savefig(out, dpi=150)
        print(f"  Top equity curves saved to {out}")
        plt.close()
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # ------------------------------------------------------------------
    # 1. Load data (once, in main process)
    # ------------------------------------------------------------------
    #csv_path = os.path.join(_STRAT_ROOT, "data", "returns_5m.csv")
    csv_path="./strategies/backtest/data/returns_5m.csv"#Ran from root of repo

    if os.path.exists(csv_path):
        price_data = load_pivoted_csv(
            csv_path,
            # symbols=['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','ADAUSDT'],  # subset
            # resample_tf='1h',      # resample for slower strategies
            start_date='2025-10-01', # slice to reduce runtime during dev
        )
    else:
        print(f"CSV not found at {csv_path} — using synthetic data.")
        price_data = load_sample_data()

    n_bars = max(len(df) for df in price_data.values())
    print(f"Data: {len(price_data)} pairs × {n_bars:,} bars\n")

    # ------------------------------------------------------------------
    # 2. Base allocator config (some jobs override individual keys)
    # ------------------------------------------------------------------
    allocator_cfg = {
        "max_single_coin":   0.30,
        "min_cash_fraction": 0.20,
        "sticky_threshold":  0.05,
        "vol_lookback":      288,
        "min_trade_usd":     200.0,
        "ewm_span":          48,
    }

    # ------------------------------------------------------------------
    # 3. Build job list
    # ------------------------------------------------------------------
    jobs = build_jobs(price_data, allocator_cfg, initial_cash=1_000_000.0)
    print(f"Running {len(jobs)} backtest jobs...\n")

    # ------------------------------------------------------------------
    # 4. Run in parallel
    #    max_workers: None = one worker per CPU core (good default).
    #    Lower it if you're RAM-constrained (each worker holds a copy of
    #    price_data in memory — 20 pairs × 896k bars ≈ ~500MB per worker).
    # ------------------------------------------------------------------
    n_workers = min(len(jobs), os.cpu_count() or 4)//2
    print(f"Using {n_workers} worker processes (CPU count: {os.cpu_count()})\n")

    t_start  = time.time()
    results  = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, job): job["label"] for job in jobs}

        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"  [{label:<35}] "
                    f"composite={result.get('composite', 'ERR'):>8.4f}  "
                    f"return={result.get('total_return', 0):>+7.2f}%  "
                    f"trades={result.get('total_trades', 0):>5}  "
                    f"({result.get('elapsed_s', 0):.1f}s)"
                )
            except Exception as e:
                results.append({"label": label, "error": str(e)})
                print(f"  [{label:<35}] ERROR: {e}")

    total_elapsed = time.time() - t_start
    print(f"\nAll jobs finished in {total_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print_summary(results)
    plot_top_equity_curves(results, top_n=5)


# Guard required for multiprocessing on Windows
if __name__ == "__main__":
    main()