'''
run_strategy.py — standalone strategy process.

Loads a strategy config from a JSON file, instantiates the strategy,
and runs it in a loop. Communicates with the allocator via SQLite only
— no direct dependency on the live trading system process.

Usage:
    python run_strategy.py --config ./config/macd_1h.json

Config file format (see examples in config/strategies/):
    {
        "strategy":  "MACD",
        "name":      "MACD_1h",
        "symbols":   ["BTC", "ETH", "BNB", "SOL"],
        "resample_tf": "1h",
        "window":    6,
        "poll_interval": 30.0,
        "db_path":   "../trading_system/config/trading_bot.db"
    }

Strategy-specific params:
    MACD:        window
    Bollinger:   window
    VWAP:        window
    XSMom:       window
    AdaptRSI:    rsi_window, percentile_window

@ MTL 21 March 2026
'''
import asyncio
import argparse
import json
import logging
import os
import signal
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.join(_FILE_DIR, '..', '..')   # SGHK_Hackathon/
_STRAT_ROOT = os.path.join(_FILE_DIR, '..')          # strategies/
for p in [_REPO_ROOT, _STRAT_ROOT]:
    p = os.path.normpath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from technical_indicator import (
    BollingerReversion,
    MACDStrategy,
    VWAPReversion,
    CrossSectionalMomentum,
    AdaptiveRSI,
)

# ---------------------------------------------------------------------------
# Logging — format includes strategy name so multi-process logs are readable
# ---------------------------------------------------------------------------
def setup_logging(name: str, level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=f'%(asctime)s [%(levelname)s] {name}: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------
STRATEGY_KINDS = {
    "MACD":      MACDStrategy,
    "Bollinger": BollingerReversion,
    "VWAP":      VWAPReversion,
    "XSMom":     CrossSectionalMomentum,
    "AdaptRSI":  AdaptiveRSI,
}

def build_strategy(cfg: dict):
    """
    Instantiate a strategy from a config dict.
    Extra keys in cfg are ignored, so you can keep notes/comments in the JSON.
    """
    kind = cfg.get("strategy")
    if kind not in STRATEGY_KINDS:
        raise ValueError(
            f"Unknown strategy '{kind}'. "
            f"Must be one of: {list(STRATEGY_KINDS.keys())}"
        )

    name        = cfg.get("name", kind)
    symbols     = cfg["symbols"]
    db_path     = cfg.get("db_path", "./trading_bot.db")
    resample_tf = cfg.get("resample_tf", None)

    common = dict(
        strategy_name=name,
        symbols=symbols,
        db_path=db_path,
        resample_tf=resample_tf,
    )

    if kind == "MACD":
        return MACDStrategy(**common, window=cfg.get("window", 26))

    elif kind == "Bollinger":
        return BollingerReversion(**common, window=cfg.get("window", 20))

    elif kind == "VWAP":
        return VWAPReversion(**common, window=cfg.get("window", 14))

    elif kind == "XSMom":
        return CrossSectionalMomentum(**common, window=cfg.get("window", 24))

    elif kind == "AdaptRSI":
        return AdaptiveRSI(
            **common,
            rsi_window=cfg.get("rsi_window", 14),
            percentile_window=cfg.get("percentile_window", 100),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(config_path: str):
    # --- Load config ---
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    name          = cfg.get("name", cfg.get("strategy", "Strategy"))
    poll_interval = float(cfg.get("poll_interval", 10.0))
    log_level     = cfg.get("log_level", "INFO")

    setup_logging(name, log_level)
    logger = logging.getLogger(name)

    logger.info(f"Config loaded from {config_path}")
    logger.info(f"Strategy:    {cfg.get('strategy')}")
    logger.info(f"Symbols:     {cfg.get('symbols')}")
    logger.info(f"Resample TF: {cfg.get('resample_tf', 'none (5min native)')}")
    logger.info(f"DB:          {cfg.get('db_path')}")
    logger.info(f"Poll:        {poll_interval}s")

    # --- Build strategy ---
    strategy = build_strategy(cfg)

    # --- Graceful shutdown on SIGINT / SIGTERM ---
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, strategy.stop)
    else:
        # Windows fallback — KeyboardInterrupt catches Ctrl-C
        pass

    # --- Run ---
    logger.info("Starting strategy loop...")
    try:
        await strategy.run(poll_interval=poll_interval)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        strategy.stop()
    finally:
        logger.info("Strategy stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single strategy as a standalone process."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to strategy JSON config file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)

    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
