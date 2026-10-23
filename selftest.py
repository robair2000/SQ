"""One-off check, no orders: runs Kronos once on live KuCoin data and prints the forecasts, decisions and
memory use.

  python selftest.py                      # BTC, ETH, SOL
  python selftest.py PEPE/USDT:USDT       # specific symbols
Run it once on Render (Shell tab) after the first deploy, before trusting the worker.
"""
import sys
import time
from datetime import datetime, timezone

import ccxt
import numpy as np

import config as C
import mem
import scanner
import signals
from kronos_engine import load_forecaster


def main(symbols):
    fc = load_forecaster()
    ex = ccxt.kucoinfutures({'enableRateLimit': True})
    iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    print(f"model {C.KRONOS_MODEL} | {C.KRONOS_SAMPLES} paths x {C.KRONOS_PRED_LEN} candles, "
          f"batch {C.KRONOS_BATCH}, lookback {C.KRONOS_LOOKBACK} | before: {mem.report()}")

    ctxs = [c for c in (scanner.prepare_symbol(ex, s) for s in symbols) if c]
    t0 = time.time()
    paths = fc.paths_many({c['symbol']: c['df'] for c in ctxs})
    print(f"forecast took {time.time() - t0:.0f}s")

    for c in ctxs:
        p = paths.get(c['symbol'])
        if p is None:
            print(f"{c['symbol']}: no forecast")
            continue
        last = float(c['row']['close'])
        summ = signals.summarize_paths(p, last)
        print(f"\n=== {c['symbol']}  last close {last:.6g}  ({p.shape[0]} paths x {p.shape[1]} candles)")
        print(f"P(up at +{p.shape[1] * 4}h) = {summ['prob_up']:.2f}")
        print("final close q10/q50/q90:", np.round(np.quantile(summ['final_close'], [.1, .5, .9]), 6))
        print(f"predicted low in first {C.ENTRY_WINDOW_CANDLES} candles q25/q50/q75:",
              np.round(np.quantile(summ['window_low'], [.25, .5, .75]), 6))
        sane = np.all(np.isfinite(p)) and np.all(p > 0) and np.all(np.abs(p / last - 1) < 0.8)
        print("sanity:", "OK" if sane else "SUSPICIOUS (non-finite, <=0 or >80% away from last close)")
        plan, log = scanner.decide(c, p, fc.name, iso)
        print("decision:", log.get('decision') if log else None, "|", log.get('reason') if log else '')

    print(f"\nafter: {mem.report()}")
    print("On a 512 MB instance keep 'container peak' comfortably below 512; if not, lower KRONOS_BATCH.")


if __name__ == '__main__':
    main(sys.argv[1:] or ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'])
