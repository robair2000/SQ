"""Scan job, run by the supervisor in two separate processes around the Kronos process:

  python scan_job.py prepare   ranking + 4h data + setups -> scan_work/contexts.pkl and scan_work/in.npz
  (python kronos_infer.py scan_work/in.npz scan_work/out.npz)
  python scan_job.py decide    forecasts -> plans_kronos.json + forecast log

Optional: --symbols BTC/USDT:USDT,ETH/USDT:USDT  (used by selftest.py; also skips writing plans/logs)
"""
import os
import pickle
import sys
from datetime import datetime, timezone

import config as C

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan_work')
CONTEXTS = os.path.join(WORK, 'contexts.pkl')
KRONOS_IN = os.path.join(WORK, 'in.npz')
KRONOS_OUT = os.path.join(WORK, 'out.npz')


def _symbols_arg(argv):
    if '--symbols' in argv:
        return [s for s in argv[argv.index('--symbols') + 1].split(',') if s]
    return None


def prepare(symbols=None):
    import ccxt
    import kronos_engine
    import scanner

    os.makedirs(WORK, exist_ok=True)
    for f in (CONTEXTS, KRONOS_IN, KRONOS_OUT):
        if os.path.exists(f):
            os.remove(f)
    ex = ccxt.kucoinfutures({'enableRateLimit': True})
    if symbols:
        contexts = [c for c in (scanner.prepare_symbol(ex, s) for s in symbols) if c]
    else:
        contexts = scanner.prepare_all(ex)
    want = scanner.wanted(contexts) if not symbols else {c['symbol']: c['df'] for c in contexts}
    syms = kronos_engine.write_inputs(want, KRONOS_IN) if (want and C.USE_KRONOS) else None
    scan_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with open(CONTEXTS, 'wb') as f:
        pickle.dump({'scan_iso': scan_iso, 'contexts': contexts, 'kronos_symbols': syms,
                     'bootstrap_symbols': list(want) if not C.USE_KRONOS else None}, f)
    print(f"📦 [SCAN] prepared {len(contexts)} coins, {len(syms or want or [])} to forecast")


def decide(write=True):
    import kronos_engine
    import scanner

    if not os.path.exists(CONTEXTS):
        print("⚠️ [SCAN] nothing prepared, decide skipped")
        return []
    with open(CONTEXTS, 'rb') as f:
        d = pickle.load(f)
    contexts = d['contexts']
    if C.USE_KRONOS:
        paths, model = kronos_engine.read_paths(KRONOS_OUT, d['kronos_symbols'], KRONOS_IN), 'kronos'
        if d['kronos_symbols'] and not paths:
            print("⚠️ [SCAN] no Kronos output this scan (see the [KRONOS] lines above) -> no new plans")
    else:
        fc = kronos_engine.BootstrapForecaster()
        dfs = {c['symbol']: c['df'] for c in contexts if c['symbol'] in (d['bootstrap_symbols'] or [])}
        paths, model = fc.paths_many(dfs), 'bootstrap'
        print("ℹ️ [SCAN] USE_KRONOS=false -> bootstrap baseline forecaster (no edge, paper trading only)")
    plans = scanner.decide_all(contexts, paths, model, d['scan_iso'], write=write)
    for f in (CONTEXTS, KRONOS_IN, KRONOS_OUT):
        if os.path.exists(f):
            os.remove(f)
    return plans


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else ''
    syms = _symbols_arg(sys.argv)
    if step == 'prepare':
        prepare(syms)
    elif step == 'decide':
        decide(write=syms is None)
    else:
        sys.exit("usage: python scan_job.py prepare|decide [--symbols A,B]")
