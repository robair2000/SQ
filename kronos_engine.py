"""Forecasters. Each returns, per coin, every sampled path: array (KRONOS_SAMPLES, KRONOS_PRED_LEN, 4).

KronosForecaster never imports torch in this process. It writes the inputs for all coins to one .npz,
runs kronos_infer.py as a child process and reads the paths back. The child exits after the scan, so
torch and the model only use memory for a few minutes every 4 hours (fits a 512 MB Render instance).

BootstrapForecaster (USE_KRONOS=false) replays random blocks of the coin's own recent candles. It has no
edge by construction: it is only for testing the pipeline and as a baseline in the forecast log.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import config as C
import mem

PRICE_COLS = ['open', 'high', 'low', 'close']


def _stamps(ts):
    """Same time features as kronos.calc_time_stamps: minute, hour, weekday, day, month."""
    return np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], axis=1).astype(np.float32)


def prepare_inputs(df):
    ctx = df.tail(C.KRONOS_LOOKBACK).reset_index(drop=True)
    ts = pd.to_datetime(ctx['timestamp'], unit='ms')
    step = pd.Timedelta(seconds=C.CANDLE_SECONDS)
    future = pd.Series([ts.iloc[-1] + step * (i + 1) for i in range(C.KRONOS_PRED_LEN)])
    ohlc = ctx[PRICE_COLS].to_numpy(np.float32)
    vol = ctx['volume'].to_numpy(np.float32)
    amount = vol * ohlc.mean(axis=1)                     # same as KronosPredictor when 'amount' is missing
    x = np.column_stack([ohlc, vol, amount]).astype(np.float32)
    return x, _stamps(ts), _stamps(future)


class KronosForecaster:
    name = 'kronos'

    def paths_many(self, dfs):
        """dfs: {symbol: DataFrame of closed 4h candles}. Returns {symbol: paths}."""
        syms = [s for s, df in dfs.items() if df is not None and len(df) >= C.KRONOS_LOOKBACK]
        if not syms:
            return {}
        xs, xst, yst = zip(*(prepare_inputs(dfs[s]) for s in syms))
        with tempfile.TemporaryDirectory(prefix='kronos_') as tmp:
            inp, out = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
            np.savez(inp, x=np.stack(xs), x_stamp=np.stack(xst), y_stamp=np.stack(yst))
            here = os.path.dirname(os.path.abspath(__file__))
            t0 = time.time()
            proc = subprocess.run([sys.executable, os.path.join(here, 'kronos_infer.py'), inp, out],
                                  cwd=here, timeout=C.KRONOS_TIMEOUT)
            if proc.returncode != 0:
                if proc.returncode in (-9, 137):
                    raise RuntimeError("Kronos process was killed (out of memory). Lower KRONOS_BATCH, "
                                       "KRONOS_LOOKBACK or KRONOS_SAMPLES, or use a bigger instance.")
                raise RuntimeError(f"Kronos process failed (exit code {proc.returncode})")
            paths = np.load(out)['paths'].astype(float)
        print(f"🧠 [SCAN] Kronos forecast {len(syms)} coins in {time.time() - t0:.0f}s | {mem.report()}")
        return dict(zip(syms, paths))

    def paths(self, df, symbol='X'):
        return self.paths_many({symbol: df})[symbol]


class BootstrapForecaster:
    """Baseline: replays random horizon-long blocks of the coin's own recent candles. No edge."""
    name = 'bootstrap'

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def paths(self, df, symbol='X'):
        ctx = df.tail(C.KRONOS_LOOKBACK).reset_index(drop=True)
        h = C.KRONOS_PRED_LEN
        c = ctx['close'].to_numpy(float)
        prev = np.r_[c[0], c[:-1]]
        rel = ctx[PRICE_COLS].to_numpy(float) / prev[:, None]   # each candle relative to the previous close
        starts = self.rng.integers(1, len(ctx) - h, size=C.KRONOS_SAMPLES)
        out = np.empty((C.KRONOS_SAMPLES, h, 4))
        for i, s in enumerate(starts):
            level = c[-1]
            for j in range(h):
                out[i, j] = rel[s + j] * level
                level = out[i, j, 3]
        return out

    def paths_many(self, dfs):
        return {s: self.paths(df) for s, df in dfs.items() if df is not None}


def load_forecaster():
    if not C.USE_KRONOS:
        print("ℹ️ USE_KRONOS=false -> bootstrap baseline forecaster (no edge, paper trading only).")
        return BootstrapForecaster()
    return KronosForecaster()


def download_models():
    """Run at build time so the weights are baked into the Render build (HF_HOME inside the project)."""
    from huggingface_hub import snapshot_download
    for repo in (C.KRONOS_TOKENIZER, C.KRONOS_MODEL):
        path = snapshot_download(repo)
        print(f"⬇️ {repo} -> {path}")


if __name__ == '__main__':
    download_models()
