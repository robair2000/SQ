"""Kronos input/output files and the baseline forecaster. Paths per coin: array (KRONOS_SAMPLES, KRONOS_PRED_LEN, 4).

torch is never imported here. scan_job.py writes the inputs for all coins with write_inputs(), the
supervisor (worker.py) runs kronos_infer.py as its own process, and scan_job.py reads the result with
read_paths(). No other job is running while Kronos runs, which is what keeps the peak under 512 MB.

BootstrapForecaster (USE_KRONOS=false) replays random blocks of the coin's own recent candles. It has no
edge by construction: it is only for testing the pipeline and as a baseline in the forecast log.
"""
import os

import numpy as np
import pandas as pd

import config as C

PRICE_COLS = ['open', 'high', 'low', 'close']


def _stamps(ts):
    """Same time features as kronos.calc_time_stamps: minute, hour, weekday, day, month."""
    return np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], axis=1).astype(np.float32)


def prepare_inputs(df):
    """Returns (x, x_stamp, y_stamp, scale). Prices are divided by the last close (scale) so every coin is
    around 1.0: Kronos normalises with (x - mean) / (std + 1e-5), and for coins priced like PEPE (~0.00001)
    the 1e-5 swamps the real price spread, flattening the input and wrecking the forecast."""
    ctx = df.tail(C.KRONOS_LOOKBACK).reset_index(drop=True)
    ts = pd.to_datetime(ctx['timestamp'], unit='ms')
    step = pd.Timedelta(seconds=C.CANDLE_SECONDS)
    future = pd.Series([ts.iloc[-1] + step * (i + 1) for i in range(C.KRONOS_PRED_LEN)])
    ohlc64 = ctx[PRICE_COLS].to_numpy(np.float64)
    scale = float(ohlc64[-1, 3])
    ohlc = (ohlc64 / scale).astype(np.float32)
    vol = ctx['volume'].to_numpy(np.float32)
    amount = vol * ohlc.mean(axis=1)                     # same as KronosPredictor when 'amount' is missing
    x = np.column_stack([ohlc, vol, amount]).astype(np.float32)
    return x, _stamps(ts), _stamps(future), scale


def write_inputs(dfs, path):
    """Writes the Kronos input file for all coins; returns the symbol order (None if nothing to do)."""
    syms = [s for s, df in dfs.items() if df is not None and len(df) >= C.KRONOS_LOOKBACK]
    if not syms:
        return None
    xs, xst, yst, scales = zip(*(prepare_inputs(dfs[s]) for s in syms))
    tmp = path + '.tmp.npz'
    np.savez(tmp, x=np.stack(xs), x_stamp=np.stack(xst), y_stamp=np.stack(yst),
             scale=np.asarray(scales, dtype=np.float64))
    os.replace(tmp, path)
    return syms


def read_paths(path, syms, inputs_path):
    """Reads kronos_infer.py output and undoes the price scaling; {} when the output is missing
    (inference failed or was skipped)."""
    if not syms or not os.path.exists(path) or not os.path.exists(inputs_path):
        return {}
    paths = np.load(path)['paths'].astype(np.float64)
    scale = np.load(inputs_path)['scale']
    paths = paths * scale[:, None, None, None]
    return dict(zip(syms, paths))


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


def download_models():
    """Run at build time so the weights are baked into the Render build (HF_HOME inside the project)."""
    from huggingface_hub import snapshot_download
    for repo in (C.KRONOS_TOKENIZER, C.KRONOS_MODEL):
        path = snapshot_download(repo)
        print(f"⬇️ {repo} -> {path}")


if __name__ == '__main__':
    download_models()
