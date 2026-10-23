"""Does Kronos actually know the direction? Scores forecast_log_kronos.csv against what really happened.

  python evaluate_forecasts.py                 # reads the local log (pulls it from GitHub if missing)

For every logged forecast whose horizon has passed it fetches the real close KRONOS_PRED_LEN candles later
and reports, per model:
  - hit rate of confident calls (P(up) >= KRONOS_MIN_PROB or <= 1 - KRONOS_MIN_PROB)
  - Brier score of P(up) vs the 0.25 of a coin flip (lower is better)
  - a calibration table: when Kronos says 70% up, is it up ~70% of the time?
A confident-call hit rate that stays near 50% over a few hundred forecasts means no edge: keep DRY_RUN on.
"""
import time

import ccxt
import numpy as np
import pandas as pd

import config as C
from github_utils import github_pull
from market_data import fetch_ohlcv_paginated


def main():
    github_pull(C.FORECAST_LOG, agent='kronos-eval')
    try:
        log = pd.read_csv(C.FORECAST_LOG)
    except FileNotFoundError:
        return print(f"No {C.FORECAST_LOG} yet.")
    log = log.dropna(subset=['prob_up'])
    now = time.time()
    log['target_ts'] = log['candle_ts'] + log['horizon'] * C.CANDLE_SECONDS
    due = log[log['target_ts'] + C.CANDLE_SECONDS <= now].copy()
    if due.empty:
        return print(f"{len(log)} forecasts logged, none matured yet (horizon {C.KRONOS_PRED_LEN} x 4h).")

    ex = ccxt.kucoinfutures({'enableRateLimit': True})
    closes = {}
    for sym, g in due.groupby('symbol'):
        bars = int((now - g['candle_ts'].min()) / C.CANDLE_SECONDS) + 5
        df = fetch_ohlcv_paginated(ex, sym, '4h', total_bars=bars)
        closes[sym] = dict(zip((df['timestamp'] // 1000).astype(int), df['close']))
    due['real_close'] = [closes.get(s, {}).get(int(t)) for s, t in zip(due['symbol'], due['target_ts'])]
    due = due.dropna(subset=['real_close'])
    due['up'] = (due['real_close'] > due['last_close']).astype(int)

    for model, g in due.groupby('model'):
        brier = float(np.mean((g['prob_up'] - g['up']) ** 2))
        conf = g[(g['prob_up'] >= C.KRONOS_MIN_PROB) | (g['prob_up'] <= 1 - C.KRONOS_MIN_PROB)]
        hits = ((conf['prob_up'] >= 0.5).astype(int) == conf['up']).mean() if len(conf) else float('nan')
        print(f"\n=== {model}: {len(g)} matured forecasts")
        print(f"Brier score {brier:.4f}  (coin flip = 0.2500, lower is better)")
        print(f"confident calls: {len(conf)}, hit rate {hits:.1%}" if len(conf) else "confident calls: 0")
        bins = pd.cut(g['prob_up'], [0, .2, .35, .45, .55, .65, .8, 1.0], include_lowest=True)
        tab = g.groupby(bins, observed=True)['up'].agg(['count', 'mean']).rename(columns={'mean': 'actual_up_rate'})
        print(tab.to_string())
        traded = g[g['decision'].isin(['long', 'short'])]
        if len(traded):
            ok = ((traded['decision'] == 'long') == (traded['up'] == 1)).mean()
            print(f"plans: {len(traded)}, direction right {ok:.1%}")


if __name__ == '__main__':
    main()
