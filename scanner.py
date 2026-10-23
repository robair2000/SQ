"""4h scanner: setups + Kronos forecast -> trade plans. Runs in its own thread inside worker.py."""
import csv
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import config as C
import mem
import signals
from asset_utils import get_combined_trading_assets
from github_utils import github_put, github_pull
from market_data import fetch_ohlcv_paginated

FORECAST_FIELDS = [
    'scan_iso', 'candle_ts', 'symbol', 'model', 'last_close', 'atr', 'percent_b', 'width_pctl', 'rsi',
    'setups_long', 'setups_short', 'prob_up', 'final_q10', 'final_q50', 'final_q90', 'horizon',
    'decision', 'reason',
]


def write_atomic(path, data_bytes):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data_bytes)
    os.replace(tmp, path)


def push(filename, agent='kronos-scanner'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            github_put(filename, f.read(), agent=agent)


def append_forecast_log(rows):
    if not rows:
        return
    if not os.path.exists(C.FORECAST_LOG):
        github_pull(C.FORECAST_LOG, agent='kronos-scanner')   # keep history across restarts
    new = not os.path.exists(C.FORECAST_LOG)
    with open(C.FORECAST_LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FORECAST_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    push(C.FORECAST_LOG)


def refresh_ranking(exchange):
    """Runs getcoins in this process on the shared exchange (a second Python interpreter would cost ~150 MB)."""
    try:
        import getcoins
        getcoins.EXCHANGE = exchange
        getcoins.fetch_top_performing_coins()
        push('kucoin_best_coins_ranking.csv')
    except Exception as e:
        print(f"⚠️ [SCAN] coin ranking skipped: {e}")


def fetch_history(exchange, symbol):
    need = max(signals.min_bars_required(), C.KRONOS_LOOKBACK)
    for attempt in range(3):
        try:
            df = fetch_ohlcv_paginated(exchange, symbol, C.TIMEFRAME, total_bars=max(C.HISTORY_BARS, need))
            if df is not None and len(df) >= need:
                return df
        except Exception as e:
            print(f"⚠️ [SCAN] fetch {symbol} attempt {attempt + 1}/3: {e}")
        time.sleep(6.0)
    return None


def prepare_symbol(exchange, symbol):
    """Phase 1: data + indicators + setups. Returns a context dict or None."""
    df = fetch_history(exchange, symbol)
    if df is None:
        print(f"❌ [SCAN] {symbol}: not enough 4h history, skipped")
        return None
    row = signals.indicators(df).iloc[-1]
    if row[['percent_b', 'bb_width_pctl', 'atr', 'rsi']].isnull().any():
        print(f"❌ [SCAN] {symbol}: indicators incomplete, skipped")
        return None
    sides, strength = signals.find_setups(row)
    return {'symbol': symbol, 'df': df, 'row': row, 'sides': sides, 'strength': strength,
            'has_setup': bool(sides['long'] or sides['short'])}


def decide(ctx, paths, model_name, scan_iso):
    """Phase 3: forecast summary -> plan. Returns (plan or None, forecast-log row)."""
    symbol, row, sides, strength = ctx['symbol'], ctx['row'], ctx['sides'], ctx['strength']
    candle_ts = int(row['timestamp']) // 1000          # candle open, seconds
    atr = float(row['atr'])
    log = {
        'scan_iso': scan_iso, 'candle_ts': candle_ts, 'symbol': symbol, 'model': model_name,
        'last_close': float(row['close']), 'atr': round(atr, 10),
        'percent_b': round(float(row['percent_b']), 3), 'width_pctl': round(float(row['bb_width_pctl']), 3),
        'rsi': round(float(row['rsi']), 1),
        'setups_long': '+'.join(sides['long']), 'setups_short': '+'.join(sides['short']),
        'horizon': C.KRONOS_PRED_LEN,
    }
    desc = (f"%B={row['percent_b']:.2f} width-pctl={row['bb_width_pctl']:.2f} RSI={row['rsi']:.0f} "
            f"| long[{'+'.join(sides['long']) or '-'}] short[{'+'.join(sides['short']) or '-'}]")
    if paths is None:
        log.update({'decision': 'none', 'reason': 'no forecast'})
        print(f"   · [SCAN] {symbol}: {desc} -> {'no forecast' if ctx['has_setup'] else 'no setup'}")
        return None, (log if ctx['has_setup'] else None)

    summary = signals.summarize_paths(paths, float(row['close']))
    fc = summary['final_close']
    log.update({
        'prob_up': round(summary['prob_up'], 3),
        'final_q10': float(np.quantile(fc, 0.1)), 'final_q50': float(np.quantile(fc, 0.5)),
        'final_q90': float(np.quantile(fc, 0.9)),
    })
    if ctx['has_setup']:
        plan, reason = signals.build_plan(row, summary, atr, sides, strength)
    else:
        plan, reason = None, 'no setup'
    log['decision'] = plan['side'] if plan else 'none'
    log['reason'] = reason
    print(f"   · [SCAN] {symbol}: {desc} | P(up)={summary['prob_up']:.2f} ({model_name}) "
          f"-> {reason if not plan else plan['side'].upper()}")

    if plan:
        plan.update({
            'symbol': symbol,
            'candle_ts': candle_ts,
            'created_at': time.time(),
            # entry window starts when the signal candle closed
            'valid_until': candle_ts + C.CANDLE_SECONDS * (1 + C.ENTRY_WINDOW_CANDLES),
            'model': model_name,
        })
        print(f"   🎯 [SCAN] PLAN {symbol} {plan['side'].upper()} [{'+'.join(plan['setups'])}] "
              f"limit {plan['entry']:.6g} (close {plan['ref_close']:.6g}, fill-prob {plan['fill_prob']:.0%}) "
              f"SL {plan['sl']:.6g} TP {plan['tp']:.6g} | R:R {plan['rr']} | P(up) {plan['prob_up']} "
              f"| score {plan['score']} | lev {plan['leverage']}x")
    return plan, log


def prepare_all(exchange):
    """Phase 1 for every coin: ranking, data, indicators, setups."""
    refresh_ranking(exchange)
    contexts = []
    for symbol in get_combined_trading_assets():
        try:
            ctx = prepare_symbol(exchange, symbol)
            if ctx:
                contexts.append(ctx)
        except Exception as e:
            print(f"❌ [SCAN] {symbol}: {e}")
        time.sleep(3.0)                         # be gentle with KuCoin rate limits
    return contexts


def wanted(contexts):
    """Coins that get a forecast this scan."""
    return {c['symbol']: c['df'] for c in contexts if c['has_setup'] or C.LOG_ALL_FORECASTS}


def decide_all(contexts, paths, model_name, scan_iso, write=True):
    """Phase 3: plans + forecast log. paths: {symbol: array}. write=False for the self-test."""
    plans, logs = [], []
    for ctx in contexts:
        try:
            p = paths.get(ctx['symbol'])
            if p is not None and not sane(p, float(ctx['row']['close'])):
                print(f"⚠️ [SCAN] {ctx['symbol']}: forecast looks broken (non-finite or >80% away), ignored")
                p = None
            plan, log = decide(ctx, p, model_name, scan_iso)
            if log:
                logs.append(log)
            if plan:
                plans.append(plan)
        except Exception as e:
            print(f"❌ [SCAN] {ctx['symbol']}: {e}")
    if write:
        write_atomic(C.PLANS_FILE, json.dumps({'scan_iso': scan_iso, 'plans': plans}, indent=2).encode())
        push(C.PLANS_FILE)
        append_forecast_log(logs)
    print(f"✅ [SCAN] done: {len(logs)} forecasts, {len(plans)} plans | {mem.report()}")
    return plans


def sane(paths, last):
    p = np.asarray(paths, dtype=float)
    return bool(np.all(np.isfinite(p)) and np.all(p > 0) and np.all(np.abs(p / last - 1) < 0.8))


def seconds_until_next_scan(now=None):
    now = time.time() if now is None else now
    next_close = (int(now // C.CANDLE_SECONDS) + 1) * C.CANDLE_SECONDS
    return max(5.0, next_close + C.SCAN_DELAY_SECONDS - now)
