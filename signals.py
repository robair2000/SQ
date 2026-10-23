"""Pure signal logic (no network): indicators, setups, forecast summary and the trade plan.

Flow per coin at every 4h close:
  1. indicators()        Bollinger %B / width percentile, ATR, RSI, volume ratio
  2. find_setups()       which sides are allowed: squeeze and/or mean reversion
  3. summarize_paths()   Kronos sampled paths -> P(up), predicted lows/highs, horizon closes
  4. build_plan()        Kronos picks the side; limit entry at the predicted low (long) / high (short),
                         stop beyond the predicted extremes, target at the predicted horizon close
"""
import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ indicators

def indicators(df):
    df = df.copy()
    close, high, low = df['close'], df['high'], df['low']

    sma = close.rolling(C.BB_PERIOD).mean()
    std = close.rolling(C.BB_PERIOD).std()
    upper, lower = sma + C.BB_STD * std, sma - C.BB_STD * std
    df['sma'], df['bb_upper'], df['bb_lower'] = sma, upper, lower
    df['percent_b'] = (close - lower) / ((upper - lower) + 1e-12)
    df['bb_width_pctl'] = ((upper - lower) / sma).rolling(C.BB_PCTL_WINDOW).rank(pct=True)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(C.ATR_PERIOD).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / C.RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / C.RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-12))

    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(C.VOL_MA_PERIOD).mean() + 1e-12)
    return df


def min_bars_required():
    return max(C.BB_PERIOD, C.VOL_MA_PERIOD, C.ATR_PERIOD + 1, C.RSI_PERIOD + 1) + C.BB_PCTL_WINDOW + 5


# ------------------------------------------------------------------ setups

def find_setups(row):
    """Returns {'long': [labels], 'short': [labels]} plus a 0..1 strength per side."""
    sides = {'long': [], 'short': []}
    strength = {'long': 0.0, 'short': 0.0}
    pctb, pctl = float(row['percent_b']), float(row['bb_width_pctl'])

    if C.ENABLE_SQUEEZE and pctl <= C.SQUEEZE_PCTL and float(row['vol_ratio']) >= C.SQUEEZE_MIN_VOL_RATIO:
        s = float(np.clip((C.SQUEEZE_PCTL - pctl) / max(C.SQUEEZE_PCTL, 1e-9), 0, 1))
        discount = C.SQUEEZE_SIDE_FILTER == 'discount'
        if not discount or pctb <= 0.5:
            sides['long'].append('squeeze')
            strength['long'] = max(strength['long'], s)
        if not discount or pctb >= 0.5:
            sides['short'].append('squeeze')
            strength['short'] = max(strength['short'], s)

    if C.ENABLE_MEAN_REVERSION:
        rsi = float(row['rsi'])
        if pctb <= C.MR_PCTB_LOW and (C.MR_RSI_LOW <= 0 or rsi <= C.MR_RSI_LOW):
            sides['long'].append('mean_reversion')
            strength['long'] = max(strength['long'], float(np.clip(0.5 + (C.MR_PCTB_LOW - pctb) / 0.4, 0, 1)))
        if pctb >= C.MR_PCTB_HIGH and (C.MR_RSI_HIGH >= 100 or rsi >= C.MR_RSI_HIGH):
            sides['short'].append('mean_reversion')
            strength['short'] = max(strength['short'], float(np.clip(0.5 + (pctb - C.MR_PCTB_HIGH) / 0.4, 0, 1)))

    return sides, strength


# ------------------------------------------------------------------ forecast summary

def summarize_paths(paths, last_close):
    """paths: array (samples, horizon, 4) of predicted open/high/low/close in price units."""
    paths = np.asarray(paths, dtype=float)
    lows = paths.min(axis=2)    # robust: lowest of o/h/l/c per predicted candle
    highs = paths.max(axis=2)
    w = int(max(1, min(C.ENTRY_WINDOW_CANDLES, paths.shape[1])))
    final_close = paths[:, -1, 3]
    return {
        'samples': int(paths.shape[0]),
        'horizon': int(paths.shape[1]),
        'last_close': float(last_close),
        'prob_up': float(np.mean(final_close > last_close)),
        'final_close': final_close,
        'window_low': lows[:, :w].min(axis=1),
        'window_high': highs[:, :w].max(axis=1),
        'horizon_low': lows.min(axis=1),
        'horizon_high': highs.max(axis=1),
        'mean_path_close': paths[:, :, 3].mean(axis=0),
    }


def kronos_side(prob_up):
    if prob_up >= C.KRONOS_MIN_PROB:
        return 'long'
    if prob_up <= 1.0 - C.KRONOS_MIN_PROB:
        return 'short'
    return None


# ------------------------------------------------------------------ plan

def score_plan(conf, rr, setup_strength, fill_prob, n_setups):
    rr_norm = float(np.clip((rr - 1.0) / 2.0, 0, 1))            # 1R -> 0, 3R -> 1
    s = 0.45 * conf + 0.25 * rr_norm + 0.15 * setup_strength + 0.15 * fill_prob
    if n_setups > 1:
        s += 0.05                                                 # squeeze AND stretched: confluence bonus
    return round(100.0 * min(s, 1.0), 1)


def build_plan(row, summary, atr, sides, strength):
    """Returns (plan or None, reason). Long: buy below the market at the predicted low; short mirrored."""
    side = kronos_side(summary['prob_up'])
    if side is None:
        return None, f"Kronos undecided (P(up)={summary['prob_up']:.2f})"
    if not sides[side]:
        return None, f"Kronos says {side} (P(up)={summary['prob_up']:.2f}) but no {side} setup"
    if not atr or atr <= 0:
        return None, 'ATR unavailable'

    last = summary['last_close']
    q = C.SNIPE_QUANTILE
    if side == 'long':
        raw = float(np.quantile(summary['window_low'], q))
        entry = float(np.clip(raw, last - C.SNIPE_MAX_ATR * atr, last - C.SNIPE_MIN_ATR * atr))
        path_sl = float(np.quantile(summary['horizon_low'], C.SL_PATH_QUANTILE)) - C.SL_BUFFER_ATR * atr
        sl = max(min(entry - C.SL_ATR_MULT * atr, path_sl), entry - C.SL_MAX_ATR * atr)
        tp = float(np.quantile(summary['final_close'], C.TP_QUANTILE))
        risk, reward = entry - sl, tp - entry
        fill_prob = float(np.mean(summary['window_low'] <= entry))
        conf = float(np.clip((summary['prob_up'] - 0.5) / 0.3, 0, 1))
    else:
        raw = float(np.quantile(summary['window_high'], 1.0 - q))
        entry = float(np.clip(raw, last + C.SNIPE_MIN_ATR * atr, last + C.SNIPE_MAX_ATR * atr))
        path_sl = float(np.quantile(summary['horizon_high'], 1.0 - C.SL_PATH_QUANTILE)) + C.SL_BUFFER_ATR * atr
        sl = min(max(entry + C.SL_ATR_MULT * atr, path_sl), entry + C.SL_MAX_ATR * atr)
        tp = float(np.quantile(summary['final_close'], 1.0 - C.TP_QUANTILE))
        risk, reward = sl - entry, entry - tp
        fill_prob = float(np.mean(summary['window_high'] >= entry))
        conf = float(np.clip((0.5 - summary['prob_up']) / 0.3, 0, 1))

    if reward <= 0:
        return None, f"{side}: predicted target {tp:.6g} not beyond entry {entry:.6g}"
    rr = reward / risk
    if rr < C.MIN_RR:
        return None, f"{side}: reward:risk {rr:.2f} < {C.MIN_RR}"

    score = score_plan(conf, rr, strength[side], fill_prob, len(sides[side]))
    if score < C.MIN_SCORE:
        return None, f"{side}: score {score} < {C.MIN_SCORE}"

    leverage = int(np.clip(round(C.BASE_LEVERAGE * (0.75 + 0.5 * conf)), C.MIN_LEVERAGE, C.MAX_LEVERAGE))
    return {
        'side': side,
        'setups': sides[side],
        'entry': entry,
        'sl': float(sl),
        'tp': tp,
        'rr': round(rr, 2),
        'risk_atr': round(risk / atr, 2),
        'atr': float(atr),
        'prob_up': round(summary['prob_up'], 3),
        'fill_prob': round(fill_prob, 2),
        'score': score,
        'leverage': leverage,
        'ref_close': last,
    }, 'ok'
