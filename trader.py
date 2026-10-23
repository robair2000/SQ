"""Trader: turns plans into resting limit entries ('snipes') and manages them until exit.

Lifecycle per symbol:  plan -> PENDING (post-only limit at the predicted low/high)
                            -> ACTIVE  (filled: SL + TP placed, breakeven at +BREAKEVEN_R, time stop)
                            -> closed  (SL / TP / time stop / exchange) -> cooldown
A pending entry is cancelled when its window expires, when price reaches the target without filling
('missed'), or when the stop level is hit before the fill ('invalidated').

DRY_RUN=true is paper trading: nothing is sent to KuCoin; fills, stops and targets are simulated
from 1-minute candles (stop assumed first when a candle touches both).
"""
import json
import math
import os
import threading
import time

import config as C
from github_utils import github_pull, github_put
from market_data import fetch_1m_since
from order_utils import (fetch_all_open_orders, cancel_orders, create_limit_entry,
                         replace_protective_orders, replace_stop_loss)
from risk_utils import DailyLossGuard, log_trade

MAKER_FEE = float(os.environ.get('EST_MAKER_FEE', 0.0002))
TAKER_FEE = float(os.environ.get('EST_TAKER_FEE', 0.0006))


def _sgn(side):
    return 1 if side == 'long' else -1


class Trader:
    def __init__(self, exchange, dry_run=C.DRY_RUN, now=time.time):
        self.ex = exchange
        self.dry = dry_run
        self.now = now
        self.guard = DailyLossGuard(exchange)
        self.pending = {}
        self.active = {}
        self.cooldowns = {}
        self.last_traded_candles = {}
        self.paper = {'realized_usdt': 0.0, 'trades': 0, 'wins': 0}
        self._queue = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._logged = set()

    # ------------------------------------------------------------ plumbing
    def log_once(self, key, msg):
        if key not in self._logged:
            self._logged.add(key)
            if len(self._logged) > 5000:
                self._logged.clear()
            print(msg)

    def submit_plan(self, plan):
        """Thread-safe: called by the scanner thread."""
        with self._lock:
            old = self._queue.get(plan['symbol'])
            if old is None or plan['candle_ts'] >= old['candle_ts']:
                self._queue[plan['symbol']] = plan

    def _take_queue(self):
        with self._lock:
            q, self._queue = self._queue, {}
        return q

    def load_status(self):
        if not github_pull(C.STATUS_FILE, timeout=10, agent='kronos-trader'):
            print("ℹ️ [TRADER] no previous status, starting clean.")
            return
        try:
            with open(C.STATUS_FILE, encoding='utf-8') as f:
                d = json.load(f)
            if bool(d.get('dry_run')) != self.dry:
                print(f"⚠️ [TRADER] status was written with DRY_RUN={d.get('dry_run')}; "
                      f"now DRY_RUN={self.dry} -> ignoring old pending/active, keeping cooldowns.")
            else:
                self.pending = d.get('pending', {})
                self.active = d.get('active', {})
                self.paper = d.get('paper', self.paper)
            self.cooldowns = d.get('cooldowns', {})
            self.last_traded_candles = {k: int(v) for k, v in d.get('last_traded_candles', {}).items()}
            self.guard.state = d.get('daily_guard', {}) or {}
            print(f"📥 [TRADER] status restored: {len(self.pending)} pending, {len(self.active)} active.")
        except Exception as e:
            print(f"⚠️ [TRADER] status unreadable ({e}), starting clean.")

    def save_status(self, force=False):
        if not (self._dirty or self.guard.dirty or force):
            return
        self._dirty = False
        self.guard.dirty = False
        data = {
            'updated_at': self.now(), 'dry_run': self.dry,
            'pending': self.pending, 'active': self.active,
            'cooldowns': self.cooldowns, 'last_traded_candles': self.last_traded_candles,
            'daily_guard': self.guard.state, 'paper': self.paper,
        }
        b = json.dumps(data, indent=2).encode()
        tmp = C.STATUS_FILE + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(b)
        os.replace(tmp, C.STATUS_FILE)
        github_put(C.STATUS_FILE, b, message=f"Auto-update {C.STATUS_FILE} [skip ci]", agent='kronos-trader')

    # ------------------------------------------------------------ market helpers
    def price(self, symbol):
        try:
            return float(self.ex.fetch_ticker(symbol)['last'])
        except Exception:
            return None

    def round_price(self, symbol, p):
        try:
            return float(self.ex.price_to_precision(symbol, p))
        except Exception:
            return float(p)

    def contract_size(self, symbol):
        try:
            return float(self.ex.market(symbol).get('contractSize') or 1.0)
        except Exception:
            return 1.0

    def contracts_for(self, symbol, price, leverage):
        target = C.TRADE_AMOUNT_USDT * leverage
        try:
            m = self.ex.market(symbol)
            cs = float(m.get('contractSize') or 1.0)
            min_amt = float((m.get('limits', {}).get('amount', {}) or {}).get('min') or 1.0)
        except Exception:
            cs, min_amt = 1.0, 1.0
        per = price * cs
        if per <= 0:
            return None
        lots = math.floor(target / per)
        if lots < max(1, min_amt):
            self.log_once(('size', symbol, leverage),
                          f"⚠️ [TRADER] {symbol}: {C.TRADE_AMOUNT_USDT} USDT x {leverage}x < 1 contract "
                          f"({per:.2f} USDT) -> skipped. Raise TRADE_AMOUNT_USDT.")
            return None
        return float(lots)

    def pnl_usdt(self, symbol, side, entry, exit_px, contracts, entry_maker=True):
        notional_in = entry * contracts * self.contract_size(symbol)
        notional_out = exit_px * contracts * self.contract_size(symbol)
        gross = _sgn(side) * (notional_out - notional_in)
        fees = notional_in * (MAKER_FEE if entry_maker else TAKER_FEE) + notional_out * TAKER_FEE
        return gross - fees

    # ------------------------------------------------------------ main step
    def step(self, symbols=None):
        queued = self._take_queue()
        now = self.now()
        for sym in [s for s, ts in self.cooldowns.items() if now - ts > C.COOLDOWN_SECONDS]:
            self.cooldowns.pop(sym, None)
            self._dirty = True

        if self.dry:
            self._paper_pending()
            self._paper_active()
            halted = False
        else:
            try:
                positions = self.ex.fetch_positions()
            except Exception as e:
                print(f"⚠️ [TRADER] fetch_positions failed, step skipped: {e}")
                return
            held = {}
            for p in positions:
                c = abs(float(p.get('contracts') or 0))
                if c > 0 and p.get('side') in ('long', 'short'):
                    held[p.get('symbol')] = p
            self._live_pending(held)
            self._live_active(held)
            halted = self.guard.check()
            if halted:
                self.log_once(('halt', self.guard.state.get('date')),
                              f"🛑 [TRADER] MAX_DAILY_LOSS_USDT reached -> no new entries today.")
                for sym, p in list(self.pending.items()):
                    self._cancel_pending(sym, 'entry_cancel_daily_loss')

        if not halted:
            for plan in sorted(queued.values(), key=lambda p: -p['score']):
                try:
                    self._place(plan)
                except Exception as e:
                    print(f"⚠️ [TRADER] placing {plan['symbol']} failed: {e}")
        self.save_status()

    # ------------------------------------------------------------ placing
    def _place(self, plan):
        sym, side = plan['symbol'], plan['side']
        now = self.now()
        tag = f"[TRADER] {sym} {side.upper()}"
        if plan.get('model') != 'kronos' and not self.dry:
            return print(f"⏭️ {tag}: plan from '{plan.get('model')}' forecaster -> live trading refused.")
        if now > plan['valid_until']:
            return print(f"⏭️ {tag}: plan window already over.")
        if sym in self.pending or sym in self.active:
            return print(f"⏭️ {tag}: already pending/active.")
        if sym in self.cooldowns:
            return print(f"⏭️ {tag}: in cooldown.")
        if plan['candle_ts'] <= self.last_traded_candles.get(sym, 0):
            return
        if len(self.pending) + len(self.active) >= C.MAX_CONCURRENT_POSITIONS:
            return print(f"⏭️ {tag}: MAX_CONCURRENT_POSITIONS ({C.MAX_CONCURRENT_POSITIONS}) reached.")
        try:
            self.ex.load_markets()
        except Exception as e:
            return print(f"⚠️ {tag}: load_markets failed: {e}")
        if sym not in getattr(self.ex, 'markets', {}):
            return print(f"⏭️ {tag}: not a KuCoin futures market.")

        px = self.price(sym)
        if px is None:
            return print(f"⚠️ {tag}: no price.")
        s = _sgn(side)
        entry, sl, tp, atr = plan['entry'], plan['sl'], plan['tp'], plan['atr']
        if s * (px - tp) >= 0:
            return print(f"⏭️ {tag}: price {px:.6g} already at target {tp:.6g} -> missed.")
        # Price already moved through our level since the close: keep sniping below (above) the market.
        if s * (px - entry) <= C.SNIPE_MIN_ATR * atr:
            entry = px - s * C.SNIPE_MIN_ATR * atr
        if s * (entry - sl) <= 0:
            return print(f"⏭️ {tag}: price {px:.6g} already through the stop {sl:.6g} -> invalidated.")
        rr = s * (tp - entry) / (s * (entry - sl))
        if rr < C.MIN_RR:
            return print(f"⏭️ {tag}: R:R {rr:.2f} at current price < {C.MIN_RR}.")

        entry, sl, tp = self.round_price(sym, entry), self.round_price(sym, sl), self.round_price(sym, tp)
        lev = int(plan['leverage'])
        amount = self.contracts_for(sym, entry, lev)
        if amount is None:
            return

        order_id, cid = None, None
        if not self.dry:
            try:
                self.ex.set_margin_mode(C.MARGIN_MODE, sym)
                self.ex.set_leverage(lev, sym)
            except Exception as e:
                return print(f"⚠️ {tag}: margin/leverage setup failed: {e}")
            try:
                order_id, cid = create_limit_entry(self.ex, sym, 'buy' if side == 'long' else 'sell',
                                                   amount, entry, C.PREFIX, post_only=C.POST_ONLY)
            except Exception as e:
                return print(f"⚠️ {tag}: limit entry rejected: {e}")

        self.pending[sym] = {
            'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr, 'amount': amount, 'leverage': lev,
            'order_id': order_id, 'cid': cid, 'valid_until': plan['valid_until'], 'placed_at': now,
            'candle_ts': plan['candle_ts'], 'score': plan['score'], 'prob_up': plan['prob_up'],
            'setups': plan['setups'], 'last_check_ms': int(now * 1000),
        }
        self.last_traded_candles[sym] = plan['candle_ts']
        self._dirty = True
        mode = 'PAPER' if self.dry else 'LIVE'
        log_trade(C.BOT, 'entry_placed' if not self.dry else 'paper_entry_placed', sym, side, price=entry,
                  amount=amount, entry=entry, sl=sl, tp=tp,
                  note=f"score={plan['score']} P(up)={plan['prob_up']} rr={rr:.2f} lev={lev} "
                       f"setups={'+'.join(plan['setups'])} market={px}")
        print(f"🎯 [{mode}] {sym} {side.upper()} limit {amount:g} @ {entry:.6g} (market {px:.6g}) "
              f"SL {sl:.6g} TP {tp:.6g} R:R {rr:.2f} lev {lev}x, valid {((plan['valid_until'] - now) / 3600):.1f}h")

    def _activate(self, sym, p, entry, contracts):
        risk = abs(entry - p['sl'])
        max_hold = C.MAX_HOLD_CANDLES or C.KRONOS_PRED_LEN
        self.active[sym] = {
            'side': p['side'], 'entry': entry, 'sl': p['sl'], 'initial_sl': p['sl'], 'tp': p['tp'],
            'risk': risk, 'contracts': contracts, 'protected_contracts': 0.0, 'breakeven': False,
            'opened_at': self.now(), 'max_hold_until': self.now() + max_hold * C.CANDLE_SECONDS,
            'score': p.get('score'), 'last_check_ms': int(self.now() * 1000),
        }
        self._dirty = True
        log_trade(C.BOT, 'paper_entry_filled' if self.dry else 'entry_filled', sym, p['side'], price=entry,
                  amount=contracts, entry=entry, sl=p['sl'], tp=p['tp'], note=f"score={p.get('score')}")
        print(f"✅ [{'PAPER' if self.dry else 'LIVE'}] {sym} {p['side'].upper()} filled {contracts:g} @ {entry:.6g}")

    def _close_record(self, sym, t, exit_px, reason, contracts=None):
        contracts = contracts or t['contracts']
        pnl = self.pnl_usdt(sym, t['side'], t['entry'], exit_px, contracts)
        if self.dry:
            self.paper['realized_usdt'] = round(self.paper['realized_usdt'] + pnl, 4)
            self.paper['trades'] += 1
            self.paper['wins'] += int(pnl > 0)
        log_trade(C.BOT, ('paper_exit_' if self.dry else 'exit_') + reason, sym, t['side'], price=exit_px,
                  amount=contracts, entry=t['entry'], sl=t['sl'], tp=t['tp'],
                  note=f"est_pnl_usdt={pnl:.4f} R={_sgn(t['side']) * (exit_px - t['entry']) / (t['risk'] or 1):.2f}")
        print(f"🏁 [{'PAPER' if self.dry else 'LIVE'}] {sym} {t['side'].upper()} exit {reason} @ {exit_px:.6g} "
              f"-> est {pnl:+.2f} USDT" + (f" | paper total {self.paper['realized_usdt']:+.2f} USDT over "
                                            f"{self.paper['trades']} trades" if self.dry else ''))
        self.active.pop(sym, None)
        self.cooldowns[sym] = self.now()
        self._dirty = True

    def _cancel_pending(self, sym, reason):
        p = self.pending.pop(sym, None)
        if p is None:
            return
        if not self.dry and p.get('order_id'):
            try:
                self.ex.cancel_order(p['order_id'], sym)
            except Exception as e:
                print(f"⚠️ [TRADER] cancel {sym} entry failed (may already be gone): {e}")
        log_trade(C.BOT, ('paper_' if self.dry else '') + reason, sym, p['side'], price=p['entry'],
                  entry=p['entry'], sl=p['sl'], tp=p['tp'])
        print(f"✖️ [{'PAPER' if self.dry else 'LIVE'}] {sym} {p['side'].upper()} entry @ {p['entry']:.6g}: {reason}")
        self._dirty = True

    # ------------------------------------------------------------ paper simulation
    def _candles(self, sym, since_ms):
        out, pages = [], 0
        while pages < 10:
            pages += 1
            try:
                df = fetch_1m_since(self.ex, sym, since_ms)
            except Exception as e:
                print(f"⚠️ [PAPER] 1m candles {sym}: {e}")
                break
            if df.empty:
                break
            out.append(df)
            if len(df) < 500:
                break
            since_ms = int(df['timestamp'].iloc[-1]) + 60_000
        if not out:
            return []
        import pandas as pd
        df = pd.concat(out).drop_duplicates('timestamp')
        now_ms = int(self.now() * 1000)
        df = df[df['timestamp'] + 60_000 <= now_ms]           # closed 1m candles only
        return df.to_dict('records')

    def _paper_pending(self):
        for sym, p in list(self.pending.items()):
            s = _sgn(p['side'])
            filled_at = None
            for c in self._candles(sym, p['last_check_ms']):
                p['last_check_ms'] = int(c['timestamp']) + 60_000
                touch = c['low'] <= p['entry'] if s > 0 else c['high'] >= p['entry']
                if touch:
                    filled_at = c
                    break
                beyond_tp = c['high'] >= p['tp'] if s > 0 else c['low'] <= p['tp']
                if beyond_tp:
                    self._cancel_pending(sym, 'entry_missed')
                    break
            if sym not in self.pending:
                continue
            if filled_at is not None:
                self.pending.pop(sym)
                self._activate(sym, p, p['entry'], p['amount'])
                # Re-check the fill candle, but only for the stop: a target touch inside it may have come first.
                self.active[sym]['last_check_ms'] = int(filled_at['timestamp'])
                self.active[sym]['fill_candle_ms'] = int(filled_at['timestamp'])
            elif self.now() > p['valid_until']:
                self._cancel_pending(sym, 'entry_expired')

    def _paper_active(self):
        for sym, t in list(self.active.items()):
            s = _sgn(t['side'])
            closed = False
            for c in self._candles(sym, t['last_check_ms']):
                t['last_check_ms'] = int(c['timestamp']) + 60_000
                hit_sl = c['low'] <= t['sl'] if s > 0 else c['high'] >= t['sl']
                hit_tp = c['high'] >= t['tp'] if s > 0 else c['low'] <= t['tp']
                if int(c['timestamp']) == t.get('fill_candle_ms'):
                    hit_tp = False
                if hit_sl:
                    self._close_record(sym, t, t['sl'], 'breakeven' if t['breakeven'] else 'stop')
                    closed = True
                    break
                if hit_tp:
                    self._close_record(sym, t, t['tp'], 'tp')
                    closed = True
                    break
                best = c['high'] if s > 0 else c['low']
                if int(c['timestamp']) == t.get('fill_candle_ms'):
                    continue
                if not t['breakeven'] and s * (best - t['entry']) >= C.BREAKEVEN_R * t['risk']:
                    t['sl'], t['breakeven'] = t['entry'], True
                    self._dirty = True
                    print(f"🔒 [PAPER] {sym}: stop to breakeven {t['entry']:.6g}")
            if not closed and self.now() >= t['max_hold_until']:
                px = self.price(sym)
                if px is not None:
                    self._close_record(sym, t, px, 'time')

    # ------------------------------------------------------------ live management
    def _live_pending(self, held):
        for sym, p in list(self.pending.items()):
            order = None
            try:
                order = self.ex.fetch_order(p['order_id'], sym)
            except Exception as e:
                self.log_once(('fo', sym, p['order_id']), f"⚠️ [TRADER] fetch_order {sym}: {e}")
            status = (order or {}).get('status')
            filled = float((order or {}).get('filled') or 0)
            pos = held.get(sym)
            have_pos = pos is not None and pos.get('side') == p['side']

            if have_pos and sym not in self.active:
                entry = float(pos.get('entryPrice') or 0) or p['entry']
                self._activate(sym, p, entry, abs(float(pos.get('contracts') or 0)))

            expired = self.now() > p['valid_until']
            if status in ('closed', 'canceled', 'cancelled', 'expired', 'rejected'):
                self.pending.pop(sym, None)
                self._dirty = True
                if not have_pos and filled == 0:
                    log_trade(C.BOT, 'entry_gone', sym, p['side'], price=p['entry'], note=f'status={status}')
                continue
            if expired:
                self._cancel_pending(sym, 'entry_expired' if not have_pos else 'entry_rest_cancelled')
                continue
            if not have_pos:
                px = self.price(sym)
                if px is not None:
                    s = _sgn(p['side'])
                    if s * (px - p['tp']) >= 0:
                        self._cancel_pending(sym, 'entry_missed')
                    elif s * (px - p['sl']) <= 0:
                        self._cancel_pending(sym, 'entry_invalidated')

    def _live_active(self, held):
        for sym, t in list(self.active.items()):
            pos = held.get(sym)
            if pos is None or pos.get('side') != t['side']:
                if sym in self.pending:          # fill not visible yet
                    continue
                try:
                    cancel_orders(self.ex, fetch_all_open_orders(self.ex, sym), C.PREFIX)
                except Exception:
                    pass
                px = self.price(sym) or t['entry']
                reason = 'tp' if _sgn(t['side']) * (px - t['tp']) >= 0 else (
                    'stop' if _sgn(t['side']) * (px - t['sl']) <= 0 else 'exchange')
                self._close_record(sym, t, px, reason)
                continue

            contracts = abs(float(pos.get('contracts') or 0))
            if contracts != t.get('protected_contracts'):
                if replace_protective_orders(self.ex, sym, t['side'], contracts, t['sl'], t['tp'], C.PREFIX):
                    t['protected_contracts'] = contracts
                    t['contracts'] = contracts
                    self._dirty = True
                elif not t.get('protected_contracts'):
                    print(f"🚨 [LIVE] {sym}: could not place the stop on a fresh fill -> closing.")
                    self._market_close(sym, t, contracts, 'no_stop')
                    continue

            px = self.price(sym)
            if px is None:
                continue
            s = _sgn(t['side'])
            if not t['breakeven'] and s * (px - t['entry']) >= C.BREAKEVEN_R * t['risk']:
                new_sl = self.round_price(sym, t['entry'])
                if replace_stop_loss(self.ex, sym, t['side'], contracts, new_sl, C.PREFIX):
                    t['sl'], t['breakeven'] = new_sl, True
                    self._dirty = True
                    print(f"🔒 [LIVE] {sym}: stop to breakeven {new_sl:.6g}")

            if s * (px - t['sl']) <= 0:
                self._market_close(sym, t, contracts, 'stop_soft')
            elif s * (px - t['tp']) >= 0:
                self._market_close(sym, t, contracts, 'tp_soft')
            elif self.now() >= t['max_hold_until']:
                self._market_close(sym, t, contracts, 'time')

    def _market_close(self, sym, t, contracts, reason):
        try:
            cancel_orders(self.ex, fetch_all_open_orders(self.ex, sym), C.PREFIX)
        except Exception:
            pass
        try:
            self.ex.create_order(sym, 'market', 'sell' if t['side'] == 'long' else 'buy', contracts, None,
                                 {'reduceOnly': True, 'marginMode': C.MARGIN_MODE})
        except Exception as e:
            print(f"⚠️ [LIVE] close {sym} failed: {e}")
            return
        self._close_record(sym, t, self.price(sym) or t['entry'], reason, contracts)
