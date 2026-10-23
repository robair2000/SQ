"""Kronos Sniper: Render background worker.

  scanner thread : every 4h close -> setups (squeeze / mean reversion) -> Kronos sampled forecast
                   -> plan with a limit entry at the predicted low (long) / high (short)
  main loop      : every TRADER_LOOP_SECONDS -> place / manage / expire entries, protect and manage positions

Start: python worker.py
"""
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt

import config as C
import scanner
from kronos_engine import load_forecaster
from trader import Trader

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

wake = threading.Event()
state = {'forecaster': None, 'last_scan': None, 'plans': 0}


def make_exchanges():
    data_ex = ccxt.kucoinfutures({'enableRateLimit': True})            # scanner thread (public data)
    trade_ex = ccxt.kucoinfutures({                                    # main thread (orders)
        'apiKey': os.getenv('KUCOIN_API_KEY', ''),
        'secret': os.getenv('KUCOIN_API_SECRET', ''),
        'password': os.getenv('KUCOIN_API_PASSWORD', ''),
        'enableRateLimit': True,
    })
    return data_ex, trade_ex


def get_forecaster():
    if state['forecaster'] is None:
        try:
            state['forecaster'] = load_forecaster()
        except Exception as e:
            print(f"❌ [SCAN] forecaster could not be loaded ({e}). No new plans until it loads; "
                  f"open positions are still managed.")
    return state['forecaster']


def scanner_loop(data_ex, trader):
    def on_plan(plan):
        trader.submit_plan(plan)
        wake.set()

    while True:
        try:
            fc = get_forecaster()
            if fc is not None:
                print("🔍 [SCAN] starting scan...")
                plans = scanner.run_scan(data_ex, fc, on_plan=on_plan)
                state['last_scan'], state['plans'] = time.time(), len(plans)
        except Exception as e:
            print(f"⚠️ [SCAN] scan loop error: {e}")
        wait = scanner.seconds_until_next_scan()
        print(f"💤 [SCAN] next scan in {int(wait // 60)} min")
        time.sleep(wait)


class Health(BaseHTTPRequestHandler):
    trader = None

    def do_GET(self):
        t = self.trader
        body = (f"Kronos Sniper | DRY_RUN={C.DRY_RUN} | pending={len(t.pending)} active={len(t.active)} "
                f"| last scan={time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(state['last_scan'])) if state['last_scan'] else '-'} "
                f"| paper={t.paper}\n")
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(body.encode())

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    mode = 'PAPER (DRY_RUN)' if C.DRY_RUN else 'LIVE'
    print(f"🚀 Kronos Sniper starting in {mode} mode | margin {C.TRADE_AMOUNT_USDT} USDT/trade, "
          f"max {C.MAX_CONCURRENT_POSITIONS} positions, Kronos {C.KRONOS_MODEL} "
          f"({C.KRONOS_SAMPLES} paths x {C.KRONOS_PRED_LEN} candles, min P {C.KRONOS_MIN_PROB})")
    if not C.DRY_RUN and not os.getenv('KUCOIN_API_KEY'):
        print("❌ DRY_RUN=false but KUCOIN_API_KEY is empty. Refusing to start.")
        sys.exit(1)

    data_ex, trade_ex = make_exchanges()
    trader = Trader(trade_ex)
    trader.load_status()

    if os.environ.get('PORT') or os.environ.get('ENABLE_HEALTH_SERVER', '').lower() in ('1', 'true', 'yes'):
        Health.trader = trader
        port = int(os.environ.get('PORT', 10000))
        threading.Thread(target=lambda: HTTPServer(('0.0.0.0', port), Health).serve_forever(), daemon=True).start()

    scan_thread = threading.Thread(target=scanner_loop, args=(data_ex, trader), name='scanner', daemon=True)
    scan_thread.start()

    while True:
        try:
            if not scan_thread.is_alive():
                print("⚠️ scanner thread died, restarting")
                scan_thread = threading.Thread(target=scanner_loop, args=(data_ex, trader), name='scanner', daemon=True)
                scan_thread.start()
            trader.step()
        except Exception as e:
            print(f"⚠️ [TRADER] loop error: {e}")
        wake.wait(C.TRADER_LOOP_SECONDS)
        wake.clear()


if __name__ == '__main__':
    main()
