"""Kronos Sniper supervisor: Render background worker (python worker.py). Standard library only (~15 MB).

It never loads ccxt, pandas or torch itself. Every piece of work is a short-lived process, run one at a
time, so peak memory = the largest single job instead of everything at once (fits 512 MB):

  every 4h close + SCAN_DELAY_SECONDS:
      scan_job.py prepare      ranking, 4h data, setups          (~200 MB, ~2-3 min)
      kronos_infer.py          torch + Kronos-mini forecasts     (~300 MB, a few min)
      scan_job.py decide       plans + forecast log              (~150 MB, seconds)
  every TRADER_LOOP_SECONDS:
      trade_job.py             place / manage entries and positions (~150 MB, seconds)

While Kronos runs the trader pauses for a few minutes: live positions are protected by their exchange
stop-loss/take-profit, and paper mode replays the missed 1m candles on its next step.
"""
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import config as C
import mem
from joblock import job_lock

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'scan_work')
state = {'last_scan': None, 'started': time.time()}

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def run(args, timeout, label):
    """Run one job (under the job lock). Returns its exit code (None on timeout)."""
    t0 = time.time()
    with job_lock():
        try:
            rc = subprocess.run([sys.executable] + args, cwd=HERE, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"⏱️ {label} timed out after {timeout}s and was stopped")
            return None
    if rc in (-9, 137):
        print(f"💥 {label} was killed: out of memory. Lower KRONOS_BATCH / KRONOS_LOOKBACK / KRONOS_SAMPLES "
              f"or SCAN_TOP_N.")
    elif rc != 0:
        print(f"⚠️ {label} exited with code {rc}")
    if label != 'trader':
        print(f"   {label} finished in {time.time() - t0:.0f}s")
    return rc


def scan_cycle():
    print("🔍 [SCAN] starting scan...")
    if run(['scan_job.py', 'prepare'], 1800, 'scan prepare') != 0:
        return
    kin, kout = os.path.join(WORK, 'in.npz'), os.path.join(WORK, 'out.npz')
    if C.USE_KRONOS and os.path.exists(kin):
        run(['kronos_infer.py', kin, kout], C.KRONOS_TIMEOUT, 'Kronos')
    run(['scan_job.py', 'decide'], 600, 'scan decide')
    state['last_scan'] = time.time()
    print(f"📏 {mem.report()}")


def next_scan_time(now):
    next_close = (int(now // C.CANDLE_SECONDS) + 1) * C.CANDLE_SECONDS
    return next_close + C.SCAN_DELAY_SECONDS


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        last = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(state['last_scan'])) if state['last_scan'] else '-'
        body = f"Kronos Sniper | DRY_RUN={C.DRY_RUN} | last scan {last} | {mem.report()}\n"
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
          f"max {C.MAX_CONCURRENT_POSITIONS} positions, Kronos {C.KRONOS_MODEL if C.USE_KRONOS else 'OFF (bootstrap)'} "
          f"({C.KRONOS_SAMPLES} paths x {C.KRONOS_PRED_LEN} candles, min P {C.KRONOS_MIN_PROB})")
    if not C.DRY_RUN and not os.getenv('KUCOIN_API_KEY'):
        print("❌ DRY_RUN=false but KUCOIN_API_KEY is empty. Refusing to start.")
        sys.exit(1)
    if os.environ.get('PORT') or os.environ.get('ENABLE_HEALTH_SERVER', '').lower() in ('1', 'true', 'yes'):
        port = int(os.environ.get('PORT', 10000))
        threading.Thread(target=lambda: HTTPServer(('0.0.0.0', port), Health).serve_forever(), daemon=True).start()

    run(['trade_job.py'], 600, 'trader')            # restore state / protect positions first
    next_scan = time.time()                            # scan right away after a (re)start
    last_beat = time.time()
    while True:
        try:
            if time.time() >= next_scan:
                scan_cycle()
                next_scan = next_scan_time(time.time())
                print(f"💤 [SCAN] next scan at {time.strftime('%H:%M UTC', time.gmtime(next_scan))}")
            run(['trade_job.py'], 600, 'trader')
            if time.time() - last_beat >= 3600:
                last_beat = time.time()
                print(f"💓 alive | {mem.report()}")
        except Exception as e:
            print(f"⚠️ supervisor loop error: {e}")
        time.sleep(max(1.0, min(C.TRADER_LOOP_SECONDS, next_scan - time.time())))


if __name__ == '__main__':
    main()
