"""One-off check, no orders: runs the real scan pipeline (data -> Kronos -> decision) for a few coins.

  python selftest.py                      # BTC, ETH, SOL
  python selftest.py PEPE/USDT:USDT       # specific symbols
Safe to run from the Render Shell while the worker runs: every step waits for the worker's current job
(shared job lock), so the two never use memory at the same time. Writes no plans or logs.
"""
import os
import subprocess
import sys
import time

import config as C
import mem
from joblock import job_lock

HERE = os.path.dirname(os.path.abspath(__file__))


def step(args, label):
    t0 = time.time()
    with job_lock():
        rc = subprocess.run([sys.executable] + args, cwd=HERE).returncode
    print(f"--- {label}: exit {rc}, {time.time() - t0:.0f}s")
    return rc


def main(symbols):
    syms = ','.join(symbols)
    print(f"model {C.KRONOS_MODEL} | {C.KRONOS_SAMPLES} paths x {C.KRONOS_PRED_LEN} candles, "
          f"batch {C.KRONOS_BATCH}, lookback {C.KRONOS_LOOKBACK}")
    if step(['scan_job.py', 'prepare', '--symbols', syms], 'prepare') != 0:
        return
    work = os.path.join(HERE, 'scan_work')
    if C.USE_KRONOS:
        rc = step(['kronos_infer.py', os.path.join(work, 'in.npz'), os.path.join(work, 'out.npz')], 'Kronos')
        if rc in (-9, 137):
            print("💥 Kronos was killed: out of memory. Lower KRONOS_BATCH, KRONOS_LOOKBACK or KRONOS_SAMPLES.")
    step(['scan_job.py', 'decide', '--symbols', syms], 'decide')
    print(f"\n{mem.report()}")
    print("Healthy if: every coin shows a P(up) value, no 'forecast looks broken' warnings, and the container "
          "peak stays well under 512 MB.")


if __name__ == '__main__':
    main(sys.argv[1:] or ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'])
